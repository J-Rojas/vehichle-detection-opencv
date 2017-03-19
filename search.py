import numpy as np
import cv2
from common import showImages

class Search:

    def __init__(self, trainer, start_size, end_size, cell_size,
                 overlap_factor=0.125, reduction_factor=0.75, scaling_factor=0.5, sample_shape=(64, 64)):
        self.trainer = trainer
        self.start_size = start_size
        self.end_size = end_size
        self.cell_size = cell_size
        self.overlap_factor = overlap_factor
        self.reduction_factor = reduction_factor
        self.scaling_factor = scaling_factor
        self.sample_shape = sample_shape

    @staticmethod
    def draw_boxes(img, bboxes, color=(255, 255, 255), alpha=.25, alphas=None, thick=6):
        # Make a copy of the image
        imcopy = np.array(img * 255, dtype=np.uint8)
        imalpha = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        # Iterate through the bounding boxes
        if alphas is None:
            alphas = []
            for i in range(len(bboxes)):
                alphas.append(alpha)
        for bbox, alp in zip(bboxes, alphas):
            imtmp = np.zeros_like(imcopy)
            bboxCv = ((bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]))
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imtmp, bboxCv[0], bboxCv[1], color, thick)
            imalpha[np.sum(imtmp, axis=2) > 0] += alp

        # Return the image copy with boxes layer drawn on top drawn
        imalpha = np.clip(imalpha, 0., 1.)
        imalpha = np.dstack((imalpha, imalpha, imalpha))
        #imalpha = np.ones_like(img, dtype=np.float32)
        return np.uint8(imcopy * (1 - imalpha) + color * imalpha)

    @staticmethod
    def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    # extract features using hog sub-sampling and make predictions
    @staticmethod
    def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

        draw_img = np.copy(img)
        img = img.astype(np.float32)/255

        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1
        nfeat_per_block = orient*cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

        return draw_img

    @staticmethod
    def extract_pixels_from_windows(img, windows, sample_shape=(64, 64)):

        subimgs = []

        for window in windows:
            #3) Extract the test window from original image
            subimg = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], sample_shape)
            subimgs.append(subimg)

        return subimgs

    def search(self, img, region):

        results_boxes = []
        results_scores = []

        # how many scales to generate? assumes square scales
        scaleStart = self.start_size
        scaleEnd = self.end_size
        cellSize = self.cell_size
        overlap = self.overlap_factor

        scaleCurr = scaleStart

        subimg = img[region[0][1]:region[1][1],region[0][0]:region[1][0]].copy()

        while scaleCurr >= scaleEnd:

            windows = Search.slide_window(subimg, xy_window=(scaleCurr, scaleCurr), xy_overlap=(overlap, overlap))
            windows = np.array(windows)
            dataset = Search.extract_pixels_from_windows(subimg, windows, sample_shape=self.sample_shape)

            X, labels = self.trainer.prepare(data=dataset)
            pred = self.trainer.predict(X)
            y = pred['labels']
            scores = pred['scores']

            vals = np.where(y == 1)[0]
            boxes = windows[vals] + region[0]

            results_boxes.extend(boxes)
            results_scores.extend(scores[vals])

            scaleCurr = float(scaleCurr) * self.reduction_factor
            #round to nearest cell size
            scaleCurr = int(cellSize * round(float(scaleCurr)/cellSize))

        return results_boxes, results_scores
