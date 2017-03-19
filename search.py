import numpy as np
import cv2
from common import showImages, color_convert
from feature import Feature

class Search:

    def __init__(self, featureEx, trainer, start_size, end_size, cell_size,
                 overlap_factor=0.125, reduction_factor=0.75, scaling_factor=0.5, sample_shape=(64, 64)):
        self.featureEx = featureEx
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

        def calcScaleCurr(scale):
            #round to nearest cell size
            scaleCurr = int(cellSize * round(float(scale)/cellSize))
            return scaleCurr

        sampleSize = self.featureEx.sample_size
        scaleCurr = calcScaleCurr(scaleStart)

        subimg = img[region[0][1]:region[1][1],region[0][0]:region[1][0]].copy()

        while scaleCurr >= scaleEnd:

            print(scaleCurr)

            windows = Search.slide_window(subimg, xy_window=(scaleCurr, scaleCurr), xy_overlap=(overlap, overlap))
            windows = np.array(windows)
            #dataset = Search.extract_pixels_from_windows(subimg, windows, sample_shape=self.sample_shape)
            cache = None
            # scale image down based on orignal scale and current scale
            xDim = subimg.shape[1]
            yDim = subimg.shape[0]
            scaledImg = cv2.resize(subimg, (int(sampleSize[0]/scaleCurr * xDim), int(sampleSize[1]/scaleCurr * yDim)))
            cellsStep = int(round(float((1. - overlap) * sampleSize[0])/cellSize))

            dataset, cache = self.featureEx.extractPatches(scaledImg, cellsStep, cache=cache)

            assert(len(dataset) == len(windows))

            #X, labels = self.trainer.prepare(data=dataset)
            #pred = self.trainer.predict(X)
            pred = self.trainer.predict(dataset)
            y = pred['labels']
            scores = pred['scores']

            vals = np.where(y == 1)[0]
            boxes = windows[vals] + region[0]

            results_boxes.extend(boxes)
            results_scores.extend(scores[vals])

            scaleCurr = float(scaleCurr) * self.reduction_factor
            scaleCurr = calcScaleCurr(scaleCurr)


        return results_boxes, results_scores
