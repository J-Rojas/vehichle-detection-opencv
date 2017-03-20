import numpy as np
import cv2
import scipy.ndimage.measurements as sklabel
from common import showImages, color_convert, area
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

        # start the search from the bottom of the region up to increase search
        # priority based on z-distance
        #y_start = y_start_stop[1] - xy_window[1]
        #ny_pix_per_step = - ny_pix_per_step

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

        window_count = 0

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

            print(len(dataset), len(windows))
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

            window_count += len(windows)

        return results_boxes, results_scores, window_count

    # Malisiewicz et al.
    def non_max_suppression(boxes, overlapThresh):
    	# if there are no boxes, return an empty list
    	if len(boxes) == 0:
    		return []

    	# if the bounding boxes integers, convert them to floats --
    	# this is important since we'll be doing a bunch of divisions
    	if boxes.dtype.kind == "i":
    		boxes = boxes.astype("float")

    	# initialize the list of picked indexes
    	pick = []

    	# grab the coordinates of the bounding boxes
    	x1 = boxes[:,0]
    	y1 = boxes[:,1]
    	x2 = boxes[:,2]
    	y2 = boxes[:,3]

    	# compute the area of the bounding boxes and sort the bounding
    	# boxes by the bottom-right y-coordinate of the bounding box
    	area = (x2 - x1 + 1) * (y2 - y1 + 1)
    	idxs = np.argsort(y2)

    	# keep looping while some indexes still remain in the indexes
    	# list
    	while len(idxs) > 0:
    		# grab the last index in the indexes list and add the
    		# index value to the list of picked indexes
    		last = len(idxs) - 1
    		i = idxs[last]
    		pick.append(i)

    		# find the largest (x, y) coordinates for the start of
    		# the bounding box and the smallest (x, y) coordinates
    		# for the end of the bounding box
    		xx1 = np.maximum(x1[i], x1[idxs[:last]])
    		yy1 = np.maximum(y1[i], y1[idxs[:last]])
    		xx2 = np.minimum(x2[i], x2[idxs[:last]])
    		yy2 = np.minimum(y2[i], y2[idxs[:last]])

    		# compute the width and height of the bounding box
    		w = np.maximum(0, xx2 - xx1 + 1)
    		h = np.maximum(0, yy2 - yy1 + 1)

    		# compute the ratio of overlap
    		overlap = (w * h) / area[idxs[:last]]

    		# delete all indexes from the index list that have
    		idxs = np.delete(idxs, np.concatenate(([last],
    			np.where(overlap > overlapThresh)[0])))

    	# return only the bounding boxes that were picked using the
    	# integer data type
    	return boxes[pick].astype("int")

    def add_heat(self, heatmap, votes, bbox_list, scores):
        # Iterate through list of bboxes
        for box, score in zip(bbox_list, scores):
            # Add += score for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += score
            if votes is not None:
                votes[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap, votes# Iterate through list of bboxes

    def filter_boxes(self, heatmap, areaThreshold=64*64):
        object_mask, nlabels = sklabel.label(heatmap)

        boxes = []

        for item in range(1, nlabels + 1):
            vals = np.where(object_mask == item)
            bbox = (
                (np.min(vals[1]), np.min(vals[0])),
                (np.max(vals[1]), np.max(vals[0]))
            )
            a = area(bbox)
            if (a > areaThreshold):
                boxes.append(bbox)

        return boxes
