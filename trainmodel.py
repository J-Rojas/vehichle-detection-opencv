import cv2
import numpy as np
import random
import os
from train import Trainer
from train import Classifier
from feature import Feature
from search import Search
from common import intersection
from common import area
from sklearn.externals import joblib

class SearchTrainer(Trainer):

    def __init__(self, featureExtractor, classifier, imgDict,
        overlap_threshold=0.75, score_threshold=2, jitter=6, stepSize=8,
        falsePositiveThreshold=1000, maxFalsePositivesPerStep=20, maxFalsePositives=2000,
    ):
        super(SearchTrainer, self).__init__(featureExtractor, classifier)
        self.search = None
        self.overlap_threshold = overlap_threshold
        self.score_threshold = score_threshold
        self.jitter = jitter
        self.imgDict = imgDict
        self.stepSize = stepSize
        self.falsePositiveThreshold = falsePositiveThreshold
        self.maxFalsePositives = maxFalsePositives
        self.maxFalsePositivesPerStep = maxFalsePositivesPerStep

    def readSamples(self, data):
        positiveSamples = {}
        for item in data:
            samples = positiveSamples.get(item['file'], None)
            if samples is None:
                samples = positiveSamples[item['file']] = []
            bbox = ((int(item['x']),int(item['y'])),(int(item['x'])+int(item['w']),int(item['y'])+int(item['h'])))
            samples.append(bbox)
        return positiveSamples

    def extractAndTest(self, region, dataTest, extract=False):
        positiveSamples = self.readSamples(dataTest)
        features_p = []
        labels_p  = []
        features_n = []
        labels_n  = []
        search = self.search

        total_window_count = 0
        false_negative_count = 0
        false_positive_count = 0

        for f, samples in positiveSamples.items():

            print(f)

            false_positives = []
            false_negatives = []

            img = self.imgDict[f]

            # test model
            results_boxes, results_scores, window_count = search.search(img, region)
            total_window_count += window_count

            heatmap = np.zeros((img.shape[0], img.shape[1]))

            # determine if hit was correct
            for bbox, score in zip(results_boxes, results_scores):
                a = area(bbox)
                for sample in samples:
                    #does it overlap within a certain margin?
                    overlap_bbox = intersection(bbox, sample)
                    overlap_area = area(overlap_bbox) if overlap_bbox is not None else 0
                    if overlap_bbox is not None and overlap_area / a >= self.overlap_threshold:
                        #print(overlap_bbox)
                        search.add_heat(heatmap, None, [overlap_bbox], [1])
                    else:
                        # false positives
                        false_positives.append([bbox, score])

            # determine if positive hit was missed
            for bbox in samples:
                #print(bbox)
                w = bbox[1][0] - bbox[0][0]
                if (w == 0):
                    continue
                #is the heatmap sufficiently 'hot' for the bounding area?
                bbox_area = heatmap[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]]
                cool = bbox_area[bbox_area < self.score_threshold]
                if len(cool) > 0:
                    #false negative
                    false_negatives.append(sample)

            false_negative_count += len(false_negatives)
            false_positive_count += len(false_positives)

            if extract:
                if len(false_negatives) > 0:
                    # gather false negative features
                    false_negative_features, false_negative_windows = self.extract_features_from_bboxes(img, false_negatives)
                    features_p.extend(false_negative_features)
                    labels_p.extend(np.full((len(false_negative_features)), 1))

                if len(false_positives) > 0:
                    # this can be very large... only add the ones with high scores
                    false_positives = np.array(false_positives)
                    sorted_false_positives = false_positives[np.argsort(false_positives[:,1])]

                    sorted_false_positives_scores = sorted_false_positives[:,1]
                    false_positives_to_use = sorted_false_positives[sorted_false_positives_scores > self.falsePositiveThreshold]
                    false_positives_to_use = np.flipud(false_positives_to_use)[:self.maxFalsePositivesPerStep]

                    assert(len(false_positives_to_use) <= self.maxFalsePositivesPerStep)

                    print(false_positives_to_use)

                    false_positives_to_use = false_positives_to_use[:,0]

                    # gather false positive features
                    false_positive_features, false_positive_windows = self.extract_features_from_bboxes(img, false_positives_to_use)
                    features_n.extend(false_positive_features)
                    labels_n.extend(np.full((len(false_positive_features)), 0))

        failed_count = false_positive_count + false_negative_count
        accuracy = (total_window_count - failed_count) / total_window_count if total_window_count > 0 else 0

        print('False Positive cases: {}'.format(false_positive_count))
        print('False Negative cases: {}'.format(false_negative_count))
        print('Accuracy: {:.3f}'.format(accuracy))

        return features_p, labels_p, features_n, labels_n

    def extractAndTrain(self, region, data, dataTest, filenameIn, filenameOut):

        positiveSamples = self.readSamples(data)
        features_p = []
        labels_p  = []
        features_n = []
        labels_n  = []
        search = self.search

        total_window_count = 0

        for f, samples in positiveSamples.items():

            img = self.imgDict[f]
            print(f)

            positive_features, positive_windows = self.extract_features_from_bboxes(img, samples)
            features_p.extend(positive_features)
            labels_p.extend(np.full((len(positive_features)), 1))

        self.train(region, (features_p, labels_p, features_n, labels_n), dataTest, filenameIn, filenameOut)

    def train(self, region, featureSet, dataTest, filenameIn, filenameOut):

        search = self.search
        features_p, labels_p, features_n, labels_n = featureSet

        X, y = [], []

        if len(features_n) > 0 or len(features_p) > 0:
            print('Training additional cases...')
            print('  additional positive set size: {}'.format(len(features_p)))
            print('  additional negative set size: {}'.format(len(features_n)))

            X.extend(features_n)
            y.extend(labels_n)
            X.extend(features_p)
            y.extend(labels_p)

            # load previous training data
            if (os.path.isfile(filenameIn)):
                X2, y2 = joblib.load(filenameIn)
                X.extend(X2)
                y.extend(y2)

            y_ = np.array(y)
            ones = y_[y_ == 1]
            zeros = y_[y_ == 0]

            print('Total training set size: {}'.format(len(y)))
            print('  positive set size: {}'.format(len(ones)))
            print('  negative set size: {}'.format(len(zeros)))

            groups = ((X, y), ([], []))

            super(SearchTrainer, self).train(groups, verbose=False)

            # test
            self.extractAndTest(region, dataTest, extract=False)

            joblib.dump((X, y), filenameOut)

    def random_jitter(self, bboxes, r, n = 4):
        for bbox in bboxes:
            for i in range(n):
                # randomly jitter window
                randx = int(random.randint(-r, r))
                randy = int(random.randint(-r, r))
                bbox[:,0] += randy
                bbox[:,1] += randx
                if (bbox[0][0] < 0):
                    bbox[:,0] -= bbox[0][0]
                if (bbox[0][1] < 0):
                    bbox[:,1] -= bbox[0][1]

        return bboxes

    def extract_negative_features_from_bboxes(self, img, bboxes):
        pass


    def extract_features_from_bboxes(self, img, bboxes):

        features = []
        sample_size = self.fEx.sample_size
        stepSize = self.stepSize
        retboxes = []

        for bbox in bboxes:

            p1 = bbox[0]
            p2 = bbox[1]
            w = p2[0] - p1[0]
            h = p2[1] - p1[1]

            if w == 0 or h == 0:
                continue

            ratio = w / h

            sf = float(sample_size[1]) / h
            rw = int(sf * img.shape[1])
            rh = int(sf * img.shape[0])
            rdim = int(max(sf * w, sf * h))

            sample_size = self.fEx.sample_size

            resized_img = cv2.resize(img, (rw, rh)) if (rw, rh) != sample_size else img

            #print(sf, rdim, rw, rh)

            overlap = 1. - float(stepSize) / sample_size[0]

            nx = [int(p1[0]*sf), int(p2[0]*sf)]
            ny = [int(p1[1]*sf), int(p2[1]*sf)]
            if (nx[1] - nx[0] < sample_size[0]):
                nx[1] = (nx[0] + sample_size[0])

            if ratio >= .8:

                #extract frames across a horizontal sliding window
                if (rw, rh) != sample_size:
                    windows = Search.slide_window(
                        resized_img,
                        x_start_stop=nx,
                        y_start_stop=ny,
                        xy_window=sample_size,
                        xy_overlap=(overlap, overlap)
                    )

                    windows = np.array(windows)
                    #jitter frames

                    if (len(windows) == 0):
                        print(nx, ny, overlap)

                    windows = np.concatenate((windows, self.random_jitter(windows, self.jitter)))
                    dataset = Search.extract_pixels_from_windows(resized_img, windows, sample_shape=sample_size)
                    assert(len(dataset) > 0)
                    X, labels = self.prepare(data=dataset, augment={'flip': True})

                    retboxes.extend(windows)
                    features.extend(X)

                #extract a square frame
                imgpatch = np.array(((nx[0],ny[0]),(nx[0]+rdim,ny[0]+rdim)))

                windows = np.concatenate(([imgpatch], self.random_jitter([imgpatch], self.jitter)))
                dataset = Search.extract_pixels_from_windows(resized_img, windows, sample_shape=sample_size)

                X, labels = self.prepare(data=dataset, augment={'flip': True})

                features.extend(X)
                retboxes.extend(windows)

            else:

                print('Skipping bbox {} with ratio {}'.format(bbox, ratio))

        return features, bboxes
