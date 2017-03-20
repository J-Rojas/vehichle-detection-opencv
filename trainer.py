import numpy as np
import cv2
import os
import sys
import glob
import csv
import matplotlib.image as mpimg

from trainmodel import SearchTrainer
from search import Search
from feature import Feature
from train import Classifier

if len(sys.argv) < 3:
    exit('Usage: train_csv test_csv [data_in] [data_out] [model] ')

train_data = sys.argv[1]
test_data = sys.argv[2]

dataIn = sys.argv[3] if len(sys.argv) > 3 else None
dataOut = sys.argv[4] if len(sys.argv) > 4 else None
model_prefix = sys.argv[5] if len(sys.argv) > 5 else None

print(train_data, test_data)

train_reader = csv.DictReader(open(train_data))
test_reader = csv.DictReader(open(test_data))

imgCache = {}

for row in train_reader:
    if imgCache.get(row['file'], None) is None:
        fName = row['file']
        imgCache[fName] = mpimg.imread('./data/video-frames/' + fName)
        assert(np.max(imgCache[fName]) == 1.)

for row in test_reader:
    if imgCache.get(row['file'], None) is None:
        fName = row['file']
        imgCache[fName] = mpimg.imread('./data/video-frames/' + fName)
        assert(np.max(imgCache[fName]) == 1.)

train_reader = csv.DictReader(open(train_data))
test_reader = csv.DictReader(open(test_data))

region = ((0, 360), (1280, 640))
classifier = Classifier()
featureExtractor = Feature(
    sample_size=(64,64),
    hist_bins=None,
    spatial_size=None,
    useMeanAndStd=False,
    cspaces=['HLS', 'YUV'],
    useHog=True,
    hog_bins=9,
    hog_cell_size=8,
    hog_block_size=2
)
trainer = SearchTrainer(featureExtractor, classifier, imgCache)
search = Search(featureExtractor, trainer, 200, 64, 8, overlap_factor=0.875)
trainer.search = search

model_files = glob.glob('./' + model_prefix + '.*.pkl')

if (len(model_files) < 2):
    trainer.extractAndTrain(region, train_reader, test_reader, dataIn, dataOut)
    classifier.save(model_prefix)
else:
    classifier.load(model_prefix)
    trainset = trainer.extractAndTest(region, train_reader, extract=True)
    trainer.train(region, trainset, test_reader, dataIn, dataOut)
