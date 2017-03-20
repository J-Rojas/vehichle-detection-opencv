import numpy as np
import cv2
import os
import sys
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from feature import Feature
from train import Trainer
from train import Classifier
from search import Search

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
trainer = Trainer(featureExtractor, classifier)
search = Search(featureExtractor, trainer, 256, 64, 8, overlap_factor=0.875)

SCORE_THRESHOLD = 9000

def pipeline(video, frame_count):

    retval, nextframe = video.read()

    #convert RGB to 1.0 scale
    nextframe = np.array(nextframe, dtype=np.float32)
    nextframe /= 255.

    if not(retval):
        return False

    windows, scores = search.search(nextframe, ((0, 360), (1280, 640)))
    heatmap = np.zeros_like(nextframe)
    votes = np.zeros_like(nextframe)
    search.add_heat(heatmap, votes, windows, scores)
    votes[votes == 0] = 1
    heatmap[heatmap < SCORE_THRESHOLD] = 0

    car_bboxs = search.filter_boxes(heatmap)
    car_bbox_img = Search.draw_boxes(nextframe, car_bboxs, color=(0, 255, 0), alpha=0.75, thick=2)

    return car_bbox_img

if len(sys.argv) < 4:
    exit('Usage: model movie_in movie_out')

model_prefix = sys.argv[1]
file1 = sys.argv[2]
file2 = sys.argv[3]

video = cv2.VideoCapture()
video.open(file1)
video_out = cv2.VideoWriter()
video_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

print(video_size, video.get(cv2.CAP_PROP_FOURCC))

if os.path.isfile(file2):
    os.remove(file2)
video_out.open(
    file2,
    cv2.VideoWriter_fourcc(*'MP42'),
    video.get(cv2.CAP_PROP_FPS),
    video_size
)

#load model
classifier.load(model_prefix)

i = 0
while True:
    frame = pipeline(video, i)
    if frame is not False:
        print('writing frame ', i)
        video_out.write(frame)
    else:
        break
    i+=1
