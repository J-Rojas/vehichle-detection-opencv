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
    cspaces=['YUV', 'HLS'],
    useHog=True,
    hog_bins=9,
    hog_cell_size=8,
    hog_block_size=2
)

trainer = Trainer(featureExtractor, classifier)
search = Search(featureExtractor, trainer, 256, 74, 8, overlap_factor=0.875)

SCORE_THRESHOLD_SINGLE = 7000
SCORE_THRESHOLD_HISTORY = 20000
HEATMAP_MAX = 5

def pipeline(frame, frame_count, heatmap, heatmap_history, detect=True, car_bboxs=None):

    #convert RGB to 1.0 scale
    frame = np.array(frame, dtype=np.float32)
    frame /= 255.

    if not(retval):
        return False

    if (detect):
        windows, scores, window_count = search.search(frame, ((0, 360), (1280, 640)))
        search.add_heat(heatmap, None, windows, scores)

        heatmap[heatmap < SCORE_THRESHOLD_SINGLE] = 0

        # average the heatmaps
        heatmap_sum = np.sum(heatmap_history, axis=0)
        heatmap_sum[heatmap_sum < SCORE_THRESHOLD_HISTORY] = 0

        heatmap_sum /= np.max(heatmap) + 0.00001
        heatmap_sum[heatmap_sum < 0.3] = 0

        car_bboxs = search.filter_boxes(heatmap_sum)

    car_bbox_img = Search.draw_boxes(frame, car_bboxs, color=(0, 255, 0), alpha=0.75, thick=2)

    return car_bbox_img, car_bboxs

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
car_bboxs = None
heatmap_history = []
while True:
    retval, nextframe = video.read()
    detect = True
    heatmap = np.zeros_like(nextframe, dtype=np.float64)
    heatmap_history.append(heatmap)
    if (len(heatmap_history) > HEATMAP_MAX):
        heatmap_history.pop(0)

    frame, car_bboxs = pipeline(nextframe, i, heatmap, heatmap_history, car_bboxs=car_bboxs, detect=detect)
    print('Cars detected: ', car_bboxs)
    if frame is not False:
        print('writing frame ', i)
        video_out.write(frame)
    else:
        break
    i+=1
