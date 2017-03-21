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
    hist_bins=64,
    spatial_size=None,
    useMeanAndStd=False,
    cspaces=['YUV'],
    useHog=True,
    hog_bins=9,
    hog_cell_size=8,
    hog_block_size=2
)

trainer = Trainer(featureExtractor, classifier)
search = Search(featureExtractor, trainer, 256, 64, 8, overlap_factor=0.875)

SCORE_THRESHOLD_SINGLE = 3000
HEATMAP_MAX = 7
SCORE_THRESHOLD_HISTORY = 6000


def pipeline(frame, frame_count, heatmap, heatmap_history, detect=True, car_bboxs=None):

    #convert RGB to 1.0 scale
    frame = np.array(frame, dtype=np.float32)
    frame /= 255.

    assert(np.max(frame) <= 1.)


    zones = []

    if (detect):
        windows, scores, window_count = search.search(frame, ((0, 384), (1280, 640)))
        search.add_heat(heatmap, None, windows, scores)

        print('Heat map max = ', np.max(heatmap))

        heatmap[heatmap < SCORE_THRESHOLD_SINGLE] = 0
        heatmap_norm = heatmap / np.max(heatmap)
        heatmap[heatmap_norm < 0.2] = 0
        heatmap[:,0:430] = 0

        # average the heatmap
        heatmap_sum = np.average(heatmap_history, axis=0)

        print('Heat map sum max = ', np.max(heatmap_sum))

        heatmap_sum[heatmap_sum < SCORE_THRESHOLD_HISTORY] = 0

        heatmap_sum /= np.max(heatmap_sum) + 0.00001
        heatmap_sum[heatmap_sum < 0.2] = 0

        car_bboxs = search.filter_boxes(heatmap_sum)

        heatmap_sum = np.average(heatmap_history, axis=0)
        for bbox in car_bboxs:
            sub = heatmap[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]]
            subhist = heatmap_sum[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]]
            zones.append({'max':np.max(sub), 'avg': np.average(sub), 'maxhist':np.max(subhist), 'avghist': np.average(subhist)})


    region_h = int(frame.shape[0]/2.5)
    region_w = int(frame.shape[1]/2.5)
    region_w2 = int(region_w * 1.5)
    heatmap_sum /= np.max(heatmap_sum)
    heatmap_img = heatmap.copy()
    heatmap_img /= np.max(heatmap)
    heatmap_sum = cv2.resize(heatmap_sum, (region_w, region_h), interpolation=cv2.INTER_AREA)
    heatmap_img = cv2.resize(heatmap_img, (region_w, region_h), interpolation=cv2.INTER_AREA)

    frame[0:region_h,0:region_w] = heatmap_sum
    frame[0:region_h,region_w2:] = heatmap_img

    car_bbox_img = Search.draw_boxes(frame, car_bboxs, color=(0, 255, 0), alpha=0.75, thick=2)

    print('Heat zones max: ', zones)

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

    nextframe = cv2.cvtColor(nextframe, cv2.COLOR_BGR2RGB)

    if not(retval):
        break
    detect = True
    heatmap = np.zeros_like(nextframe, dtype=np.float64)
    heatmap_history.append(heatmap)
    if (len(heatmap_history) > HEATMAP_MAX):
        heatmap_history.pop(0)
    assert(len(heatmap_history) <= HEATMAP_MAX)

    frame, car_bboxs = pipeline(nextframe, i, heatmap, heatmap_history, car_bboxs=car_bboxs, detect=detect)
    print('Cars detected: ', car_bboxs)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if frame is not False:
        print('writing frame ', i)
        video_out.write(frame)
    else:
        break
    i+=1
