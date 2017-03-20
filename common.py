import matplotlib.pyplot as plt
import cv2
import numpy as np

def showImages(images, imgs_row, imgs_col, col_titles=None, cmap=None):
    fig, axes = plt.subplots(imgs_row, imgs_col, figsize=(35, 35),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    if imgs_row == 1 and imgs_col == 1:
        axes.imshow(images[0], cmap=cmap)
    else:
        i = 0
        for ax, image in zip(axes.flat, images):
            if i < imgs_col and col_titles is not None:
                ax.set_title(col_titles[i], fontsize=50)
            ax.imshow(image, cmap=cmap)
            i += 1

    plt.show()
    plt.close()

def color_convert(image, cspace):
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            feature_image[:,:,0] = np.array(feature_image)[:,:,0] / 360.
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            feature_image[:,:,0] = np.array(feature_image)[:,:,0] / 360.
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    else: feature_image = np.copy(image)
    return feature_image

def intersection(a,b):
    x = max(a[0][0], b[0][0])
    y = max(a[0][1], b[0][1])
    w = min(a[1][0], b[1][0]) - x
    h = min(a[1][1], b[1][1]) - y
    if w<=0 or h<=0: return None
    return ((x, y), (x+w, y+h))

def area(bbox):
    w = (bbox[1][0] - bbox[0][0])
    h = (bbox[1][1] - bbox[0][1])
    area_bbox = w * h
    return area_bbox
