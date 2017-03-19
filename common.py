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
