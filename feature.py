import matplotlib.image as mpimg
import numpy as np
import sklearn
import sklearn.preprocessing as skprocess
import skimage.feature as skimg
import cv2

class Feature:

    def __init__(self, hist_bins=None, spatial_size=None, useMeanAndStd=False,
                 cspaces=['RGB'], useHog=False, hog_bins=13, hog_cell_size=8, hog_block_size=2):
        self.scaler = skprocess.StandardScaler()
        self.cspaces=cspaces
        self.hist_bins = hist_bins
        self.spatial_size = spatial_size
        self.useMeanAndStd = useMeanAndStd
        self.useHog=useHog
        self.hog_bins = hog_bins
        self.hog_cell_size = hog_cell_size
        self.hog_block_size = hog_block_size

    @staticmethod
    def color_hist(img, nbins=32, bins_range=(0, 1.0)):
        # Compute the histogram of the RGB channels separately
        rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Generating bin centers
        bin_edges = rhist[1]
        bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return rhist, ghist, bhist, bin_centers, hist_features

    # Define a function to compute binned color features
    @staticmethod
    def bin_spatial(img, size=(64, 64)):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features

    @staticmethod
    def mean_and_std(img):
        mean = np.mean(img)
        std = np.std(img)
        return mean, std

    @staticmethod
    def hog(img, orient_bins=13, cell_size=8, block_size=2, visualize=False, useFeatureVector=True):
        hog = skimg.hog(img, orientations=orient_bins, pixels_per_cell=(cell_size, cell_size), cells_per_block=(block_size, block_size), visualise=visualize, feature_vector=useFeatureVector)
        return hog

    def extract_hog(img, orient_bins=13, cell_size=8, block_size=2, useFeatureVector=True):
        hog_0 = Feature.hog(img[:,:,0], orient_bins=orient_bins, cell_size=cell_size, block_size=block_size, useFeatureVector=useFeatureVector)
        hog_hist_0 = np.sum(np.array(hog_0).reshape((-1,orient_bins)), axis=0)
        hog_1 = Feature.hog(img[:,:,1], orient_bins=orient_bins, cell_size=cell_size, block_size=block_size, useFeatureVector=useFeatureVector)
        hog_hist_1 = np.sum(np.array(hog_1).reshape((-1,orient_bins)), axis=0)
        hog_2 = Feature.hog(img[:,:,2], orient_bins=orient_bins, cell_size=cell_size, block_size=block_size, useFeatureVector=useFeatureVector)
        hog_hist_2 = np.sum(np.array(hog_2).reshape((-1,orient_bins)), axis=0)

        hog = np.concatenate((hog_0, hog_1, hog_2))
        hog_hist = np.concatenate((hog_hist_0, hog_hist_1, hog_hist_2))

        return hog, hog_hist, hog_0, hog_1, hog_2

    @staticmethod
    def extract_features_from_image(img,
                            hist_bins=None, spatial_size=None, useMeanAndStd=False, useHog=False,
                            hog_bins=13, hog_cell_size=8, hog_block_size=2):

        feature_vec = []

        if spatial_size is not None:
            # Apply bin_spatial() to get spatial color features
            spatial_features = Feature.bin_spatial(img, size=spatial_size)
            feature_vec = np.concatenate((feature_vec, spatial_features))

            if useMeanAndStd:
                spatial_mean_std = np.ravel([
                    Feature.mean_and_std(spatial_features.reshape((spatial_size[0], spatial_size[1], 3))[:,:,0]),
                    Feature.mean_and_std(spatial_features.reshape((spatial_size[0], spatial_size[1], 3))[:,:,1]),
                    Feature.mean_and_std(spatial_features.reshape((spatial_size[0], spatial_size[1], 3))[:,:,2])
                ])

                feature_vec = np.concatenate((feature_vec, spatial_mean_std))

        if hist_bins is not None:
            # Apply color_hist() also with a color space option now
            color_hist = Feature.color_hist(img, nbins=hist_bins, bins_range=(0.0, 1.0))
            hist_features = color_hist[4]
            feature_vec = np.concatenate((feature_vec, hist_features))

        if useHog:

            hog_features, hog_hist, hog0, hog1, hog2 = Feature.extract_hog(img,
                orient_bins=hog_bins, cell_size=hog_cell_size, block_size=hog_block_size,
                useFeatureVector=True
            )
            feature_vec = np.concatenate((feature_vec, hog_features, hog_hist))

            if useMeanAndStd:
                hog_mean_std = np.ravel([
                    Feature.mean_and_std(hog0),
                    Feature.mean_and_std(hog1),
                    Feature.mean_and_std(hog2)
                ])

                feature_vec = np.concatenate((feature_vec, hog_mean_std))

        return feature_vec

    # Define a function to extract features from a list of images
    @staticmethod
    def extract_features(imgs, cspaces=['RGB'],
                            hist_bins=None, spatial_size=None, useMeanAndStd=False, useHog=False,
                            hog_bins=13, hog_cell_size=8, hog_block_size=2, flip=False):
        # Create a list to append feature vectors to
        features = []
        spatial_features = None
        color_hist = None
        hog_hist = None
        hist_features = None
        hog_features = None
        hog_mean_std = None
        spatial_mean_std = None

        # Iterate through the list of images
        for image in imgs:

            feature_images = []

            for cspace in cspaces:
                # apply color conversion if other than 'RGB'
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
                feature_images.append(feature_image)

            for feature_image in feature_images:
                feature_vec = Feature.extract_features_from_image(feature_image,
                    hist_bins=hist_bins, spatial_size=spatial_size,
                    useMeanAndStd=useMeanAndStd, useHog=useHog,
                    hog_bins=hog_bins, hog_cell_size=hog_cell_size, hog_block_size=hog_block_size)

            # Append the new feature vector to the features list
            features.append(feature_vec)

            if flip:
                for feature_image in feature_images:
                    feature_vec = Feature.extract_features_from_image(np.fliplr(feature_image),
                        hist_bins=hist_bins, spatial_size=spatial_size,
                        useMeanAndStd=useMeanAndStd, useHog=useHog,
                        hog_bins=hog_bins, hog_cell_size=hog_cell_size, hog_block_size=hog_block_size)
                # Append the new feature vector to the features list
                features.append(feature_vec)

        # Return list of feature vectors
        return np.array(features)

    def extract(self, imgPaths, flip=False):
        features = Feature.extract_features(imgPaths, self.cspaces,
            hist_bins=self.hist_bins, spatial_size=self.spatial_size,
            useMeanAndStd=self.useMeanAndStd,
            useHog=self.useHog, hog_bins=self.hog_bins, flip=flip,
            hog_cell_size=self.hog_cell_size, hog_block_size=self.hog_block_size)
        return features

    def transform(self, features, standardize='fit'):
        if standardize == 'fit':
            self.scaler.fit(features)
        return self.scaler.transform(features)
