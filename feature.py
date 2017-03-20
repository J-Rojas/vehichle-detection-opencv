import matplotlib.image as mpimg
import numpy as np
import sklearn
import skimage.feature as skimg
import cv2
from common import color_convert

class Feature:

    def __init__(self, sample_size=(64, 64), hist_bins=None, spatial_size=None, useMeanAndStd=False,
                 cspaces=['RGB'], useHog=False, hog_bins=13, hog_cell_size=8, hog_block_size=2):
        self.cspaces=cspaces
        self.sample_size = sample_size
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

    @staticmethod
    def extract_hog(img, orient_bins=13, cell_size=8, block_size=2, useFeatureVector=True):
        hog_0 = Feature.hog(img[:,:,0], orient_bins=orient_bins, cell_size=cell_size, block_size=block_size, useFeatureVector=useFeatureVector)
        hog_1 = Feature.hog(img[:,:,1], orient_bins=orient_bins, cell_size=cell_size, block_size=block_size, useFeatureVector=useFeatureVector)
        hog_2 = Feature.hog(img[:,:,2], orient_bins=orient_bins, cell_size=cell_size, block_size=block_size, useFeatureVector=useFeatureVector)
        hog = np.concatenate((hog_0, hog_1, hog_2))
        return hog, hog_0, hog_1, hog_2

    @staticmethod
    def extract_hog_hist(hog_0, hog_1, hog_2, orient_bins=13):
        hog_hist_0 = np.sum(np.array(hog_0).reshape((-1,orient_bins)), axis=0)
        hog_hist_1 = np.sum(np.array(hog_1).reshape((-1,orient_bins)), axis=0)
        hog_hist_2 = np.sum(np.array(hog_2).reshape((-1,orient_bins)), axis=0)
        hog_hist = np.concatenate((hog_hist_0, hog_hist_1, hog_hist_2))
        return hog_hist


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

            hog_features, hog0, hog1, hog2 = Feature.extract_hog(img,
                orient_bins=hog_bins, cell_size=hog_cell_size, block_size=hog_block_size,
                useFeatureVector=True
            )
            hog_hist = Feature.extract_hog_hist(hog0, hog1, hog2, orient_bins=hog_bins)

            #print('Size of hog_features {}, size of hog_hist {}'.format(len(hog_features), len(hog_hist)))

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
            feature_vec = []

            for cspace in cspaces:
                # apply color conversion if other than 'RGB'
                feature_image = color_convert(image, cspace)
                feature_images.append(feature_image)

            for feature_image in feature_images:
                vec = Feature.extract_features_from_image(feature_image,
                    hist_bins=hist_bins, spatial_size=spatial_size,
                    useMeanAndStd=useMeanAndStd, useHog=useHog,
                    hog_bins=hog_bins, hog_cell_size=hog_cell_size, hog_block_size=hog_block_size)
                feature_vec = np.concatenate((feature_vec, vec))

            # Append the new feature vector to the features list
            features.append(feature_vec)

            if flip:
                feature_vec = []

                for feature_image in feature_images:
                    vec = Feature.extract_features_from_image(np.fliplr(feature_image),
                        hist_bins=hist_bins, spatial_size=spatial_size,
                        useMeanAndStd=useMeanAndStd, useHog=useHog,
                        hog_bins=hog_bins, hog_cell_size=hog_cell_size, hog_block_size=hog_block_size)
                    feature_vec = np.concatenate((feature_vec, vec))

                # Append the new feature vector to the features list
                features.append(feature_vec)

        # Return list of feature vectors
        return np.array(features)

    # extract features using hog sub-sampling and make predictions
    def extractPatches(self, img, cells_per_step, cache=None):

        pix_per_cell = self.hog_cell_size
        orient = self.hog_bins
        cell_per_block = self.hog_block_size
        window = self.sample_size[0]

        images = cache['images'] if cache is not None else []
        hogs = cache['hogs'] if cache is not None else []

        for cspace in self.cspaces:
            if cache is None:
                image = color_convert(img, cspace)
                images.append(image)
                # Compute individual channel HOG features for the entire image
                hog = Feature.extract_hog(image,
                    orient_bins=orient, cell_size=pix_per_cell, block_size=cell_per_block,
                    useFeatureVector=False
                )
                hogs.append(hog)

        # Define blocks and steps as above
        nxcells = (images[0].shape[1] // pix_per_cell)
        nycells = (images[0].shape[0] // pix_per_cell)
        nfeat_per_block = orient*cell_per_block**2

        ncells_per_window = (window // pix_per_cell)
        nblocks_per_window = ncells_per_window - cell_per_block + 1
        nxsteps = (nxcells - ncells_per_window + 1) // cells_per_step
        nysteps = (nycells - ncells_per_window + 1) // cells_per_step

        results = []

        for yb in range(nysteps):
            for xb in range(nxsteps):

                feature_vec = []

                for feature_image, hog in zip(images, hogs):
                    ypos = yb*cells_per_step
                    xpos = xb*cells_per_step

                    xleft = xpos*pix_per_cell
                    ytop = ypos*pix_per_cell

                    # Extract the other features
                    subimg = feature_image[ytop:ytop+window, xleft:xleft+window]

                    feature_vec = np.concatenate((feature_vec,
                        Feature.extract_features_from_image(subimg,
                            hist_bins=self.hist_bins, spatial_size=self.spatial_size,
                            useMeanAndStd=self.useMeanAndStd, useHog=False)
                    ))

                    if self.useHog:

                        hog1, hog2, hog3 = hog[1], hog[2], hog[3]

                        # Extract HOG for this patch
                        #print('extractng hog... ({}, {}), ({}, {})'.format(xpos, ypos, xpos+nblocks_per_window, ypos+nblocks_per_window))
                        hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                        hog_hist = Feature.extract_hog_hist(hog_feat1, hog_feat2, hog_feat3, orient_bins=orient)
                        hog_features = np.concatenate((hog_feat1, hog_feat2, hog_feat3, hog_hist))

                        #print(len(hog_feat1), ncells_per_window)
                        assert(len(hog_feat1) == nfeat_per_block * nblocks_per_window**2)

                        if self.useMeanAndStd:
                            hog_mean_std = np.ravel([
                                Feature.mean_and_std(hog_feat1),
                                Feature.mean_and_std(hog_feat2),
                                Feature.mean_and_std(hog_feat3)
                            ])

                            hog_features = np.concatenate((hog_features, hog_mean_std))

                        feature_vec = np.concatenate((feature_vec, hog_features))

                results.append(feature_vec)

        return results, cache if cache is not None else {'images': images, 'hogs': hogs}

    def extract(self, imgPaths, flip=False):
        features = Feature.extract_features(imgPaths, self.cspaces,
            hist_bins=self.hist_bins, spatial_size=self.spatial_size,
            useMeanAndStd=self.useMeanAndStd,
            useHog=self.useHog, hog_bins=self.hog_bins, flip=flip,
            hog_cell_size=self.hog_cell_size, hog_block_size=self.hog_block_size)
        return features
