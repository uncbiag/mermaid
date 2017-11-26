"""
Various methods to manipulate images
"""

import numpy as np

class IntensityNormalizeImage(object):
    def __init__(self):
        """
        Constructor
        """
        self.default_normalization_mode = 'percentile_normalization'
        """Default intensity normalization method"""

    def percentile_normalization(self,I,perc=95):
        """
        Linearly normalized image intensities so that the 95-th percentile gets mapped to 0.95; 0 stays 0
        :param I: input image
        :param perc: desired percentile
        :return: returns the normalized image
        """
        I = I / np.percentile(I, 95) * 0.95
        return I

    def defaultIntensityNormalization(self,I):
        """
        Intensity normalizes an image using the default intensity normalization method
        :param I: input image
        :return: intensity normalized image
        """
        if self.default_normalization_mode == 'percentile_normalization':
            return self.percentile_normalization(I)
        else:
            print('ERROR: unknown normalization mode: ' + self.default_normalization_mode )
            print('ERROR: returning un-normalized image')
            return I
