"""
Various methods to manipulate images
"""
from __future__ import print_function

from builtins import object
import numpy as np

class IntensityNormalizeImage(object):
    def __init__(self):
        """
        Constructor
        """
        self.default_normalization_mode = 'percentile_normalization'
        """Default intensity normalization method"""

    def max_normalization(self,I):
        # first zero out negative values
        I = I - I.min()
        np.clip(I, 0, None, out=I)
        I = I/np.max(I)
        return I

    def percentile_normalization(self,I,perc=99.):
        """
        Linearly normalized image intensities so that the 95-th percentile gets mapped to 0.95; 0 stays 0
        :param I: input image
        :param perc: desired percentile
        :return: returns the normalized image
        """
        # first zero out negative values
        I =I - I.min()
        np.clip(I, 0, None, out=I)
        # then normalize the 99th percentile
        percI = np.percentile(I, perc)
        #np.clip (I,None,percI,out=I)
        if percI == 0:
            print('Cannot normalize based on percentile; as 99-th percentile is 0. Ignoring normalization')
            return I
        else:
            I = I / percI * perc/100.
            return I

    def default_intensity_normalization(self, I):
        """
        Intensity normalizes an image using the default intensity normalization method
        :param I: input image
        :return: intensity normalized image
        """
        if self.default_normalization_mode == 'percentile_normalization':
            return self.percentile_normalization(I)
        elif self.default_normalization_mode == 'max_normalization':
            return self.max_normalization(I)
        else:
            print('ERROR: unknown normalization mode: ' + self.default_normalization_mode )
            print('ERROR: returning un-normalized image')
            return I
