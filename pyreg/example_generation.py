"""
Module to create example images to test image registrations
"""

# TODO: convert everything to simple itk and its image format (including spacing)
# for now, simply get the 3D versions up and running

from abc import ABCMeta, abstractmethod
import numpy as np
import utils

class CreateExample(object):
    __metaclass__ = ABCMeta

    def __init__(self,sz):
        self.dim = len(sz)
        self.sz = sz

    @abstractmethod
    def create_image_pair(self,params):
        """
        Abstract method to create example image pairs
        :param params: dictionary which contains parameters to create the images 
        :return: will return two images
        """
        pass

class CreateSquares(CreateExample):
    """
    Class to create two example squares in arbitrary dimension as registration test cases
    """
    def __init__(self,sz):
        super(CreateSquares, self).__init__(sz)

    def create_image_pair(self,params=None):
        if params is None:
            params = dict() # simply set it to an empty dictionary

        I0 = np.zeros(self.sz, dtype='float32')
        I1 = np.zeros(self.sz, dtype='float32')
        # get parameters and replace with defaults if necessary
        len_s = utils.getpar( params, 'len_s', self.sz.min()/6 ) # half of side-length for small square
        len_l = utils.getpar( params, 'len_l', self.sz.min()/3) # half of side-length for large square
        c = self.sz/2 # center coordinates
        # create small and large squares
        if self.dim==1:
            I0[c[0]-len_s:c[0]+len_s]=1
            I1[c[0]-len_l:c[0]+len_l]=1
        elif self.dim==2:
            I0[c[0]-len_s:c[0]+len_s, c[1]-len_s:c[1]+len_s] = 1
            I1[c[0]-len_l:c[0]+len_l, c[1]-len_l:c[1]+len_l] = 1
        elif self.dim==3:
            I0[c[0] - len_s:c[0] + len_s, c[1] - len_s:c[1] + len_s, c[2]-len_s:c[2]+len_s] = 1
            I1[c[0] - len_l:c[0] + len_l, c[1] - len_l:c[1] + len_l, c[2]-len_l:c[2]+len_l] = 1
        else:
            raise ValueError('Square examples only supported in dimensions 1-3.')

        return I0,I1
