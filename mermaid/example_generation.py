"""
Package to create example images to test the image registration algorithms
"""
from __future__ import absolute_import

from builtins import object
from abc import ABCMeta, abstractmethod
import numpy as np
from . import fileio
from future.utils import with_metaclass

class CreateExample(with_metaclass(ABCMeta, object)):
    """
    Abstract base class.
    """

    def __init__(self,dim):
        """
        Constructor
        
        :param dim: Desired dimension of the example image 
        """
        self.dim = dim
        """Spatial dimension"""

    @abstractmethod
    def create_image_pair(self,sz=None,params=None):
        """
        Abstract method to create example image pairs
        
        :param params: Dictionary which contains parameters to create the images 
        :return: Will return two images
        """
        pass

class CreateSquares(CreateExample):
    """
    Class to create two example squares in arbitrary dimension as registration test cases
    """
    def __init__(self,dim, add_noise_to_bg=False):
        super(CreateSquares, self).__init__(dim)
        self.add_noise_to_bg = add_noise_to_bg

    def create_image_pair(self,sz,params):
        """
        Creates two square images in dimensions 1-3
        
        :param sz: Desired size, e.g., [5,10] 
        :param params: Parameter dictionary. Uses 'len_s' and 'len_l' to define the side-length of the small and 
            the large squares which will be generated 
        :return: Returns two images of squares and the spacing (I0,I1,spacing)
        """
        if not self.add_noise_to_bg:
            I0 = np.zeros(sz, dtype='float32')
            I1 = np.zeros(sz, dtype='float32')
        else:
            I0 = np.random.rand(*sz).astype(np.float32)/5.
            I1 = np.random.rand(*sz).astype(np.float32)/5.
        # get parameters and replace with defaults if necessary

        # create a new category if it does not exist
        params[('square_example_images', {}, 'Controlling the size of a nD cube')]
        len_s = params['square_example_images'][('len_s',int(sz.min()//6),'Mimimum side-length of square')]
        len_l = params['square_example_images'][('len_l',int(sz.max()//4),'Maximum side-length of square')]

        c = sz//2 # center coordinates
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

        # now transform from single-channel to multi-channel image format
        I0 = I0.reshape([1, 1] + list(I0.shape))
        I1 = I1.reshape([1, 1] + list(I1.shape))

        sz = np.array(I0.shape)
        spacing = 1. / (sz[2::] - 1)  # the first two dimensions are batch size and number of image channels

        return I0,I1,spacing




class CreateRealExampleImages(CreateExample):
    """
    Class to create two example brain images. Currently only supported in 2D
    """
    def __init__(self,dim=2,s_path=None,t_path=None):
        super(CreateRealExampleImages, self).__init__(dim)
        if s_path is None:
            self.s_path = '../mermaid_test_data/brain_slices/ws_slice.nrrd'
            self.t_path = '../mermaid_test_data/brain_slices/wt_slice.nrrd'
        else:
            self.s_path = s_path
            self.t_path = t_path
        

    def create_image_pair(self,sz=None,params=None):
        """
        Loads the two brain images using SimpleITK, normalizes them so that the 95-th percentile is as 0.95 and returns them.
        
        :param sz: Ignored 
        :param params: Ignored
        :return: Returns the two brain slices.
        """

        # create small and large squares
        if self.dim==2:
            I0,_,_,squeezed_spacing = fileio.ImageIO().read_to_nc_format(filename=self.s_path,intensity_normalize=True,squeeze_image=True)
            I1,_,_,squeezed_spacing = fileio.ImageIO().read_to_nc_format(filename=self.t_path,intensity_normalize=True,squeeze_image=True)
        else:
            raise ValueError('Real examples only supported in dimension 2 at the moment.')

        return I0,I1,squeezed_spacing
