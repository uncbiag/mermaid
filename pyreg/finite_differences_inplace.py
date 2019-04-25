"""
*finite_difference.py* is the main package to compute finite differences in 
1D, 2D, and 3D on numpy arrays (class FD_np) and pytorch tensors (class FD_torch).
The package supports first and second order derivatives and Neumann and linear extrapolation
boundary conditions (though the latter have not been tested extensively yet).
"""
from __future__ import absolute_import

# from builtins import object
from abc import ABCMeta, abstractmethod

import torch
from torch.autograd import Variable
from .data_wrapper import MyTensor
import numpy as np
from future.utils import with_metaclass

class FD_multi_channel(with_metaclass(ABCMeta, object)):
    """
    *FD* is the abstract class for finite differences. It includes most of the actual finite difference code, 
    but requires the definition (in a derived class) of the methods *get_dimension*, *create_zero_array*, and *get_size_of_array*.
    In this way the numpy and pytorch versions can easily be derived. All the method expect BxXxYxZ format (i.e., they process a batch at a time)
    """

    def __init__(self,spacing, bcNeumannZero=True):
        """
        Constructor        
        :param spacing: 1D numpy array defining the spatial spacing, e.g., [0.1,0.1,0.1] for a 3D image  
        :param bcNeumannZero: Defines the boundary condition. If set to *True* (default) zero Neumann boundary conditions
            are imposed. If set to *False* linear extrapolation is used (this is still experimental, but may be beneficial 
            for better boundary behavior)
        """
        self.dim = spacing.size
        """spatial dimension"""
        self.spacing = np.ones(self.dim)
        """spacing"""
        self.bcNeumannZero = bcNeumannZero  # if false then linear interpolation
        self.order_smooth =True
        self.strict_bcNeumannZero = False
        """should Neumann boundary conditions be used? (otherwise linear extrapolation)"""
        if spacing.size == 1:
            self.spacing[0] = spacing[0]
        elif spacing.size == 2:
            self.spacing[0] = spacing[0]
            self.spacing[1] = spacing[1]
        elif spacing.size == 3:
            self.spacing = spacing
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')









    @abstractmethod
    def getdimension(self,I):
        """
        Abstract method to return the dimension of an input image I
        
        :param I: Input image 
        :return: Returns the dimension of the image I
        """
        pass

    @abstractmethod
    def create_zero_array(self, sz):
        """
        Abstract method to create a zero array of a given size, sz. E.g., sz=[10,2,5]
        
        :param sz: Size array
        :return: Returns a zero array of the specified size
        """
        pass

    @abstractmethod
    def get_size_of_array(self, A):
        """
        Abstract method to return the size of an array (as a vector)
        
        :param A: Input array
        :return: Returns its size (e.g., [5,10] or [3,4,6]
        """
        pass







    def zero_x_boundary(self, I):
        rx = I
        ndim = self.getdimension(I)
        if ndim <= 5:
            rx[:, :,  0] = 0.
            rx[:, :, -1] = 0.
        elif ndim == 4 + 2:
            rx[:, :, :, 0, :, :] = 0.
            rx[:, :, :, -1, :, :] = 0.
        return rx

    def interpolate_x_boundary(self,I):
        rx = I
        ndim = self.getdimension(I)
        if ndim <=5:  # 2+1,2+2,2+3
            rx[:, :, 0] = I[:, :, 1]+I[:, :, 1]-I[:, :, 2]
            rx[:, :, -1] = I[:, :, -2]+I[:, :, -2]-I[:, :, -3]
        elif ndim == 4 + 2:
            rx[:, :, :, 0, :, :] = 0.
            rx[:, :, :, -1, :, :] = 0.
        return rx

    def zero_y_boundary(self, I):
        ry = I
        ndim = self.getdimension(I)
        if ndim >= 4 and ndim <= 5:  # 2+1,2+2,2+3
            ry[:, :, :, 0] = 0.
            ry[:, :, :,-1] = 0.
        elif ndim == 4 + 2:
            ry[:, :, :, :, 0, :] = 0.
            ry[:, :, :, :,-1, :] = 0.
        return ry

    def interpolate_y_boundary(self,I):
        ry = I
        ndim = self.getdimension(I)
        if ndim>=4 and ndim <=5:  # 2+1,2+2,2+3
            ry[:, :,:, 0] = I[:, :,:, 1]+I[:, :,:, 1]-I[:, :,:, 2]
            ry[:, :,:, -1] = I[:, :,:, -2]+I[:, :,:, -2]-I[:, :,:, -3]
        elif ndim == 4 + 2:
            ry[:, :, :,:, 0, :, :] = 0.
            ry[:, :, :,:, -1, :, :] = 0.
        return ry

    def zero_z_boundary(self, I):
        rz =I
        ndim = self.getdimension(I)
        if ndim ==3+2:
            rz[:, :, :, :,0] = 0.
            rz[:, :, :, :,-1] = 0.
        elif ndim == 4 + 2:
            rz[:, :, :, :, :,0] = 0.
            rz[:, :, :, :, :,-1] = 0.
        return rz

    def interpolate_z_boundary(self,I):
        rz = I
        ndim = self.getdimension(I)
        if ndim==5:  # 2+1,2+2,2+3
            rz[:, :,:,:, 0] = I[:, :,:,:, 1]+I[:, :,:,:, 1]-I[:, :,:,:, 2]
            rz[:, :,:,:, -1] = I[:, :,:,:, -2]+I[:, :,:,:, -2]-I[:, :,:,:, -3]
        elif ndim == 4 + 2:
            rz[:, :, :,:,:, 0, :, :] = 0.
            rz[:, :, :,:,:, -1, :, :] = 0.
        return rz


    def dXc(self,I):
        rxc =  (I[:, :, 2:,1:-1,1:-1]-I[:,:,0:-2,1:-1,1:-1])*(0.5/self.spacing[0])
        return rxc
    def dYc(self,I):
        ryc = (I[:, :,1:-1, 2:,1:-1] - I[:, :,1:-1, 0:-2,1:-1]) * (0.5 / self.spacing[1])
        return ryc
    def dZc(self,I):
        rzc = (I[:, :, 1:-1,1:-1, 2:] - I[:, :, 1:-1,1:-1, 0:-2]) * (0.5 / self.spacing[2])
        return rzc


    def shrink_boundary(self,I):
        ndim = len(I.shape[2:])
        sz = I.shape
        if ndim ==1:
            return I[:,:,1:-1]
        elif ndim==2:
            return I[:,:,1:-1,1:-1]
        elif ndim ==3:
            return I[:,:,1:-1,1:-1,1:-1]
        elif ndim==4:
            return I[:,:,:,1:-1,1:-1,1:-1]


    def cover_boundary(self,I):
        sz = list(I.shape)
        sz = sz[:2]+[sz[i]+2 for i in range(2,len(sz))]
        I_cover = torch.empty(*sz,dtype=I.dtype,device=I.device)
        if self.order_smooth:
            I_cover= self.interpolate_x_boundary(I_cover)
            I_cover= self.interpolate_y_boundary(I_cover)
            I_cover=self.interpolate_z_boundary(I_cover)
        else:
            I_cover=self.zero_x_boundary(I_cover)
            I_cover=self.zero_y_boundary(I_cover)
            I_cover=self.zero_z_boundary(I_cover)
        return I_cover






class FD_np_multi_channel(FD_multi_channel):
    """
    Defnitions of the abstract methods for numpy
    """

    def __init__(self,dim,bcNeumannZero=True):
        """
        Constructor for numpy finite differences
        :param spacing: spatial spacing (array with as many entries as there are spatial dimensions)
        :param bcNeumannZero: Specifies if zero Neumann conditions should be used (if not, uses linear extrapolation)
        """
        super(FD_np_multi_channel, self).__init__(dim,bcNeumannZero)

    def getdimension(self,I):
        """
        Returns the dimension of an image
        :param I: input image
        :return: dimension of the input image
        """
        return I.ndim

    def create_zero_array(self, sz):
        """
        Creates a zero array
        :param sz: size of the zero array, e.g., [3,4,2]
        :return: the zero array
        """
        return np.zeros( sz )

    def get_size_of_array(self, A):
        """
        Returns the size (shape in numpy) of an array
        :param A: input array
        :return: shape/size
        """
        return A.shape


class FD_torch_multi_channel(FD_multi_channel):
    """
    Defnitions of the abstract methods for torch
    """

    def __init__(self,dim,bcNeumannZero=True):
        """
          Constructor for torch finite differences
          :param spacing: spatial spacing (array with as many entries as there are spatial dimensions)
          :param bcNeumannZero: Specifies if zero Neumann conditions should be used (if not, uses linear extrapolation)
          """
        super(FD_torch_multi_channel, self).__init__(dim,bcNeumannZero)

    def getdimension(self,I):
        """
        Returns the dimension of an image
        :param I: input image
        :return: dimension of the input image 
        """
        return I.dim()

    def create_zero_array(self, sz):
        """
        Creats a zero array
        :param sz: size of the array, e.g., [3,4,2]
        :return: the zero array
        """
        return  MyTensor(sz).zero_()

    def get_size_of_array(self, A):
        """
        Returns the size (size()) of an array
        :param A: input array
        :return: shape/size
        """
        return A.size()
