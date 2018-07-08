"""
*finite_difference.py* is the main package to compute finite differences in 
1D, 2D, and 3D on numpy arrays (class FD_np) and pytorch tensors (class FD_torch).
The package supports first and second order derivatives and Neumann and linear extrapolation
boundary conditions (though the latter have not been tested extensively yet).
"""
from __future__ import absolute_import

from builtins import object
from abc import ABCMeta, abstractmethod

import torch
from torch.autograd import Variable
from .data_wrapper import MyTensor
import numpy as np
from future.utils import with_metaclass

class FD(with_metaclass(ABCMeta, object)):
    """
    *FD* is the abstract class for finite differences. It includes most of the actual finite difference code, 
    but requires the definition (in a derived class) of the methods *get_dimension*, *create_zero_array*, and *get_size_of_array*.
    In this way the numpy and pytorch versions can easily be derived. All the method expect BxXxYxZ format (i.e., they process a batch at a time)
    """

    def __init__(self, spacing, bcNeumannZero=True):
        """
        Constructor        
        :param spacing: 1D numpy array defining the spatial spacing, e.g., [0.1,0.1,0.1] for a 3D image  
        :param bcNeumannZero: Defines the boundary condition. If set to *True* (default) zero Neumann boundary conditions
            are imposed. If set to *False* linear extrapolation is used (this is still experimental, but may be beneficial 
            for better boundary behavior)
        """

        self.dim = spacing.size
        """spatial dimension"""
        self.spacing = np.ones( self.dim )
        """spacing"""
        self.bcNeumannZero = bcNeumannZero # if false then linear interpolation
        """should Neumann boundary conditions be used? (otherwise linear extrapolation)"""
        if spacing.size==1:
            self.spacing[0] = spacing[0]
        elif spacing.size==2:
            self.spacing[0] = spacing[0]
            self.spacing[1] = spacing[1]
        elif spacing.size==3:
            self.spacing = spacing
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')

    def dXb(self,I):
        """
        Backward difference in x direction:
        :math:`\\frac{dI(i)}{dx}\\approx\\frac{I_i-I_{i-1}}{h_x}`

        :param I: Input image  
        :return: Returns the first derivative in x direction using backward differences
        """
        return (I-self.xm(I))/self.spacing[0]

    def dXf(self,I):
        """
        Forward difference in x direction:
        :math:`\\frac{dI(i)}{dx}\\approx\\frac{I_{i+1}-I_{i}}{h_x}`
        
        :param I: Input image
        :return: Returns the first derivative in x direction using forward differences
        """
        return (self.xp(I)-I)/self.spacing[0]

    def dXc(self,I):
        """
        Central difference in x direction:
        :math:`\\frac{dI(i)}{dx}\\approx\\frac{I_{i+1}-I_{i-1}}{2h_x}`
        
        :param I: Input image
        :return: Returns the first derivative in x direction using central differences
        """
        return (self.xp(I)-self.xm(I))/(2*self.spacing[0])

    def ddXc(self,I):
        """
        Second deriative in x direction
        
        :param I: Input image
        :return: Returns the second derivative in x direction
        """
        return (self.xp(I)-2*I+self.xm(I))/(self.spacing[0]**2)

    def dYb(self,I):
        """
        Same as dXb, but for the y direction
        
        :param I: Input image
        :return: Returns the first derivative in y direction using backward differences
        """
        return (I-self.ym(I))/self.spacing[1]

    def dYf(self,I):
        """
        Same as dXf, but for the y direction
        
        :param I: Input image
        :return: Returns the first derivative in y direction using forward differences
        """
        return (self.yp(I)-I)/self.spacing[1]

    def dYc(self,I):
        """
        Same as dXc, but for the y direction
        
        :param I: Input image
        :return: Returns the first derivative in y direction using central differences
        """
        return (self.yp(I)-self.ym(I))/(2*self.spacing[1])

    def ddYc(self,I):
        """
        Same as ddXc, but for the y direction
        
        :param I: Input image
        :return: Returns the second derivative in the y direction
        """
        return (self.yp(I)-2*I+self.ym(I))/(self.spacing[1]**2)

    def dZb(self,I):
        """
        Same as dXb, but for the z direction
        
        :param I: Input image 
        :return: Returns the first derivative in the z direction using backward differences
        """
        return (I - self.zm(I))/self.spacing[2]

    def dZf(self, I):
        """
        Same as dXf, but for the z direction
        
        :param I: Input image
        :return: Returns the first derivative in the z direction using forward differences
        """
        return (self.zp(I)-I)/self.spacing[2]

    def dZc(self, I):
        """
        Same as dXc, but for the z direction
        
        :param I: Input image
        :return: Returns the first derivative in the z direction using central differences
        """
        return (self.zp(I)-self.zm(I))/(2*self.spacing[2])

    def ddZc(self,I):
        """
        Same as ddXc, but for the z direction
        
        :param I: Input iamge
        :return: Returns the second derivative in the z direction 
        """
        return (self.zp(I)-2*I+self.zm(I))/(self.spacing[2]**2)

    def lap(self, I):
        """
        Compute the Lapacian of an image
        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.
        
        :param I: Input image [batch, X,Y,Z]
        :return: Returns the Laplacian
        """
        ndim = self.getdimension(I)
        if ndim == 1+1:
            return self.ddXc(I)
        elif ndim == 2+1:
            return (self.ddXc(I) + self.ddYc(I))
        elif ndim == 3+1:
            return (self.ddXc(I) + self.ddYc(I) + self.ddZc(I))
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')

    def grad_norm_sqr_c(self, I):
        """
        Computes the gradient norm of an image
        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.

        :param I: Input image [batch, X,Y,Z]
        :return: returns ||grad I||^2
        """
        ndim = self.getdimension(I)
        if ndim == 1 + 1:
            return self.dXc(I)**2
        elif ndim == 2 + 1:
            return (self.dXc(I)**2 + self.dYc(I)**2)
        elif ndim == 3 + 1:
            return (self.dXc(I)**2 + self.dYc(I)**2 + self.dZc(I)**2)
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')

    def grad_norm_sqr_f(self, I):
        """
        Computes the gradient norm of an image
        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.

        :param I: Input image [batch, X,Y,Z]
        :return: returns ||grad I||^2
        """
        ndim = self.getdimension(I)
        if ndim == 1 + 1:
            return self.dXf(I)**2
        elif ndim == 2 + 1:
            return (self.dXf(I)**2 + self.dYf(I)**2)
        elif ndim == 3 + 1:
            return (self.dXf(I)**2 + self.dYf(I)**2 + self.dZf(I)**2)
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')

    def grad_norm_sqr_b(self, I):
        """
        Computes the gradient norm of an image
        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.

        :param I: Input image [batch, X,Y,Z]
        :return: returns ||grad I||^2
        """
        ndim = self.getdimension(I)
        if ndim == 1 + 1:
            return self.dXb(I)**2
        elif ndim == 2 + 1:
            return (self.dXb(I)**2 + self.dYb(I)**2)
        elif ndim == 3 + 1:
            return (self.dXb(I)**2 + self.dYb(I)**2 + self.dZb(I)**2)
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

    def xp(self,I):
        """

        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.


        Returns the values for x-index incremented by one (to the right in 1D)
        
        :param I: Input image [batch, X, Y,Z]
        :return: Image with values at an x-index one larger
        """
        rxp = self.create_zero_array( self.get_size_of_array( I ) )
        ndim = self.getdimension(I)
        if ndim == 1+1:
            rxp[:,0:-1] = I[:,1:]
            if self.bcNeumannZero:
                rxp[:,-1] = I[:,-1]
            else:
                rxp[:,-1] = 2*I[:,-1]-I[:,-2]
        elif ndim == 2+1:
            rxp[:,0:-1,:] = I[:,1:,:]
            if self.bcNeumannZero:
                rxp[:,-1,:] = I[:,-1,:]
            else:
                rxp[:,-1,:] = 2*I[:,-1,:]-I[:,-2,:]
        elif ndim == 3+1:
            rxp[:,0:-1,:,:] = I[:,1:,:,:]
            if self.bcNeumannZero:
                rxp[:,-1,:,:] = I[:,-1,:,:]
            else:
                rxp[:,-1,:,:] = 2*I[:,-1,:,:]-I[:,-2,:,:]
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')
        return rxp

    def xm(self,I):
        """

        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.

        Returns the values for x-index decremented by one (to the left in 1D)
        
        :param I: Input image [batch, X, Y, Z]
        :return: Image with values at an x-index one smaller
        """
        rxm = self.create_zero_array( self.get_size_of_array( I ) )
        ndim = self.getdimension(I)
        if ndim == 1+1:
            rxm[:,1:] = I[:,0:-1]
            if self.bcNeumannZero:
                rxm[:,0] = I[:,0]
            else:
                rxm[:,0] = 2*I[:,0]-I[:,1]
        elif ndim == 2+1:
            rxm[:,1:,:] = I[:,0:-1,:]
            if self.bcNeumannZero:
                rxm[:,0,:] = I[:,0,:]
            else:
                rxm[:,0,:] = 2*I[:,0,:]-I[:,1,:]
        elif ndim == 3+1:
            rxm[:,1:,:,:] = I[:,0:-1,:,:]
            if self.bcNeumannZero:
                rxm[:,0,:,:] = I[:,0,:,:]
            else:
                rxm[:,0,:,:] = 2*I[:,0,:,:]-I[:,1,:,:]
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')
        return rxm

    def yp(self, I):
        """


        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.

        Same as xp, but for the y direction
        
        :param I: Input image
        :return: Image with values at y-index one larger
        """
        ryp = self.create_zero_array( self.get_size_of_array( I ) )
        ndim = self.getdimension(I)
        if ndim == 2+1:
            ryp[:,:,0:-1] = I[:,:,1:]
            if self.bcNeumannZero:
                ryp[:,:,-1] = I[:,:,-1]
            else:
                ryp[:,:,-1] = 2*I[:,:,-1]-I[:,:,-2]
        elif ndim == 3+1:
            ryp[:,:,0:-1,:] = I[:,:,1:,:]
            if self.bcNeumannZero:
                ryp[:,:,-1,:] = I[:,:,-1,:]
            else:
                ryp[:,:,-1,:] = 2*I[:,:,-1,:]-I[:,:,-2,:]
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')
        return ryp

    def ym(self, I):
        """
        Same as xm, but for the y direction



        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.

        Returns the values for x-index decremented by one (to the left in 1D)

        :param I: Input image [batch, X, Y, Z]
        :return: Image with values at y-index one smaller
        """
        rym = self.create_zero_array( self.get_size_of_array( I ) )
        ndim = self.getdimension(I)
        if ndim == 2+1:
            rym[:,:,1:] = I[:,:,0:-1]
            if self.bcNeumannZero:
                rym[:,:,0] = I[:,:,0]
            else:
                rym[:,:,0] = 2*I[:,:,0]-I[:,:,1]
        elif ndim == 3+1:
            rym[:,:,1:,:] = I[:,:,0:-1,:]
            if self.bcNeumannZero:
                rym[:,:,0,:] = I[:,:,0,:]
            else:
                rym[:,:,0,:] = 2*I[:,:,0,:]-I[:,:,1,:]
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')
        return rym

    def zp(self, I):
        """
        Same as xp, but for the z direction
        
        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.

        Returns the values for x-index decremented by one (to the left in 1D)

        :param I: Input image [batch, X, Y, Z]
        :return: Image with values at z-index one larger
        """
        rzp = self.create_zero_array( self.get_size_of_array( I ) )
        ndim = self.getdimension(I)
        if ndim == 3+1:
            rzp[:,:,:,0:-1] = I[:,:,:,1:]
            if self.bcNeumannZero:
                rzp[:,:,:,-1] = I[:,:,:,-1]
            else:
                rzp[:,:,:,-1] = 2*I[:,:,:,-1]-I[:,:,:,-2]
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')
        return rzp

    def zm(self, I):
        """
        Same as xm, but for the z direction
        
        !!!!!!!!!!!
        IMPORTANT:
        ALL THE FOLLOWING IMPLEMENTED CODE ADD 1 ON DIMENSION, WHICH REPRESENT BATCH DIMENSION.
        THIS IS FOR COMPUTATIONAL EFFICIENCY.

        Returns the values for x-index decremented by one (to the left in 1D)

        :param I: Input image [batch, X, Y, Z]
        :return: Image with values at z-index one smaller
        """
        rzm = self.create_zero_array( self.get_size_of_array( I ) )
        ndim = self.getdimension(I)
        if ndim == 3+1:
            rzm[:,:,:,1:] = I[:,:,:,0:-1]
            if self.bcNeumannZero:
                rzm[:,:,:,0] = I[:,:,:,0]
            else:
                rzm[:,:,:,0] = 2*I[:,:,:,0]-I[:,:,:,1]
        else:
            raise ValueError('Finite differences are only supported in dimensions 1 to 3')
        return rzm


class FD_np(FD):
    """
    Defnitions of the abstract methods for numpy
    """

    def __init__(self,spacing,bcNeumannZero=True):
        """
        Constructor for numpy finite differences
        :param spacing: spatial spacing (array with as many entries as there are spatial dimensions)
        :param bcNeumannZero: Specifies if zero Neumann conditions should be used (if not, uses linear extrapolation)
        """
        super(FD_np, self).__init__(spacing,bcNeumannZero)

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


class FD_torch(FD):
    """
    Defnitions of the abstract methods for torch
    """

    def __init__(self,spacing,bcNeumannZero=True):
        """
          Constructor for torch finite differences
          :param spacing: spatial spacing (array with as many entries as there are spatial dimensions)
          :param bcNeumannZero: Specifies if zero Neumann conditions should be used (if not, uses linear extrapolation)
          """
        super(FD_torch, self).__init__(spacing,bcNeumannZero)

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
        return  Variable(MyTensor(sz).zero_(), requires_grad=False)

    def get_size_of_array(self, A):
        """
        Returns the size (size()) of an array
        :param A: input array
        :return: shape/size
        """
        return A.size()
