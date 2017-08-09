"""
This package implements various types of smoothers.
"""

from abc import ABCMeta, abstractmethod

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import finite_differences as fd

import utils

import custom_pytorch_extensions as ce

class Smoother(object):
    """
    Abstract base class defining the general smoother interface.
    """
    __metaclass__ = ABCMeta

    def __init__(self, sz, spacing, params):
        self.sz = sz
        """image size"""
        self.spacing = spacing
        """image spacing"""
        self.fdt = fd.FD_torch( self.spacing )
        """finite difference support for torch"""
        self.volumeElement = self.spacing.prod()
        """volume of pixel/voxel"""
        self.dim = len(spacing)
        """dimension"""
        self.params = params
        """ParameterDict() parameter object holding various configuration options"""

    @abstractmethod
    def smooth_scalar_field(self, I, Iout=None):
        """
        Abstract method to smooth a scalar field. Only this method should be overwritten in derived classes.
        
        :param I: input image to smooth 
        :param Iout: if not None then result is returned in this variable
        :return: should return the a smoothed scalar field, image dimension XxYxZ
        """
        pass

    @abstractmethod
    def inverse_smooth_scalar_field(self, I, Iout=None):
        """
        Experimental abstract method (NOT PROPERLY TESTED) to apply "inverse"-smoothing to a scalar field
        
        :param I: input image to inverse smooth 
        :param Iout: if not None then result is returned in this variable
        :return: should return the inverse-smoothed scalar field, image dimension XxYxZ
        """
        pass

    def smooth_scalar_field_multiNC(self, I, Iout=None):
        """
        Smoothes a scalar field of dimension BxCxXxYxZ (i.e, can smooth a batch and multi-channel images)
        
        :param I: input image to smooth 
        :param Iout: if not None then result is returned in this variable
        :return: smoothed image
        """
        sz = I.size()
        if Iout is not None:
            Is = Iout
        else:
            Is = Variable(torch.zeros(sz), requires_grad=False)

        for nrI in range(sz[0]):  # loop over all the images
            Is[nrI, ...] = self.smooth_scalar_field_multiC(I[nrI, ...])
        return Is

    def smooth_scalar_field_multiC(self, I, Iout=None):
        """
        Smoothes a scalar field of dimension CxXxYxZ (i.e, can smooth multi-channel images)

        :param I: input image to smooth 
        :param Iout: if not None then result is returned in this variable
        :return: smoothed image
        """
        sz = I.size()
        if Iout is not None:
            Is = Iout
        else:
            Is = Variable(torch.zeros(sz), requires_grad=False)

        for nrC in range(sz[0]):  # loop over all the channels, just advect them all the same
            Is[nrC, ...] = self.smooth_scalar_field(I[nrC, ...])
        return Is

    def inverse_smooth_vector_field_multiN(self, v, vout=None):
        """
        Inverse-smoothes (EXPERIMENTAL, NOT PROPERLY TESTED) a vector field of dimension NxdimxXxYxZ
        
        :param v: vector field to smooth
        :param vout: if not None then result is returned in this variable
        :return: smoothed vector field
        """
        sz = v.size()
        if vout is not None:
            ISv = vout
        else:
            ISv = Variable(torch.FloatTensor(v.size()))

        for nrI in range(sz[0]): # loop over all images
            ISv[nrI,...] = self.inverse_smooth_vector_field(v[nrI, ...])
        return ISv

    def inverse_smooth_vector_field(self, v, vout=None):
        """
        Inverse-smoothes (EXPERIMENTAL, NOT PROPERLY TESTED) a vector field of dimension dimxXxYxZ

        :param v: vector field to smooth
        :param vout: if not None then result is returned in this variable
        :return: smoothed vector field
        """

        if self.dim==1:
            if vout is None:
                return self.inverse_smooth_scalar_field(v) # if one dimensional, default to scalar-field smoothing
            else:
                return self.inverse_smooth_scalar_field(v,vout) # if one dimensional, default to scalar-field smoothing
        else:
            if vout is not None:
                ISv = vout
            else:
                ISv = Variable( torch.FloatTensor( v.size() ) )

            # smooth every dimension individually
            for d in range(0, self.dim):
                ISv[d,...] = self.inverse_smooth_scalar_field(v[d, ...])
            return ISv

    def smooth_vector_field_multiN(self, v, vout=None):
        """
        Smoothes a vector field of dimension NxdimxXxYxZ

        :param v: vector field to smooth
        :param vout: if not None then result is returned in this variable
        :return: smoothed vector field
        """
        sz = v.size()
        if vout is not None:
            Sv = vout
        else:
            Sv = Variable(torch.FloatTensor(v.size()))

        for nrI in range(sz[0]): #loop over all the images
            Sv[nrI,...] = self.smooth_vector_field(v[nrI, ...])
        return Sv

    def smooth_vector_field(self, v, vout=None):
        """
        Smoothes a vector field of dimension dimxXxYxZ

        :param v: vector field to smooth
        :param vout: if not None then result is returned in this variable
        :return: smoothed vector field
        """

        if self.dim==1:
            if vout is None:
                return self.smooth_scalar_field(v) # if one dimensional, default to scalar-field smoothing
            else:
                return self.smooth_scalar_field(v,vout) # if one dimensional, default to scalar-field smoothing
        else:
            if vout is not None:
                Sv = vout
            else:
                Sv = Variable( torch.FloatTensor( v.size() ) )

            # smooth every dimension individually
            for d in range(0, self.dim):
                Sv[d,...] = self.smooth_scalar_field(v[d, ...])
            return Sv

class DiffusionSmoother(Smoother):
    """
    Smoothing by solving the diffusion equation iteratively.
    """

    def __init__(self, sz, spacing, params):
        super(DiffusionSmoother,self).__init__(sz,spacing,params)
        self.iter = params[('iter', 5, 'Number of iterations' )]
        """number of iterations"""

    def set_iter(self,iter):
        """
        Set the number of iterations for the diffusion smoother
        
        :param iter: number of iterations 
        :return: returns the number of iterations
        """
        self.iter = iter
        self.params['iter'] = self.iter

    def get_iter(self):
        """
        Returns the set number of iterations
        
        :return: number of iterations
        """
        return self.iter

    def smooth_scalar_field(self, v, vout=None):
        """
        Smoothes a scalar field of dimension XxYxZ
        
        :param v: input image 
        :param vout: if not None returns the result in this variable
        :return: smoothed image
        """
        # basically just solving the heat equation for a few steps
        if vout is not None:
            Sv = vout
        else:
            Sv = v.clone()

        # now iterate and average based on the neighbors
        for i in range(0,self.iter*2**self.dim): # so that we smooth the same indepdenent of dimension
            # multiply with smallest h^2 and divide by 2^dim to assure stability
            Sv = Sv + 0.5/(2**self.dim)*self.fdt.lap(Sv)*self.spacing.min()**2 # multiply with smallest h^2 to assure stability
        return Sv

    def inverse_smooth_scalar_field(self, v, vout=None):
        raise ValueError('Sorry: Inversion of smoothing only supported for Fourier-based filters at the moment')

class GaussianSmoother(Smoother):
    """
    Gaussian smoothing in the spatial domain (hence, SLOW in high dimensions on the CPU at least).
    
    .. todo::
        Clean up the two implementations (spatial and Fourier of the Gaussian smoothers).
        In particular, assure that all computions are done in physical coordinates. For now these are just in [-1,1]^d
    """
    # TODO

    def __init__(self, sz, spacing, params):
        super(GaussianSmoother,self).__init__(sz,spacing,params)

class GaussianSpatialSmoother(GaussianSmoother):

    def __init__(self, sz, spacing, params):
        super(GaussianSpatialSmoother,self).__init__(sz,spacing,params)
        self.k_sz_h = params[('k_sz_h', None, 'size of the kernel' )]
        """size of half the smoothing kernel"""
        self.filter = None
        """smoothing filter"""

    def set_k_sz_h(self,k_sz_h):
        """
        Set the size of half the smoothing kernel
        
        :param k_sz_h: size of half the kernel as array 
        """
        self.k_sz_h = k_sz_h
        self.params['k_sz_h'] = self.k_sz_h

    def get_k_sz_h(self):
        """
        Returns the size of half the smoothing kernel
        
        :return: return half the smoothing kernel size 
        """
        return self.k_sz_h

    def _create_filter(self):

        if self.k_sz_h is None:
            self.k_sz = (2 * 5 + 1) * np.ones(self.dim, dtype='int')  # default kernel size
        else:
            self.k_sz = k_sz_h * 2 + 1  # this is to assure that the kernel is odd size

        self.smoothingKernel = self._create_smoothing_kernel(self.k_sz)
        self.required_padding = (self.k_sz-1)/2

        if self.dim==1:
            self.filter = Variable(torch.from_numpy(self.smoothingKernel).view([sz[0],sz[1],k_sz[0]]))
        elif self.dim==2:
            self.filter = Variable(torch.from_numpy(self.smoothingKernel).view([sz[0],sz[1],k_sz[0],k_sz[1]]))
        elif self.dim==3:
            self.filter = Variable(torch.from_numpy(self.smoothingKernel).view([sz[0],sz[1],k_sz[0],k_sz[1],k_sz[2]]))
        else:
            raise ValueError('Can only create the smoothing kernel in dimensions 1-3')

    def _create_smoothing_kernel(self, k_sz):
        mus = np.zeros(self.dim)
        stds = np.ones(self.dim)
        id = utils.identity_map(k_sz)
        g = utils.compute_normalized_gaussian(id, mus, stds)

        return g

    def _filter_input_with_padding(self, I, Iout=None):
        if self.dim==1:
            I_4d = I.view([1,1,1]+list(I.size()))
            I_pad = F.pad(I_4d,(self.required_padding[0],self.required_padding[0],0,0),mode='replicate').view(1,1,-1)
            if Iout is not None:
                Iout = F.conv1d(I_pad,self.filter).view(I.size())
                return Iout
            else:
                return F.conv1d(I_pad,self.filter).view(I.size())
        elif self.dim==2:
            I_pad = F.pad(I,(self.required_padding[0],self.required_padding[0],
                                self.required_padding[1],self.required_padding[1]),mode='replicate')
            if Iout is not None:
                Iout = F.conv2d(I_pad,self.filter).view(I.size())
                return Iout
            else:
                return F.conv2d(I_pad,self.filter).view(I.size())
        elif self.dim==3:
            I_pad = F.pad(I, (self.required_padding[0], self.required_padding[0],
                                 self.required_padding[1], self.required_padding[1],
                                 self.required_padding[2], self.required_padding[2]), mode='replicate')
            if Iout is not None:
                Iout = F.conv3d(I_pad, self.filter).view(I.size())
            else:
                return F.conv3d(I_pad, self.filter).view(I.size())
        else:
            raise ValueError('Can only perform padding in dimensions 1-3')

    def smooth_scalar_field(self, v, vout=None):
        """
        Smooth the scalar field using Gaussian smoothing in the spatial domain
        
        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :return: smoothed image
        """
        if self.filter is None:
            self._create_filter()
        # just doing a Gaussian smoothing
        return self._filter_input_with_padding(v, vout)

    def inverse_smooth_scalar_field(self, v, vout=None):
        """
        Not yet implemented 
        """
        raise ValueError('Sorry: Inversion of smoothing only supported for Fourier-based filters at the moment')


class GaussianFourierSmoother(GaussianSmoother):
    """
    Performs Gaussian smoothing via convolution in the Fourier domain. Much faster for large dimensions
    than spatial Gaussian smoothing on the CPU in large dimensions.
    """

    def __init__(self, sz, spacing, params):
        super(GaussianFourierSmoother,self).__init__(sz,spacing,params)
        self.gaussianStd = params[('gaussian_std', 0.15,'std for the Gaussian' )]
        """stanard deviation of Gaussian"""
        self.FFilter = None
        """filter in Fourier domain"""

    def _create_filter(self):

        mus = np.zeros(self.dim)
        stds = self.gaussianStd*np.ones(self.dim)
        id = utils.identity_map(self.sz)
        g = utils.compute_normalized_gaussian(id, mus, stds)

        self.FFilter = ce.create_complex_fourier_filter(g, self.sz)

    def set_gaussian_std(self,gstd):
        """
        Set the standard deviation of the Gaussian filter
        
        :param gstd: standard deviation 
        """
        self.gaussianStd = gstd
        self.params['gaussian_std'] = self.gaussianStd

    def get_gaussian_std(self):
        """
        Return the standard deviation of the Gaussian filter
        
        :return: standard deviation of Gaussian filter 
        """
        return self.gaussianStd

    def smooth_scalar_field(self, v, vout=None):
        """
        Smooth the scalar field using Gaussian smoothing in the Fourier domain

        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :return: smoothed image
        """

        # just doing a Gaussian smoothing
        # we need to instantiate a new filter function here every time for the autograd to work
        if self.FFilter is None:
            self._create_filter()
        if vout is not None:
            vout = ce.fourier_convolution(v, self.FFilter)
            return vout
        else:
            return ce.fourier_convolution(v, self.FFilter)

    def inverse_smooth_scalar_field(self, v, vout=None):
        """
        Inverse-smooth the scalar field using Gaussian smoothing in the Fourier domain
        (with the inverse of the Fourier transform of the Gaussian; EXPERIMENTAL, requires regularization
        to avoid divisions by zero. DO NOT USE AT THE MOMENT.)

        :param v: image to inverse-smooth
        :param vout: if not None returns the result in this variable
        :return: inverse-smoothed image
        """
        if self.FFilter is None:
            self._create_filter()
        if vout is not None:
            vout = ce.inverse_fourier_convolution(v, self.FFilter)
            return vout
        else:
            return ce.inverse_fourier_convolution(v, self.FFilter)

class SmootherFactory(object):
    """
    Factory to quickly create different types of smoothers.
    """

    def __init__(self,sz,spacing):
        self.spacing = spacing
        """spatial spacing of image"""
        self.sz = sz
        """size of image"""
        self.dim = len( spacing )
        """dimension of image"""
        self.default_smoother_type = 'gaussian'
        """default smoother used for smoothing"""

    def set_default_smoother_type_to_gaussian(self):
        """
        Set the default smoother type to Gaussian smoothing in the Fourier domain
        """
        self.default_smoother_type = 'gaussian'

    def set_default_smoother_type_to_diffusion(self):
        """
        Set the default smoother type to diffusion smoothing 
        """
        self.default_smoother_type = 'diffusion'

    def set_default_smoother_type_to_gaussianSpatial(self):
        """
        Set the default smoother type to Gaussian smoothing in the spatial domain
        """
        self.default_smoother_type = 'gaussianSpatial'

    def create_smoother(self, params):
        """
        Create the desired smoother
        :param params: ParamterDict() object to hold paramters which should be passed on
        :return: returns the smoother
        """

        cparams = params[('smoother',{})]
        smootherType = cparams[('type', self.default_smoother_type,
                                          'type of smoother (difusion/gaussian/gaussianSpatial)' )]
        if smootherType=='diffusion':
            return DiffusionSmoother(self.sz,self.spacing,cparams)
        elif smootherType=='gaussian':
            return GaussianFourierSmoother(self.sz,self.spacing,cparams)
        elif smootherType=='gaussianSpatial':
            return GaussianSpatialSmoother(self.sz,self.spacing,cparams)
        else:
            raise ValueError( 'Smoother: ' + smootherName + ' not known')
