"""
This package implements various types of smoothers.
"""

from abc import ABCMeta, abstractmethod

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from data_wrapper import USE_CUDA, MyTensor, AdaptVal
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
        self.optimizer_params = None

    def get_optimization_parameters(self):
        """
        Returns the optimizer parameters for a smoother. Returns None of there are none, or if optimization is disabled.
        :return: Optimizer parameters
        """
        return None

    def get_custom_optimizer_output_string(self):
        """
        Returns a customized string describing a smoother's setting. Will be displayed during optimization.
        Useful to overwrite if optimizing over smoother parameters.
        :return: string
        """
        return ''

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
        Experimental abstract method (NOT PROPERLY TESTED) to apply "inverse"-smoothing to a scalar field.

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
            Is = Variable(MyTensor(sz).zero_(), requires_grad=False)

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
            Is = Variable(MyTensor(sz).zero_(), requires_grad=False)

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
            ISv = Variable(MyTensor(v.size()).zero_())

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
                ISv = Variable(MyTensor(v.size()).zero_())

            # smooth every dimension individually
            for d in range(0, self.dim):
                ISv[d,...] = self.inverse_smooth_scalar_field(v[d, ...])
            return ISv

    def smooth_vector_field_multiN(self, v, vout=None):
        """
        Smoothes a vector field of dimension NxdimxXxYxZ

        a new version, where channels are separately calcualted during filtering time, so channel loop is not needed

        :param v: vector field to smooth
        :param vout: if not None then result is returned in this variable
        :return: smoothed vector field
        """
        sz = v.size()
        if vout is not None:
            Sv = vout
        else:
            Sv = Variable(MyTensor(v.size()).zero_())

        # if USE_CUDA:
        Sv[:] = self.smooth_scalar_field(v)    # here must use :, very important !!!!
        # else:
        #     for nrI in range(sz[0]): #loop over all the images
        #         Sv[nrI,...] = self.smooth_vector_field(v[nrI, ...])

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
                Sv = Variable(MyTensor(v.size()).zero_())

            #smooth every dimension individually
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
            for c in range(Sv.size()[1]):
                Sv[:,c] = Sv[:,c] + 0.5/(2**self.dim)*self.fdt.lap(Sv[:,c])*self.spacing.min()**2 # multiply with smallest h^2 to assure stability
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
        self.k_sz_h = None # params[('k_sz_h', None, 'size of the kernel' )]
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
            self.k_sz = self.k_sz_h * 2 + 1  # this is to assure that the kernel is odd size

        self.smoothingKernel = self._create_smoothing_kernel(self.k_sz)
        self.required_padding = (self.k_sz-1)/2

        if self.dim==1:
            self.filter =AdaptVal(Variable(torch.from_numpy(self.smoothingKernel)))
        elif self.dim==2:
            self.filter = AdaptVal(Variable(torch.from_numpy(self.smoothingKernel)))
        elif self.dim==3:
            self.filter = AdaptVal(Variable(torch.from_numpy(self.smoothingKernel)))
        else:
            raise ValueError('Can only create the smoothing kernel in dimensions 1-3')

    def _create_smoothing_kernel(self, k_sz):
        mus = np.zeros(self.dim)
        stds = np.ones(self.dim)
        id = utils.identity_map(k_sz,self.spacing)
        g = utils.compute_normalized_gaussian(id, mus, stds)

        return g

    def _filter_input_with_padding(self, I, Iout=None):

        if self.dim==1:
            I_4d = I.view([1]+list(I.size()))
            I_pad = F.pad(I_4d,(self.required_padding[0],self.required_padding[0],0,0),mode='replicate').view(1,1,-1)
            # 1D will be available in pytorch 0.4
            # I_pad = F.pad(I, (self.required_padding[0], self.required_padding[0]), mode='replicate')
            I_sz = I_pad.size()
            sm_filter = self.filter.repeat(I_sz[1], 1, 1)  # output_ch input_chh h, w
            if Iout is not None:
                Iout = F.conv1d(I_pad,sm_filter, groups=I_sz[1])
                return Iout
            else:
                return F.conv1d(I_pad,sm_filter, groups=I_sz[1])
        elif self.dim==2:
            I_pad = F.pad(I,(self.required_padding[0],self.required_padding[0],
                                self.required_padding[1],self.required_padding[1]),mode='replicate')
            I_sz = I_pad.size()
            sm_filter = self.filter.repeat(I_sz[1], 1, 1, 1)  # output_ch input_chh h, w
            if Iout is not None:
                Iout = F.conv2d(I_pad,sm_filter, groups=I_sz[1])
                return Iout
            else:
                return F.conv2d(I_pad,sm_filter, groups=I_sz[1])
        elif self.dim==3:
            I_pad = F.pad(I, (self.required_padding[0], self.required_padding[0],
                                 self.required_padding[1], self.required_padding[1],
                                 self.required_padding[2], self.required_padding[2]), mode='replicate')
            I_sz = I_pad.size()
            sm_filter = self.filter.repeat(I_sz[1], 1, 1, 1, 1)  # output_ch input_chh h, w
            if Iout is not None:
                Iout = F.conv3d(I_pad, sm_filter, groups=I_sz[1])
            else:
                return F.conv3d(I_pad, sm_filter, groups=I_sz[1])
        else:
            raise ValueError('Can only perform padding in dimensions 1-3')

    def smooth_scalar_field(self, v, vout=None):
        """
        Smooth the scalar field using Gaussian smoothing in the spatial domain
        
        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :return: smoothed image
        """

        self.sz = v.size()
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

    __metaclass__ = ABCMeta

    def __init__(self, sz, spacing, params):
        super(GaussianFourierSmoother, self).__init__(sz, spacing, params)
        self.FFilter = None
        """filter in Fourier domain"""

    @abstractmethod
    def _create_filter(self):
        """
        Creates the Gaussian filter in the Fourier domain (needs to be assigned to self.FFilter)
        :return: n/a
        """
        pass

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


class AdaptiveSingleGaussianFourierSmoother(GaussianSmoother):
    """
    Performs Gaussian smoothing via convolution in the Fourier domain. Much faster for large dimensions
    than spatial Gaussian smoothing on the CPU in large dimensions.
    """

    def __init__(self, sz, spacing, params):
        super(AdaptiveSingleGaussianFourierSmoother, self).__init__(sz, spacing, params)
        self.gaussianStd = np.array(params[('gaussian_std', [0.15], 'std for the Gaussian')])
        """standard deviation of Gaussian"""
        self.gaussianStd_min = params[('gaussian_std_min', 0.00001, 'minimal allowed std for the Gaussian')]
        """minimal allowed standard deviation during optimization"""
        self.optimize_over_smoother_parameters = params[('optimize_over_smoother_parameters', False, 'if set to true the smoother will be optimized')]
        """determines if we should optimize over the smoother parameters"""

        self.gaussian_fourier_filter_generator = ce.GaussianFourierFilterGenerator(sz,spacing)

        self.optimizer_params = self._create_optimization_vector_parameters()

    def get_custom_optimizer_output_string(self):
        return ", smooth(std)= " + np.array_str(self.get_gaussian_std()[0].data.numpy(),precision=3)

    def get_optimization_parameters(self):
        if self.optimize_over_smoother_parameters:
            return self.optimizer_params
        else:
            return None

    def _get_gaussian_std_from_optimizer_params(self):
        # project if needed
        if self.optimizer_params.data[0]<self.gaussianStd_min:
            self.optimizer_params.data[0] = self.gaussianStd_min
        return self.optimizer_params[0]

    def _set_gaussian_std_optimizer_params(self,g_std):
        self.optimizer_params.data[0]=g_std

    def set_gaussian_std(self, gstd):
        """
        Set the standard deviation of the Gaussian filter

        :param gstd: standard deviation 
        """
        self.params['gaussian_std'] = gstd
        self._set_gaussian_std_optimizer_params(gstd)

    def get_gaussian_std(self):
        """
        Return the standard deviation of the Gaussian filter

        :return: standard deviation of Gaussian filter 
        """
        gaussianStd = self._get_gaussian_std_from_optimizer_params()
        return gaussianStd

    def smooth_scalar_field(self, v, vout=None):
        """
        Smooth the scalar field using Gaussian smoothing in the Fourier domain

        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :return: smoothed image
        """

        # just doing a Gaussian smoothing
        if vout is not None:
            vout = ce.fourier_single_gaussian_convolution(v,self.gaussian_fourier_filter_generator,self.get_gaussian_std())
            return vout
        else:
            return ce.fourier_single_gaussian_convolution(v,self.gaussian_fourier_filter_generator,self.get_gaussian_std())

    def inverse_smooth_scalar_field(self, v, vout=None):
        """
        Inverse-smooth the scalar field using Gaussian smoothing in the Fourier domain
        (with the inverse of the Fourier transform of the Gaussian; EXPERIMENTAL, requires regularization
        to avoid divisions by zero. DO NOT USE AT THE MOMENT.)

        :param v: image to inverse-smooth
        :param vout: if not None returns the result in this variable
        :return: inverse-smoothed image
        """

        raise ValueError("Not yet implemented")

    def _create_optimization_vector_parameters(self):
        self.optimizer_params = utils.create_vector_parameter(1)
        self.optimizer_params.data[0] = self.gaussianStd[0]
        return self.optimizer_params


class SingleGaussianFourierSmoother(GaussianFourierSmoother):
    """
    Performs Gaussian smoothing via convolution in the Fourier domain. Much faster for large dimensions
    than spatial Gaussian smoothing on the CPU in large dimensions.
    """

    def __init__(self, sz, spacing, params):
        super(SingleGaussianFourierSmoother,self).__init__(sz,spacing,params)
        self.gaussianStd = params[('gaussian_std', 0.15 ,'std for the Gaussian' )]
        """standard deviation of Gaussian"""
        self.gaussianStd_min = params[('gaussian_std_min', 0.00001 ,'minimal allowed std for the Gaussian' )]
        """minimal allowed standard deviation during optimization"""

    def _create_filter(self):

        mus = np.zeros(self.dim)
        stds = self.gaussianStd*np.ones(self.dim)
        id = utils.identity_map(self.sz,self.spacing)
        g = utils.compute_normalized_gaussian(id, mus, stds)

        self.FFilter,_ = ce.create_complex_fourier_filter(g, self.sz)

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


class MultiGaussianFourierSmoother(GaussianFourierSmoother):
    """
    Performs multi Gaussian smoothing via convolution in the Fourier domain. Much faster for large dimensions
    than spatial Gaussian smoothing on the CPU in large dimensions.
    """

    def __init__(self, sz, spacing, params):
        super(MultiGaussianFourierSmoother, self).__init__(sz, spacing, params)
        self.multi_gaussian_stds = np.array( params[('multi_gaussian_stds', [0.05,0.1,0.15,0.2,0.25], 'std deviations for the Gaussians')] )
        default_multi_gaussian_weights = self.multi_gaussian_stds
        default_multi_gaussian_weights /= default_multi_gaussian_weights.sum()
        """standard deviations of Gaussians"""
        self.multi_gaussian_weights = np.array( params[('multi_gaussian_weights', default_multi_gaussian_weights.tolist(), 'weights for the multiple Gaussians')] )
        """weights for the Gaussians"""

        assert len(self.multi_gaussian_weights)==len(self.multi_gaussian_stds)

        weight_sum = self.multi_gaussian_weights.sum()
        if weight_sum!=1.:
            print('WARNING: multi-Gaussian weights do not sum to one. Projecting them.')
            self.multi_gaussian_weights += (1.-weight_sum)/len(self.multi_gaussian_weights)
            params['multi_gaussian_weights'] = self.multi_gaussian_weights.tolist()

        assert (np.array(self.multi_gaussian_weights)).sum()==1.

    def _create_filter(self):

        mus = np.zeros(self.dim)
        id = utils.identity_map(self.sz,self.spacing)

        assert len(self.multi_gaussian_stds)>0
        assert len(self.multi_gaussian_weights)>0

        nr_of_gaussians = len(self.multi_gaussian_stds)

        for nr in range(nr_of_gaussians):
            stds = self.multi_gaussian_stds[nr] * np.ones(self.dim)
            g = self.multi_gaussian_weights[nr] * utils.compute_normalized_gaussian(id, mus, stds)

            if nr==0:
                self.FFilter,_ = ce.create_complex_fourier_filter(g, self.sz)
            else:
                cFilter,_ = ce.create_complex_fourier_filter(g, self.sz)
                self.FFilter += cFilter


class AdaptiveMultiGaussianFourierSmoother(GaussianSmoother):
    """
    Performs Gaussian smoothing via convolution in the Fourier domain. Much faster for large dimensions
    than spatial Gaussian smoothing on the CPU in large dimensions.
    """

    def __init__(self, sz, spacing, params):
        super(AdaptiveMultiGaussianFourierSmoother, self).__init__(sz, spacing, params)

        self.multi_gaussian_stds = np.array(params[('multi_gaussian_stds', [0.05, 0.1, 0.15, 0.2, 0.25], 'std deviations for the Gaussians')])
        default_multi_gaussian_weights = self.multi_gaussian_stds
        default_multi_gaussian_weights /= default_multi_gaussian_weights.sum()
        """standard deviations of Gaussians"""
        self.multi_gaussian_weights = np.array(params[('multi_gaussian_weights', default_multi_gaussian_weights.tolist(), 'weights for the multiple Gaussians')])
        """weights for the Gaussians"""
        self.gaussianStd_min = params[('gaussian_std_min', 0.00001, 'minimal allowed std for the Gaussians')]
        """minimal allowed standard deviation during optimization"""
        self.optimize_over_smoother_parameters = params[('optimize_over_smoother_parameters', False, 'if set to true the smoother will be optimized')]
        """determines if we should optimize over the smoother parameters"""

        assert len(self.multi_gaussian_weights) == len(self.multi_gaussian_stds)

        weight_sum = self.multi_gaussian_weights.sum()
        if weight_sum != 1.:
            print('WARNING: multi-Gaussian weights do not sum to one. Projecting them.')
            self.multi_gaussian_weights += (1. - weight_sum) / len(self.multi_gaussian_weights)
            params['multi_gaussian_weights'] = self.multi_gaussian_weights.tolist()

        assert (np.array(self.multi_gaussian_weights)).sum() == 1.

        self.nr_of_gaussians = len(self.multi_gaussian_stds)
        self.gaussian_fourier_filter_generator = ce.GaussianFourierFilterGenerator(sz,spacing,self.nr_of_gaussians)
        self.optimizer_params = self._create_optimization_vector_parameters()

    def get_custom_optimizer_output_string(self):
        return ", smooth(stds)= " + np.array_str(self.get_gaussian_stds().data.numpy(),precision=3) + \
               ", smooth(weights)= " + np.array_str(self.get_gaussian_weights().data.numpy(),precision=3)

    def get_optimization_parameters(self):
        if self.optimize_over_smoother_parameters:
            return self.optimizer_params
        else:
            return None

    def _project_parameter_vector_if_necessary(self):
        # all standard deviations need to be positive and the weights need to be non-negative
        for i in range(self.nr_of_gaussians):
            if self.optimizer_params.data[i] <= self.gaussianStd_min:
                self.optimizer_params.data[i] = self.gaussianStd_min
            if self.optimizer_params.data[i + self.nr_of_gaussians] < 0:
                self.optimizer_params.data[i + self.nr_of_gaussians] = 0

        # now make sure the weights sum up to one and if not project them back
        weight_sum = self.optimizer_params.data[self.nr_of_gaussians:].sum()
        if weight_sum != 1.:
            self.optimizer_params.data[self.nr_of_gaussians:] += (1. - weight_sum) / self.nr_of_gaussians

    def _get_gaussian_weights_from_optimizer_params(self):
        # project if needed
        self._project_parameter_vector_if_necessary()
        return self.optimizer_params[self.nr_of_gaussians:]

    def _set_gaussian_weights_optimizer_params(self,gweights):
        self.optimizer_params.data[self.nr_of_gaussians:]=gweights

    def set_gaussian_weights(self, gweights):
        """
        Sets the weights for the multi-Gaussian smoother
        :param gweights: vector of weights
        :return: n/a
        """
        self.params['multi_gaussian_weights'] = gweights
        self._set_gaussian_weights_optimizer_params(gweights)

    def get_gaussian_weights(self):
        """
        Returns the weights for the multi-Gaussian smoother
        :return: vector of weights
        """
        gaussianWeights = self._get_gaussian_weights_from_optimizer_params()
        return gaussianWeights

    def _get_gaussian_stds_from_optimizer_params(self):
        # project if needed
        self._project_parameter_vector_if_necessary()
        return self.optimizer_params[0:self.nr_of_gaussians]

    def _set_gaussian_stds_optimizer_params(self,g_stds):
        self.optimizer_params.data[0:self.nr_of_gaussians]=g_stds

    def set_gaussian_stds(self, gstds):
        """
        Set the standard deviation of the Gaussian filter

        :param gstd: standard deviation
        """
        self.params['multi_gaussian_stds'] = gstds
        self._set_gaussian_std_optimizer_params(gstds)

    def get_gaussian_stds(self):
        """
        Return the standard deviations of the Gaussian filters

        :return: standard deviation of Gaussian filter
        """
        gaussianStds = self._get_gaussian_stds_from_optimizer_params()
        return gaussianStds

    def smooth_scalar_field(self, v, vout=None):
        """
        Smooth the scalar field using Gaussian smoothing in the Fourier domain

        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :return: smoothed image
        """

        # just doing a Gaussian smoothing
        if vout is not None:
            vout = ce.fourier_multi_gaussian_convolution(v,self.gaussian_fourier_filter_generator,self.get_gaussian_stds(),self.get_gaussian_weights())
            return vout
        else:
            return ce.fourier_multi_gaussian_convolution(v,self.gaussian_fourier_filter_generator,self.get_gaussian_stds(),self.get_gaussian_weights())

    def inverse_smooth_scalar_field(self, v, vout=None):
        """
        Inverse-smooth the scalar field using Gaussian smoothing in the Fourier domain
        (with the inverse of the Fourier transform of the Gaussian; EXPERIMENTAL, requires regularization
        to avoid divisions by zero. DO NOT USE AT THE MOMENT.)

        :param v: image to inverse-smooth
        :param vout: if not None returns the result in this variable
        :return: inverse-smoothed image
        """

        raise ValueError("Not yet implemented")

    def _create_optimization_vector_parameters(self):
        self.optimizer_params = utils.create_vector_parameter(2 * self.nr_of_gaussians)
        for i in range(self.nr_of_gaussians):
            self.optimizer_params.data[i] = self.multi_gaussian_stds[i]
            self.optimizer_params.data[i + self.nr_of_gaussians] = self.multi_gaussian_weights[i]
        return self.optimizer_params


class AdaptiveSmoother(Smoother):
    """
    Performs Gaussian smoothing via convolution in the Fourier domain. Much faster for large dimensions
    than spatial Gaussian smoothing on the CPU in large dimensions.
    """

    def __init__(self, sz, spacing, params):
        super(AdaptiveSmoother, self).__init__(sz, spacing, params)
        """standard deviation of Gaussian"""
        self.net_sched = params[('net_sched', 'm_fixed', 'std for the Gaussian')]
        self.sz = sz
        self.spacing = spacing
        self.FFilter = None
        self.moving, self.target = params['input']
        inputs ={}
        inputs['s'] = self.moving
        inputs['t'] = self.target
        self.smoother = utils.AdpSmoother(inputs, len(sz))
        utils.init_weights(self.smoother, init_type='normal')
        if USE_CUDA:
           self.smoother = self.smoother.cuda()
        #self.sm_filter = utils.nn.Conv2d(2, 2, 3, 1, padding=1,groups=2, bias=False).cuda()

        """filter in Fourier domain"""

    def adaptive_smooth(self, m, phi, using_map=True):
        """

        :param m:
        :param phi:
        :param using_map:
        :return:
        """
        if using_map:
            I = utils.compute_warped_image_multiNC(self.moving, phi, self.spacing)
        else:
            I = None
        v = self.smoother(m, I.detach())
        return v


    def smooth_scalar_field(self, v, vout=None):
        pass

    def inverse_smooth_scalar_field(self, v, vout=None):
        """
        Inverse-smooth the scalar field using Gaussian smoothing in the Fourier domain
        (with the inverse of the Fourier transform of the Gaussian; EXPERIMENTAL, requires regularization
        to avoid divisions by zero. DO NOT USE AT THE MOMENT.)

        :param v: image to inverse-smooth
        :param vout: if not None returns the result in this variable
        :return: inverse-smoothed image
        """

        raise ValueError("inverse smooth scalar field is not implemented")


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
        self.default_smoother_type = 'multiGaussian'
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
                                          'type of smoother (diffusion|gaussian|adaptive_gaussian|multiGaussian|adaptive_multiGaussian|gaussianSpatial|adaptiveNet)' )]
        if smootherType=='diffusion':
            return DiffusionSmoother(self.sz,self.spacing,cparams)
        elif smootherType=='gaussian':
            return SingleGaussianFourierSmoother(self.sz,self.spacing,cparams)
        elif smootherType=='adaptive_gaussian':
            return AdaptiveSingleGaussianFourierSmoother(self.sz,self.spacing,cparams)
        elif smootherType=='multiGaussian':
            return MultiGaussianFourierSmoother(self.sz,self.spacing,cparams)
        elif smootherType=='adaptive_multiGaussian':
            return AdaptiveMultiGaussianFourierSmoother(self.sz,self.spacing,cparams)
        elif smootherType=='gaussianSpatial':
            return GaussianSpatialSmoother(self.sz,self.spacing,cparams)
        elif smootherType=='adaptiveNet':
            return AdaptiveSmoother(self.sz, self.spacing, cparams)
        else:
            raise ValueError( 'Smoother: ' + smootherType + ' not known')
