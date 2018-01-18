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
        self.multi_gaussian_optimizer_params = None
        """parameters that will be exposed to the optimizer"""
        self.ISource = None
        """For smoothers that make use of the map, stores the source image to which the map can be applied"""

    def set_source_image(self,ISource):
        """
        Sets the source image. Useful for smoothers that have as an input the map and need to compute a warped source image.

        :param ISource: source image
        :return: n/a
        """

        self.ISource = ISource

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

    def get_custom_optimizer_output_values(self):
        """
        Returns a customized dictionary with additional values describing a smoother's setting.
        Will become part of the optimization history.
        Useful to overwrite if optimizing over smoother parameters.

        :return: string
        """
        return None

    @abstractmethod
    def apply_smooth(self, v, vout=None, I_or_phi=None, variables_from_optimizer=None):
        """
        Abstract method to smooth a vector field. Only this method should be overwritten in derived classes.

        :param v: input field to smooth  BxCxXxYxZ
        :param vout: if not None then result is returned in this variable
        :param I_or_phi: list that either only contains an image or a map as first entry and False or True as the second entry for image/map respectively
                Can be used to design image-dependent smoothers; typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
        :return: should return the a smoothed scalar field, image dimension BxCXxYxZ
        """
        pass



    def smooth(self, v, vout=None, I_or_phi=None, variables_from_optimizer=None):
        """
        Smoothes a vector field of dimension BxCxXxYxZ,

        :param v: vector field to smooth BxCxXxYxZ
        :param vout: if not None then result is returned in this variable
        :param I_or_phi: list that either only contains an image or a map as first entry and False or True as the second entry for image/map respectively
                Can be used to design image-dependent smoothers; typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
        :return: smoothed vector field BxCxXxYxZ
        """
        sz = v.size()
        if vout is not None:
            Sv = vout
        else:
            Sv = Variable(MyTensor(v.size()).zero_())

        Sv[:] = self.apply_smooth(v,vout,I_or_phi,variables_from_optimizer)    # here must use :, very important !!!!
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

    def apply_smooth(self, v, vout=None, I_or_phi=None, variables_from_optimizer=None):
        """
        Smoothes a scalar field of dimension XxYxZ

        :param v: input image
        :param vout: if not None returns the result in this variable
        :param I_or_phi: list that either only contains an image or a map as first entry and False or True as the second entry for image/map respectively
                Can be used to design image-dependent smoothers; typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
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
        centered_id = utils.centered_identity_map(k_sz,self.spacing)
        g = utils.compute_normalized_gaussian(centered_id, mus, stds)

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

    def apply_smooth(self, v, vout=None, I_or_phi=None, variables_from_optimizer=None):
        """
        Smooth the scalar field using Gaussian smoothing in the spatial domain

        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :param I_or_phi: list that either only contains an image or a map as first entry and False or True as the second entry for image/map respectively
                Can be used to design image-dependent smoothers; typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
        :return: smoothed image
        """

        self.sz = v.size()
        if self.filter is None:
            self._create_filter()
        # just doing a Gaussian smoothing
        return self._filter_input_with_padding(v, vout)



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

    def apply_smooth(self, v, vout=None, I_or_phi=None, variables_from_optimizer=None):
        """
        Smooth the scalar field using Gaussian smoothing in the Fourier domain

        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :param I_or_phi: list that either only contains an image or a map as first entry and False or True as the second entry for image/map respectively
                Can be used to design image-dependent smoothers; typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
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



class AdaptiveSingleGaussianFourierSmoother(GaussianSmoother):
    """
    Performs Gaussian smoothing via convolution in the Fourier domain. Much faster for large dimensions
    than spatial Gaussian smoothing on the CPU in large dimensions.
    """

    def __init__(self, sz, spacing, params):
        super(AdaptiveSingleGaussianFourierSmoother, self).__init__(sz, spacing, params)
        self.gaussianStd = np.array(params[('gaussian_std', [0.15], 'std for the Gaussian')])
        """standard deviation of Gaussian"""
        self.gaussianStd_min = params[('gaussian_std_min', 0.01, 'minimal allowed std for the Gaussian')]
        """minimal allowed standard deviation during optimization"""
        self.optimize_over_smoother_parameters = params[('optimize_over_smoother_parameters', False, 'if set to true the smoother will be optimized')]
        """determines if we should optimize over the smoother parameters"""
        self.start_optimize_over_smoother_parameters_at_iteration = \
            params[('start_optimize_over_smoother_parameters_at_iteration', 0, 'Does not optimize the parameters before this iteration')]

        self.gaussian_fourier_filter_generator = ce.GaussianFourierFilterGenerator(sz,spacing)

        self.optimizer_params = self._create_optimization_vector_parameters()

    def get_custom_optimizer_output_string(self):
        return ", smooth(std)= " + np.array_str(self.get_gaussian_std()[0].data.numpy(),precision=3)

    def get_custom_optimizer_values(self):
        return {'smoother_std': self.get_gaussian_std()[0].data.numpy().copy()}

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

    def apply_smooth(self, v, vout=None, I_or_phi=None, variables_from_optimizer=None):
        """
        Smooth the scalar field using Gaussian smoothing in the Fourier domain

        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :param I_or_phi: list that either only contains an image or a map as first entry and False or True as the second entry for image/map respectively
                Can be used to design image-dependent smoothers; typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
        :return: smoothed image
        """

        if not self.optimize_over_smoother_parameters or (variables_from_optimizer is None):
            compute_std_gradient = False
        else:
            if self.start_optimize_over_smoother_parameters_at_iteration <= variables_from_optimizer['iter']:
                compute_std_gradient = True
            else:
                compute_std_gradient = False

        # just doing a Gaussian smoothing
        if vout is not None:
            vout = ce.fourier_single_gaussian_convolution(v,self.gaussian_fourier_filter_generator,self.get_gaussian_std(),compute_std_gradient)
            return vout
        else:
            return ce.fourier_single_gaussian_convolution(v,self.gaussian_fourier_filter_generator,self.get_gaussian_std(),compute_std_gradient)


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
        self.gaussianStd_min = params[('gaussian_std_min', 0.01 ,'minimal allowed std for the Gaussian' )]
        """minimal allowed standard deviation during optimization"""

    def _create_filter(self):

        mus = np.zeros(self.dim)
        stds = self.gaussianStd*np.ones(self.dim)
        centered_id = utils.centered_identity_map(self.sz,self.spacing)
        g = utils.compute_normalized_gaussian(centered_id, mus, stds)

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
        centered_id = utils.centered_identity_map(self.sz,self.spacing)

        assert len(self.multi_gaussian_stds)>0
        assert len(self.multi_gaussian_weights)>0

        nr_of_gaussians = len(self.multi_gaussian_stds)

        for nr in range(nr_of_gaussians):
            stds = self.multi_gaussian_stds[nr] * np.ones(self.dim)
            g = self.multi_gaussian_weights[nr] * utils.compute_normalized_gaussian(centered_id, mus, stds)

            if nr==0:
                self.FFilter,_ = ce.create_complex_fourier_filter(g, self.sz)
            else:
                cFilter,_ = ce.create_complex_fourier_filter(g, self.sz)
                self.FFilter += cFilter


class ParameterizedMultiGaussianFourierSmoother(GaussianSmoother):
    """
    Base class for adaptive smoothers making use of multiple Gaussians
    """

    def __init__(self, sz, spacing, params):
        super(ParameterizedMultiGaussianFourierSmoother, self).__init__(sz, spacing, params)

        self.multi_gaussian_stds = np.array(params[('multi_gaussian_stds', [0.05, 0.1, 0.15, 0.2, 0.25], 'std deviations for the Gaussians')])
        default_multi_gaussian_weights = self.multi_gaussian_stds
        default_multi_gaussian_weights /= default_multi_gaussian_weights.sum()
        """standard deviations of Gaussians"""
        self.multi_gaussian_weights = np.array(params[('multi_gaussian_weights', default_multi_gaussian_weights.tolist(), 'weights for the multiple Gaussians')])
        """weights for the Gaussians"""
        self.gaussianStd_min = params[('gaussian_std_min', 0.01, 'minimal allowed std for the Gaussians')]
        """minimal allowed standard deviation during optimization"""
        self.optimize_over_smoother_parameters = params[('optimize_over_smoother_parameters', False, 'if set to true the smoother will be optimized')]
        """determines if we should optimize over the smoother parameters"""
        self.start_optimize_over_smoother_parameters_at_iteration = \
            params[('start_optimize_over_smoother_parameters_at_iteration', 0, 'Does not optimize the parameters before this iteration')]

        assert len(self.multi_gaussian_weights) == len(self.multi_gaussian_stds)

        weight_sum = self.multi_gaussian_weights.sum()
        if weight_sum != 1.:
            print('WARNING: multi-Gaussian weights do not sum to one. Projecting them.')
            self.multi_gaussian_weights += (1. - weight_sum) / len(self.multi_gaussian_weights)
            params['multi_gaussian_weights'] = self.multi_gaussian_weights.tolist()

        assert (np.array(self.multi_gaussian_weights)).sum() == 1.

        self.nr_of_gaussians = len(self.multi_gaussian_stds)
        self.gaussian_fourier_filter_generator = ce.GaussianFourierFilterGenerator(sz,spacing,self.nr_of_gaussians)
        self.multi_gaussian_optimizer_params = self._create_multi_gaussian_optimization_vector_parameters()

    def get_custom_optimizer_output_string(self):
        return ", smooth(stds)= " + np.array_str(self.get_gaussian_stds().data.numpy(),precision=3) + \
               ", smooth(weights)= " + np.array_str(self.get_gaussian_weights().data.numpy(),precision=3)

    def get_custom_optimizer_output_values(self):
        return {'smoother_stds': self.get_gaussian_stds().data.numpy().copy(), 'smoother_weights': self.get_gaussian_weights().data.numpy().copy()}

    def _project_parameter_vector_if_necessary(self):
        # all standard deviations need to be positive and the weights need to be non-negative
        for i in range(self.nr_of_gaussians):
            if self.multi_gaussian_optimizer_params.data[i] <= self.gaussianStd_min:
                self.multi_gaussian_optimizer_params.data[i] = self.gaussianStd_min
            if self.multi_gaussian_optimizer_params.data[i + self.nr_of_gaussians] < 0:
                self.multi_gaussian_optimizer_params.data[i + self.nr_of_gaussians] = 0

        # now make sure the weights sum up to one and if not project them back
        weight_sum = self.multi_gaussian_optimizer_params.data[self.nr_of_gaussians:].sum()
        if weight_sum != 1.:
            self.multi_gaussian_optimizer_params.data[self.nr_of_gaussians:] += (1. - weight_sum) / self.nr_of_gaussians

    def _get_gaussian_weights_from_optimizer_params(self):
        # project if needed
        self._project_parameter_vector_if_necessary()
        return self.multi_gaussian_optimizer_params[self.nr_of_gaussians:]

    def _set_gaussian_weights_optimizer_params(self,gweights):
        self.multi_gaussian_optimizer_params.data[self.nr_of_gaussians:]=gweights

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
        return self.multi_gaussian_optimizer_params[0:self.nr_of_gaussians]

    def _set_gaussian_stds_optimizer_params(self,g_stds):
        self.multi_gaussian_optimizer_params.data[0:self.nr_of_gaussians]=g_stds

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

    def _create_multi_gaussian_optimization_vector_parameters(self):
        self.multi_gaussian_optimizer_params = utils.create_vector_parameter(2 * self.nr_of_gaussians)
        for i in range(self.nr_of_gaussians):
            self.multi_gaussian_optimizer_params.data[i] = self.multi_gaussian_stds[i]
            self.multi_gaussian_optimizer_params.data[i + self.nr_of_gaussians] = self.multi_gaussian_weights[i]
        return self.multi_gaussian_optimizer_params

# CONTINUE HERE. ALSO TEST THAT THE ADAPTIVE MULTI-GAUSSIAN SMOOTHER STILL WORKS

# todo: clean this class up. Only this one optimizes directly over the weights, but not the learned version
class AdaptiveMultiGaussianFourierSmoother(ParameterizedMultiGaussianFourierSmoother):
    """
    Adaptive multi-Gaussian Fourier smoother. Allows optimization over weights and standard deviations
    """

    def __init__(self, sz, spacing, params):
        super(AdaptiveMultiGaussianFourierSmoother, self).__init__(sz, spacing, params)

    def get_optimization_parameters(self):
        if self.optimize_over_smoother_parameters:
            return self.multi_gaussian_optimizer_params
        else:
            return None

    def apply_smooth(self, v, vout=None, I_or_phi=None, variables_from_optimizer=None):
        """
        Smooth the scalar field using Gaussian smoothing in the Fourier domain

        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :param I_or_phi: list that either only contains an image or a map as first entry and False or True as the second entry for image/map respectively
                Can be used to design image-dependent smoothers; typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
        :return: smoothed image
        """

        # just do a multi-Gaussian smoothing

        if not self.optimize_over_smoother_parameters or (variables_from_optimizer is None):
            compute_weight_and_std_gradients = False
        else:
           if self.start_optimize_over_smoother_parameters_at_iteration<=variables_from_optimizer['iter']:
               compute_weight_and_std_gradients = True
           else:
               compute_weight_and_std_gradients = False

        if vout is not None:
            vout = ce.fourier_multi_gaussian_convolution(v,self.gaussian_fourier_filter_generator,
                                                         self.get_gaussian_stds(),self.get_gaussian_weights(),compute_weight_and_std_gradients)
            return vout
        else:
            return ce.fourier_multi_gaussian_convolution(v,self.gaussian_fourier_filter_generator,
                                                         self.get_gaussian_stds(),self.get_gaussian_weights(),compute_weight_and_std_gradients)



class LearnedMultiGaussianCombinationFourierSmoother(ParameterizedMultiGaussianFourierSmoother):
    """
    Adaptive multi-Gaussian Fourier smoother. Allows optimization over weights and standard deviations
    """

    def __init__(self, sz, spacing, params):
        super(LearnedMultiGaussianCombinationFourierSmoother, self).__init__(sz, spacing, params)

    def get_optimization_parameters(self):
        if self.optimize_over_smoother_parameters:
            return self.multi_gaussian_optimizer_params
        else:
            return None

    def apply_smooth(self, v, vout=None, I_or_phi=None, variables_from_optimizer=None):
        """
        Smooth the scalar field using Gaussian smoothing in the Fourier domain

        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :param I_or_phi: list that either only contains an image or a map as first entry and False or True as the second entry for image/map respectively
                Can be used to design image-dependent smoothers; typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
        :return: smoothed image
        """

        # just do a multi-Gaussian smoothing

        if not self.optimize_over_smoother_parameters or (variables_from_optimizer is None):
            compute_std_gradients = False
        else:
           if self.start_optimize_over_smoother_parameters_at_iteration<=variables_from_optimizer['iter']:
               compute_std_gradients = True
           else:
               compute_std_gradients = False

        # collection of smoothed vector fields
        vcollection = ce.fourier_set_of_gaussian_convolutions(v, self.gaussian_fourier_filter_generator,
                                                       self.get_gaussian_stds(), compute_std_gradients)

        # todo: now we need a small neural net that can summarize this
        # todo: this needs to be such that in the end there are a combination of weight fields that sum up
        # todo: to one to create a new value, to make sure this is a proper smoothing

        raise ValueError('Not yet implemented')

        if vout is not None:
            vout = 0 # todo: change
            return vout
        else:
            return 0 # todo: change



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
        EXPERIMENTAL: DO NOT YET USE! #todo: fix this

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


    def apply_smooth(self, v, vout=None, I_or_phi=None, variables_from_optimizer=None):
        pass



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
        elif smootherType=='learned_multiGaussianCombination':
            return LearnedMultiGaussianCombinationFourierSmoother(self.sz,self.spacing,cparams)
        elif smootherType=='gaussianSpatial':
            return GaussianSpatialSmoother(self.sz,self.spacing,cparams)
        elif smootherType=='adaptiveNet':
            return AdaptiveSmoother(self.sz, self.spacing, cparams)
        else:
            raise ValueError( 'Smoother: ' + smootherType + ' not known')
