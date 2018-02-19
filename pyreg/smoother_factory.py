"""
This package implements various types of smoothers.
"""

from abc import ABCMeta, abstractmethod

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from data_wrapper import USE_CUDA, MyTensor, AdaptVal

import finite_differences as fd
import utils
import custom_pytorch_extensions as ce
import module_parameters as pars
import deep_smoothers

import collections

def get_state_dict_for_module(state_dict,module_name):

    res_dict = collections.OrderedDict()
    for k in state_dict.keys():
        if k.startswith(module_name + '.'):
            adapted_key = k[len(module_name)+1:]
            res_dict[adapted_key] = state_dict[k]
    return res_dict

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

    def associate_parameters_with_module(self,module):
        """
        Associates parameters that should be optimized with the given module.

        :param module: module to associate the parameters to
        :return: set of parameters that were associated
        """
        return set()


    def get_penalty(self):
        """
        Can be overwritten by a smoother to return a custom penalty which will be added to the
        optimization cost. For example to put penalities on smoother paramters that are being optimized over.
        :return: scalar value; penalty for smoother
        """
        return 0

    def set_state_dict(self,state_dict):
        """
        If the smoother contains a torch state-dict, this function allows setting it externally (to initialize as needed).
        This is typically not needed as it will be set via a registration model for example, but can be useful
        for external testing of the smoother. This is also different from smoother parameters and only
        affects parameters that may be optimized over.

        :param state_dict: OrderedDict containing the state information
        :return: n/a
        """
        pass

    def get_state_dict(self):
        """
        If the smoother contains a torch state-dict, this function returns it

        :return: state dict as an OrderedDict
        """
        raise ValueError('Not yet implemented')
        return None

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
    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None):
        """
        Abstract method to smooth a vector field. Only this method should be overwritten in derived classes.

        :param v: input field to smooth  BxCxXxYxZ
        :param vout: if not None then result is returned in this variable
        :param pars: dictionary that can contain various extra variables; for smoother this will for example be
            the current image 'I' or the current map 'phi'. typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
        :return: should return the a smoothed scalar field, image dimension BxCXxYxZ
        """
        pass



    def smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None):
        """
        Smoothes a vector field of dimension BxCxXxYxZ,

        :param v: vector field to smooth BxCxXxYxZ
        :param vout: if not None then result is returned in this variable
        :param pars: dictionary that can contain various extra variables; for smoother this will for example be
            the current image 'I' or the current map 'phi'. typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
        :return: smoothed vector field BxCxXxYxZ
        """
        sz = v.size()
        if vout is not None:
            Sv = vout
        else:
            Sv = Variable(MyTensor(v.size()).zero_())

        Sv[:] = self.apply_smooth(v,vout,pars,variables_from_optimizer)    # here must use :, very important !!!!
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

    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None):
        """
        Smoothes a scalar field of dimension XxYxZ

        :param v: input image
        :param vout: if not None returns the result in this variable
        :param pars: dictionary that can contain various extra variables; for smoother this will for example be
            the current image 'I' or the current map 'phi'. typically not used.
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

    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None):
        """
        Smooth the scalar field using Gaussian smoothing in the spatial domain

        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :param pars: dictionary that can contain various extra variables; for smoother this will for example be
            the current image 'I' or the current map 'phi'. typically not used.
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

    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None):
        """
        Smooth the scalar field using Gaussian smoothing in the Fourier domain

        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :param pars: dictionary that can contain various extra variables; for smoother this will for example be
            the current image 'I' or the current map 'phi'. typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
        :return: smoothed image
        """

        # just doing a Gaussian smoothing
        # we need to instantiate a new filter function here every time for the autograd to work
        if self.FFilter is None:
            self._create_filter()
        if vout is not None:
            vout[:] = ce.fourier_convolution(v, self.FFilter)
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

    def associate_parameters_with_module(self,module):
        module.register_parameter('multi_gaussian_std_and_weights',self.optimizer_params)
        return set({'multi_gaussian_std_and_weights'})

    def get_custom_optimizer_output_string(self):
        return ", smooth(std)= " + np.array_str(self.get_gaussian_std()[0].data.cpu().numpy(),precision=3)

    def get_custom_optimizer_values(self):
        return {'smoother_std': self.get_gaussian_std()[0].data.cpu().numpy().copy()}

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

    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None):
        """
        Smooth the scalar field using Gaussian smoothing in the Fourier domain

        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :param pars: dictionary that can contain various extra variables; for smoother this will for example be
            the current image 'I' or the current map 'phi'. typically not used.
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
            vout[:] = ce.fourier_single_gaussian_convolution(v,self.gaussian_fourier_filter_generator,self.get_gaussian_std(),compute_std_gradient)
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
        default_multi_gaussian_weights = self.multi_gaussian_stds.copy()
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


class AdaptiveMultiGaussianFourierSmoother(GaussianSmoother):
    """
    Adaptive multi-Gaussian smoother which allows optimizing over weights and standard deviations
    """

    def __init__(self, sz, spacing, params):
        super(AdaptiveMultiGaussianFourierSmoother, self).__init__(sz, spacing, params)

        self.multi_gaussian_stds = np.array(params[('multi_gaussian_stds', [0.05, 0.1, 0.15, 0.2, 0.25], 'std deviations for the Gaussians')])
        """standard deviations of Gaussians"""

        self.gaussianStd_min = params[('gaussian_std_min', 0.01, 'minimal allowed std for the Gaussians')]
        """minimal allowed standard deviation during optimization"""

        self.default_weight_penalty = params[('default_weight_penalty', 1.0, 'factor by which the deviation from the dafault weights is penalized')]
        """penalty factor for deviation from default weights"""

        self.optimize_over_smoother_stds = params[('optimize_over_smoother_stds', False, 'if set to true the smoother will optimize over standard deviations')]
        """determines if we should optimize over the smoother standard deviations"""

        self.optimize_over_smoother_weights = params[('optimize_over_smoother_weights', False, 'if set to true the smoother will optimize over the *global* weights')]
        """determines if we should optimize over the smoother global weights"""

        # todo: maybe make this more generic; there is an explicit float here
        self.default_multi_gaussian_weights = AdaptVal(Variable(torch.from_numpy(self.multi_gaussian_stds.copy()).float(), requires_grad=False))
        self.default_multi_gaussian_weights /= self.default_multi_gaussian_weights.sum()

        self.multi_gaussian_weights = np.array(params[('multi_gaussian_weights', self.default_multi_gaussian_weights.data.cpu().numpy().tolist(),'weights for the multiple Gaussians')])
        """global weights for the Gaussians"""
        self.gaussianWeight_min = params[('gaussian_weight_min', 0.0001, 'minimal allowed weight for the Gaussians')]
        """minimal allowed weight during optimization"""

        weight_sum = self.multi_gaussian_weights.sum()
        if weight_sum != 1.:
            print('WARNING: multi-Gaussian weights do not sum to one. Projecting them.')
            self.multi_gaussian_weights += (1. - weight_sum) / len(self.multi_gaussian_weights)
            params['multi_gaussian_weights'] = self.multi_gaussian_weights.tolist()

        assert (np.array(self.multi_gaussian_weights)).sum() == 1.
        assert len(self.multi_gaussian_weights) == len(self.multi_gaussian_stds)

        self.nr_of_gaussians = len(self.multi_gaussian_stds)
        """number of Gaussians"""

        self.gaussian_fourier_filter_generator = ce.GaussianFourierFilterGenerator(sz, spacing, self.nr_of_gaussians)
        """creates the smoothed vector fields"""

        self.multi_gaussian_stds_optimizer_params = self._create_multi_gaussian_stds_optimization_vector_parameters()
        self.multi_gaussian_weights_optimizer_params = self._create_multi_gaussian_weights_optimization_vector_parameters()

        self.start_optimize_over_smoother_parameters_at_iteration = \
            params[('start_optimize_over_smoother_parameters_at_iteration', 0, 'Does not optimize the parameters before this iteration')]

        self.omt_power = params[('omt_power',2.0,'Power for the optimal mass transport (i.e., to which power distances are penalized')]
        """optimal mass transport power"""

    def get_default_multi_gaussian_weights(self):
        # todo: check, should it really return this?
        return self.multi_gaussian_weights_optimizer_params

    def associate_parameters_with_module(self, module):
        s = set()
        if self.optimize_over_smoother_stds:
            module.register_parameter('multi_gaussian_stds', self.multi_gaussian_stds_optimizer_params)
            s.add('multi_gaussian_stds')
        if self.optimize_over_smoother_weights:
            module.register_parameter('multi_gaussian_weights', self.multi_gaussian_weights_optimizer_params)
            s.add('multi_gaussian_weights')

        return s

    def get_custom_optimizer_output_string(self):
        output_str = ""
        if self.optimize_over_smoother_stds:
            output_str += ", smooth(stds)= " + np.array_str(self.get_gaussian_stds().data.cpu().numpy(), precision=3)
        if self.optimize_over_smoother_weights:
            output_str += ", smooth(weights)= " + np.array_str(self.get_gaussian_weights().data.cpu().numpy(),precision=3)

        output_str += ", smooth(penalty)= " + np.array_str(self.get_penalty().data.cpu().numpy(),precision=3)

        return output_str

    def get_custom_optimizer_output_values(self):
        return {'smoother_stds': self.get_gaussian_stds().data.cpu().numpy().copy(),
                'smoother_weights': self.get_gaussian_weights().data.cpu().numpy().copy(),
                'smoother_penalty': self.get_penalty().data.cpu().numpy().copy()}

    def set_state_dict(self, state_dict):

        if state_dict.has_key('multi_gaussian_stds'):
            self.multi_gaussian_stds_optimizer_params.data[:] = state_dict['multi_gaussian_stds']
        if state_dict.has_key('multi_gaussian_weights'):
            self.multi_gaussian_weights_optimizer_params.data[:] = state_dict['multi_gaussian_weights']

    def _project_parameter_vector_if_necessary(self):
        # all standard deviations need to be positive and the weights need to be non-negative
        for i in range(self.nr_of_gaussians):
            if self.multi_gaussian_stds_optimizer_params.data[i] <= self.gaussianStd_min:
                self.multi_gaussian_stds_optimizer_params.data[i] = self.gaussianStd_min

            if self.multi_gaussian_weights_optimizer_params.data[i] < self.gaussianWeight_min:
                self.multi_gaussian_weights_optimizer_params.data[i] = self.gaussianWeight_min

        # todo: change this normalization for the adaptive multi-Gaussian smoother
        # now make sure the weights sum up to one and if not project them back
        weight_sum = self.multi_gaussian_weights_optimizer_params.data.sum()
        if weight_sum != 1.:
            self.multi_gaussian_weights_optimizer_params.data /= weight_sum

    def _get_gaussian_weights_from_optimizer_params(self):
        # project if needed
        self._project_parameter_vector_if_necessary()
        return self.multi_gaussian_weights_optimizer_params

    def _set_gaussian_weights_optimizer_params(self, gweights):
        self.multi_gaussian_weights_optimizer_params.data[:] = gweights

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
        return self.multi_gaussian_stds_optimizer_params

    def _set_gaussian_stds_optimizer_params(self, g_stds):
        self.multi_gaussian_stds_optimizer_params.data[:] = g_stds

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

    def _create_multi_gaussian_stds_optimization_vector_parameters(self):
        self.multi_gaussian_stds_optimizer_params = utils.create_vector_parameter(self.nr_of_gaussians)
        for i in range(self.nr_of_gaussians):
            self.multi_gaussian_stds_optimizer_params.data[i] = self.multi_gaussian_stds[i]
        return self.multi_gaussian_stds_optimizer_params

    def _create_multi_gaussian_weights_optimization_vector_parameters(self):
        self.multi_gaussian_weights_optimizer_params = utils.create_vector_parameter(self.nr_of_gaussians)
        for i in range(self.nr_of_gaussians):
            self.multi_gaussian_weights_optimizer_params.data[i] = self.multi_gaussian_weights[i]
        return self.multi_gaussian_weights_optimizer_params

    def _compute_omt_penalty_for_weight_vectors(self,weights,multi_gaussian_stds):

        penalty = Variable(MyTensor(1).zero_(), requires_grad=False)

        # todo: check that this is properly handled for the learned optimizer (i.e., as a variable for optimization opposed to a constant)
        max_std = torch.max(multi_gaussian_stds)
        for i, s in enumerate(multi_gaussian_stds):
            penalty += weights[i] * ((s - max_std) ** self.omt_power)

        return penalty

    def get_penalty(self):
        # puts an squared two-norm penalty on the weights as deviations from the baseline
        # also adds a penalty for the network parameters

        current_penalty = self._compute_omt_penalty_for_weight_vectors(self.get_gaussian_weights(),self.get_gaussian_stds())
        penalty = current_penalty*self.default_weight_penalty

        return penalty

    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None):
        """
        Smooth the scalar field using Gaussian smoothing in the Fourier domain

        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :param pars: dictionary that can contain various extra variables; for smoother this will for example be
            the current image 'I' or the current map 'phi'. typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
        :return: smoothed image
        """

        # just do a multi-Gaussian smoothing
        compute_weight_gradients = self.optimize_over_smoother_weights
        compute_std_gradients = self.optimize_over_smoother_stds

        if variables_from_optimizer is not None:
            if self.start_optimize_over_smoother_parameters_at_iteration>variables_from_optimizer['iter']:
                compute_std_gradients = False
                compute_weight_gradients = False

        if vout is not None:
            vout[:] = ce.fourier_multi_gaussian_convolution(v,self.gaussian_fourier_filter_generator,
                                                         self.get_gaussian_stds(),self.get_gaussian_weights(),
                                                            compute_std_gradients=compute_std_gradients,
                                                            compute_weight_gradients=compute_weight_gradients)
            return vout
        else:
            return ce.fourier_multi_gaussian_convolution(v,self.gaussian_fourier_filter_generator,
                                                         self.get_gaussian_stds(),self.get_gaussian_weights(),
                                                         compute_std_gradients=compute_std_gradients,
                                                         compute_weight_gradients=compute_weight_gradients)


class LearnedMultiGaussianCombinationFourierSmoother(GaussianSmoother):
    """
    Adaptive multi-Gaussian Fourier smoother. Allows optimization over weights and standard deviations
    """

    def __init__(self, sz, spacing, params):
        super(LearnedMultiGaussianCombinationFourierSmoother, self).__init__(sz, spacing, params)

        self.multi_gaussian_stds = np.array(
            params[('multi_gaussian_stds', [0.05, 0.1, 0.15, 0.2, 0.25], 'std deviations for the Gaussians')])
        """standard deviations of Gaussians"""
        self.gaussianStd_min = params[('gaussian_std_min', 0.01, 'minimal allowed std for the Gaussians')]
        """minimal allowed standard deviation during optimization"""

        self.default_weight_penalty = params[('default_weight_penalty', 1.0, 'factor by which the deviation from the dafault weights is penalized')]
        """penalty factor for deviation from default weights"""

        self.network_penalty = params[('network_penalty', 0.0, 'factor by which the L2 norm of network weights is penalized')]
        """penalty factor for L2 norm of network weights"""

        self.encourage_spatial_weight_consistency = params[('encourage_spatial_weight_consistency',True,'If True tries adds an averaging term in the network to make weights spatially consistent')]
        """If set to true predicted weights are first spatially averaged before normalization in network"""

        #self.optimize_over_smoother_stds = params[('optimize_over_smoother_stds', False, 'if set to true the smoother will optimize over standard deviations')]
        #"""determines if we should optimize over the smoother standard deviations"""
        self.optimize_over_smoother_stds = False # disabled for now, as this is not being used

        #self.optimize_over_smoother_weights = params[(
        #'optimize_over_smoother_weights', False, 'if set to true the smoother will optimize over the *global* weights')]
        #"""determines if we should optimize over the smoother global weights"""
        self.optimize_over_smoother_weights = False # disabled for now, as this is not being used

        self.start_optimize_over_smoother_parameters_at_iteration = \
            params[('start_optimize_over_smoother_parameters_at_iteration', 0,
                    'Does not optimize the parameters before this iteration')]
        """at what iteration the optimization over weights or stds should start"""

        # todo: maybe make this more generic; there is an explicit float here
        self.default_multi_gaussian_weights = AdaptVal(Variable(torch.from_numpy(self.multi_gaussian_stds.copy()).float(),requires_grad=False))
        self.default_multi_gaussian_weights /= self.default_multi_gaussian_weights.sum()

        self.multi_gaussian_weights = np.array(params[('multi_gaussian_weights', self.default_multi_gaussian_weights.data.cpu().numpy().tolist(), 'weights for the multiple Gaussians')])
        """global weights for the Gaussians"""
        self.gaussianWeight_min = params[('gaussian_weight_min', 0.0001, 'minimal allowed weight for the Gaussians')]
        """minimal allowed weight during optimization"""

        self.nr_of_gaussians = len(self.multi_gaussian_stds)
        """number of Gaussians"""

        self.gaussian_fourier_filter_generator = ce.GaussianFourierFilterGenerator(sz, spacing, self.nr_of_gaussians)
        """creates the smoohted vector fields"""

        self.multi_gaussian_stds_optimizer_params = self._create_multi_gaussian_stds_optimization_vector_parameters()
        self.multi_gaussian_weights_optimizer_params = self._create_multi_gaussian_weights_optimization_vector_parameters()

        #self.ws = deep_smoothers.ConsistentWeightedSmoothingModel(self.nr_of_gaussians,self.multi_gaussian_stds,self.dim,params=params)
        self.ws = deep_smoothers.DeepSmootherFactory(nr_of_gaussians=self.nr_of_gaussians,gaussian_stds=self.multi_gaussian_stds,dim=self.dim).create_deep_smoother(params)
        """learned mini-network to predict multi-Gaussian smoothing weights"""

        self.debug_retain_computed_local_weights = False
        self.debug_computed_local_weights = None

    def get_default_multi_gaussian_weights(self):
        # todo: check, should it really return this?
        return self.multi_gaussian_weights_optimizer_params

    def get_debug_computed_local_weights(self):
        return self.debug_computed_local_weights

    def set_debug_retain_computed_local_weights(self,val):
        self.debug_retain_computed_local_weights = True

    def associate_parameters_with_module(self,module):
        s = set()
        if self.optimize_over_smoother_stds:
            module.register_parameter('multi_gaussian_stds',self.multi_gaussian_stds_optimizer_params)
            s.add('multi_gaussian_stds')
        if self.optimize_over_smoother_weights:
            module.register_parameter('multi_gaussian_weights', self.multi_gaussian_weights_optimizer_params)
            s.add('multi_gaussian_weights')
        module.add_module('weighted_smoothing_net',self.ws)
        sd = self.ws.state_dict()
        for key in sd:
            s.add('weighted_smoothing_net.' + str(key))

        return s

    def get_custom_optimizer_output_string(self):
        output_str = ""
        if self.optimize_over_smoother_stds:
            output_str += ", smooth(stds)= " + np.array_str(self.get_gaussian_stds().data.cpu().numpy(), precision=3)
        if self.optimize_over_smoother_weights:
            output_str += ", smooth(weights)= " + np.array_str(self.get_gaussian_weights().data.cpu().numpy(),precision=3)

        output_str += ", smooth(penalty)= " + np.array_str(self.get_penalty().data.cpu().numpy(),precision=3)

        return output_str

    def get_custom_optimizer_output_values(self):
        return {'smoother_stds': self.get_gaussian_stds().data.cpu().numpy().copy(),
                'smoother_weights': self.get_gaussian_weights().data.cpu().numpy().copy(),
                'smoother_penalty': self.get_penalty().data.cpu().numpy().copy()}

    def set_state_dict(self,state_dict):

        if state_dict.has_key('multi_gaussian_stds'):
            self.multi_gaussian_stds_optimizer_params.data[:] = state_dict['multi_gaussian_stds']
        if state_dict.has_key('multi_gaussian_weights'):
            self.multi_gaussian_weights_optimizer_params.data[:] = state_dict['multi_gaussian_weights']
        # first check if the learned smoother has already been initialized
        if len(self.ws.state_dict())==0:
            # has not been initialized, we need to initialize it before we can load the dictionary
            nr_of_image_channels = self.ws.get_number_of_image_channels_from_state_dict(state_dict,self.dim)
            self.ws._init(nr_of_image_channels,self.dim)
        self.ws.load_state_dict(get_state_dict_for_module(state_dict,'weighted_smoothing_net'))

    def _project_parameter_vector_if_necessary(self):
        # all standard deviations need to be positive and the weights need to be non-negative
        for i in range(self.nr_of_gaussians):
            if self.multi_gaussian_stds_optimizer_params.data[i] <= self.gaussianStd_min:
                self.multi_gaussian_stds_optimizer_params.data[i] = self.gaussianStd_min

            if self.multi_gaussian_weights_optimizer_params.data[i] < self.gaussianWeight_min:
                self.multi_gaussian_weights_optimizer_params.data[i] = self.gaussianWeight_min

        # todo: change this normalization for the adaptive multi-Gaussian smoother
        # now make sure the weights sum up to one and if not project them back
        weight_sum = self.multi_gaussian_weights_optimizer_params.data.sum()
        if weight_sum != 1.:
            self.multi_gaussian_weights_optimizer_params.data /= weight_sum

    def _get_gaussian_weights_from_optimizer_params(self):
        # project if needed
        self._project_parameter_vector_if_necessary()
        return self.multi_gaussian_weights_optimizer_params

    def _set_gaussian_weights_optimizer_params(self,gweights):
        self.multi_gaussian_weights_optimizer_params.data[:]=gweights

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
        return self.multi_gaussian_stds_optimizer_params

    def _set_gaussian_stds_optimizer_params(self, g_stds):
        self.multi_gaussian_stds_optimizer_params.data[:] = g_stds

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

    def _create_multi_gaussian_stds_optimization_vector_parameters(self):
        self.multi_gaussian_stds_optimizer_params = utils.create_vector_parameter(self.nr_of_gaussians)
        for i in range(self.nr_of_gaussians):
            self.multi_gaussian_stds_optimizer_params.data[i] = self.multi_gaussian_stds[i]
        return self.multi_gaussian_stds_optimizer_params

    def _create_multi_gaussian_weights_optimization_vector_parameters(self):
        self.multi_gaussian_weights_optimizer_params = utils.create_vector_parameter(self.nr_of_gaussians)
        for i in range(self.nr_of_gaussians):
            self.multi_gaussian_weights_optimizer_params.data[i] = self.multi_gaussian_weights[i]
        return self.multi_gaussian_weights_optimizer_params

    def get_penalty(self):
        # puts an squared two-norm penalty on the weights as deviations from the baseline
        # also adds a penalty for the network parameters

        penalty = self.ws.get_current_penalty()*self.default_weight_penalty*self.spacing.prod()

        #print('omt penalty = ' + str(penalty.data.cpu().numpy()))

        total_number_of_parameters = 1
        par_penalty = Variable(MyTensor(1).zero_(),requires_grad=False)
        for p in self.ws.parameters():
            total_number_of_parameters += p.numel()
            par_penalty += (p ** 2).sum()

        par_penalty *= self.network_penalty/total_number_of_parameters
        penalty += par_penalty

        return penalty

    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None):
        """
        Smooth the scalar field using Gaussian smoothing in the Fourier domain

        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :param pars: dictionary that can contain various extra variables; for smoother this will for example be
            the current image 'I' or the current map 'phi'. typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
        :return: smoothed image
        """

        compute_std_gradients = self.optimize_over_smoother_stds

        if variables_from_optimizer is not None:
            if self.start_optimize_over_smoother_parameters_at_iteration > variables_from_optimizer['iter']:
                compute_std_gradients = False
                compute_weight_gradients = False

        # collection of smoothed vector fields
        vcollection = ce.fourier_set_of_gaussian_convolutions(v, self.gaussian_fourier_filter_generator,
                                                       self.get_gaussian_stds(), compute_std_gradients)

        # we now use a small neural net to learn the weighting

        # needs an image as its input
        if not pars.has_key('I'):
            raise ValueError('Smoother requires an image as an input')

        is_map = pars.has_key('phi')
        if is_map:
            # todo: for a map input we simply create the input image by applying the map
            raise ValueError('Only implemented for image input at the moment')
        else:
            I = pars['I']

        # apply the network selecting the smoothing from the set of smoothed results (vcollection) and
        # the input image, which may provide some guidance on where to smooth

        if self.debug_retain_computed_local_weights:
            # v is actually the vector-valued momentum here; changed the interface to pass this also
            smoothed_v = self.ws(vcollection, I, v, self.get_gaussian_weights(), self.encourage_spatial_weight_consistency, self.debug_retain_computed_local_weights)
            self.debug_computed_local_weights = self.ws.get_computed_weights()
        else:
            smoothed_v = self.ws(vcollection, I, v, self.get_gaussian_weights(), self.encourage_spatial_weight_consistency)

        if vout is not None:
            vout[:] = smoothed_v
            return vout
        else:
            return smoothed_v



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


    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None):
        pass


def _print_smoothers(smoothers):
    print('\nKnown smoothers are:')
    print('------------------------------')
    for key in smoothers:
        print('{smoother_name:>40s}: {smoother_description}'.format(smoother_name=key, smoother_description=smoothers[key][1]))


class AvailableSmoothers(object):

    def __init__(self):
        # (smoother, description)
        self.smoothers = {
            'diffusion': (DiffusionSmoother,'smoothing via iterative solution of the diffusion equation'),
            'gaussian': (SingleGaussianFourierSmoother, 'Gaussian smoothing in the Fourier domain'),
            'adaptive_gaussian': (AdaptiveSingleGaussianFourierSmoother, 'Gaussian smoothing in the Fourier domain w/ optimization over std'),
            'multiGaussian': (MultiGaussianFourierSmoother, 'Multi Gaussian smoothing in the Fourier domain'),
            'adaptive_multiGaussian': (AdaptiveMultiGaussianFourierSmoother, 'Adaptive multi Gaussian smoothing in the Fourier domain w/ optimization over weights and stds'),
            'learned_multiGaussianCombination': (LearnedMultiGaussianCombinationFourierSmoother, 'Experimental learned smoother'),
            'gaussianSpatial': (GaussianSpatialSmoother, 'Gaussian smoothing in the spatial domain'),
            'adaptiveNet': (AdaptiveSmoother,'Epxerimental learned smoother')
        }
        """dictionary defining all the smoothers"""

    def get_smoothers(self):
        """
        Returns all available smoothers as a dictionary which has as keys the smoother name and tuple entries of the form
        (smootherclass,explanation_string)
        :return: the model dictionary
        """
        return self.smoothers

    def print_available_smoothers(self):
        """
        Prints the smoothers that are available and can be created with `create_smoother`
        """

        _print_smoothers(self.smoothers)

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
        self.smoothers = AvailableSmoothers().get_smoothers()

    def get_smoothers(self):
        """
        Returns all available smoothers as a dictionary which has as keys the smoother name and tuple entries of the form
        (smootherclass,,explanation_string)
        :return: the smoother dictionary
        """
        return self.smoothers

    def print_available_smoothers(self):
        """
        Prints the smoothers that are available and can be created with `create_smoother`
        """

        _print_smoothers(self.smoothers)

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

    def create_smoother_by_name(self, smoother_name, params=None):
        """
        Creates a smoother by specifying the smoother name (convenience function).

        :param smoother_name: (string) specifies the smoother name
        :param params: ParamterDict() object to hold paramters which should be passed on
        :return:
        """

        if params is None:
            params = pars.ParameterDict()

        params['smoother']['type'] = smoother_name
        return self.create_smoother(params)

    def create_smoother(self, params ):
        """
        Create the desired smoother
        :param params: ParamterDict() object to hold paramters which should be passed on
        :return: returns the smoother
        """

        cparams = params[('smoother',{})]
        smootherType = cparams[('type', self.default_smoother_type,
                                          'type of smoother (diffusion|gaussian|adaptive_gaussian|multiGaussian|adaptive_multiGaussian|gaussianSpatial|adaptiveNet)' )]

        if self.smoothers.has_key(smootherType):
            return self.smoothers[smootherType][0](self.sz,self.spacing,cparams)
        else:
            self.print_available_smoothers()
            raise ValueError('Smoother: ' + smootherType + ' not known')
