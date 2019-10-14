"""
This package implements various types of smoothers.
"""
from __future__ import print_function
from __future__ import absolute_import

from builtins import str
from builtins import range
from builtins import object
from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
import numpy as np
import numpy.testing as npt

from .data_wrapper import USE_CUDA, MyTensor, AdaptVal
from . import finite_differences as fd
from . import utils
# if float(torch.__version__[:3])<=1.1:
#     from . import custom_pytorch_extensions as ce
# else:
#     from . import custom_pytorch_extensions_module_version as ce
from . import custom_pytorch_extensions_module_version as ce
from . import module_parameters as pars
from . import deep_smoothers
from . import deep_networks
import collections
from future.utils import with_metaclass
from .deep_loss import AdaptiveWeightLoss


def get_compatible_state_dict_for_module(state_dict,module_name,target_state_dict):

    res_dict = collections.OrderedDict()
    for k in list(target_state_dict.keys()):
        current_parameter_name = module_name + '.' + k
        if current_parameter_name in state_dict:
            res_dict[k] = state_dict[current_parameter_name]
        else:
            print('WARNING: needed key ' + k + ' but could not find it. IGNORING it.')

    return res_dict


def get_state_dict_for_module(state_dict,module_name):

    res_dict = collections.OrderedDict()
    for k in list(state_dict.keys()):
        if k.startswith(module_name + '.'):
            adapted_key = k[len(module_name)+1:]
            res_dict[adapted_key] = state_dict[k]
    return res_dict

class Smoother(with_metaclass(ABCMeta, object)):
    """
    Abstract base class defining the general smoother interface.
    """

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
        self.batch_size = None
        """batch size of what is being smoothed"""


    def associate_parameters_with_module(self,module):
        """
        Associates parameters that should be optimized with the given module.

        :param module: module to associate the parameters to
        :return: set of parameters that were associated
        """
        return set()

    def write_parameters_to_settings(self):
        """
        If called will take the current parameter state and write it back into the initial setting configuration.
        This should be called from the model it uses the smoother and will for example allow to write back optimized
        weitght parameters into the smoother
        :param module:
        :return:
        """

        pass

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

    def _do_CFL_clamping_if_necessary(self,v, clampCFL_dt):
        """
        Takes a velocity field and clamps it according to the CFL condition (assuming Runge-Kutta 4)
        :param v: velocity field; dimension BxCxXxYxZ
        :return: clampled velocity field
        """

        rk4_factor = 2*np.sqrt(2)/self.dim*0.75 # 0.75 is saftey margin (see paper by Polzin et al. for this RK4 stability condition)

        if clampCFL_dt is not None:

            # only clamp if necessary
            need_to_clamp = False
            for d in range(self.dim):
                if (torch.abs(v[:,d,...].data)).max()>=self.spacing[d]/clampCFL_dt*rk4_factor:
                    need_to_clamp = True
                    break

            if need_to_clamp:
                v_ret = torch.zeros_like(v)
                for d in range(self.dim):
                    cmax = self.spacing[d]/clampCFL_dt*rk4_factor
                    v_ret[:,d,...] = torch.clamp(v[:,d,...],-cmax,cmax)
                return v_ret
            else:
                # clamping was not necessary
                return v

        else:
            return v

    @abstractmethod
    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None, smooth_to_compute_regularizer_energy=False, clampCFL_dt=None):
        """
        Abstract method to smooth a vector field. Only this method should be overwritten in derived classes.

        :param v: input field to smooth  BxCxXxYxZ
        :param vout: if not None then result is returned in this variable
        :param pars: dictionary that can contain various extra variables; for smoother this will for example be
            the current image 'I' or the current map 'phi'. typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
        :param smooth_to_compute_regularizer_energy: in certain cases smoothing to compute a smooth velocity field should
                be different than the smoothing for the regularizer (this flag allows smoother implementations reacting to this difference)
        :param clampCFL_dt: If specified the result of the smoother is clampled according to the CFL condition, based on the given time-step
        :return: should return the a smoothed scalar field, image dimension BxCXxYxZ
        """
        pass



    def smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None, smooth_to_compute_regularizer_energy=False, clampCFL_dt=None,multi_output=False):
        """
        Smoothes a vector field of dimension BxCxXxYxZ,

        :param v: vector field to smooth BxCxXxYxZ
        :param vout: if not None then result is returned in this variable
        :param pars: dictionary that can contain various extra variables; for smoother this will for example be
            the current image 'I' or the current map 'phi'. typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
        :param smooth_to_compute_regularizer_energy: in certain cases smoothing to compute a smooth velocity field should
                be different than the smoothing for the regularizer (this flag allows smoother implementations reacting to this difference)
        :param clampCFL_dt: If specified the result of the smoother is clampled according to the CFL condition, based on the given time-step
        :return: smoothed vector field BxCxXxYxZ
        """
        sz = v.size()
        self.batch_size = sz[0]
        if not multi_output:
            if vout is not None:
                Sv = vout
            else:
                Sv = MyTensor(v.size()).zero_()

            Sv[:] = self.apply_smooth(v,vout,pars,variables_from_optimizer, smooth_to_compute_regularizer_energy, clampCFL_dt)    # here must use :, very important !!!!
            return Sv
        else:
            output = self.apply_smooth(v,vout,pars,variables_from_optimizer, smooth_to_compute_regularizer_energy, clampCFL_dt)
            return output



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

    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None, smooth_to_compute_regularizer_energy=False, clampCFL_dt=None):
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


        Sv = self._do_CFL_clamping_if_necessary(Sv,clampCFL_dt=clampCFL_dt)

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
        self.k_sz_h = params[('k_sz_h', 5*np.ones(self.dim, dtype='int'), 'size of the kernel' )]
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
        self.required_padding = (self.k_sz-1)//2

        if self.dim==1:
            self.filter =AdaptVal(torch.from_numpy(self.smoothingKernel))
        elif self.dim==2:
            self.filter = AdaptVal(torch.from_numpy(self.smoothingKernel))
        elif self.dim==3:
            self.filter = AdaptVal(torch.from_numpy(self.smoothingKernel))
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

    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None, smooth_to_compute_regularizer_energy=False, clampCFL_dt=None):
        """
        Smooth the scalar field using Gaussian smoothing in the spatial domain

        :param v: image to smooth
        :param vout: if not None returns the result in this variable
        :param pars: dictionary that can contain various extra variables; for smoother this will for example be
            the current image 'I' or the current map 'phi'. typically not used.
        :param variables_from_optimizer: variables that can be passed from the optimizer (for example iteration count)
        :return: smoothed image
        """

        if self.filter is None:
            self._create_filter()
        # just doing a Gaussian smoothing

        smoothed_v = self._filter_input_with_padding(v, vout)
        smoothed_v = self._do_CFL_clamping_if_necessary(smoothed_v,clampCFL_dt=clampCFL_dt)

        return smoothed_v



class GaussianFourierSmoother(with_metaclass(ABCMeta, GaussianSmoother)):
    """
    Performs Gaussian smoothing via convolution in the Fourier domain. Much faster for large dimensions
    than spatial Gaussian smoothing on the CPU in large dimensions.
    """

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

    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None, smooth_to_compute_regularizer_energy=False, clampCFL_dt=None):
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

        smoothed_v = ce.fourier_convolution(v, self.FFilter)
        smoothed_v = self._do_CFL_clamping_if_necessary(smoothed_v,clampCFL_dt=clampCFL_dt)

        if vout is not None:
            vout[:] = smoothed_v
            return vout
        else:
            return smoothed_v

class AdaptiveSingleGaussianFourierSmoother(GaussianSmoother):
    """
    Performs Gaussian smoothing via convolution in the Fourier domain. Much faster for large dimensions
    than spatial Gaussian smoothing on the CPU in large dimensions.
    """

    def __init__(self, sz, spacing, params):
        super(AdaptiveSingleGaussianFourierSmoother, self).__init__(sz, spacing, params)
        self.gaussianStd = np.array(params[('gaussian_std', [0.15], 'std for the Gaussian')])
        """standard deviation of Gaussian"""
        self.gaussianStd_min = params[('gaussian_std_min', 0.001, 'minimal allowed std for the Gaussian')]
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
        return ", smooth(std)= " + np.array_str(self.get_gaussian_std()[0].detach().cpu().numpy(),precision=3)

    def get_custom_optimizer_values(self):
        return {'smoother_std': self.get_gaussian_std()[0].detach().cpu().numpy().copy()}

    def write_parameters_to_settings(self):
        self.params['gaussian_std'] = self.get_gaussian_sts()[0].detach().cpu().numpy()

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

    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None, smooth_to_compute_regularizer_energy=False, clampCFL_dt=None):
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

        smoothed_v = ce.fourier_single_gaussian_convolution(v,self.gaussian_fourier_filter_generator,self.get_gaussian_std(),compute_std_gradient)
        smoothed_v = self._do_CFL_clamping_if_necessary(smoothed_v,clampCFL_dt=clampCFL_dt)

        # just doing a Gaussian smoothing
        if vout is not None:
            vout[:] = smoothed_v
            return vout
        else:
            return smoothed_v

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
        self.gaussianStd_min = params[('gaussian_std_min', 0.001 ,'minimal allowed std for the Gaussian' )]
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

        npt.assert_almost_equal((np.array(self.multi_gaussian_weights)).sum(),1.)

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



def _compute_omt_penalty_for_weight_vectors(weights,multi_gaussian_stds,omt_power=2.0,use_log_transform=False):

    penalty = MyTensor(1).zero_()

    # todo: check that this is properly handled for the learned optimizer (i.e., as a variable for optimization opposed to a constant)
    max_std = torch.max(multi_gaussian_stds)
    min_std = torch.min(multi_gaussian_stds)

    if omt_power == 2:
        for i, s in enumerate(multi_gaussian_stds):
            if use_log_transform:
                penalty += weights[i] * ((torch.log(max_std/s)) ** omt_power)
            else:
                penalty += weights[i] * ((s - max_std) ** omt_power)
        if use_log_transform:
            penalty /= (torch.log(max_std/min_std)) ** omt_power
        else:
            penalty /= (max_std - min_std) ** omt_power
    else:
        for i, s in enumerate(multi_gaussian_stds):
            if use_log_transform:
                penalty += weights[i] * (torch.abs(torch.log(max_std/s)) ** omt_power)
            else:
                penalty += weights[i] * (torch.abs(s - max_std) ** omt_power)

        if use_log_transform:
            penalty /= torch.abs(torch.log(max_std/min_std))**omt_power
        else:
            penalty /= torch.abs(max_std-min_std)**omt_power

    return penalty

class AdaptiveMultiGaussianFourierSmoother(GaussianSmoother):
    """
    Adaptive multi-Gaussian smoother which allows optimizing over weights and standard deviations
    """

    def __init__(self, sz, spacing, params):
        super(AdaptiveMultiGaussianFourierSmoother, self).__init__(sz, spacing, params)

        self.multi_gaussian_stds = np.array(params[('multi_gaussian_stds', [0.05, 0.1, 0.15, 0.2, 0.25], 'std deviations for the Gaussians')])
        """standard deviations of Gaussians"""

        self.gaussianStd_min = params[('gaussian_std_min', 0.001, 'minimal allowed std for the Gaussians')]
        """minimal allowed standard deviation during optimization"""

        self.omt_weight_penalty = self.params[('omt_weight_penalty', 25.0, 'Penalty for the optimal mass transport')]
        """penalty factor for the optimal mass transport term"""

        self.omt_use_log_transformed_std = self.params[('omt_use_log_transformed_std', False, 'If set to true the standard deviations are log transformed for the computation of OMT')]
        """if set to true the standard deviations are log transformed for the OMT computation"""

        self.optimize_over_smoother_stds = params[('optimize_over_smoother_stds', False, 'if set to true the smoother will optimize over standard deviations')]
        """determines if we should optimize over the smoother standard deviations"""

        self.optimize_over_smoother_weights = params[('optimize_over_smoother_weights', False, 'if set to true the smoother will optimize over the *global* weights')]
        """determines if we should optimize over the smoother global weights"""

        self.nr_of_gaussians = len(self.multi_gaussian_stds)
        """number of Gaussians"""
        # todo: maybe make this more generic; there is an explicit float here

        self.default_multi_gaussian_weights = AdaptVal(torch.from_numpy(np.ones(self.nr_of_gaussians) / self.nr_of_gaussians).float())
        self.default_multi_gaussian_weights /= self.default_multi_gaussian_weights.sum()

        self.multi_gaussian_weights = np.array(params[('multi_gaussian_weights', self.default_multi_gaussian_weights.detach().cpu().numpy().tolist(),'weights for the multiple Gaussians')])
        """global weights for the Gaussians"""
        self.gaussianWeight_min = params[('gaussian_weight_min', 0.001, 'minimal allowed weight for the Gaussians')]
        """minimal allowed weight during optimization"""

        weight_sum = self.multi_gaussian_weights.sum()
        if weight_sum != 1.:
            print('WARNING: multi-Gaussian weights do not sum to one. Projecting them.')
            self.multi_gaussian_weights += (1. - weight_sum) / len(self.multi_gaussian_weights)
            params['multi_gaussian_weights'] = self.multi_gaussian_weights.tolist()

        npt.assert_almost_equal((np.array(self.multi_gaussian_weights)).sum(),1.)
        assert len(self.multi_gaussian_weights) == len(self.multi_gaussian_stds)

        self.gaussian_fourier_filter_generator = ce.GaussianFourierFilterGenerator(sz, spacing, nr_of_slots=self.nr_of_gaussians)
        """creates the smoothed vector fields"""

        self.multi_gaussian_stds_optimizer_params = self._create_multi_gaussian_stds_optimization_vector_parameters()
        self.multi_gaussian_weights_optimizer_params = self._create_multi_gaussian_weights_optimization_vector_parameters()

        self.start_optimize_over_smoother_parameters_at_iteration = \
            params[('start_optimize_over_smoother_parameters_at_iteration', 0, 'Does not optimize the parameters before this iteration')]

        self.omt_power = params[('omt_power',1.0,'Power for the optimal mass transport (i.e., to which power distances are penalized')]
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
            output_str += ", smooth(stds)= " + np.array_str(self.get_gaussian_stds().detach().cpu().numpy(), precision=3)
        if self.optimize_over_smoother_weights:
            output_str += ", smooth(weights)= " + np.array_str(self.get_gaussian_weights().detach().cpu().numpy(),precision=3)

        output_str += ", smooth(penalty)= " + np.array_str(self.get_penalty().detach().cpu().numpy(),precision=3)

        return output_str

    def get_custom_optimizer_output_values(self):
        return {'smoother_stds': self.get_gaussian_stds().detach().cpu().numpy().copy(),
                'smoother_weights': self.get_gaussian_weights().detach().cpu().numpy().copy(),
                'smoother_penalty': self.get_penalty().detach().cpu().numpy().copy()}

    def set_state_dict(self, state_dict):

        if 'multi_gaussian_stds' in state_dict:
            self.multi_gaussian_stds_optimizer_params.data[:] = state_dict['multi_gaussian_stds']
        if 'multi_gaussian_weights' in state_dict:
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

    def write_parameters_to_settings(self):
        if self.optimize_over_smoother_stds:
            self.params['multi_gaussian_stds'] = self.get_gaussian_stds().detach().cpu().numpy().tolist()

        if self.optimize_over_smoother_weights:
            self.params['multi_gaussian_weights'] = self.get_gaussian_weights().detach().cpu().numpy().tolist()

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

    def get_penalty(self):
        # puts an squared two-norm penalty on the weights as deviations from the baseline
        # also adds a penalty for the network parameters

        current_penalty = _compute_omt_penalty_for_weight_vectors(self.get_gaussian_weights(),self.get_gaussian_stds(),self.omt_power,self.omt_use_log_transformed_std)
        penalty = current_penalty*self.omt_weight_penalty

        return penalty

    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None, smooth_to_compute_regularizer_energy=False, clampCFL_dt=None):
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

        smoothed_v = ce.fourier_multi_gaussian_convolution(v,self.gaussian_fourier_filter_generator,
                                                         self.get_gaussian_stds(),self.get_gaussian_weights(),
                                                            compute_std_gradients=compute_std_gradients,
                                                            compute_weight_gradients=compute_weight_gradients)

        smoothed_v = self._do_CFL_clamping_if_necessary(smoothed_v,clampCFL_dt=clampCFL_dt)

        if vout is not None:
            vout[:] = smoothed_v
            return vout
        else:
            return smoothed_v


class LearnedMultiGaussianCombinationFourierSmoother(GaussianSmoother):
    """
    Adaptive multi-Gaussian Fourier smoother. Allows optimization over weights and standard deviations
    """

    def __init__(self, sz, spacing, params):
        super(LearnedMultiGaussianCombinationFourierSmoother, self).__init__(sz, spacing, params)

        self.multi_gaussian_stds = AdaptVal(torch.from_numpy(np.array(params[('multi_gaussian_stds', [0.05, 0.1, 0.15, 0.2, 0.25], 'std deviations for the Gaussians')],dtype='float32')))
        """standard deviations of Gaussians"""
        self.gaussianStd_min = params[('gaussian_std_min', 0.001, 'minimal allowed std for the Gaussians')]
        """minimal allowed standard deviation during optimization"""
        self.smallest_gaussian_std = self.multi_gaussian_stds.min()
        """The smallest of the standard deviations"""

        self.optimize_over_smoother_stds = params[('optimize_over_smoother_stds', False, 'if set to true the smoother will optimize over standard deviations')]
        """determines if we should optimize over the smoother standard deviations"""

        self.optimize_over_smoother_weights = params[('optimize_over_smoother_weights', False, 'if set to true the smoother will optimize over the *global* weights')]
        """determines if we should optimize over the smoother global weights"""

        self.scale_global_parameters = params[('scale_global_parameters',False,'If set to True the global parameters are scaled for the global parameters, to make sure energies decay similarly as for the deep-network weight estimation')]
        """If set to True the global parameters are scaled for the global parameters, to make sure energies decay similarly as for the deep-network weight estimation'"""

        self.optimize_over_deep_network = params[('optimize_over_deep_network', False, 'if set to true the smoother will optimize over the deep network parameters; otherwise will ignore the deep network')]
        """determines if we should optimize over the smoother global weights"""

        self.evaluate_but_do_not_optimize_over_shared_registration_parameters = params[('evaluate_but_do_not_optimize_over_shared_registration_parameters',False,'If set to true then shared registration parameters (e.g., the network or global weights) are evaluated (should have been loaded from a previously computed optimized state), but are not being optimized over')]
        """If set to true then the network is evaluated (should have been loaded from a previously computed optimized state), but the network weights are not being optimized over"""

        self.freeze_parameters = params[('freeze_parameters', False, 'if set to true then all the parameters that are optimized over are frozen (but they still influence the optimization indirectly; they just do not change themselves)')]
        """Freezes parameters; this, for example, allows optimizing for a few extra steps without changing their current value"""

        self.start_optimize_over_smoother_parameters_at_iteration = \
            params[('start_optimize_over_smoother_parameters_at_iteration', 0,
                    'Does not optimize the parameters before this iteration')]
        """at what iteration the optimization over weights or stds should start"""

        self.start_optimize_over_nn_smoother_parameters_at_iteration = \
            params[('start_optimize_over_nn_smoother_parameters_at_iteration', 0,
                    'Does not optimize the nn smoother parameters before this iteration')]
        """at what iteration the optimization over nn parameters should start"""

        self.nr_of_gaussians = len(self.multi_gaussian_stds)
        """number of Gaussians"""
        # todo: maybe make this more generic; there is an explicit float here

        self.default_multi_gaussian_weights = AdaptVal(torch.from_numpy(np.ones(self.nr_of_gaussians,dtype='float32')/self.nr_of_gaussians).float())
        self.default_multi_gaussian_weights /= self.default_multi_gaussian_weights.sum()

        self.multi_gaussian_weights = AdaptVal(torch.from_numpy(np.array(params[('multi_gaussian_weights', self.default_multi_gaussian_weights.detach().cpu().numpy().tolist(), 'weights for the multiple Gaussians')],dtype='float32')))
        """global weights for the Gaussians"""
        self.gaussianWeight_min = params[('gaussian_weight_min', 0.001, 'minimal allowed weight for the Gaussians')]
        """minimal allowed weight during optimization"""

        self.gaussian_fourier_filter_generator = ce.GaussianFourierFilterGenerator(sz, spacing, nr_of_slots=self.nr_of_gaussians)
        self.gaussian_fourier_filter_generator.get_gaussian_filters(self.multi_gaussian_stds)
        """creates the smoothed vector fields"""

        self.ws = deep_smoothers.DeepSmootherFactory(nr_of_gaussians=self.nr_of_gaussians,gaussian_stds=self.multi_gaussian_stds,nr_of_image_channels=0,dim=self.dim,spacing=self.spacing,im_sz=self.sz).create_deep_smoother(params)
        """learned mini-network to predict multi-Gaussian smoothing weights"""

        last_kernel_size = self.ws.get_last_kernel_size()
        if self.scale_global_parameters:
            self.global_parameter_scaling_factor = params[('scale_global_parameters_scaling_factor',0.05,'value that is used to scale the global parameters, to make sure energies decay similarly as for the deep-network weight estimation')]
            """If set to True the global parameters are scaled, to make sure energies decay similarly as for the deep-network weight estimation'"""
            #self.global_parameter_scaling_factor = float(np.sqrt(float(last_kernel_size**self.dim) / self.sz.prod()))
        else:
            self.global_parameter_scaling_factor = 1.0

        self.pre_multi_gaussian_stds_optimizer_params = self._create_pre_multi_gaussian_stds_optimization_vector_parameters()
        self.pre_multi_gaussian_weights_optimizer_params = self._create_pre_multi_gaussian_weights_optimization_vector_parameters()

        self.weighting_type = self.ws.get_weighting_type() # 'w_K','w_K_w','sqrt_w_K_sqrt_w'

        if self.weighting_type=='w_K_w' or self.weighting_type=='sqrt_w_K_sqrt_w':
            self.use_multi_gaussian_regularization = self.params[('use_multi_gaussian_regularization',False,'If set to true then the regularization for w_K_w or sqrt_w_K_sqrt_w will use multi-Gaussian smoothing (not the velocity) of the deep smoother')]
            """If set to true then the regularization for w_K_w or sqrt_w_K_sqrt_w will use multi-Gaussian smoothing (not the velocity) of the deep smoother"""
        else:
            self.use_multi_gaussian_regularization = False

        if self.weighting_type=='w_K' or self.use_multi_gaussian_regularization:
            # this setting only matter for the w_K registration model
            self.only_use_smallest_standard_deviation_for_regularization_energy = \
                params[('only_use_smallest_standard_deviation_for_regularization_energy', True,
                        'When set to True the regularization energy only used the Gaussian with smallest standard deviation to compute the velocity field for the energy computation')]
        else:
            self.only_use_smallest_standard_deviation_for_regularization_energy = False

        self.load_dnn_parameters_from_this_file = params[('load_dnn_parameters_from_this_file','',
                                                          'If not empty, this is the file the DNN parameters are read from; useful to run a pre-initialized network')]
        """To allow pre-initializing a network"""
        if self.load_dnn_parameters_from_this_file!='' and self.load_dnn_parameters_from_this_file is not None:
            print('INFO: Loading network configuration from {:s}'.format(self.load_dnn_parameters_from_this_file))
            print('WARNING: If start_from_previously_saved_parameters is set to True then this setting may get ignored; current HACK: overwrites shared parameters in the current results directory')
            self.set_state_dict(torch.load(self.load_dnn_parameters_from_this_file))

        self.omt_weight_penalty = self.ws.get_omt_weight_penalty()
        """penalty factor for the optimal mass transport term"""

        self.omt_use_log_transformed_std = self.params[('omt_use_log_transformed_std', False, 'If set to true the standard deviations are log transformed for the computation of OMT')]
        """if set to true the standard deviations are log transformed for the OMT computation"""

        self.omt_power = self.ws.get_omt_power()
        """power for the optimal mass transport term"""

        self.preweight_input_range_weight_penalty = self.params[('preweight_input_range_weight_penalty', 10.0,
                                                            'Penalty for the input to the preweight computation; weights should be between 0 and 1. If they are not they get quadratically penalized; use this with weighted_linear_softmax only.')]

        self.debug_retain_computed_local_weights = False
        self.debug_computed_local_weights = None
        self.debug_computed_local_pre_weights = None

        self._is_optimizing_over_deep_network = True

        self._nn_hooks = None
        self._nn_check_hooks = None

        self.weight_input_range_loss = deep_networks.WeightInputRangeLoss()

        self.penalty_deatched= None

        print("ATTENTION!!!! THE DEEP SMOOTHER SHOULD ONLY INITIALIZED ONCE")

    def _compute_weights_from_preweights(self,pre_weights):
        weights = deep_smoothers.weighted_linear_softmax(pre_weights*self.global_parameter_scaling_factor,dim=0,weights=self.multi_gaussian_weights)
        proj_weights = deep_smoothers._project_weights_to_min(weights, self.gaussianWeight_min, norm_type='sum',dim=0)
        return proj_weights

    def _compute_weight_input_range_loss(self):
        pre_weights = self.pre_multi_gaussian_weights_optimizer_params*self.global_parameter_scaling_factor
        return self.weight_input_range_loss(pre_weights, spacing=None,
                                            use_weighted_linear_softmax=True,
                                            weights=self.multi_gaussian_weights,
                                            min_weight=self.gaussianWeight_min,
                                            max_weight=1.0,
                                            dim=0)

    def _compute_stds_from_prestds(self,pre_stds):
        stds = torch.clamp(pre_stds*self.global_parameter_scaling_factor, min=self.gaussianStd_min)
        return stds

    def _compute_std_input_range_loss(self):
        pre_stds = self.pre_multi_gaussian_stds_optimizer_params*self.global_parameter_scaling_factor
        std_loss = ((pre_stds-self._compute_stds_from_prestds(pre_stds=pre_stds))**2).sum()
        return std_loss

    def get_default_multi_gaussian_weights(self):
        # todo: check, should it really return this?
        return self.multi_gaussian_weights

    def get_debug_computed_local_weights(self):
        return self.debug_computed_local_weights

    def get_debug_computed_local_pre_weights(self):
        return self.debug_computed_local_pre_weights

    def set_debug_retain_computed_local_weights(self,val=True):
        self.debug_retain_computed_local_weights = val

    def _remove_all_nn_hooks(self):

        if self._nn_hooks is None:
            return
        print("the gradient mask will be removed in deep smoother")
        for h in self._nn_hooks:
            h.remove()

        self._nn_hooks = None


    def disable_penalty_computation(self):
        self.ws.compute_the_penalty=False
    def enable_accumulated_penalty(self):
        self.ws.accumulate_the_penalty=True

    def reset_penalty(self):
        self.ws.current_penalty=0.
        self.ws.compute_the_penalty= True

    def __print_grad_hook(self,grad):
        print(torch.sum(torch.abs(grad)))
        return grad
    def __debug_grad_exist(self):
        self._nn_check_hooks = []
        for child in self.ws.children():
            for cur_param in child.parameters():
                current_hook = cur_param.register_hook(self.__print_grad_hook)
                self._nn_check_hooks.append(current_hook)


    def _enable_force_nn_gradients_to_zero_hooks(self):

        if self._nn_hooks is None:
            print("the gradient mask will be added in deep smoother")
            self._nn_hooks = []

            for child in self.ws.children():
                for cur_param in child.parameters():
                    current_hook = cur_param.register_hook(lambda grad: grad*0)
                    self._nn_hooks.append(current_hook)

    def associate_parameters_with_module(self,module):
        s = set()

        # freeze parameters if needed
        freeze_shared_parameters = self.freeze_parameters or self.evaluate_but_do_not_optimize_over_shared_registration_parameters

        if self.optimize_over_smoother_stds:
            self.pre_multi_gaussian_stds_optimizer_params.requires_grad = not freeze_shared_parameters
            module.register_parameter('pre_multi_gaussian_stds', self.pre_multi_gaussian_stds_optimizer_params)
            s.add('pre_multi_gaussian_stds')

        if self.optimize_over_smoother_weights:
            self.pre_multi_gaussian_weights_optimizer_params.requires_grad = not freeze_shared_parameters
            module.register_parameter('pre_multi_gaussian_weights', self.pre_multi_gaussian_weights_optimizer_params)
            s.add('pre_multi_gaussian_weights')

        # todo: is it possible that the following code not properly disable parameter updates
        if self.optimize_over_deep_network:
            for child in self.ws.children():
                for cur_param in child.parameters():
                    cur_param.requires_grad = not freeze_shared_parameters

        if self.optimize_over_deep_network or self.evaluate_but_do_not_optimize_over_shared_registration_parameters:
            module.add_module('weighted_smoothing_net',self.ws)
            sd = self.ws.state_dict()
            for key in sd:
                s.add('weighted_smoothing_net.' + str(key))

        if self.evaluate_but_do_not_optimize_over_shared_registration_parameters:
             print('INFO: Setting network to evaluation mode')
             self.ws.network.eval()

        return s

    def get_custom_optimizer_output_string(self):
        output_str = ""
        if self.optimize_over_smoother_stds:
            output_str += ", smooth(stds)= " + np.array_str(self.get_gaussian_stds().detach().cpu().numpy(), precision=3)
        if self.optimize_over_smoother_weights:
            output_str += ", smooth(weights)= " + np.array_str(self.get_gaussian_weights().detach().cpu().numpy(),precision=3)

        output_str += ", smooth(penalty)= " + np.array_str(self.penalty_deatched.cpu().numpy(),precision=3)

        return output_str

    def get_custom_optimizer_output_values(self):
        return {'smoother_stds': self.get_gaussian_stds().detach().cpu().numpy().copy(),
                'smoother_weights': self.get_gaussian_weights().detach().cpu().numpy().copy(),
                'smoother_penalty': self.penalty_deatched.cpu().numpy().copy()}

    def write_parameters_to_settings(self):
        if self.optimize_over_smoother_stds:
            self.params['multi_gaussian_stds'] = self.get_gaussian_stds().detach().cpu().numpy().tolist()

        if self.optimize_over_smoother_weights:
            self.params['multi_gaussian_weights'] = self.get_gaussian_weights().detach().cpu().numpy().tolist()

    def set_state_dict(self,state_dict):

        if 'pre_multi_gaussian_stds' in state_dict:
            self.pre_multi_gaussian_stds_optimizer_params.data[:] = state_dict['pre_multi_gaussian_stds']
        if 'pre_multi_gaussian_weights' in state_dict:
            self.pre_multi_gaussian_weights_optimizer_params.data[:] = state_dict['pre_multi_gaussian_weights']

        # first check if the learned smoother has already been initialized
        if len(self.ws.state_dict())==0:
            # has not been initialized, we need to initialize it before we can load the dictionary
            nr_of_image_channels = self.ws.get_number_of_image_channels_from_state_dict(state_dict,self.dim)
            self.ws._init(nr_of_image_channels,self.dim)
        self.ws.load_state_dict(get_compatible_state_dict_for_module(state_dict,'weighted_smoothing_net',self.ws.state_dict()))

    def _get_gaussian_weights_from_optimizer_params(self):
        return self._compute_weights_from_preweights(pre_weights=self.pre_multi_gaussian_weights_optimizer_params)

    def _set_gaussian_weights_optimizer_params(self,gweights):
        self.pre_multi_gaussian_weights_optimizer_params.data[:] = gweights/self.global_parameter_scaling_factor

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
    def get_deep_smoother_weights(self):
        return self.ws.get_weights()

    def get_deep_smoother_preweights(self):
        return self.ws.get_pre_weights()

    def _get_gaussian_stds_from_optimizer_params(self):
        return self._compute_stds_from_prestds(pre_stds=self.pre_multi_gaussian_stds_optimizer_params)

    def _set_gaussian_stds_optimizer_params(self, g_stds):
        self.pre_multi_gaussian_stds_optimizer_params.data[:] = g_stds/self.global_parameter_scaling_factor

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

    def _create_pre_multi_gaussian_stds_optimization_vector_parameters(self):
        self.pre_multi_gaussian_stds_optimizer_params = utils.create_vector_parameter(self.nr_of_gaussians)
        self.pre_multi_gaussian_stds_optimizer_params.data[:] = self.multi_gaussian_stds/self.global_parameter_scaling_factor
        return self.pre_multi_gaussian_stds_optimizer_params

    def _create_pre_multi_gaussian_weights_optimization_vector_parameters(self):
        self.pre_multi_gaussian_weights_optimizer_params = utils.create_vector_parameter(self.nr_of_gaussians)
        self.pre_multi_gaussian_weights_optimizer_params.data.zero_()
        return self.pre_multi_gaussian_weights_optimizer_params

    def get_penalty(self):
        # puts an squared two-norm penalty on the weights as deviations from the baseline
        # also adds a penalty for the network parameters

        if not self._is_optimizing_over_deep_network:
            current_penalty = _compute_omt_penalty_for_weight_vectors(self.get_gaussian_weights(),
                                                                      self.get_gaussian_stds(), omt_power=self.omt_power, use_log_transform=self.omt_use_log_transformed_std)

            penalty = current_penalty * self.omt_weight_penalty
            if self.optimize_over_smoother_weights:
                current_weight_penalty = self.preweight_input_range_weight_penalty*self._compute_weight_input_range_loss()
                penalty += current_weight_penalty
                #print('Current weight penalty = {}'.format(current_weight_penalty.item()))
            if self.optimize_over_smoother_stds:
                current_std_penalty = self.preweight_input_range_weight_penalty*self._compute_std_input_range_loss()
                penalty += current_std_penalty
                #print('Current std penalty = {}'.format(current_std_penalty.item()))
            penalty *= self.spacing.prod()*float(self.sz.prod())
            penalty *= self.batch_size

            #print('global OMT_penalty = {}'.format(penalty.detach().cpu().numpy()))

        else:

            # norrmalize by  batch size to make it consistent with the global approach above
            penalty = self.ws.get_current_penalty()
            #penalty += self.ws.compute_l2_parameter_weight_penalty()
        self.penalty_deatched = penalty.detach()
        return penalty

    def _smooth_via_deep_network(self,I,additional_inputs,iter=0,retain_computed_local_weights=False):
        if retain_computed_local_weights:
            # v is actually the vector-valued momentum here; changed the interface to pass this also
            smoothed_v = self.ws(I=I, additional_inputs=additional_inputs,
                                 global_multi_gaussian_weights=self.get_gaussian_weights(),
                                 gaussian_fourier_filter_generator=self.gaussian_fourier_filter_generator,
                                 iter=iter,
                                 retain_weights=retain_computed_local_weights)

            self.debug_computed_local_weights = self.ws.get_computed_weights()
            self.debug_computed_local_pre_weights = self.ws.get_computed_pre_weights()
        else:
            smoothed_v = self.ws(I=I, additional_inputs=additional_inputs,
                                 global_multi_gaussian_weights=self.get_gaussian_weights(),
                                 gaussian_fourier_filter_generator=self.gaussian_fourier_filter_generator,
                                 iter=iter)
            self.debug_computed_local_pre_weights = None
            self.debug_computed_local_pre_weights = None


        return smoothed_v

    def _smooth_via_smallest_gaussian(self, v, compute_std_gradients):

        # only smooth with the smallest standard deviation
        smoothed_v = ce.fourier_set_of_gaussian_convolutions(v, self.gaussian_fourier_filter_generator,
                                                             self.smallest_gaussian_std.view(1),
                                                             compute_std_gradients)
        return smoothed_v

    def _smooth_via_std_multi_gaussian(self, v, compute_std_gradients):

        # we can smooth over everything
        # collection of smoothed vector fields
        vcollection = ce.fourier_set_of_gaussian_convolutions(v, self.gaussian_fourier_filter_generator,
                                                              self.get_gaussian_stds(), compute_std_gradients)

        # just do global weighting here
        smoothed_v = torch.zeros_like(vcollection[0, ...])
        for i, w in enumerate(self.get_gaussian_weights()):
            smoothed_v += w * vcollection[i, ...]

        return smoothed_v

    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None, smooth_to_compute_regularizer_energy=False, clampCFL_dt=None):
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
            # if we deal with epochs then use them to indicate iteration progress, otherwise use iterations
            if variables_from_optimizer['epoch'] is not None:
                iter_or_epoch = variables_from_optimizer['epoch']
            else:
                iter_or_epoch = variables_from_optimizer['iter']
        else:
            iter_or_epoch = None

        self.ws.set_cur_epoch(iter_or_epoch)

        if variables_from_optimizer is not None:
            if self.start_optimize_over_smoother_parameters_at_iteration > iter_or_epoch:
                compute_std_gradients = False
                compute_weight_gradients = False

        # we now use a small neural net to learn the weighting

        # needs an image as its input
        if 'I' not in pars:
            raise ValueError('Smoother requires an image as an input')

        is_map = 'phi' in pars
        if is_map:
            # todo: for a map input we simply create the input image by applying the map
            raise ValueError('Only implemented for image input at the moment')
        else:
            I = pars['I']

        # apply the network selecting the smoothing from the set of smoothed results (vcollection) and
        # the input image, which may provide some guidance on where to smooth

        # todo: make this more generic later
        additional_inputs = {'m':v,'I0':pars['I0'],'I1':pars['I1']}
        #print(pars['I0'].shape, pars['I1'].shape)

        # if variables_from_optimizer is not None:
        #     if self.start_optimize_over_smoother_parameters_at_iteration > variables_from_optimizer['iter']:
        #         # just apply the global weights for smoothing
        #         self._is_optimizing_over_deep_network = False
        #     else:
        #         self._is_optimizing_over_deep_network = True
        # else:
        #     self._is_optimizing_over_deep_network = True
        #
        # if not self.optimize_over_deep_network:
        #     self._is_optimizing_over_deep_network = False

        #print('Current weight l2 = {}'.format(self._compute_current_nn_weight_l2().item()))

        self._is_optimizing_over_deep_network = self.optimize_over_deep_network   #TODO  Should be recovered ###########################################################3
        if self._is_optimizing_over_deep_network:
            if variables_from_optimizer is not None:
                if self.start_optimize_over_nn_smoother_parameters_at_iteration > iter_or_epoch:
                    print('INFO: disabling the deep smoother network gradients, i.e., FORCING them to zero')
                    # just apply the global weights for smoothing
                    self._enable_force_nn_gradients_to_zero_hooks()
                else:
                    if self.start_optimize_over_nn_smoother_parameters_at_iteration == iter_or_epoch:
                        print('INFO: Allowing optimization over deep smoother network (assuming we are not in evaluation-only mode)')
                        self._remove_all_nn_hooks()  # todo we put this line under if statement, so that it will only remove hooks when hit the start epoch

        # if self. _nn_check_hooks is None:
        #     self.__debug_grad_exist()


        # distiniguish the two cases where we compute the vector field (for the deformation) versus where we compute for regularization
        if smooth_to_compute_regularizer_energy:
            # we need to distinguish here the standard model, versus the different flavors for the deep network
            if self._is_optimizing_over_deep_network:

                if self.weighting_type=='w_K' or self.use_multi_gaussian_regularization:
                    if self.only_use_smallest_standard_deviation_for_regularization_energy:
                        # only use the smallest std for regularization
                        smoothed_v = self._smooth_via_smallest_gaussian(v=v,compute_std_gradients=compute_std_gradients)
                    else:
                        # use standard multi-Gaussian for regularization
                        smoothed_v = self._smooth_via_std_multi_gaussian(v=v,compute_std_gradients=compute_std_gradients)

                elif self.weighting_type=='sqrt_w_K_sqrt_w':
                    smoothed_v = self._smooth_via_deep_network(I=I, additional_inputs=additional_inputs, iter=iter_or_epoch)

                elif self.weighting_type=='w_K_w':
                    smoothed_v = self._smooth_via_deep_network(I=I, additional_inputs=additional_inputs, iter=iter_or_epoch)
                else:
                    raise ValueError('Unknown weighting type')

            else: # standard SVF
                smoothed_v = self._smooth_via_std_multi_gaussian(v=v,compute_std_gradients=compute_std_gradients)

        else: # here this will be the velocity driving the deformation
            # we need to distinguish here the standard model, versus the different flavors for the deep network
            if self._is_optimizing_over_deep_network:
                smoothed_v = self._smooth_via_deep_network(I=I,
                                                           additional_inputs=additional_inputs,
                                                           iter=iter_or_epoch,
                                                           retain_computed_local_weights=self.debug_retain_computed_local_weights)
            else:  # standard SVF
                smoothed_v = self._smooth_via_std_multi_gaussian(v=v,compute_std_gradients=compute_std_gradients)


        smoothed_v = self._do_CFL_clamping_if_necessary(smoothed_v,clampCFL_dt=clampCFL_dt) # TODO check if we need to remove this when use dopri

        if vout is not None:
            vout[:] = smoothed_v
            return vout
        else:
            return smoothed_v


class LocalFourierSmoother(Smoother):
    """Performs multi Gaussian smoothing via convolution in the Fourier domain. Much faster for large dimensions
    than spatial Gaussian smoothing on the CPU in large dimensions.

    the local fourier smoother is designed for  optimization version, in this case, only local Fourier smoother
    need to be used, but currently, not support global std, weights optimization.

    Besides, it can be jointly used with deep smoother, in this case, we  would call import_outside_var during the init
    to share the vars with the deep smoother.
    """

    def __init__(self, sz, spacing, params):
        super(LocalFourierSmoother, self).__init__( sz, spacing, params)
        self.multi_gaussian_stds = AdaptVal(torch.from_numpy(
            np.array(params[('multi_gaussian_stds', [0.05, 0.1, 0.15, 0.2, 0.25], 'std deviations for the Gaussians')],
                     dtype='float32')))
        self.multi_gaussian_weights = AdaptVal(torch.from_numpy(
            np.array(params[('multi_gaussian_weights', [0.05, 0.1, 0.15, 0.2, 0.25], 'weights for the Gaussians std')],
                     dtype='float32')))
        self.nr_of_gaussians = len(self.multi_gaussian_stds)
        self.weighting_type = params['deep_smoother'][
            ('weighting_type', 'sqrt_w_K_sqrt_w', 'Type of weighting: w_K|w_K_w|sqrt_w_K_sqrt_w')]
        self.gaussian_fourier_filter_generator = ce.GaussianFourierFilterGenerator(self.sz, self.spacing,
                                                                                   nr_of_slots=self.nr_of_gaussians)
        self.gaussian_fourier_filter_generator.get_gaussian_filters(self.multi_gaussian_stds)
        self.loss = AdaptiveWeightLoss(self.nr_of_gaussians, self.multi_gaussian_stds, self.dim, spacing, sz,
                                       omt_power=None, params=params)
        self.accumulate_the_penalty = False
        self.compute_the_penalty = False
        self.current_penalty = 0.


    def set_debug_retain_computed_local_weights(self, val=True):
        pass

    def get_default_multi_gaussian_weights(self):
        return self.multi_gaussian_weights

    def get_gaussian_stds(self):
        return self.multi_gaussian_stds

    def get_gaussian_weights(self):
        return self.multi_gaussian_weights

    def get_custom_optimizer_output_string(self):
        output_str = ""
        output_str += ", smooth(penalty)= " + np.array_str(self.current_penalty.detach().cpu().numpy(), precision=3)

        return output_str

    def get_custom_optimizer_output_values(self):
        return {'smoother_stds': self.multi_gaussian_stds.detach().cpu().numpy().copy(),
                'smoother_weights': self.multi_gaussian_weights.detach().cpu().numpy().copy(),
                'smoother_penalty': self.current_penalty.detach().cpu().numpy().copy()}

    def import_outside_var(self, multi_gaussian_stds, multi_gaussian_weights, gaussian_fourier_filter_generator, loss):
        """This function is to deal with situation like the optimization of the multi-gaussian-stds,
        multi-gaussian_weight, here we also take the  gaussian_fourier_filter_generator loss as the input to reduce
        the head cost; this function only needs to be called once at the init.

        :param multi_gaussian_stds:
        :param multi_gaussian_weights:
        :param weighting_type:
        :param gaussian_fourier_filter_generator:
        :param loss:
        :return:

        """
        self.multi_gaussian_stds = multi_gaussian_stds
        self.multi_gaussian_weights = multi_gaussian_weights
        self.gaussian_fourier_filter_generator = gaussian_fourier_filter_generator
        self.loss = loss

    def enable_accumulated_penalty(self):
        self.accumulate_the_penalty = True

    def enable_penalty_computation(self):
        self.compute_the_penalty = True

    def disable_penalty_computation(self):
        self.compute_the_penalty = False

    def reset_penalty(self):
        self.current_penalty = 0.
        self.compute_the_penalty = True

    def get_nr_of_gaussians(self):
        return self.nr_of_gaussians

    def debugging(self, input, t):
        x = utils.checkNan(input)
        if np.sum(x):
            print(input[0])
            print(input[1])
            print("flag m: {}, ".format(x[0]))
            print("flag smooth_m: {},".format(x[1]))
            raise ValueError("nan error")

    def get_penalty(self):
        return self.current_penalty

    def set_epoch(self, epoch):
        self.epoch = epoch

    def compute_penalty(self, I, weights, pre_weights, input_to_pre_weights=None):

        if self.compute_the_penalty:
            # todo if the accumluate penalty need to be considered, here should need an accumulated term
            self.loss.epoch = self.epoch
            current_penalty, _, _, _, _ = self.loss._compute_penalty_from_weights_preweights_and_input_to_preweights(
                I=I, weights=weights,
                pre_weights=pre_weights,
                input_to_preweights=input_to_pre_weights,
                global_multi_gaussian_weights=self.multi_gaussian_weights)
            if not self.accumulate_the_penalty:
                self.current_penalty = current_penalty
            else:
                self.current_penalty += current_penalty

    def apply_smooth(self, v, vout=None, pars=dict(), variables_from_optimizer=None, smooth_to_compute_regularizer_energy=False, clampCFL_dt=None):
        from . import deep_smoothers as DS
        momentum = v
        weights=pars['w']
        weights = torch.clamp((weights), min=1e-3)

        if self.weighting_type == 'sqrt_w_K_sqrt_w':
            sqrt_weights = torch.sqrt(weights)
            sqrt_weighted_multi_smooth_v = DS.compute_weighted_multi_smooth_v(momentum=momentum, weights=sqrt_weights,
                                                                              gaussian_stds=self.multi_gaussian_stds,
                                                                              gaussian_fourier_filter_generator=self.gaussian_fourier_filter_generator)
            extra_ret = sqrt_weighted_multi_smooth_v
            # if EV.debug_mode_on:
            #     pass #self.debugging([sqrt_weights,sqrt_weighted_multi_smooth_v],0)
        elif self.weighting_type == 'w_K_w':
            # now create the weighted multi-smooth-v
            weighted_multi_smooth_v = DS.compute_weighted_multi_smooth_v(momentum=momentum, weights=weights,
                                                                         gaussian_stds=self.multi_gaussian_stds,
                                                                         gaussian_fourier_filter_generator=self.gaussian_fourier_filter_generator)
            extra_ret = weighted_multi_smooth_v
        elif self.weighting_type == 'w_K':
            # todo: check if we can do a more generic datatype conversion than using .float()
            multi_smooth_v = ce.fourier_set_of_gaussian_convolutions(momentum,
                                                                     gaussian_fourier_filter_generator=self.gaussian_fourier_filter_generator,
                                                                     sigma=self.multi_gaussian_stds,
                                                                     compute_std_gradients=False)
            extra_ret = multi_smooth_v
        else:
            raise ValueError('Unknown weighting_type: {}'.format(self.weighting_type))
        sz_m = momentum.size()
        ret = AdaptVal(MyTensor(*sz_m))

        for n in range(self.dim):
            if self.weighting_type == 'sqrt_w_K_sqrt_w':
                # sqrt_weighted_multi-smooth_v should be:  batch x K x dim x X x Y x ...
                # roc should be: batch x multi_v x X x Y
                roc = sqrt_weighted_multi_smooth_v[:, :, n, ...]
                yc = torch.sum(roc * sqrt_weights, dim=1)
            elif self.weighting_type == 'w_K_w':
                # roc should be: batch x multi_v x X x Y
                roc = weighted_multi_smooth_v[:, :, n, ...]
                yc = torch.sum(roc * weights, dim=1)
            elif self.weighting_type == 'w_K':
                # roc should be: batch x multi_v x X x Y
                roc = torch.transpose(multi_smooth_v[:, :, n, ...], 0, 1)
                yc = torch.sum(roc * weights, dim=1)
            else:
                raise ValueError('Unknown weighting_type: {}'.format(self.weighting_type))

            ret[:, n, ...] = yc  # ret is: batch x channels x X x Y
        return ret, extra_ret

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
            'localAdaptive':(LocalFourierSmoother,'Experimental local smoother')
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
        """size of image (X,Y,...); does not include batch-size or number of channels"""
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

    def create_smoother(self, params,smooth_type=None):
        """
        Create the desired smoother
        :param params: ParamterDict() object to hold paramters which should be passed on
        :return: returns the smoother
        """

        cparams = params[('smoother',{})]
        if smooth_type is None:
            smootherType = cparams[('type', self.default_smoother_type,
                                          'type of smoother (diffusion|gaussian|adaptive_gaussian|multiGaussian|adaptive_multiGaussian|gaussianSpatial|adaptiveNet)' )]
        else:
            smootherType = smooth_type
        if smootherType in self.smoothers:
            return self.smoothers[smootherType][0](self.sz,self.spacing,cparams)
        else:
            self.print_available_smoothers()
            raise ValueError('Smoother: ' + smootherType + ' not known')
