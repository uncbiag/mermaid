"""
This package enables easy single-scale and multi-scale optimization support.
"""
from __future__ import print_function
from __future__ import absolute_import

# from builtins import zip
# from builtins import str
# from builtins import range
# from builtins import object
from abc import ABCMeta, abstractmethod
import os
import time
import copy
from . import utils
from . import visualize_registration_results as vizReg
from . import custom_optimizers as CO
import numpy as np
import torch
from .data_wrapper import USE_CUDA, AdaptVal, MyTensor
from . import model_factory as MF
from . import image_sampling as IS
from .metrics import get_multi_metric
from .res_recorder import XlsxRecorder
from .data_utils import make_dir
from torch.utils.data import Dataset, DataLoader
from . import optimizer_data_loaders as OD
from . import fileio as FIO
from . import model_evaluation

from collections import defaultdict
from future.utils import with_metaclass

from termcolor import colored, cprint

# add some convenience functionality
class SimpleRegistration(with_metaclass(ABCMeta, object)):
    """
           Abstract optimizer base class.
    """

    def __init__(self,ISource,ITarget,spacing,sz,params,compute_inverse_map=False, default_learning_rate=None):
        """
        :param ISource: source image
        :param ITarget: target image
        :param spacing: image spacing
        :param params: parameters
        :param compute_inverse_map: for map-based method the inverse map can be computed on the fly
        """
        self.params = params
        self.use_map = self.params['model']['deformation'][('use_map', True, '[True|False] either do computations via a map or directly using the image')]
        self.map_low_res_factor = self.params['model']['deformation'][('map_low_res_factor', 1.0, 'Set to a value in (0,1) if a map-based solution should be computed at a lower internal resolution (image matching is still at full resolution')]
        self.spacing = spacing
        self.ISource = ISource
        self.ITarget = ITarget
        self.sz = sz
        self.compute_inverse_map = compute_inverse_map
        self.default_learning_rate=default_learning_rate
        self.optimizer = None

    def get_history(self):
        """
        Returns the optimization history as a dictionary. Keeps track of energies, iterations counts, and additonal custom measures.

        :return: history dictionary
        """

        if self.optimizer is not None:
            return self.optimizer.get_history()
        else:
            return None

    def write_parameters_to_settings(self):
        """
        Allows currently computed parameters (if they were optimized) to be written back to an output parameter file
        :return:
        """
        if self.optimizer is not None:
            self.optimizer.write_parameters_to_settings()

    @abstractmethod
    def register(self):
        """
        Abstract method to register the source to the target image
        :return: 
        """
        pass

    def get_optimizer(self):
        """
        Returns the optimizer being used (can be used to customize the simple registration if desired)
        :return: optimizer
        """
        return self.optimizer

    def get_energy(self):
        """
        Returns the current energy
        :return: Returns a tuple (energy, similarity energy, regularization energy)
        """
        if self.optimizer is not None:
            return self.optimizer.get_energy()
        else:
            return None

    def get_warped_label(self):
        """
        Returns the warped label
        :return: the warped label
        """
        if self.optimizer is not None:
            return self.optimizer.get_warped_label()
        else:
            return None

    def get_warped_image(self):
        """
        Returns the warped image
        :return: the warped image
        """
        if self.optimizer is not None:
            return self.optimizer.get_warped_image()
        else:
            return None

    def set_initial_map(self,map0,initial_inverse_map=None):
        """
        Sets the initial map for the registrations; by default (w/o setting anything) this will be the identity
        map, but by setting it to a different initial condition one can concatenate transformations.

        :param map0:
        :return: n/a
        """
        if self.optimizer is not None:
            self.optimizer.set_initial_map(map0, initial_inverse_map)
            # self.optimizer.set_initial_inverse_map(initial_inverse_map)

    def set_weight_map(self,weight_map):
        if self.optimizer is not None:
            self.optimizer.set_initial_map(weight_map)

    def get_initial_map(self):
        """
        Returns the initial map; this will typically be the identity map, but can be set to a different initial
        condition using set_initial_map

        :return: returns the initial map (if applicable)
        """

        if self.optimizer is not None:
            return self.optimizer.get_initial_map()
        else:
            return None

    def get_initial_inverse_map(self):
        """
        Returns the initial inverse map; this will typically be the identity map, but can be set to a different initial
        condition using set_initial_map

        :return: returns the initial map (if applicable)
        """

        if self.optimizer is not None:
            return self.optimizer.get_initial_inverse_map()
        else:
            return None

    def get_map(self):
        """
        Returns the deformation map
        :return: deformation map
        """
        if self.optimizer is not None:
            return self.optimizer.get_map()

    def get_inverse_map(self):
        """
        Returns the inverse deformation map if available
        :return: deformation map
        """
        if self.optimizer is not None:
            return self.optimizer.get_inverse_map()

    def get_model_parameters(self):
        """
        Returns the parameters of the model

        :return: model parameters 
        """
        return self.optimizer.get_model_parameters()

    def set_model_parameters(self,p):
        """
        Sets the parameters of a model

        :param p: model parameters
        :return:
        """
        self.optimizer.set_model_parameters(p)

    def get_model_state_dict(self):
        """
        Returns the state dictionary of the mode

        :return: state dictionary
        """
        return self.optimizer.get_model_state_dict()

    def set_model_state_dict(self,sd):
        """
        Sets the state dictionary of the model

        :param sd: state dictionary
        :return:
        """
        self.optimizer.set_model_state_dict(sd)



class SimpleSingleScaleRegistration(SimpleRegistration):
    """
    Simple single scale registration
    """
    def __init__(self,ISource,ITarget,spacing,sz,params,compute_inverse_map=False, default_learning_rate=None):
        super(SimpleSingleScaleRegistration, self).__init__(ISource,ITarget,spacing,sz,params,compute_inverse_map=compute_inverse_map,default_learning_rate=default_learning_rate)
        self.optimizer = SingleScaleRegistrationOptimizer(self.sz,self.spacing,self.use_map,self.map_low_res_factor,self.params,compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)

    def register(self):
        """
        Registers the source to the target image
        :return: n/a
        """
        self.optimizer.register(self.ISource, self.ITarget)


class SimpleSingleScaleConsensusRegistration(SimpleRegistration):
    """
    Single scale registration making use of consensus optimization (to allow for multiple independent registration
    that can share parameters).
    """
    def __init__(self,ISource,ITarget,spacing,sz,params,compute_inverse_map=False, default_learning_rate=None):
        super(SimpleSingleScaleConsensusRegistration, self).__init__(ISource,ITarget,spacing,sz,params,compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)
        self.optimizer = SingleScaleConsensusRegistrationOptimizer(self.sz,self.spacing,self.use_map,self.map_low_res_factor,self.params,compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)

    def register(self):
        """
        Registers the source to the target image
        :return: n/a
        """
        self.optimizer.register(self.ISource, self.ITarget)


class SimpleSingleScaleBatchRegistration(SimpleRegistration):
    """
    Single scale registration making use of batch optimization (to allow optimizing over many or large images).
    """
    def __init__(self,ISource,ITarget,spacing,sz,params,compute_inverse_map=False, default_learning_rate=None):
        super(SimpleSingleScaleBatchRegistration, self).__init__(ISource,ITarget,spacing,sz,params,compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)
        self.optimizer = SingleScaleBatchRegistrationOptimizer(self.sz,self.spacing,self.use_map,self.map_low_res_factor,self.params,compute_inverse_map=compute_inverse_map,default_learning_rate=default_learning_rate)

    def register(self):
        """
        Registers the source to the target image
        :return: n/a
        """
        self.optimizer.register(self.ISource, self.ITarget)


class SimpleMultiScaleRegistration(SimpleRegistration):
    """
    Simple multi scale registration
    """
    def __init__(self,ISource,ITarget,spacing,sz,params,compute_inverse_map=False, default_learning_rate=None):
        super(SimpleMultiScaleRegistration, self).__init__(ISource, ITarget, spacing,sz,params,compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)
        self.optimizer = MultiScaleRegistrationOptimizer(self.sz,self.spacing,self.use_map,self.map_low_res_factor,self.params,compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)

    def register(self):
        """
        Registers the source to the target image
        :return: n/a
        """
        self.optimizer.register(self.ISource,self.ITarget)


class Optimizer(with_metaclass(ABCMeta, object)):
    """
       Abstract optimizer base class.
    """

    def __init__(self, sz, spacing, useMap, mapLowResFactor, params, compute_inverse_map=False, default_learning_rate=None):
        """
        Constructor.
        
        :param sz: image size in BxCxXxYxZ format
        :param spacing: spatial spacing, e.g., [0.1,0.1,0.1] in 3D
        :param useMap: boolean, True if a coordinate map is evolved to warp images, False otherwise
        :param map_low_res_factor: if <1 evolutions happen at a lower resolution; >=1 ignored 
        :param params: ParametersDict() instance to hold parameters
        :param compute_inverse_map: for map-based models the inverse map can be computed on the fly
        """
        self.sz = sz
        """image size"""
        self.spacing = spacing
        """image spacing"""
        self.lowResSize = None
        """low res image size"""
        self.lowResSpacing = None
        """low res image spacing"""
        self.useMap = useMap
        """makes use of map"""
        self.mapLowResFactor = mapLowResFactor
        """if <1 then evolutions are at a lower resolution, but image is compared at the same resolution; >=1 ignored"""
        if self.mapLowResFactor is not None:
            if self.mapLowResFactor>1:
                print('mapLowResFactor needs to be <=1 but is set to ' + str( self.mapLowResFactor ) + '; ignoring it')
                self.mapLowResFactor = None
            elif self.mapLowResFactor==1:
                print('mapLowResFactor = 1: performing computations at original resolution.')
                self.mapLowResFactor = None

        self.compute_inverse_map = compute_inverse_map
        """If set to True the inverse map is computed on the fly for map-based models"""
        self.default_learning_rate = default_learning_rate
        """If set, this will be the learning rate that the optimizers used (otherwise, as specified in the json configuration, via params)"""

        self.params = params
        """general parameters"""
        self.rel_ftol = 1e-4
        """relative termination tolerance for optimizer"""
        self.last_successful_step_size_taken = None
        """Records the last successful step size an optimizer took (possible use: propogate step size between multiscale levels"""

        self.external_optimizer_parameter_loss = None

        if (self.mapLowResFactor is not None):
            self.lowResSize = utils._get_low_res_size_from_size( sz, self.mapLowResFactor )
            self.lowResSpacing = utils._get_low_res_spacing_from_spacing(self.spacing,sz,self.lowResSize)
        self.sampler = IS.ResampleImage()

        self.params[('optimizer', {}, 'optimizer settings')]
        self.params[('model', {}, 'general model settings')]
        self.params['model'][('deformation', {}, 'model describing the desired deformation model')]
        self.params['model'][('registration_model', {}, 'general settings for the registration model')]

        self.params['model']['deformation']['use_map']= (useMap, '[True|False] either do computations via a map or directly using the image')
        self.params['model']['deformation']['map_low_res_factor'] = (mapLowResFactor, 'Set to a value in (0,1) if a map-based solution should be computed at a lower internal resolution (image matching is still at full resolution')

        self.compute_similarity_measure_at_low_res = self.params['model']['deformation'][('compute_similarity_measure_at_low_res',False,'If set to true map is not upsampled and the entire computations proceeds at low res')]

        self.rel_ftol = self.params['optimizer']['single_scale'][('rel_ftol',self.rel_ftol,'relative termination tolerance for optimizer')]

        self.spline_order = params['model']['registration_model'][('spline_order', 1, 'Spline interpolation order; 1 is linear interpolation (default); 3 is cubic spline')]
        """order of the spline for interpolations"""

        self.show_iteration_output = True
        self.history = dict()

        self.optimizer_has_been_initialized = False
        """
            Needs to be set before the actual optimization commences; allows to keep track if all parameters have been set
            and for example to delay external parameter settings
        """

    def write_parameters_to_settings(self):
        """
        Writes current state of optimized parameters back to the json setting file (for example to keep track of optimized weights)
        :return:
        """
        pass

    def turn_iteration_output_on(self):
        self.show_iteration_output = True

    def turn_iteration_output_off(self):
        self.show_iteration_output = False

    def get_history(self):
        """
        Returns the optimization history as a dictionary. Keeps track of energies, iterations counts, and additonal custom measures.

        :return: history dictionary
        """
        return self.history

    def _add_to_history(self,key,value):
        """
        Adds an element to the optimizer history

        :param key: history key
        :param value: value that is associated with it
        :return: n/a
        """
        if key not in self.history:
            self.history[key] = [value]
        else:
            self.history[key].append(value)

    def set_last_successful_step_size_taken(self,lr):
        """
        Function to let the optimizer know what step size has been successful previously.
        Useful for example to retain optimization "memory" across scales in a multi-scale implementation
        :param lr: step size
        :return: n/a
        """
        self.last_successful_step_size_taken=lr

    def get_last_successful_step_size_taken(self):
        """
        Returns the last successful step size the optimizer has taken (if the optimizer supports querying the step size)
        :return: last successful step size
        """
        return self.last_successful_step_size_taken

    def set_rel_ftol(self, rel_ftol):
        """Sets the relative termination tolerance: :math:`|f(x_i)-f(x_{i-1})|/f(x_i)<tol`
        
        :param rel_ftol: relative termination tolerance for optimizer
        """
        self.rel_ftol = rel_ftol
        self.params['optimizer']['single_scale']['rel_ftol'] = (rel_ftol,'relative termination tolerance for optimizer')
        self.rel_ftol = self.params['optimizer']['single_scale']['rel_ftol']

    def get_rel_ftol(self):
        """
        Returns the optimizer termination tolerance
        """
        return self.rel_ftol



    @abstractmethod
    def set_model(self, modelName):
        """
        Abstract method to select the model which should be optimized by name
        
        :param modelName: name (string) of the model that should be solved
        """
        pass

    @abstractmethod
    def optimize(self):
        """
        Abstract method to start the optimization
        """
        pass

    def get_last_successful_step_size_taken(self):
        return self.last_successful_step_size_taken

    def get_checkpoint_dict(self):
        """
        Returns a dict() object containing the information for the current checkpoint.
        :return: checpoint dictionary
        """
        return dict()

    def load_checkpoint_dict(self,d,load_optimizer_state=False):
        """
        Takes the dictionary from a checkpoint and loads it as the current state of optimizer and model

        :param d: dictionary
        :param load_optimizer_state: if set to True the optimizer state will be restored
        :return: n/a
        """
        pass

    def save_checkpoint(self,filename):
        torch.save(self.get_checkpoint_dict(),filename)

    def load_checkpoint(self,filename):
        d = torch.load(filename)
        self.load_checkpoint_dict(d)

    def set_external_optimizer_parameter_loss(self,opt_parameter_loss):
        """
        Allows to set an external method as an optimizer parameter loss
        :param opt_parameter_loss: method which takes shared_model_parameters as its only input
        :return: returns a scalar value which is the loss
        """
        self.external_optimizer_parameter_loss = opt_parameter_loss

    def get_external_optimizer_parameter_loss(self):
        """
        Returns the externally set method for parameter loss. Will be None if none was set.
        :return: method
        """
        return self.external_optimizer_parameter_loss

    def compute_optimizer_parameter_loss(self,shared_model_parameters):
        """
        Returns the optimizer parameter loss. This is the method that should be called to compute this loss.
        Will either evaluate the method optimizer_parameter_loss or if one was externally defined, the
        externally defined one will have priority.

        :param shared_model_parameters: paramters that have been declared shared in a model
        :return: parameter loss
        """
        if self.external_optimizer_parameter_loss is not None:
            return self.external_optimizer_parameter_loss(shared_model_parameters)
        else:
            return self.optimizer_parameter_loss(shared_model_parameters)

    def optimizer_parameter_loss(self,shared_model_parameters):
        """
        This allows to define additional terms for the loss which are based on parameters that are shared
        between models (for example for the smoother). Can be used to define a form of consensus optimization.
        :param shared_model_parameters: paramters that have been declared shared in a model
        :return: 0 by default, otherwise the corresponding penalty
        """
        return MyTensor(1).zero_()

class ImageRegistrationOptimizer(Optimizer):
    """
    Optimization class for image registration.
    """

    def __init__(self, sz, spacing, useMap, mapLowResFactor, params, compute_inverse_map=False, default_learning_rate=None):
        super(ImageRegistrationOptimizer, self).__init__(sz, spacing, useMap, mapLowResFactor, params, compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)
        self.ISource = None
        """source image"""
        self.lowResISource = None
        """if mapLowResFactor <1, a lowres soure image needs to be created to parameterize some of the registration algorithms"""
        self.lowResITarget = None
        """if mapLowResFactor <1, a lowres target image may need to be created to be used as additonal inputs for registration algorithms"""
        self.ITarget = None
        """target image"""
        self.LSource = None
        """ source label """
        self.LTarget = None
        """  target label """
        self.lowResLSource = None
        """if mapLowResFactor <1, a lowres soure label image needs to be created to parameterize some of the registration algorithms"""
        self.lowResLTarget = None
        """if mapLowResFactor <1, a lowres target label image needs to be created to parameterize some of the registration algorithms"""
        self.initialMap = None
        """  initial map"""
        self.initialInverseMap = None
        """ initial inverse map"""
        self.weight_map =None
        """ initial weight map"""
        self.multi_scale_info_dic = None
        """ dicts containing full resolution image and label"""
        self.optimizer_name = None #''lbfgs_ls'
        """name of the optimizer to use"""
        self.optimizer_params = {}
        """parameters that should be passed to the optimizer"""
        self.optimizer = None
        """optimizer object itself (to be instantiated)"""
        self.visualize = True
        """if True figures are created during the run"""
        self.visualize_step = 10
        """how often the figures are updated; each self.visualize_step-th iteration"""
        self.nrOfIterations = None
        """the maximum number of iterations for the optimizer"""
        self.current_epoch = None
        """Can be set externally, so the optimizer knows in which epoch we are"""
        self.save_fig=False
        """ save fig during the visualization"""
        self.save_fig_path=None
        """ the path for saving figures"""
        self.save_fig_num =-1
        """ the max num of the fig to be saved during one call, set -1 to save all"""
        self.pair_name=None
        """ name list of the registration pair """
        self.iter_count = 0
        """ count of the iterations over multi-resolution"""
        self.recording_step = None
        """sets the step-size for recording all intermediate results to the history"""

    def set_recording_step(self, step):
        assert step > 0, 'Recording step needs to be larger than 0'
        self.recording_step = step
        self.history['recording'] = []

    def set_current_epoch(self,current_epoch):
        self.current_epoch = current_epoch

    def get_current_epoch(self):
        return self.current_epoch



    def turn_visualization_on(self):
        """
        Turns on visualization during the run
        """
        self.visualize = True

    def turn_visualization_off(self):
        """
        Turns off visualization during the run
        """
        self.visualize = False

    def set_visualization(self, vis):
        """
        Set if visualization should be on (True) or off (False)

        :param vis: visualization status on (True) or off (False)
        """
        self.visualize = vis

    def get_visualization(self):
        """
        Returns the visualization status

        :return: Returns True if visualizations will be displayed and False otherwise
        """
        return self.visualize

    def set_visualize_step(self, nr_step):
        """
        Set after how many steps a visualization should be updated

        :param nr_step:
        """
        self.visualize_step = nr_step


    def get_visualize_step(self):
        """
        Returns after how many steps visualizations are updated

        :return: after how many steps visualizations are updated
        """
        return self.visualize_step

    def set_save_fig(self,save_fig):
        """
        :param save_fig: True: save the visualized figs
        :return:
        """
        self.save_fig = save_fig
    def get_save_fig(self):
        """
        :param save_fig: True: get the visualized figs
        :return:
        """
        return self.save_fig

    def set_save_fig_path(self, save_fig_path):
        """
        the path of saved figures, default is the ../data/expr_name
        :param save_fig_path:
        :return:
        """
        self.save_fig_path = save_fig_path



    def get_save_fig_path(self):
        """
        the path of saved figures, default is the ../data/expr_name
        :param save_fig_path:
        :return:
        """
        return self.save_fig_path


    def set_save_fig_num(self, save_fig_num=-1):
        """
        set the num of the fig to save
        :param save_fig_num:
        :return:
        """
        self.save_fig_num = save_fig_num

    def get_save_fig_num(self):
        """
        set the num of the fig to save
        :param save_fig_num:
        :return:
        """
        return self.save_fig_num

    def set_expr_name(self, expr_name):
        """
        the name of experiments
        :param expr_name:
        :return:
        """
        self.expr_name = expr_name

    def get_expr_name(self): 
        """
        the name of experiments
        :param expr_name:
        :return:
        """
        return self.expr_name

    def set_pair_name(self, pair_name):
        self.pair_name = pair_name


    def get_pair_name(self):
        return self.pair_name


    def register(self, ISource, ITarget):
        """
        Registers the source to the target image
        :param ISource: source image
        :param ITarget: target image
        :return: n/a
        """
        self.set_source_image(ISource)
        self.set_target_image(ITarget)
        self.optimize()
        self.write_parameters_to_settings()

    def set_source_image(self, I):
        """
        Setting the source image which should be deformed to match the target image

        :param I: source image
        """
        self.ISource = I

    def set_multi_scale_info(self, ISource, ITarget, spacing, LSource=None, LTarget=None):
        """provide full resolution of Image and Label"""
        self.multi_scale_info_dic = {'ISource': ISource, 'ITarget': ITarget, 'spacing': spacing, 'LSource': LSource,
                                     'LTarget': LTarget}

    def _compute_low_res_image(self,I,params,spacing=None):
        low_res_image = None
        if self.mapLowResFactor is not None:
            low_res_image,_ = self.sampler.downsample_image_to_size(I,spacing,self.lowResSize[2::],self.spline_order)
        return low_res_image

    def _compute_low_res_label_map(self,label_map,params, spacing=None):
        low_res_label_map = None
        if self.mapLowResFactor is not None:
            low_res_image, _ = self.sampler.downsample_image_to_size(label_map, spacing, self.lowResSize[2::],
                                                                     0)
        return low_res_label_map

    def compute_low_res_image_if_needed(self):
        """To be called before the optimization starts"""
        if self.multi_scale_info_dic is None:
            ISource = self.ISource
            ITarget = self.ITarget
            LSource = self.LSource
            LTarget = self.LTarget
            spacing = self.spacing
        else:
            ISource, ITarget, LSource, LTarget, spacing = self.multi_scale_info_dic['ISource'], self.multi_scale_info_dic['ITarget'],\
                                                          self.multi_scale_info_dic['LSource'],self.multi_scale_info_dic['LTarget'],self.multi_scale_info_dic['spacing']
        if self.mapLowResFactor is not None:
            self.lowResISource = self._compute_low_res_image(ISource,self.params,spacing)
            # todo: can be removed to save memory; is more experimental at this point
            self.lowResITarget = self._compute_low_res_image(ITarget,self.params,spacing)
            if self.LSource is not None and self.LTarget is not None:
                self.lowResLSource = self._compute_low_res_label_map(LSource,self.params,spacing)
                self.lowResLTarget = self._compute_low_res_label_map(LTarget, self.params,spacing)

    def set_source_label(self, LSource):
        """
        :param LSource:
        :return:
        """
        self.LSource = LSource


    def set_target_label(self, LTarget):
        """
        :param LTarget:
        :return:
        """
        self.LTarget = LTarget


    def get_source_label(self):
        return self.LSource

    def get_target_label(self):
        return self.LTarget

    def set_target_image(self, I):
        """
        Setting the target image which the source image should match after registration

        :param I: target image
        """
        self.ITarget = I


    def set_optimizer_by_name(self, optimizer_name):
        """
        Set the desired optimizer by name (only lbfgs and adam are currently supported)

        :param optimizer_name: name of the optimizer (string) to be used
        """
        self.optimizer_name = optimizer_name
        self.params['optimizer']['name'] = optimizer_name

    def get_optimizer_by_name(self):
        """
        Get the name (string) of the optimizer that was selected

        :return: name (string) of the optimizer
        """
        return self.optimizer_name

    def set_optimizer(self, opt):
        """
        Set the optimizer. Not by name, but instead by passing the optimizer object which should be instantiated

        :param opt: optimizer object
        """
        self.optimizer = opt

    def get_optimizer(self):
        """
        Returns the optimizer object which was set to perform the optimization

        :return: optimizer object
        """
        return self.optimizer

    def set_optimizer_params(self, opt_params):
        """
        Set the desired parameters of the optimizer. This is done by passing a dictionary, for example, dict(lr=0.01)

        :param opt_params: dictionary holding the parameters of an optimizer
        """
        self.optimizer_params = opt_params


class SingleScaleRegistrationOptimizer(ImageRegistrationOptimizer):
    """
    Optimizer operating on a single scale. Typically this will be the full image resolution.

    .. todo::
        Check what the best way to adapt the tolerances for the pre-defined optimizers;
        tying it to rel_ftol is not really correct.
    """

    def __init__(self, sz, spacing, useMap, mapLowResFactor, params, compute_inverse_map=False, default_learning_rate=None):
        super(SingleScaleRegistrationOptimizer, self).__init__(sz, spacing, useMap, mapLowResFactor, params,compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)

        if self.mapLowResFactor is not None:
            # computes model at a lower resolution than the image similarity
            if self.compute_similarity_measure_at_low_res:
                self.mf = MF.ModelFactory(self.lowResSize, self.lowResSpacing, self.lowResSize, self.lowResSpacing )
            else:
                self.mf = MF.ModelFactory(self.sz, self.spacing, self.lowResSize, self.lowResSpacing )
        else:
            # computes model and similarity at the same resolution
            self.mf = MF.ModelFactory(self.sz, self.spacing, self.sz, self.spacing)
        """model factory which will be used to create the model and its loss function"""

        self.model = None
        """the model itself"""
        self.criterion = None
        """the loss function"""

        self.initialMap = None
        """initial map, will be needed for map-based solutions; by default this will be the identity map, but can be set to something different externally"""
        self.initialInverseMap = None
        """initial inverse map; will be the same as the initial map, unless it was set externally"""
        self.map0_inverse_external = None
        """initial inverse map, set externally, will be needed for map-based solutions; by default this will be the identity map, but can be set to something different externally"""
        self.map0_external = None
        """intial map, set externally"""
        self.lowResInitialMap = None
        """low res initial map, by default the identity map, will be needed for map-based solutions which are computed at lower resolution"""
        self.lowResInitialInverseMap = None
        """low res initial inverse map, by default the identity map, will be needed for map-based solutions which are computed at lower resolution"""
        self.weight_map =None
        """init_weight map, which only used by metric learning models"""
        self.optimizer_instance = None
        """the optimizer instance to perform the actual optimization"""

        c_params = self.params[('optimizer', {}, 'optimizer settings')]
        self.weight_clipping_type = c_params[('weight_clipping_type','none','Type of weight clipping that should be used [l1|l2|l1_individual|l2_individual|l1_shared|l2_shared|None]')]
        self.weight_clipping_type = self.weight_clipping_type.lower()
        """Type of weight clipping; applied to weights and bias indepdenendtly; norm restricted to weight_clipping_value"""
        if self.weight_clipping_type=='none':
            self.weight_clipping_type = None
        if self.weight_clipping_type!='pre_lsm_weights':
            self.weight_clipping_value = c_params[('weight_clipping_value', 1.0, 'Value to which the norm is being clipped')]
            """Desired norm after clipping"""

        extent = self.spacing * self.sz[2:]
        max_extent = max(extent)

        clip_params = c_params[('gradient_clipping',{},'clipping settings for the gradient for optimization')]
        self.clip_display = clip_params[('clip_display', True, 'If set to True displays if clipping occurred')]
        self.clip_individual_gradient = clip_params[('clip_individual_gradient',False,'If set to True, the gradient for the individual parameters will be clipped')]
        self.clip_individual_gradient_value = clip_params[('clip_individual_gradient_value',max_extent,'Value to which the gradient for the individual parameters is clipped')]
        self.clip_shared_gradient = clip_params[('clip_shared_gradient', True, 'If set to True, the gradient for the shared parameters will be clipped')] # todo recover the clip gradient,or it may cause unstable
        self.clip_shared_gradient_value = clip_params[('clip_shared_gradient_value', 1.0, 'Value to which the gradient for the shared parameters is clipped')]

        self.scheduler = None # for the step size scheduler
        self.patience = None # for the step size scheduler
        self._use_external_scheduler = False

        self.rec_energy = None
        self.rec_similarityEnergy = None
        self.rec_regEnergy = None
        self.rec_opt_par_loss_energy = None
        self.rec_phiWarped = None
        self.rec_phiInverseWarped = None
        self.rec_IWarped = None
        self.last_energy = None
        self.rel_f = None
        self.rec_custom_optimizer_output_string = ''
        """the evaluation information"""
        self.rec_custom_optimizer_output_values = None

        self.delayed_model_parameters = None
        self.delayed_model_parameters_still_to_be_set = False
        self.delayed_model_state_dict = None
        self.delayed_model_state_dict_still_to_be_set = False

        # to be able to transfer state and parameters
        self._sgd_par_list = None # holds the list of parameters
        self._sgd_par_names = None # holds the list of names associated with these parameters
        self._sgd_name_to_model_par = None # allows mapping from name to model parameter
        self._sgd_split_shared = None # keeps track if the shared states were split or not
        self._sgd_split_individual = None # keeps track if the individual states were split or not
        self.over_scale_iter_count = None #accumulated iter count over different scales
        self.n_scale = None #the index of  current scale, torename and document  todo


    def write_parameters_to_settings(self):
        if self.model is not None:
            self.model.write_parameters_to_settings()

    def get_sgd_split_shared(self):
        return self._sgd_split_shared

    def get_sgd_split_indvidual(self):
        return self._sgd_split_individual

    def get_checkpoint_dict(self):
        if self.model is not None and self.optimizer_instance is not None:
            d = super(SingleScaleRegistrationOptimizer, self).get_checkpoint_dict()
            d['model'] = dict()
            d['model']['parameters'] = self.model.get_registration_parameters_and_buffers()
            d['model']['size'] = self.model.sz
            d['model']['spacing'] = self.model.spacing
            d['optimizer_state'] = self.optimizer_instance.state_dict()
            return d
        else:
            raise ValueError('Unable to create checkpoint, because either the model or the optimizer have not been initialized')

    def load_checkpoint_dict(self,d,load_optimizer_state=False):
        if self.model is not None and self.optimizer_instance is not None:
            self.model.set_registration_parameters(d['model']['parameters'],d['model']['size'],d['model']['spacing'])
            if load_optimizer_state:
                try:
                    self.optimizer_instance.load_state_dict(d['optimizer_state'])
                    print('INFO: Was able to load the previous optimzer state from checkpoint data')
                except:
                    print('INFO: Could not load the previous optimizer state')
            else:
                print('WARNING: Turned off the loading of the optimizer state')
        else:
            raise ValueError('Cannot load checkpoint dictionary, because either the model or the optimizer have not been initialized')

    def get_opt_par_energy(self):
        """
        Energy for optimizer parameters

        :return:
        """
        return self.rec_opt_par_loss_energy.cpu().item()

    def get_custom_output_values(self):
        """
        Custom output values

        :return:
        """
        return self.rec_custom_optimizer_output_values

    def get_energy(self):
        """
        Returns the current energy
        :return: Returns a tuple (energy, similarity energy, regularization energy)
        """
        return self.rec_energy.cpu().item(), self.rec_similarityEnergy.cpu().item(), self.rec_regEnergy.cpu().item()

    def get_warped_image(self):
        """
        Returns the warped image
        :return: the warped image
        """
        if self.useMap:
            cmap = self.get_map()
            # and now warp it
            return utils.compute_warped_image_multiNC(self.ISource, cmap, self.spacing, self.spline_order,zero_boundary=True)
        else:
            return self.rec_IWarped

    def get_warped_label(self):
        """
        Returns the warped label
        :return: the warped label
        """
        if self.useMap:
            cmap = self.get_map()
            return utils.get_warped_label_map(self.LSource, cmap, self.spacing)
        else:
            return None

    def get_map(self):
        """
        Returns the deformation map
        :return: deformation map
        """
        return self.rec_phiWarped

    def get_inverse_map(self):
        """
        Returns the deformation map
        :return: deformation map
        """
        return self.rec_phiInverseWarped

    def set_n_scale(self, n_scale):
        """
        the path of saved figures, default is the ../data/expr_name
        :param save_fig_path:
        :return:
        """
        self.n_scale = n_scale

    def set_over_scale_iter_count(self, iter_count):
        self.over_scale_iter_count = iter_count


    def _create_initial_maps(self):
        if self.useMap:
            # create the identity map [-1,1]^d, since we will use a map-based implementation
            if self.map0_external is not None:
                self.initialMap = self.map0_external
            else:
                id = utils.identity_map_multiN(self.sz, self.spacing)
                self.initialMap = AdaptVal(torch.from_numpy(id))

            if self.map0_inverse_external is not None:
                self.initialInverseMap = self.map0_inverse_external
            else:
                id =utils.identity_map_multiN(self.sz, self.spacing)
                self.initialInverseMap =  AdaptVal(torch.from_numpy(id))

            if self.mapLowResFactor is not None:
                # create a lower resolution map for the computations
                if self.map0_external is None:
                    lowres_id = utils.identity_map_multiN(self.lowResSize, self.lowResSpacing)
                    self.lowResInitialMap = AdaptVal(torch.from_numpy(lowres_id))
                else:
                    sampler = IS.ResampleImage()
                    lowres_id, _ = sampler.downsample_image_to_size(self.initialMap , self.spacing,self.lowResSize[2::] , 1,zero_boundary=False)
                    self.lowResInitialMap = AdaptVal(lowres_id)

                if self.map0_inverse_external is None:
                    lowres_id = utils.identity_map_multiN(self.lowResSize, self.lowResSpacing)
                    self.lowResInitialInverseMap = AdaptVal(torch.from_numpy(lowres_id))
                else:
                    sampler = IS.ResampleImage()
                    lowres_inverse_id, _ = sampler.downsample_image_to_size(self.initialInverseMap, self.spacing, self.lowResSize[2::],
                                                                    1, zero_boundary=False)
                    self.lowResInitialInverseMap = AdaptVal(lowres_inverse_id)

    def set_model(self, modelName):
        """
        Sets the model that should be solved

        :param modelName: name of the model that should be solved (string)
        """

        self.params['model']['registration_model']['type'] = ( modelName, "['svf'|'svf_quasi_momentum'|'svf_scalar_momentum'|'svf_vector_momentum'|'lddmm_shooting'|'lddmm_shooting_scalar_momentum'] all with '_map' or '_image' suffix" )

        self.model, self.criterion = self.mf.create_registration_model(modelName, self.params['model'],compute_inverse_map=self.compute_inverse_map)
        print(self.model)

        self._create_initial_maps()

    def set_initial_map(self,map0,map0_inverse=None):
        """
        Sets the initial map (overwrites the default identity map)
        :param map0: intial map
        :param map0_inverse: initial inverse map
        :return: n/a
        """

        self.map0_external = map0
        self.map0_inverse_external = map0_inverse

        if self.initialMap is not None:
            # was already set, so let's modify it
            self._create_initial_maps()

    def set_initial_weight_map(self,weight_map,freeze_weight=False):
        """
        Sets the initial map (overwrites the default identity map)
        :param map0: intial map
        :param map0_inverse: initial inverse map
        :return: n/a
        """
        if self.mapLowResFactor is not None:
            sampler = IS.ResampleImage()
            weight_map, _ = sampler.downsample_image_to_size(weight_map, self.spacing, self.lowResSize[2::], 1,
                                                            zero_boundary=False)
        self.model.local_weights.data = weight_map
        if freeze_weight:
            self.model.freeze_adaptive_regularizer_param()

    def get_initial_map(self):
        """
        Returns the initial map

        :return: initial map
        """

        if self.initialMap is not None:
            return self.initialMap
        elif self.map0_external is not None:
            return self.map0_external
        else:
            return None

    def get_initial_inverse_map(self):
        """
        Returns the initial inverse map

        :return: initial inverse map
        """

        if self.initialInverseMap is not None:
            return self.initialInverseMap
        elif self.map0_inverse_external is not None:
            return self.map0_inverse_external
        else:
            return None

    def add_similarity_measure(self, sim_name, sim_measure):
        """
        Adds a custom similarity measure.

        :param sim_name: name of the similarity measure (string)
        :param sim_measure: similarity measure itself (class object that can be instantiated)
        """
        self.criterion.add_similarity_measure(sim_name, sim_measure)
        self.params['model']['registration_model']['similarity_measure']['type'] = (sim_name, 'was customized; needs to be expplicitly instantiated, cannot be loaded')

    def add_model(self, model_name, model_network_class, model_loss_class, use_map, model_description='custom model'):
        """
        Adds a custom model and its loss function

        :param model_name: name of the model to be added (string)
        :param model_network_class: registration model itself (class object that can be instantiated)
        :param model_loss_class: registration loss (class object that can be instantiated)
        :param use_map: True/False: specifies if model uses a map or not
        :param model_description: optional model description
        """
        self.mf.add_model(model_name, model_network_class, model_loss_class, use_map, model_description)
        self.params['model']['registration_model']['type'] = (model_name, 'was customized; needs to be explicitly instantiated, cannot be loaded')

    def set_model_state_dict(self,sd):
        """
        Sets the state dictionary of the model

        :param sd: state dictionary
        :return: n/a
        """

        if self.optimizer_has_been_initialized:
            self.model.load_state_dict(sd)
            self.delayed_model_state_dict_still_to_be_set = False
        else:
            self.delayed_model_state_dict_still_to_be_set = True
            self.delayed_model_state_dict = sd

    def get_model_state_dict(self):
        """
        Returns the state dictionary of the model

        :return: state dictionary
        """
        return self.model.state_dict()

    def set_model_parameters(self, p):
        """
        Set the parameters of the registration model

        :param p: parameters
        """

        if self.optimizer_has_been_initialized:
            if (self.useMap) and (self.mapLowResFactor is not None):
                self.model.set_registration_parameters(p, self.lowResSize, self.lowResSpacing)
            else:
                self.model.set_registration_parameters(p, self.sz, self.spacing)
            self.delayed_model_parameters_still_to_be_set = False
        else:
            self.delayed_model_parameters_still_to_be_set = True
            self.delayed_model_parameters = p

    def _is_vector(self,d):
        sz = d.size()
        if len(sz)==1:
            return True
        else:
            return False

    def _is_tensor(self,d):
        sz = d.size()
        if len(sz)>1:
            return True
        else:
            return False

    def _aux_do_weight_clipping_norm(self,pars,desired_norm):
        """does weight clipping but only for conv or bias layers (assuming they are named as such); be careful with the namimg here"""
        if self.weight_clipping_value > 0:
            for key in pars:
                # only do the clipping if it is a conv layer or a bias term
                if key.lower().find('conv')>0 or key.lower().find('bias')>0:
                    p = pars[key]
                    if self._is_vector(p.data):
                        # just normalize this vector component-by-component, norm does not matter here as these are only scalars
                        p.data = p.data.clamp_(-self.weight_clipping_value, self.weight_clipping_value)
                    elif self._is_tensor(p.data):
                        # normalize sample-by-sample individually
                        for b in range(p.data.size()[0]):
                            param_norm = p.data[b, ...].norm(desired_norm)
                            if param_norm > self.weight_clipping_value:
                                clip_coef = self.weight_clipping_value / param_norm
                                p.data[b, ...].mul_(clip_coef)
                    else:
                        raise ValueError('Unknown data type; I do not know how to clip this')

    def _do_shared_weight_clipping_pre_lsm(self):
        multi_gaussian_weights = self.params['model']['registration_model']['forward_model']['smoother'][('multi_gaussian_weights', -1, 'the used multi gaussian weights')]
        if multi_gaussian_weights==-1:
            raise ValueError('The multi-gaussian weights should have been set before')
        multi_gaussian_weights = np.array(multi_gaussian_weights)

        sp = self.get_shared_model_parameters()
        for key in sp:
            if key.lower().find('pre_lsm_weights') > 0:
                p = sp[key]
                sz = p.size() #0 dim is weight dimension
                if sz[0]!=len(multi_gaussian_weights):
                    raise ValueError('Number of multi-Gaussian weights needs to be {}, but got {}'.format(sz[0],len(multi_gaussian_weights)))
                for w in range(sz[0]):
                    # this is to assure that the weights are always between 0 and 1 (when using the WeightedLinearSoftmax
                    p[w,...].data.clamp_(0.0-multi_gaussian_weights[w],1.0-multi_gaussian_weights[w])
                
    def _do_individual_weight_clipping_l1(self):
        ip = self.get_individual_model_parameters()
        self._aux_do_weight_clipping_norm(pars=ip,desired_norm=1)

    def _do_shared_weight_clipping_l1(self):
        sp = self.get_shared_model_parameters()
        self._aux_do_weight_clipping_norm(pars=sp,desired_norm=1)

    def _do_individual_weight_clipping_l2(self):
        ip = self.get_individual_model_parameters()
        self._aux_do_weight_clipping_norm(pars=ip, desired_norm=2)

    def _do_shared_weight_clipping_l2(self):
        sp = self.get_shared_model_parameters()
        self._aux_do_weight_clipping_norm(pars=sp, desired_norm=2)

    def _do_weight_clipping(self):
        """performs weight clipping, if desired"""
        if self.weight_clipping_type is not None:
            possible_modes = ['l1', 'l2', 'l1_individual', 'l2_individual', 'l1_shared', 'l2_shared', 'pre_lsm_weights']
            if self.weight_clipping_type in possible_modes:
                if self.weight_clipping_type=='l1':
                    self._do_shared_weight_clipping_l1()
                    self._do_individual_weight_clipping_l1()
                elif self.weight_clipping_type=='l2':
                    self._do_shared_weight_clipping_l2()
                    self._do_individual_weight_clipping_l2()
                elif self.weight_clipping_type=='l1_individual':
                    self._do_individual_weight_clipping_l1()
                elif self.weight_clipping_type=='l2_individual':
                    self._do_individual_weight_clipping_l2()
                elif self.weight_clipping_type=='l1_shared':
                    self._do_shared_weight_clipping_l1()
                elif self.weight_clipping_type=='l2_shared':
                    self._do_shared_weight_clipping_l2()
                elif self.weight_clipping_type=='pre_lsm_weights':
                    self._do_shared_weight_clipping_pre_lsm()
                else:
                    raise ValueError('Illegal weight clipping type: {}'.format(self.weight_clipping_type))
            else:
                raise ValueError('Weight clipping needs to be: [None|l1|l2|l1_individual|l2_individual|l1_shared|l2_shared]')

    def get_model_parameters(self):
        """
        Returns the parameters of the model

        :return: model parameters
        """
        return self.model.get_registration_parameters()

    def set_shared_model_parameters(self,p):
        """
        Set only the shared parameters of the model

        :param p: shared registration parameters as an ordered dict
        :return: n/a
        """

        self.model.set_shared_registration_parameters(p)

    def get_shared_model_parameters_and_buffers(self):
        """
        Returns only the model parameters that are shared between models and the shared buffers associated w/ it.

        :return: shared model parameters and buffers
        """
        return self.model.get_shared_registration_parameters_and_buffers()

    def get_shared_model_parameters(self):
        """
        Returns only the model parameters that are shared between models.

        :return: shared model parameters
        """
        return self.model.get_shared_registration_parameters()

    def set_individual_model_parameters(self,p):
        """
        Set only the individual parameters of the model

        :param p: individual registration parameters as an ordered dict
        :return: n/a
        """

        self.model.set_individual_registration_parameters(p)

    def get_individual_model_parameters(self):
        """
        Returns only the model parameters that individual to a model (i.e., not shared).

        :return: individual model parameters
        """
        return self.model.get_individual_registration_parameters()

    def _collect_individual_or_shared_parameters_in_list(self,pars):
        pl = []
        for p_key in pars:
            pl.append(pars[p_key])
        return pl

    def load_shared_state_dict(self,sd):
        """
        Loads the shared part of a state dictionary
        :param sd: shared state dictionary
        :return: n/a
        """
        self.model.load_shared_state_dict(sd)

    def shared_state_dict(self):
        """
        Returns the shared part of a state dictionary
        :return:
        """
        return self.model.shared_state_dict()

    def load_individual_state_dict(self):
        raise ValueError('Not yet implemented')

    def individual_state_dict(self):
        raise ValueError('Not yet implemented')

    def upsample_model_parameters(self, desiredSize):
        """
        Upsamples the model parameters

        :param desiredSize: desired size after upsampling, e.g., [100,20,50]
        :return: returns a tuple (upsampled_parameters,upsampled_spacing)
        """
        return self.model.upsample_registration_parameters(desiredSize)

    def downsample_model_parameters(self, desiredSize):
        """
        Downsamples the model parameters

        :param desiredSize: desired size after downsampling, e.g., [50,50,40]
        :return: returns a tuple (downsampled_parameters,downsampled_spacing)
        """
        return self.model.downsample_registration_parameters(desiredSize)

    def _set_number_of_iterations_from_multi_scale(self, nrIter):
        """
        Same as set_number_of_iterations with the exception that this is not recored in the parameter structure since it comes from the multi-scale setting
        :param nrIter: number of iterations
        """
        self.nrOfIterations = nrIter

    def set_number_of_iterations(self, nrIter):
        """
        Set the number of iterations of the optimizer

        :param nrIter: number of iterations
        """
        self.params['optimizer'][('single_scale', {}, 'single scale settings')]
        self.params['optimizer']['single_scale']['nr_of_iterations'] = (nrIter, 'number of iterations')

        self.nrOfIterations = nrIter

    def get_number_of_iterations(self):
        """
        Returns the number of iterations of the solver

        :return: number of set iterations
        """
        return self.nrOfIterations

    def _closure(self):
        self.optimizer_instance.zero_grad()
        # 1) Forward pass: Compute predicted y by passing x to the model
        # 2) Compute loss

        # first define variables that will be passed to the model and the criterion (for further use)

        over_scale_iter_count = self.iter_count if self.over_scale_iter_count is None else self.over_scale_iter_count + self.iter_count
        opt_variables = {'iter': self.iter_count, 'epoch': self.current_epoch, 'scale': self.n_scale,
                         'over_scale_iter_count': over_scale_iter_count}

        self.rec_IWarped, self.rec_phiWarped, self.rec_phiInverseWarped = model_evaluation.evaluate_model_low_level_interface(
            model=self.model,
            I_source=self.ISource,
            opt_variables=opt_variables,
            use_map=self.useMap,
            initial_map=self.initialMap,
            compute_inverse_map=self.compute_inverse_map,
            initial_inverse_map=self.initialInverseMap,
            map_low_res_factor=self.mapLowResFactor,
            sampler=self.sampler,
            low_res_spacing=self.lowResSpacing,
            spline_order=self.spline_order,
            low_res_I_source=self.lowResISource,
            low_res_initial_map=self.lowResInitialMap,
            low_res_initial_inverse_map=self.lowResInitialInverseMap,
            compute_similarity_measure_at_low_res=self.compute_similarity_measure_at_low_res)

        # compute the respective losses
        if self.useMap:
            if self.mapLowResFactor is not None and self.compute_similarity_measure_at_low_res:
                loss_overall_energy, sim_energy, reg_energy = self.criterion(self.lowResInitialMap, self.rec_phiWarped,
                                                                             self.lowResISource, self.lowResITarget,
                                                                             self.lowResISource,
                                                                             self.model.get_variables_to_transfer_to_loss_function(),
                                                                             opt_variables)
            else:
                loss_overall_energy,sim_energy,reg_energy = self.criterion(self.initialMap, self.rec_phiWarped, self.ISource, self.ITarget, self.lowResISource,
                                                                           self.model.get_variables_to_transfer_to_loss_function(),
                                                                           opt_variables)
        else:
            loss_overall_energy,sim_energy,reg_energy = self.criterion(self.rec_IWarped, self.ISource, self.ITarget,
                                  self.model.get_variables_to_transfer_to_loss_function(),
                                  opt_variables )

        # to support consensus optimization we have the option of adding a penalty term
        # based on shared parameters
        opt_par_loss_energy = self.compute_optimizer_parameter_loss(self.model.get_shared_registration_parameters())
        loss_overall_energy  = loss_overall_energy + opt_par_loss_energy
        loss_overall_energy.backward()

        # do gradient clipping
        if self.clip_individual_gradient:
            current_individual_grad_norm = torch.nn.utils.clip_grad_norm_(
                self._collect_individual_or_shared_parameters_in_list(self.get_individual_model_parameters()),
                self.clip_individual_gradient_value)

            if self.clip_display:
                if current_individual_grad_norm>self.clip_individual_gradient_value:
                    print('INFO: Individual gradient was clipped: {} -> {}'.format(current_individual_grad_norm,self.clip_individual_gradient_value))

        if self.clip_shared_gradient:
            current_shared_grad_norm = torch.nn.utils.clip_grad_norm_(
                self._collect_individual_or_shared_parameters_in_list(self.get_shared_model_parameters()),
                self.clip_shared_gradient_value)

            if self.clip_display:
                if current_shared_grad_norm > self.clip_shared_gradient_value:
                    print('INFO: Shared gradient was clipped: {} -> {}'.format(current_shared_grad_norm,
                                                                                   self.clip_shared_gradient_value))

        self.rec_custom_optimizer_output_string = self.model.get_custom_optimizer_output_string()
        self.rec_custom_optimizer_output_values = self.model.get_custom_optimizer_output_values()

        self.rec_energy = loss_overall_energy
        self.rec_similarityEnergy = sim_energy
        self.rec_regEnergy = reg_energy
        self.rec_opt_par_loss_energy = opt_par_loss_energy

        # if self.useMap:
        #
        #    if self.iter_count % 1 == 0:
        #        self.rec_energy, self.rec_similarityEnergy, self.rec_regEnergy = self.criterion.get_energy(
        #            self.identityMap, self.rec_phiWarped, self.ISource, self.ITarget, self.lowResISource, self.model.get_variables_to_transfer_to_loss_function())
        # else:
        #    if self.iter_count % 1 == 0:
        #        self.rec_energy, self.rec_similarityEnergy, self.rec_regEnergy = self.criterion.get_energy(
        #            self.rec_IWarped, self.ISource, self.ITarget, self.model.get_variables_to_transfer_to_loss_function())

        return loss_overall_energy

    def analysis(self, energy, similarityEnergy, regEnergy, opt_par_energy, phi_or_warped_image, custom_optimizer_output_string ='', custom_optimizer_output_values=None, force_visualization=False):
        """
        print out the and visualize the result
        :param energy:
        :param similarityEnergy:
        :param regEnergy:
        :param opt_par_energy
        :param phi_or_warped_image:
        :return: returns tuple: first entry True if termination tolerance was reached, otherwise returns False; second entry if the image was visualized
        """

        current_batch_size = phi_or_warped_image.size()[0]

        was_visualized = False
        reached_tolerance = False

        cur_energy = utils.t2np(energy.float())
        # energy analysis

        self._add_to_history('iter', self.iter_count)
        self._add_to_history('energy', cur_energy[0])
        self._add_to_history('similarity_energy', utils.t2np(similarityEnergy.float()))
        self._add_to_history('regularization_energy', utils.t2np(regEnergy.float()))
        self._add_to_history('opt_par_energy', utils.t2np(opt_par_energy.float())[0])

        if custom_optimizer_output_values is not None:
            for key in custom_optimizer_output_values:
                self._add_to_history(key,custom_optimizer_output_values[key])

        if self.last_energy is not None:

            # relative function tolerance: |f(xi)-f(xi+1)|/(1+|f(xi)|)
            self.rel_f = abs(self.last_energy - cur_energy) / (1 + abs(cur_energy))
            self._add_to_history('relF', self.rel_f[0])

            if self.show_iteration_output:
                cprint('{iter:5d}-Tot: E={energy:08.4f} | simE={similarityE:08.4f} | regE={regE:08.4f} | optParE={optParE:08.4f} | relF={relF:08.4f} | {cos}'
                       .format(iter=self.iter_count,
                               energy=utils.get_scalar(cur_energy),
                               similarityE=utils.get_scalar(utils.t2np(similarityEnergy.float())),
                               regE=utils.get_scalar(utils.t2np(regEnergy.float())),
                               optParE=utils.get_scalar(utils.t2np(opt_par_energy.float())),
                               relF=utils.get_scalar(self.rel_f),
                               cos=custom_optimizer_output_string), 'red')
                cprint('{iter:5d}-Img: E={energy:08.4f} | simE={similarityE:08.4f} | regE={regE:08.4f} |'
                       .format(iter=self.iter_count,
                               energy=utils.get_scalar(cur_energy) / current_batch_size,
                               similarityE=utils.get_scalar(utils.t2np(similarityEnergy.float())) / current_batch_size,
                               regE=utils.get_scalar(utils.t2np(regEnergy.float())) / current_batch_size), 'blue')

            # check if relative convergence tolerance is reached
            if self.rel_f < self.rel_ftol:
                if self.show_iteration_output:
                    print('Reached relative function tolerance of = ' + str(self.rel_ftol))
                reached_tolerance = True

        else:
            self._add_to_history('relF', None)
            if self.show_iteration_output:
                cprint('{iter:5d}-Tot: E={energy:08.4f} | simE={similarityE:08.4f} | regE={regE:08.4f} | optParE={optParE:08.4f} | relF=  n/a    | {cos}'
                      .format(iter=self.iter_count,
                              energy=utils.get_scalar(cur_energy),
                              similarityE=utils.get_scalar(utils.t2np(similarityEnergy.float())),
                              regE=utils.get_scalar(utils.t2np(regEnergy.float())),
                              optParE=utils.get_scalar(utils.t2np(opt_par_energy.float())),
                              cos=custom_optimizer_output_string), 'red')
                cprint('{iter:5d}-Img: E={energy:08.4f} | simE={similarityE:08.4f} | regE={regE:08.4f} |'
                      .format(iter=self.iter_count,
                              energy=utils.get_scalar(cur_energy)/current_batch_size,
                              similarityE=utils.get_scalar(utils.t2np(similarityEnergy.float()))/current_batch_size,
                              regE=utils.get_scalar(utils.t2np(regEnergy.float()))/current_batch_size),'blue')

        iter_count = self.iter_count
        self.last_energy = cur_energy

        if self.recording_step is not None:
            if iter_count % self.recording_step == 0 or iter_count == 0:
                if self.useMap:
                    if self.compute_similarity_measure_at_low_res:
                        I1Warped = utils.compute_warped_image_multiNC(self.lowResISource,
                                                                      phi_or_warped_image,
                                                                      self.lowResSpacing,
                                                                      self.spline_order,
                                                                      zero_boundary=False)
                        lowResLWarped = utils.get_warped_label_map(self.lowResLSource,
                                                                   phi_or_warped_image,
                                                                   self.spacing)
                        self.history['recording'].append({
                            'iter': iter_count,
                            'iS': utils.t2np(self.ISource),
                            'iT': utils.t2np(self.ITarget),
                            'iW': utils.t2np(I1Warped),
                            'iSL': utils.t2np(self.lowResLSource) if self.lowResLSource is not None else None,
                            'iTL': utils.t2np(self.lowResLTarget) if self.lowResLTarget is not None else None,
                            'iWL': utils.t2np(lowResLWarped) if self.lowResLWarped is not None else None,
                            'phiWarped': utils.t2np(phi_or_warped_image)
                        })
                    else:
                        I1Warped = utils.compute_warped_image_multiNC(self.ISource,
                                                                      phi_or_warped_image,
                                                                      self.spacing,
                                                                      self.spline_order,
                                                                      zero_boundary=False)
                        LWarped = None
                        if self.LSource is not None and self.LTarget is not None:
                            LWarped = utils.get_warped_label_map(self.LSource,
                                                                 phi_or_warped_image,
                                                                 self.spacing)
                        self.history['recording'].append({
                            'iter': iter_count,
                            'iS': utils.t2np(self.ISource),
                            'iT': utils.t2np(self.ITarget),
                            'iW': utils.t2np(I1Warped),
                            'iSL': utils.t2np(self.LSource) if self.LSource is not None else None,
                            'iTL': utils.t2np(self.LTarget) if self.LTarget is not None else None,
                            'iWL': utils.t2np(LWarped) if LWarped is not None else None,
                            'phiWarped': utils.t2np(phi_or_warped_image)
                        })
                else:
                    self.history['recording'].append({
                        'iter': iter_count,
                        'iS': utils.t2np(self.ISource),
                        'iT': utils.t2np(self.ITarget),
                        'iW': utils.t2np(phi_or_warped_image)
                    })

        if self.visualize or self.save_fig:
            visual_param = {}
            visual_param['visualize'] = self.visualize
            visual_param['save_fig'] = self.save_fig
            visual_param['save_fig_num'] = self.save_fig_num
            if self.save_fig:
                visual_param['save_fig_path'] = self.save_fig_path
                visual_param['save_fig_path_byname'] = os.path.join(self.save_fig_path, 'byname')
                visual_param['save_fig_path_byiter'] = os.path.join(self.save_fig_path, 'byiter')
                visual_param['pair_name'] = self.pair_name
                visual_param['iter'] = 'scale_'+str(self.n_scale) + '_iter_' + str(self.iter_count)

            if self.visualize_step and (iter_count % self.visualize_step == 0) or (iter_count == self.nrOfIterations-1) or force_visualization:
                was_visualized = True
                if self.useMap and self.mapLowResFactor is not None:
                    vizImage, vizName = self.model.get_parameter_image_and_name_to_visualize(self.lowResISource)
                else:
                    vizImage, vizName = self.model.get_parameter_image_and_name_to_visualize(self.ISource)

                if self.useMap:
                    if self.compute_similarity_measure_at_low_res:
                        I1Warped = utils.compute_warped_image_multiNC(self.lowResISource,
                                                                      phi_or_warped_image,
                                                                      self.lowResSpacing,
                                                                      self.spline_order,
                                                                      zero_boundary=False)
                        lowResLWarped = utils.get_warped_label_map(self.lowResLSource,
                                                                   phi_or_warped_image,
                                                                   self.spacing)
                        vizReg.show_current_images(iter=iter_count,
                                                   iS=self.lowResISource,
                                                   iT=self.lowResITarget,
                                                   iW=I1Warped,
                                                   iSL=self.lowResLSource,
                                                   iTL=self.lowResLTarget,
                                                   iWL=lowResLWarped,
                                                   vizImages=vizImage,
                                                   vizName=vizName,
                                                   phiWarped=phi_or_warped_image,
                                                   visual_param=visual_param)

                    else:
                        I1Warped = utils.compute_warped_image_multiNC(self.ISource,
                                                                      phi_or_warped_image,
                                                                      self.spacing,
                                                                      self.spline_order,
                                                                      zero_boundary=False)
                        vizImage = vizImage if len(vizImage)>2 else None
                        LWarped = None
                        if self.LSource is not None  and self.LTarget is not None:
                            LWarped = utils.get_warped_label_map(self.LSource,
                                                                 phi_or_warped_image,
                                                                 self.spacing)

                        vizReg.show_current_images(iter=iter_count,
                                                   iS=self.ISource,
                                                   iT=self.ITarget,
                                                   iW=I1Warped,
                                                   iSL=self.LSource,
                                                   iTL=self.LTarget,
                                                   iWL=LWarped,
                                                   vizImages=vizImage,
                                                   vizName=vizName,
                                                   phiWarped=phi_or_warped_image,
                                                   visual_param=visual_param)
                else:
                    vizReg.show_current_images(iter=iter_count,
                                               iS=self.ISource,
                                               iT=self.ITarget,
                                               iW=phi_or_warped_image,
                                               vizImages=vizImage,
                                               vizName=vizName,
                                               phiWarped=None,
                                               visual_param=visual_param)

        return reached_tolerance, was_visualized

    def _debugging_saving_intermid_img(self,img=None,is_label_map=False, append=''):
        folder_path = os.path.join(self.save_fig_path,'debugging')
        folder_path = os.path.join(folder_path, self.pair_name[0])
        make_dir(folder_path)
        file_name = 'scale_'+str(self.n_scale) + '_iter_' + str(self.iter_count)+append
        file_name=file_name.replace('.','_')
        if is_label_map:
            file_name += '_label'
        path = os.path.join(folder_path,file_name+'.nii.gz')
        im_io = FIO.ImageIO()
        im_io.write(path, np.squeeze(img.detach().cpu().numpy()))

    # todo: write these parameter/optimizer functions also for shared parameters and all parameters
    def set_sgd_shared_model_parameters_and_optimizer_states(self, pars):
        """
               Set the individual model parameters and states that may be stored by the optimizer such as the momentum.
               Expects as input what get_sgd_individual_model_parameters_and_optimizer_states creates as output,
               but potentially multiple copies of it (as generated by a pyTorch dataloader). I.e., it takes in a dataloader sample.
               NOTE: currently only supports SGD

               :param pars: parameter list as produced by get_sgd_individual_model_parameters_and_optimizer_states
               :return: n/a
               """
        if self.optimizer_instance is None:
            raise ValueError('Optimizer not yet created')

        if (self._sgd_par_list is None) or (self._sgd_par_names is None):
            raise ValueError(
                'sgd par list and/or par names not available; needs to be created before passing it to the optimizer')

        if len(pars) == 0:
            print('WARNING: found no values')
            return

        # the optimizer (if properly initialized) already holds pointers to the model parameters and the optimizer states
        # so we can set everything in one swoop here

        # loop over the SGD parameter groups (this is modeled after the code in the SGD optimizer)
        # this input will represent a sample from a pytorch dataloader

        # wrap the parameters in a list if needed (so we can mirror the setup from get_sgd_...
        if type(pars) == list:
            use_pars = pars
        else:
            use_pars = [pars]

        for p in use_pars:
            if 'is_shared' in p:
                if p['is_shared']:
                    current_name = p['name']

                    assert (torch.is_tensor(p['model_params']))
                    current_model_params = p['model_params']

                    if 'momentum_buffer' in p:
                        assert (torch.is_tensor(p['momentum_buffer']))
                        current_momentum_buffer = p['momentum_buffer']
                    else:
                        current_momentum_buffer = None

                    # now we need to match this with the parameters and the state of the SGD optimizer
                    model_par = self._sgd_name_to_model_par[current_name]
                    model_par.data.copy_(current_model_params)

                    # and now do the same with the state
                    param_state = self.optimizer_instance.state[model_par]
                    if 'momentum_buffer' in param_state:
                        param_state['momentum_buffer'].copy_(current_momentum_buffer)

    def set_sgd_individual_model_parameters_and_optimizer_states(self, pars):
        """
        Set the individual model parameters and states that may be stored by the optimizer such as the momentum.
        Expects as input what get_sgd_individual_model_parameters_and_optimizer_states creates as output,
        but potentially multiple copies of it (as generated by a pyTorch dataloader). I.e., it takes in a dataloader sample.
        NOTE: currently only supports SGD

        :param pars: parameter list as produced by get_sgd_individual_model_parameters_and_optimizer_states
        :return: n/a
        """
        if self.optimizer_instance is None:
            raise ValueError('Optimizer not yet created')

        if (self._sgd_par_list is None) or (self._sgd_par_names is None):
            raise ValueError(
                'sgd par list and/or par names not available; needs to be created before passing it to the optimizer')

        if len(pars) == 0:
            print('WARNING: found no values')
            return

        # the optimizer (if properly initialized) already holds pointers to the model parameters and the optimizer states
        # so we can set everything in one swoop here

        # loop over the SGD parameter groups (this is modeled after the code in the SGD optimizer)
        # this input will represent a sample from a pytorch dataloader

        # wrap the parameters in a list if needed (so we can mirror the setup from get_sgd_...
        if type(pars)==list:
            use_pars = pars
        else:
            use_pars = [pars]

        for p in use_pars:
            if 'is_shared' in p:
                if not p['is_shared'][0]: # need to grab the first one, because the dataloader replicated these entries
                    current_name = p['name'][0]

                    assert( torch.is_tensor(p['model_params']))
                    current_model_params = p['model_params']

                    if 'momentum_buffer' in p:
                        assert( torch.is_tensor(p['momentum_buffer']) )
                        current_momentum_buffer = p['momentum_buffer']
                    else:
                        current_momentum_buffer = None

                    # now we need to match this with the parameters and the state of the SGD optimizer
                    model_par = self._sgd_name_to_model_par[current_name]
                    model_par.data.copy_(current_model_params)

                    # and now do the same with the state
                    param_state = self.optimizer_instance.state[model_par]
                    if 'momentum_buffer' in param_state:
                        param_state['momentum_buffer'].copy_(current_momentum_buffer)

    def _convert_obj_with_parameters_to_obj_with_tensors(self, p):
        """
        Converts structures that consist of lists and dictionaries with parameters to tensors

        :param p: parameter structure
        :return: object with parameters converted to tensors
        """

        if type(p) == list:
            ret_p = []
            for e in p:
                ret_p.append(self._convert_obj_with_parameters_to_obj_with_tensors(e))
            return ret_p
        elif type(p) == dict:
            ret_p = dict()
            for key in p:
                ret_p[key] = self._convert_obj_with_parameters_to_obj_with_tensors((p[key]))
            return ret_p
        elif type(p) == torch.nn.parameter.Parameter:
            return p.data
        else:
            return p

    def get_sgd_shared_model_parameters(self):
        """
        Gets the model parameters that are shared.

        :return:
        """

        if self.optimizer_instance is None:
            raise ValueError('Optimizer not yet created')

        if (self._sgd_par_list is None) or (self._sgd_par_names is None):
            raise ValueError(
                'sgd par list and/or par names not available; needs to be created before passing it to the optimizer')

        d = []

        # loop over the SGD parameter groups (this is modeled after the code in the SGD optimizer)
        for group in self.optimizer_instance.param_groups:

            group_dict = dict()
            group_dict['params'] = []

            for p in group['params']:
                current_group_params = dict()
                # let's first see if this is a shared state
                if self._sgd_par_names[p]['is_shared']:
                    # keep track of the names so we can and batch, so we can read it back in
                    current_group_params.update(self._sgd_par_names[p])
                    # now deal with the optimizer state if available
                    current_group_params['model_params'] = self._convert_obj_with_parameters_to_obj_with_tensors(p)

                    group_dict['params'].append(current_group_params)

            d.append(group_dict)

        return d


    def get_sgd_individual_model_parameters_and_optimizer_states(self):
        """
        Gets the individual model parameters and states that may be stored by the optimizer such as the momentum.
        NOTE: currently only supports SGD

        :return:
        """
        if self.optimizer_instance is None:
            raise ValueError('Optimizer not yet created')

        if (self._sgd_par_list is None) or (self._sgd_par_names is None):
            raise ValueError(
                'sgd par list and/or par names not available; needs to be created before passing it to the optimizer')

        d = []

        # loop over the SGD parameter groups (this is modeled after the code in the SGD optimizer)
        for group in self.optimizer_instance.param_groups:

            group_dict = dict()
            group_dict['weight_decay'] = group['weight_decay']
            group_dict['momentum'] = group['momentum']
            group_dict['dampening'] = group['dampening']
            group_dict['nesterov'] = group['nesterov']
            group_dict['lr'] = group['lr']

            group_dict['params'] = []

            for p in group['params']:
                current_group_params = dict()
                # let's first see if this is a shared state
                if not self._sgd_par_names[p]['is_shared']:
                    # keep track of the names so we can and batch, so we can read it back in
                    current_group_params.update(self._sgd_par_names[p])
                    # now deal with the optimizer state if available
                    current_group_params['model_params'] = self._convert_obj_with_parameters_to_obj_with_tensors(p)
                    if group['momentum'] != 0:
                        param_state = self.optimizer_instance.state[p]
                        if 'momentum_buffer' in param_state:
                            current_group_params['momentum_buffer'] = self._convert_obj_with_parameters_to_obj_with_tensors(param_state['momentum_buffer'])

                    group_dict['params'].append(current_group_params)

            d.append(group_dict)

        return d

    def _remove_state_variables_for_individual_parameters(self,individual_pars):
        """
        Removes the optimizer state for individual parameters.
        This is required at the beginning as we do not want to reuse the SGD momentum for example for an unrelated registration.

        :param individual_pars: individual parameters are returned by get_sgd_individual_model_parameters_and_optimizer_states
        :return: n/a
        """

        if self.optimizer_instance is None:
            raise ValueError('Optimizer not yet created')

        if (self._sgd_par_list is None) or (self._sgd_par_names is None):
            raise ValueError(
                'sgd par list and/or par names not available; needs to be created before passing it to the optimizer')

        # loop over the SGD parameter groups (this is modeled after the code in the SGD optimizer)
        for group in self.optimizer_instance.param_groups:

            for p in group['params']:
                # let's first see if this is a shared state
                if not self._sgd_par_names[p]['is_shared']:
                    # we want to delete the state of this one
                    self.optimizer_instance.state.pop(p)


    def _create_optimizer_parameter_dictionary(self,individual_pars, shared_pars,
                                              settings_individual=dict(), settings_shared=dict()):

        par_list = []
        """List of parameters that can directly be passed to an optimizer; different list elements define different parameter groups"""
        par_names = dict()
        """dictionary which maps from a parameters id (i.e., memory) to its description: name/is_shared"""
        # name is the name of the variable
        # is_shared keeps track of if a parameter was declared shared (opposed to individual, which we need for registrations)

        names_to_par = dict()
        """dictionary which maps from a parameter name back to the parameter"""

        # first deal with the individual parameters
        pl_ind, par_to_name_ind = utils.get_parameter_list_and_par_to_name_dict_from_parameter_dict(individual_pars)
        #cd = {'params': pl_ind}
        cd = {'params': [p for p in pl_ind if p.requires_grad]}
        cd.update(settings_individual)
        par_list.append(cd)
        # add all the names
        for current_par, key in zip(pl_ind, par_to_name_ind):
            par_names[key] = {'name': par_to_name_ind[key], 'is_shared': False}
            names_to_par[par_to_name_ind[key]] = current_par

        # now deal with the shared parameters
        pl_shared, par_to_name_shared = utils.get_parameter_list_and_par_to_name_dict_from_parameter_dict(shared_pars)
        #cd = {'params': pl_shared}
        cd = {'params': [p for p in pl_shared if p.requires_grad]}
        cd.update(settings_shared)
        par_list.append(cd)
        for current_par, key in zip(pl_shared, par_to_name_shared):
            par_names[key] = {'name': par_to_name_shared[key], 'is_shared': True}
            names_to_par[par_to_name_shared[key]] = current_par

        return par_list, par_names, names_to_par

    def _write_out_shared_parameters(self, model_pars, filename):

        # just write out the ones that are shared
        for group in model_pars:
            if 'params' in group:
                was_shared_group = False  # there can only be one
                # create lists that will hold the information for the different batches
                cur_pars = []

                # now iterate through the current parameter list
                for p in group['params']:
                    needs_to_be_saved = True
                    if 'is_shared' in p:
                        if not p['is_shared']:
                            needs_to_be_saved = False

                    if needs_to_be_saved:
                        # we found a shared entry
                        was_shared_group = True
                        cur_pars.append(p)

                # now we have the parameter list for one of the elements of the batch and we can write it out
                if was_shared_group:  # otherwise will be overwritten by a later parameter group
                    torch.save(cur_pars, filename)


    def _write_out_individual_parameters(self, model_pars, filenames):

        batch_size = len(filenames)

        # just write out the ones that are individual
        for group in model_pars:
            if 'params' in group:
                was_individual_group = False  # there can only be one
                # create lists that will hold the information for the different batches
                for b in range(batch_size):
                    cur_pars = []

                    # now iterate through the current parameter list
                    for p in group['params']:
                        if 'is_shared' in p:
                            # we found an individual entry
                            if not p['is_shared']:
                                was_individual_group = True
                                # now go through this dictionary, extract the current batch info in it,
                                # and append it to the current batch parameter list
                                cur_dict = dict()
                                for p_el in p:
                                    if p_el == 'name':
                                        cur_dict['name'] = p[p_el]
                                    elif p_el == 'is_shared':
                                        cur_dict['is_shared'] = p[p_el]
                                    else:
                                        # this will be a tensor so we need to extract the information for the current batch
                                        cur_dict[p_el] = p[p_el][b, ...]

                                cur_pars.append(cur_dict)

                    # now we have the parameter list for one of the elements of the batch and we can write it out
                    if was_individual_group:  # otherwise will be overwritten by a later parameter group
                        torch.save(cur_pars, filenames[b])

    def _get_optimizer_instance(self):

        if (self.model is None) or (self.criterion is None):
            raise ValueError('Please specify a model to solve with set_model first')

        # first check if an optimizer was specified externally

        if self.optimizer is not None:
            # simply instantiate it
            if self.optimizer_name is not None:
                print('Warning: optimizer name = ' + str(self.optimizer_name) +
                      ' specified, but ignored since optimizer was set explicitly')
            opt_instance = self.optimizer(self.model.parameters(), **self.optimizer_params)
            return opt_instance
        else:
            # select it by name
            # TODO: Check what the best way to adapt the tolerances is here; tying it to rel_ftol is not really correct
            if self.optimizer_name is None:
                raise ValueError('Need to select an optimizer')
            elif self.optimizer_name == 'lbfgs_ls':
                if self.last_successful_step_size_taken is not None:
                    desired_lr = self.last_successful_step_size_taken
                else:
                    desired_lr = 1.0
                max_iter = self.params['optimizer']['lbfgs'][('max_iter',1,'maximum number of iterations')]
                max_eval = self.params['optimizer']['lbfgs'][('max_eval',5,'maximum number of evaluation')]
                history_size = self.params['optimizer']['lbfgs'][('history_size',5,'Size of the optimizer history')]
                line_search_fn = self.params['optimizer']['lbfgs'][('line_search_fn','backtracking','Type of line search function')]

                opt_instance = CO.LBFGS_LS(self.model.parameters(),
                                           lr=desired_lr, max_iter=max_iter, max_eval=max_eval,
                                           tolerance_grad=self.rel_ftol * 10, tolerance_change=self.rel_ftol,
                                           history_size=history_size, line_search_fn=line_search_fn)
                return opt_instance
            elif self.optimizer_name == 'sgd':
                #if self.last_successful_step_size_taken is not None:
                #    desired_lr = self.last_successful_step_size_taken
                #else:

                if self.default_learning_rate is not None:
                    current_default_learning_rate = self.default_learning_rate
                    self.params['optimizer']['sgd']['individual']['lr'] = current_default_learning_rate
                    self.params['optimizer']['sgd']['shared']['lr'] = current_default_learning_rate

                else:
                    current_default_learning_rate = 0.01

                desired_lr_individual = self.params['optimizer']['sgd']['individual'][('lr',current_default_learning_rate,'desired learning rate')]
                sgd_momentum_individual = self.params['optimizer']['sgd']['individual'][('momentum',0.9,'sgd momentum')]
                sgd_dampening_individual = self.params['optimizer']['sgd']['individual'][('dampening',0.0,'sgd dampening')]
                sgd_weight_decay_individual = self.params['optimizer']['sgd']['individual'][('weight_decay',0.0,'sgd weight decay')]
                sgd_nesterov_individual = self.params['optimizer']['sgd']['individual'][('nesterov',True,'use Nesterove scheme')]

                desired_lr_shared = self.params['optimizer']['sgd']['shared'][('lr', current_default_learning_rate, 'desired learning rate')]
                sgd_momentum_shared = self.params['optimizer']['sgd']['shared'][('momentum', 0.9, 'sgd momentum')]
                sgd_dampening_shared = self.params['optimizer']['sgd']['shared'][('dampening', 0.0, 'sgd dampening')]
                sgd_weight_decay_shared = self.params['optimizer']['sgd']['shared'][('weight_decay', 0.0, 'sgd weight decay')]
                sgd_nesterov_shared = self.params['optimizer']['sgd']['shared'][('nesterov', True, 'use Nesterove scheme')]

                settings_shared = {'momentum': sgd_momentum_shared,
                                   'dampening': sgd_dampening_shared,
                                   'weight_decay': sgd_weight_decay_shared,
                                   'nesterov': sgd_nesterov_shared,
                                   'lr': desired_lr_shared}

                settings_individual = {'momentum': sgd_momentum_individual,
                                   'dampening': sgd_dampening_individual,
                                   'weight_decay': sgd_weight_decay_individual,
                                   'nesterov': sgd_nesterov_individual,
                                   'lr': desired_lr_individual}

                self._sgd_par_list, self._sgd_par_names, self._sgd_name_to_model_par = self._create_optimizer_parameter_dictionary(
                    self.model.get_individual_registration_parameters(),
                    self.model.get_shared_registration_parameters(),
                    settings_individual=settings_individual,
                    settings_shared=settings_shared)

                opt_instance = torch.optim.SGD(self._sgd_par_list)

                return opt_instance
            elif self.optimizer_name == 'adam':
                if self.last_successful_step_size_taken is not None:
                    desired_lr = self.last_successful_step_size_taken
                else:
                    if self.default_learning_rate is not None:
                        current_default_learning_rate = self.default_learning_rate
                        self.params['optimizer']['adam']['lr'] = current_default_learning_rate
                    else:
                        current_default_learning_rate = 0.01
                    desired_lr = self.params['optimizer']['adam'][('lr',current_default_learning_rate,'desired learning rate')]

                adam_betas = self.params['optimizer']['adam'][('betas',[0.9,0.999],'adam betas')]
                adam_eps = self.params['optimizer']['adam'][('eps',self.rel_ftol,'adam eps')]
                adam_weight_decay = self.params['optimizer']['adam'][('weight_decay',0.0,'adam weight decay')]
                opt_instance = torch.optim.Adam(self.model.parameters(), lr=desired_lr,
                                                betas=adam_betas,
                                                eps=adam_eps,
                                                weight_decay=adam_weight_decay)
                return opt_instance
            else:
                raise ValueError('Optimizer = ' + str(self.optimizer_name) + ' not yet supported')

    def _set_all_still_missing_parameters(self):

        if self.optimizer_name is None:
            self.optimizer_name = self.params['optimizer'][('name','lbfgs_ls','Optimizer (lbfgs|adam|sgd)')]

        if self.model is None:
            model_name = self.params['model']['registration_model'][('type', 'lddmm_shooting_map', "['svf'|'svf_quasi_momentum'|'svf_scalar_momentum'|'svf_vector_momentum'|'lddmm_shooting'|'lddmm_shooting_scalar_momentum'] all with suffix '_map' or '_image'")]
            self.params['model']['deformation'][('use_map', True, 'use a map for the solution or not True/False' )]
            self.set_model( model_name )

        if self.nrOfIterations is None: # not externally set, so this will not be a multi-scale solution
            self.params['optimizer'][('single_scale', {}, 'single scale settings')]
            self.nrOfIterations = self.params['optimizer']['single_scale'][('nr_of_iterations', 10, 'number of iterations')]

        # get the optimizer
        if self.optimizer_instance is None:
            self.optimizer_instance = self._get_optimizer_instance()

        if USE_CUDA:
            self.model = self.model.cuda()

        self.compute_low_res_image_if_needed()
        self.optimizer_has_been_initialized = True

    def set_scheduler_patience(self,patience):
        self.params['optimizer']['scheduler']['patience'] = patience
        self.scheduler_patience = patience

    def set_scheduler_patience_silent(self,patience):
        self.scheduler_patience = patience

    def get_scheduler_patience(self):
        return self.scheduler_patience

    def _set_use_external_scheduler(self):
        self._use_external_scheduler = True

    def _set_use_internal_scheduler(self):
        self._use_external_scheduler = False

    def _get_use_external_scheduler(self):
        return self._use_external_scheduler

    def _get_dictionary_to_pass_to_integrator(self):
        """
        This is experimental to allow passing additional parameters to integrators/smoothers, etc.

        :return: dictionary
        """

        d = dict()

        if self.mapLowResFactor is not None:
            d['I0'] = self.lowResISource
            d['I1'] = self.lowResITarget
        else:
            d['I0'] = self.ISource
            d['I1'] = self.ITarget

        return d

    def optimize(self):
        """
        Do the single scale optimization
        """

        self._set_all_still_missing_parameters()

        # in this way model parameters can be "set" before the optimizer has been properly initialized
        if self.delayed_model_parameters_still_to_be_set:
            print('Setting model parameters, delayed')
            self.set_model_parameters(self.delayed_model_parameters)

        if self.delayed_model_state_dict_still_to_be_set:
            print('Setting model state dict, delayed')
            self.set_model_state_dict(self.delayed_model_state_dict)

        # this allows passing addtional parameters to the smoothers for all models and smoothers
        self.model.set_dictionary_to_pass_to_integrator(self._get_dictionary_to_pass_to_integrator())
        self.criterion.set_dictionary_to_pass_to_smoother(self._get_dictionary_to_pass_to_integrator())

        # optimize for a few steps
        start = time.time()

        self.last_energy = None
        could_not_find_successful_step = False

        if not self._use_external_scheduler:
            self.use_step_size_scheduler = self.params['optimizer'][('use_step_size_scheduler',True,'If set to True the step sizes are reduced if no progress is made')]

            if self.use_step_size_scheduler:
                self.params['optimizer'][('scheduler', {}, 'parameters for the ReduceLROnPlateau scheduler')]
                self.scheduler_verbose = self.params['optimizer']['scheduler'][
                    ('verbose', True, 'if True prints out changes in learning rate')]
                self.scheduler_factor = self.params['optimizer']['scheduler'][('factor', 0.5, 'reduction factor')]
                self.scheduler_patience = self.params['optimizer']['scheduler'][
                    ('patience', 10, 'how many steps without reduction before LR is changed')]

            if self.use_step_size_scheduler and self.scheduler is None:
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_instance, 'min',
                                                                            verbose=self.scheduler_verbose,
                                                                            factor=self.scheduler_factor,
                                                                            patience=self.scheduler_patience)

        self.iter_count = 0
        for iter in range(self.nrOfIterations):

            # take a step of the optimizer
            # for p in self.optimizer_instance._params:
            #     p.data = p.data.float()

            current_loss = self.optimizer_instance.step(self._closure)

            # do weight clipping if it is desired
            self._do_weight_clipping()

            # an external scheduler may for example be used in batch optimization
            if not self._use_external_scheduler:
                if self.use_step_size_scheduler:
                    self.scheduler.step(current_loss.data[0])

            if hasattr(self.optimizer_instance,'last_step_size_taken'):
                self.last_successful_step_size_taken = self.optimizer_instance.last_step_size_taken()

            if self.last_successful_step_size_taken==0.0:
                print('Optimizer was not able to find a successful step. Stopping iterations.')
                could_not_find_successful_step = True
                if iter==0:
                    print('The gradient was likely too large or the optimization started from an optimal point.')
                    print('If this behavior is unexpected try adjusting the settings of the similiarity measure or allow the optimizer to try out smaller steps.')

                # to make sure warped images and the map is correct, call closure once more
                self._closure()

            if self.useMap:
                vis_arg = self.rec_phiWarped
            else:
                vis_arg = self.rec_IWarped

            tolerance_reached, was_visualized = self.analysis(self.rec_energy, self.rec_similarityEnergy,
                                                              self.rec_regEnergy, self.rec_opt_par_loss_energy,
                                                              vis_arg,
                                                              self.rec_custom_optimizer_output_string,
                                                              self.rec_custom_optimizer_output_values)

            if tolerance_reached or could_not_find_successful_step:
                if tolerance_reached:
                    print('Terminating optimization, because the desired tolerance was reached.')

                # force the output of the last image in this case, if it has not been visualized previously
                if not was_visualized and (self.visualize or self.save_fig):
                    _, _ = self.analysis(self.rec_energy, self.rec_similarityEnergy,
                                              self.rec_regEnergy, self.rec_opt_par_loss_energy,
                                              vis_arg,
                                              self.rec_custom_optimizer_output_string,
                                              self.rec_custom_optimizer_output_values,
                                              force_visualization=True)
                break

            self.iter_count = iter+1

        if self.show_iteration_output:
            cprint('-->Elapsed time {:.5f}[s]'.format(time.time() - start),  'green')


class SingleScaleBatchRegistrationOptimizer(ImageRegistrationOptimizer):

    def __init__(self, sz, spacing, useMap, mapLowResFactor, params, compute_inverse_map=False, default_learning_rate=None):

        super(SingleScaleBatchRegistrationOptimizer, self).__init__(sz, spacing, useMap, mapLowResFactor, params, compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)

        self.params[('optimizer', {}, 'optimizer settings')]
        cparams = self.params['optimizer']
        cparams[('batch_settings', {}, 'settings for the batch or optimizer')]
        cparams = cparams['batch_settings']

        self.batch_size = cparams[('batch_size',2,'how many images per batch (if set larger or equal to the number of images, it will be processed as one batch')]
        """how many images per batch"""

        self.shuffle = cparams[('shuffle', True, 'if batches should be shuffled between epochs')]
        """shuffle batches between epochshow many images per batch"""

        self.num_workers = cparams[('num_workers',0,'Number of workers to read the data. Set it to zero on the GPU or use >0 at your own risk.')]
        """number of workers to read the data"""

        self.nr_of_epochs = cparams[('nr_of_epochs', 1,'how many epochs')]
        """how many iterations for batch; i.e., how often to iterate over the entire dataset = epochs"""

        self.parameter_output_dir = cparams[('parameter_output_dir','parameters','output directory to store the shared and the individual parameters during the iterations')]
        """output directory to store the shared and the individual parameters during the iterations"""

        self.individual_parameter_output_dir = os.path.join(self.parameter_output_dir,'individual')
        self.shared_parameter_output_dir = os.path.join(self.parameter_output_dir,'shared')

        self.start_from_previously_saved_parameters = cparams[('start_from_previously_saved_parameters',True,'If set to true checks already for the first batch of files in the output directories exist and uses them if they do.')]
        """If true then checks if previously saved parameter files exists and load them at the beginning already"""

        self.individual_checkpoint_output_directory = os.path.join(self.individual_parameter_output_dir,'checkpoints')
        self.shared_checkpoint_output_directory = os.path.join(self.shared_parameter_output_dir,'checkpoints')

        self.checkpoint_interval = cparams[('checkpoint_interval',0,'after how many epochs, checkpoints are saved; if set to 0, checkpoint will not be saved')]
        """after how many epochs checkpoints are saved"""

        self.verbose_output = cparams[('verbose_output',False,'turns on verbose output')]

        self.show_sample_optimizer_output = cparams[('show_sample_optimizer_output',False,'If true shows the energies during optimizaton of a sample')]
        """Shows iterations for each sample being optimized"""

        self.also_eliminate_shared_state_between_samples_during_first_epoch = \
            self.params['optimizer']['sgd'][('also_eliminate_shared_state_between_samples_during_first_epoch', False,
                                             'if set to true all states are eliminated, otherwise only the individual ones')]

        self.use_step_size_scheduler = self.params['optimizer'][('use_step_size_scheduler', True, 'If set to True the step sizes are reduced if no progress is made')]
        self.scheduler = None

        if self.use_step_size_scheduler:
            self.params['optimizer'][('scheduler', {}, 'parameters for the ReduceLROnPlateau scheduler')]
            self.scheduler_verbose = self.params['optimizer']['scheduler'][
                ('verbose', True, 'if True prints out changes in learning rate')]
            self.scheduler_factor = self.params['optimizer']['scheduler'][('factor', 0.75, 'reduction factor')]
            self.scheduler_patience = self.params['optimizer']['scheduler'][
                ('patience', 5, 'how many steps without reduction before LR is changed')]

        self.model_name = None
        self.add_model_name = None
        self.add_model_networkClass = None
        self.add_model_lossClass = None
        self.addSimName = None
        self.addSimMeasure = None

        self.ssOpt = None

    def write_parameters_to_settings(self):
        if self.ssOpt is not None:
            self.ssOpt.write_parameters_to_settings()

    def add_similarity_measure(self, simName, simMeasure):
        """
        Adds a custom similarity measure

        :param simName: name of the similarity measure (string)
        :param simMeasure: the similarity measure itself (an object that can be instantiated)
        """
        self.addSimName = simName
        self.addSimMeasure = simMeasure


    def set_model(self, modelName):
        """
        Sets the model that should be solved

        :param modelName: name of the model that should be solved (string)
        """

        self.model_name = modelName

    def add_model(self, add_model_name, add_model_networkClass, add_model_lossClass):
        """
        Adds a custom model to be optimized over

        :param add_model_name: name of the model (string)
        :param add_model_networkClass: network model itself (as an object that can be instantiated)
        :param add_model_lossClass: loss of the model (as an object that can be instantiated)
        """
        self.add_model_name = add_model_name
        self.add_model_networkClass = add_model_networkClass
        self.add_model_lossClass = add_model_lossClass


    def get_checkpoint_dict(self):
        d = super(SingleScaleBatchRegistrationOptimizer, self).get_checkpoint_dict()
        if self.ssOpt is not None:
            d['shared_parameters'] = self.ssOpt.get_shared_model_parameters_and_buffers()
        return d

    def load_checkpoint_dict(self, d, load_optimizer_state=False):
        super(SingleScaleBatchRegistrationOptimizer, self).load_checkpoint_dict(d)
        if 'shared_parameters' in d:
            if self.ssOpt is not None:
                self.ssOpt.set_shared_model_parameters(d['shared_parameters'])
        else:
            raise ValueError('checkpoint does not contain: consensus_dual')

    def get_warped_image(self):
        """
        Returns the warped image
        :return: the warped image
        """

        p = dict()
        p['warped_images'] = []

        print('get_warped_image: not yet implemented')

        return p

    def get_map(self):
        """
        Returns the deformation map
        :return: deformation map
        """

        p = dict()
        p['phi'] = []

        print('get_map: not yet implemented')

        return p

    def get_inverse_map(self):
        """
        Returns the inverse deformation map
        :return: deformation map
        """

        p = dict()
        p['phi_inv'] = []

        print('get_inverse_map: not yet implemented')

        return p


    def get_model_parameters(self):
        """
        Returns the parameters of the model

        :return: model parameters
        """
        p = dict()
        if self.ssOpt is not None:
            p['shared_parameters'] = self.ssOpt.get_shared_model_parameters_and_buffers()

        return p

    def set_model_parameters(self, p):
        raise ValueError('Setting model parameters not yet supported by batch optimizer')

    def _set_all_still_missing_parameters(self):
        if self.model_name is None:
            model_name = self.params['model']['registration_model'][('type', 'lddmm_shooting_map',
                                                                     "['svf'|'svf_quasi_momentum'|'svf_scalar_momentum'|'svf_vector_momentum'|'lddmm_shooting'|'lddmm_shooting_scalar_momentum'] all with suffix '_map' or '_image'")]
            self.params['model']['deformation'][('use_map', True, 'use a map for the solution or not True/False')]
            self.set_model(model_name)

        if self.optimizer_name is None:
            self.optimizer_name = self.params['optimizer'][('name', 'sgd', 'Optimizer (lbfgs|adam|sgd)')]

        self.optimizer_has_been_initialized = True

    def _create_single_scale_optimizer(self,batch_size):
        ssOpt = SingleScaleRegistrationOptimizer(batch_size, self.spacing, self.useMap, self.mapLowResFactor, self.params, compute_inverse_map=self.compute_inverse_map, default_learning_rate=self.default_learning_rate)

        if ((self.add_model_name is not None) and
                (self.add_model_networkClass is not None) and
                (self.add_model_lossClass is not None)):
            ssOpt.add_model(self.add_model_name, self.add_model_networkClass, self.add_model_lossClass)

        # now set the actual model we want to solve
        ssOpt.set_model(self.model_name)

        if (self.addSimName is not None) and (self.addSimMeasure is not None):
            ssOpt.add_similarity_measure(self.addSimName, self.addSimMeasure)

        if self.optimizer_name is not None:
            ssOpt.set_optimizer_by_name(self.optimizer_name)
        else:
            raise ValueError('Optimizers need to be specified by name of consensus optimization at the moment.')

        ssOpt.set_rel_ftol(self.get_rel_ftol())

        ssOpt.set_visualization(self.get_visualization())
        ssOpt.set_visualize_step(self.get_visualize_step())

        return ssOpt

    def _get_individual_checkpoint_filenames(self,output_directory,idx,epoch_iter):
        filenames = []
        for v in idx:
            filenames.append(os.path.join(output_directory,'checkpoint_individual_parameter_pair_{:05d}_epoch_{:05d}.pt'.format(v,epoch_iter)))
        return filenames

    def _get_shared_checkpoint_filename(self,output_directory,epoch_iter):

        filename = os.path.join(output_directory,'checkpoint_shared_parameters_epoch_{:05d}.pt'.format(epoch_iter))
        return filename

    def _create_all_output_directories(self):

        if not os.path.exists(self.parameter_output_dir):
            os.makedirs(self.parameter_output_dir)
            print('Creating directory: ' + self.parameter_output_dir)

        if not os.path.exists(self.individual_parameter_output_dir):
            os.makedirs(self.individual_parameter_output_dir)
            print('Creating directory: ' + self.individual_parameter_output_dir)

        if not os.path.exists(self.shared_parameter_output_dir):
            os.makedirs(self.shared_parameter_output_dir)
            print('Creating directory: ' + self.shared_parameter_output_dir)

        if not os.path.exists(self.individual_checkpoint_output_directory):
            os.makedirs(self.individual_checkpoint_output_directory)
            print('Creating directory: ' + self.individual_checkpoint_output_directory)

        if not os.path.exists(self.shared_checkpoint_output_directory):
            os.makedirs(self.shared_checkpoint_output_directory)
            print('Creating directory: ' + self.shared_checkpoint_output_directory)


    def _get_shared_parameter_filename(self,output_dir):
        return os.path.join(output_dir,'shared_parameters.pt')

    def optimize(self):
        """
        The optimizer to optimize over batches of images

        :return: n/a
        """

        #todo: maybe switch loading and writing individual parameters to individual states; this would assure that all states (such as running averages, etc.) are included and not only parameters

        if self.optimizer is not None:
            raise ValueError('Custom optimizers are currently not supported for batch optimization.\
                                  Set the optimizer by name (e.g., in the json configuration) instead. Should be some form of stochastic gradient descent.')


        self._set_all_still_missing_parameters()
        self._create_all_output_directories()

        iter_offset = 0

        if torch.is_tensor(self.ISource) or torch.is_tensor(self.ITarget):
            raise ValueError('Batch optimizer expects lists of filenames as inputs for the source and target images')

        registration_data_set = OD.PairwiseRegistrationDataset(output_directory=self.individual_parameter_output_dir,
                                                               source_image_filenames=self.ISource,
                                                               target_image_filenames=self.ITarget,
                                                               params=self.params)

        nr_of_datasets = len(registration_data_set)
        if nr_of_datasets<self.batch_size:
            print('INFO: nr of datasets is smaller than batch-size. Reducing batch size to ' + str(nr_of_datasets))
            self.batch_size=nr_of_datasets

        if nr_of_datasets%self.batch_size!=0:
            raise ValueError('nr_of_datasets = {}; batch_size = {}: Number of registration pairs needs to be divisible by the batch size.'.format(nr_of_datasets,self.batch_size))

        dataloader = DataLoader(registration_data_set, batch_size=self.batch_size,
                                shuffle=self.shuffle, num_workers=self.num_workers)

        self.ssOpt = None
        last_batch_size = None

        nr_of_samples = nr_of_datasets//self.batch_size

        last_energy = None
        last_sim_energy = None
        last_reg_energy = None
        last_opt_energy = None

        shared_parameter_filename = self._get_shared_parameter_filename(self.shared_parameter_output_dir)

        load_individual_parameters_during_first_epoch = False
        load_shared_parameters_before_first_epoch = False

        if self.start_from_previously_saved_parameters:
            # check if there are files in the output_directory
            has_all_filenames = True
            for idx in range(len(self.ISource)):
                cur_filename = registration_data_set._get_parameter_filename(idx)
                if not os.path.isfile(cur_filename):
                    has_all_filenames = False
                    break

            load_individual_parameters_during_first_epoch =  has_all_filenames
            load_shared_parameters_before_first_epoch = os.path.isfile(shared_parameter_filename)

            if load_individual_parameters_during_first_epoch:
                print('INFO: Will load the individual parameters from the previous run in directory ' + self.individual_parameter_output_dir + ' for initialization.')
            else:
                print('INFO: Will NOT load the individual parameters from the previous run in directory ' + self.individual_parameter_output_dir + ' for initialization.')

            if load_shared_parameters_before_first_epoch:
                print('INFO: Will load the shared parameter file ' + shared_parameter_filename + ' before computing the first epoch')
            else:
                print('INFO: Will NOT load the shared parameter file ' + shared_parameter_filename + ' before computing the first epoch')

        for iter_epoch in range(iter_offset,self.nr_of_epochs+iter_offset):
            if self.verbose_output:
                print('Computing epoch ' + str(iter_epoch + 1) + ' of ' + str(iter_offset+self.nr_of_epochs))

            cur_running_energy = 0.0
            cur_running_sim_energy = 0.0
            cur_running_reg_energy = 0.0
            cur_running_opt_energy = 0.0

            cur_min_energy = None
            cur_max_energy = None
            cur_min_sim_energy = None
            cur_max_sim_energy = None
            cur_min_reg_energy = None
            cur_max_reg_energy = None
            cur_min_opt_energy = None
            cur_max_opt_energy = None

            for i, sample in enumerate(dataloader, 0):

                # get the data from the dataloader
                current_source_batch = AdaptVal(sample['ISource'])
                current_target_batch = AdaptVal(sample['ITarget'])

                # create the optimizer
                batch_size = current_source_batch.size()
                if (batch_size != last_batch_size) and (last_batch_size is not None):
                    raise ValueError('Ooops, this should not have happened.')

                initialize_optimizer = False
                if (batch_size != last_batch_size) or (self.ssOpt is None):
                    initialize_optimizer = True
                    # we need to create a new optimizer; otherwise optimizer already exists
                    self.ssOpt = self._create_single_scale_optimizer(batch_size)

                # images need to be set before calling _set_all_still_missing_parameters
                self.ssOpt.set_source_image(current_source_batch)
                self.ssOpt.set_target_image(current_target_batch)
                self.ssOpt.set_current_epoch(iter_epoch)

                if initialize_optimizer:
                    # to make sure we have the model initialized, force parameter installation
                    self.ssOpt._set_all_still_missing_parameters()
                    # since this is chunked-up we increase the patience
                    self.ssOpt._set_use_external_scheduler()

                    if self.show_sample_optimizer_output:
                        self.ssOpt.turn_iteration_output_on()
                    else:
                        self.ssOpt.turn_iteration_output_off()

                    if self.use_step_size_scheduler and self.scheduler is None:
                        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.ssOpt.optimizer_instance, 'min',
                                                                                    verbose=self.scheduler_verbose,
                                                                                    factor=self.scheduler_factor,
                                                                                    patience=self.scheduler_patience)

                    if load_shared_parameters_before_first_epoch:
                        print('Loading the shared parameters/state.')
                        self.ssOpt.load_shared_state_dict(torch.load(shared_parameter_filename))

                last_batch_size = batch_size

                if iter_epoch!=0 or load_individual_parameters_during_first_epoch: # only load the individual parameters after the first epoch
                    if 'individual_parameter' in sample:
                        current_individual_parameters = sample['individual_parameter']
                        if current_individual_parameters is not None:
                            if self.verbose_output:
                                print('INFO: loading current individual optimizer state')
                            self.ssOpt.set_sgd_individual_model_parameters_and_optimizer_states(current_individual_parameters)
                    else:
                        print('WARNING: could not find previous parameter file')
                else:
                    # this is the case when optimization is run for the first time for a batch or if previous results should not be used
                    # In this case we want to have a fresh start for the initial conditions
                    par_file = os.path.join(self.individual_parameter_output_dir,'default_init.pt')
                    if i==0:
                        # this is the first time, so we store the individual parameters
                        torch.save(self.ssOpt.get_individual_model_parameters(),par_file)
                    else:
                        # now we load them
                        if self.verbose_output:
                            print('INFO: forcing the initial individual parameters to default')
                        self.ssOpt.set_individual_model_parameters(torch.load(par_file))
                        # and we need to kill the optimizer state (to get rid of the previous momentum)
                        if self.also_eliminate_shared_state_between_samples_during_first_epoch:
                            if self.verbose_output:
                                print('INFO: discarding the entire optimizer state')
                            self.ssOpt.optimizer_instance.state = defaultdict(dict)
                        else:
                            if self.verbose_output:
                                print('INFO: discarding current *individual* optimizer states only')
                            self.ssOpt._remove_state_variables_for_individual_parameters(self.ssOpt.get_sgd_individual_model_parameters_and_optimizer_states())


                if self.visualize:
                    if i == 0:
                        # to avoid excessive graphical output
                        self.ssOpt.turn_visualization_on()
                    else:
                        self.ssOpt.turn_visualization_off()
                else:
                    self.ssOpt.turn_visualization_off()

                self.ssOpt.optimize()

                cur_energy,cur_sim_energy,cur_reg_energy = self.ssOpt.get_energy()
                cur_opt_energy = self.ssOpt.get_opt_par_energy()

                cur_running_energy += 1./nr_of_samples*cur_energy
                cur_running_sim_energy += 1./nr_of_samples*cur_sim_energy
                cur_running_reg_energy += 1./nr_of_samples*cur_reg_energy
                cur_running_opt_energy += 1./nr_of_samples*cur_opt_energy

                if i==0:
                    cur_min_energy = cur_energy
                    cur_max_energy = cur_energy
                    cur_min_sim_energy = cur_sim_energy
                    cur_max_sim_energy = cur_sim_energy
                    cur_min_reg_energy = cur_reg_energy
                    cur_max_reg_energy = cur_reg_energy
                    cur_min_opt_energy = cur_opt_energy
                    cur_max_opt_energy = cur_opt_energy
                else:
                    cur_min_energy = min(cur_energy,cur_min_energy)
                    cur_max_energy = max(cur_energy,cur_max_energy)
                    cur_min_sim_energy = min(cur_sim_energy,cur_min_sim_energy)
                    cur_max_sim_energy = max(cur_sim_energy,cur_max_sim_energy)
                    cur_min_reg_energy = min(cur_reg_energy,cur_min_reg_energy)
                    cur_max_reg_energy = max(cur_reg_energy,cur_max_reg_energy)
                    cur_min_opt_energy = min(cur_opt_energy,cur_min_opt_energy)
                    cur_max_opt_energy = max(cur_opt_energy,cur_max_opt_energy)

                # need to save this index by index so we can shuffle
                self.ssOpt._write_out_individual_parameters(self.ssOpt.get_sgd_individual_model_parameters_and_optimizer_states(),sample['individual_parameter_filename'])

                if self.checkpoint_interval>0:
                    if (iter_epoch%self.checkpoint_interval==0) or (iter_epoch==self.nr_of_epochs+iter_offset-1):
                        if self.verbose_output:
                            print('Writing out individual checkpoint data for epoch ' + str(iter_epoch) + ' for sample ' + str(i+1) + '/' + str(nr_of_samples))
                        individual_filenames = self._get_individual_checkpoint_filenames(self.individual_checkpoint_output_directory,sample['idx'],iter_epoch)
                        self.ssOpt._write_out_individual_parameters(self.ssOpt.get_sgd_individual_model_parameters_and_optimizer_states(),individual_filenames)

                        if i==nr_of_samples-1:
                            if self.verbose_output:
                                print('Writing out shared checkpoint data for epoch ' + str(iter_epoch))
                            shared_filename = self._get_shared_checkpoint_filename(self.shared_checkpoint_output_directory,iter_epoch)
                            self.ssOpt._write_out_shared_parameters(self.ssOpt.get_sgd_shared_model_parameters(),shared_filename)

            if self.show_sample_optimizer_output:
                if (last_energy is not None) and (last_sim_energy is not None) and (last_reg_energy is not None):
                    print('\n\nEpoch {:05d}: Last energies   : E=[{:2.5f}], simE=[{:2.5f}], regE=[{:2.5f}], optE=[{:2.5f}]'\
                          .format(iter_epoch-1,last_energy,last_sim_energy,last_reg_energy,last_opt_energy))
                    print('    / image: Last energies   : E=[{:2.5f}], simE=[{:2.5f}], regE=[{:2.5f}]' \
                        .format(last_energy/batch_size[0], last_sim_energy/batch_size[0], last_reg_energy/batch_size[0]))
                else:
                    print('\n\n')

            last_energy = cur_running_energy
            last_sim_energy = cur_running_sim_energy
            last_reg_energy = cur_running_reg_energy
            last_opt_energy = cur_running_opt_energy

            if self.show_sample_optimizer_output:
                print('Epoch {:05d}: Current energies: E=[{:2.5f}], simE=[{:2.5f}], regE=[{:2.5f}], optE=[{:2.5f}]'\
                  .format(iter_epoch,last_energy, last_sim_energy,last_reg_energy,last_opt_energy))
                print('    / image: Current energies: E=[{:2.5f}], simE=[{:2.5f}], regE=[{:2.5f}]' \
                      .format(last_energy/batch_size[0], last_sim_energy/batch_size[0], last_reg_energy/batch_size[0]))
            else:
                print('Epoch {:05d}: Current energies: E={:2.5f}:[{:1.2f},{:1.2f}], simE={:2.5f}:[{:1.2f},{:1.2f}], regE={:2.5f}:[{:1.2f},{:1.2f}], optE={:1.2f}:[{:1.2f},{:1.2f}]'\
                      .format(iter_epoch, last_energy, cur_min_energy, cur_max_energy,
                              last_sim_energy, cur_min_sim_energy, cur_max_sim_energy,
                              last_reg_energy, cur_min_reg_energy, cur_max_reg_energy,
                              last_opt_energy, cur_min_opt_energy, cur_max_opt_energy))
                print('    / image: Current energies: E={:2.5f}:[{:1.2f},{:1.2f}], simE={:2.5f}:[{:1.2f},{:1.2f}], regE={:2.5f}:[{:1.2f},{:1.2f}]' \
                    .format(last_energy/batch_size[0], cur_min_energy/batch_size[0], cur_max_energy/batch_size[0],
                            last_sim_energy/batch_size[0], cur_min_sim_energy/batch_size[0], cur_max_sim_energy/batch_size[0],
                            last_reg_energy/batch_size[0], cur_min_reg_energy/batch_size[0], cur_max_reg_energy/batch_size[0]))

            if self.show_sample_optimizer_output:
                print('\n\n')

            if self.use_step_size_scheduler:
                self.scheduler.step(last_energy)

        print('Writing out shared parameter/state file to ' + shared_parameter_filename )
        torch.save(self.ssOpt.shared_state_dict(),shared_parameter_filename)


class SingleScaleConsensusRegistrationOptimizer(ImageRegistrationOptimizer):

    def __init__(self, sz, spacing, useMap, mapLowResFactor, params, compute_inverse_map=False, default_learning_rate=None):

        super(SingleScaleConsensusRegistrationOptimizer, self).__init__(sz, spacing, useMap, mapLowResFactor, params, compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)

        self.params[('optimizer', {}, 'optimizer settings')]
        cparams = self.params['optimizer']
        cparams[('consensus_settings', {}, 'settings for the consensus optimizer')]
        cparams = cparams['consensus_settings']

        self.sigma = cparams[('sigma', 1.0, 'sigma/2 is multiplier for squared augmented Lagrangian penalty')]
        """Multiplier for squared augmented Lagrangian penalty"""

        self.nr_of_epochs = cparams[('nr_of_epochs', 1, 'how many iterations for consensus; i.e., how often to iterate over the entire dataset')]
        """how many iterations for consensus; i.e., how often to iterate over the entire dataset"""
        self.batch_size = cparams[('batch_size',1,'how many images per batch (if set larger or equal to the number of images, it will be processed as one batch')]
        """how many images per batch"""
        self.save_intermediate_checkpoints = cparams[('save_intermediate_checkpoints',False,'when set to True checkpoints are retained for each batch iterations')]
        """when set to True checkpoints are retained for each batch iterations"""

        self.checkpoint_output_directory = cparams[('checkpoint_output_directory','checkpoints','directory where the checkpoints will be stored')]
        """output directory where the checkpoints will be saved"""

        self.save_consensus_state_checkpoints = cparams[('save_consensus_state_checkpoints',True,'saves the current consensus state; typically only the individual states are saved as checkpoints')]
        """saves the current consensus state; typically only the individual states are saved as checkpoints"""

        self.continue_from_last_checkpoint = cparams[('continue_from_last_checkpoint',False,'If true then iterations are resumed from last checkpoint. Allows restarting an optimization')]
        """allows restarting an optimization by continuing from the last checkpoint"""

        self.load_optimizer_state_from_checkpoint = cparams[('load_optimizer_state_from_checkpoint',True,'If set to False only the state of the model is loaded when resuming from a checkpoint')]
        """If set to False only the state of the model is loaded when resuming from a checkpoint"""

        self.nr_of_batches = None
        self.nr_of_images = None

        self.current_consensus_state = None
        self.current_consensus_dual = None
        self.next_consensus_state = None
        self.last_shared_state = None

        self.model_name = None
        self.add_model_name = None
        self.add_model_networkClass = None
        self.add_model_lossClass = None
        self.addSimName = None
        self.addSimMeasure = None

        self.iter_offset = None

        self.ssOpt = None

    def write_parameters_to_settings(self):
        if self.ssOpt is not None:
            self.ssOpt.write_parameters_to_settings()

    def _consensus_penalty_loss(self,shared_model_parameters):
        """
        This allows to define additional terms for the loss which are based on parameters that are shared
        between models (for example for the smoother). Can be used to define a form of consensus optimization.
        :param shared_model_parameters: parameters that have been declared shared in a model
        :return: 0 by default, otherwise the corresponding penalty
        """
        additional_loss = MyTensor(1).zero_()
        total_number_of_parameters = 1
        for k in shared_model_parameters:
            total_number_of_parameters += shared_model_parameters[k].numel()
            additional_loss += ((shared_model_parameters[k]\
                               -self.current_consensus_state[k]\
                               -self.current_consensus_dual[k])**2).sum()


        additional_loss *= self.sigma/(2.0*total_number_of_parameters)

        #print('sigma=' + str(self.sigma) + '; additional loss = ' + str( additional_loss.data.cpu().numpy()))

        return additional_loss

    def _set_state_to_zero(self,state):
        # set all the individual parameters to zero
        for k in state:
            state[k].zero_()

    def _add_scaled_difference_to_state(self,state,model_shared_state,current_dual,scaling_factor):
        for k in state:
            state[k] += scaling_factor*(model_shared_state[k]-current_dual[k])

    def _create_single_scale_optimizer(self,batch_size,consensus_penalty):

        ssOpt = SingleScaleRegistrationOptimizer(batch_size, self.spacing, self.useMap, self.mapLowResFactor, self.params, compute_inverse_map=self.compute_inverse_map, default_learning_rate=self.default_learning_rate)

        if ((self.add_model_name is not None) and
                (self.add_model_networkClass is not None) and
                (self.add_model_lossClass is not None)):
            ssOpt.add_model(self.add_model_name, self.add_model_networkClass, self.add_model_lossClass)

        # now set the actual model we want to solve
        ssOpt.set_model(self.model_name)

        if (self.addSimName is not None) and (self.addSimMeasure is not None):
            ssOpt.add_similarity_measure(self.addSimName, self.addSimMeasure)

        # setting the optimizer
        #if self.optimizer is not None:
        #    ssOpt.set_optimizer(self.optimizer)
        #    ssOpt.set_optimizer_params(self.optimizer_params)
        #elif self.optimizer_name is not None:
        if self.optimizer_name is not None:
            ssOpt.set_optimizer_by_name(self.optimizer_name)
        else:
            raise ValueError('Optimizers need to be specified by name of consensus optimization at the moment.')

        ssOpt.set_rel_ftol(self.get_rel_ftol())

        ssOpt.set_visualization(self.get_visualization())
        ssOpt.set_visualize_step(self.get_visualize_step())

        if consensus_penalty:
            ssOpt.set_external_optimizer_parameter_loss(self._consensus_penalty_loss)

        return ssOpt

    def _initialize_consensus_variables_if_needed(self,ssOpt):
        if self.current_consensus_state is None:
            self.current_consensus_state = copy.deepcopy(ssOpt.get_shared_model_parameters())
            self._set_state_to_zero(self.current_consensus_state)

        if self.current_consensus_dual is None:
            self.current_consensus_dual = copy.deepcopy(self.current_consensus_state)
            self._set_state_to_zero(self.current_consensus_dual)

        if self.last_shared_state is None:
            self.last_shared_state = copy.deepcopy(self.current_consensus_state)
            self._set_state_to_zero(self.last_shared_state)

        if self.next_consensus_state is None:
            self.next_consensus_state = copy.deepcopy(self.current_consensus_dual)  # also make it zero
            self._set_state_to_zero(self.next_consensus_state)

    def add_similarity_measure(self, simName, simMeasure):
        """
        Adds a custom similarity measure

        :param simName: name of the similarity measure (string)
        :param simMeasure: the similarity measure itself (an object that can be instantiated)
        """
        self.addSimName = simName
        self.addSimMeasure = simMeasure


    def set_model(self, modelName):
        """
        Sets the model that should be solved

        :param modelName: name of the model that should be solved (string)
        """

        self.model_name = modelName

    def add_model(self, add_model_name, add_model_networkClass, add_model_lossClass):
        """
        Adds a custom model to be optimized over

        :param add_model_name: name of the model (string)
        :param add_model_networkClass: network model itself (as an object that can be instantiated)
        :param add_model_lossClass: loss of the model (as an object that can be instantiated)
        """
        self.add_model_name = add_model_name
        self.add_model_networkClass = add_model_networkClass
        self.add_model_lossClass = add_model_lossClass

    def get_checkpoint_dict(self):
        d = super(SingleScaleConsensusRegistrationOptimizer, self).get_checkpoint_dict()
        d['consensus_dual'] = self.current_consensus_dual
        return d

    def load_checkpoint_dict(self, d, load_optimizer_state=False):
        super(SingleScaleConsensusRegistrationOptimizer, self).load_checkpoint_dict(d)
        if 'consensus_dual' in d:
            self.current_consensus_dual = d['consensus_dual']
        else:
            raise ValueError('checkpoint does not contain: consensus_dual')

    def _custom_load_checkpoint(self,ssOpt,filename):
        d = torch.load(filename)
        ssOpt.load_checkpoint_dict(d)
        self.load_checkpoint_dict(d)

    def _custom_single_batch_load_checkpoint(self,ssOpt,filename):
        d = torch.load(filename)
        if self.load_optimizer_state_from_checkpoint:
            ssOpt.load_checkpoint_dict(d,load_optimizer_state=True)

    def _custom_save_checkpoint(self,ssOpt,filename):
        sd = ssOpt.get_checkpoint_dict()

        # todo: maybe make this optional to save storage
        sd['res'] = dict()
        sd['res']['Iw'] = ssOpt.get_warped_image()
        sd['res']['phi'] = ssOpt.get_map()

        cd = self.get_checkpoint_dict()
        # now merge these two dictionaries
        sd.update(cd)
        # and now save it
        torch.save(sd,filename)

    def _copy_state(self,state_to,state_from):

        for key in state_to:
            if key in state_from:
                state_to[key].copy_(state_from[key])
            else:
                raise ValueError('Could not copy key ' + key)


    def _set_all_still_missing_parameters(self):

        if self.model_name is None:
            model_name = self.params['model']['registration_model'][('type', 'lddmm_shooting_map', "['svf'|'svf_quasi_momentum'|'svf_scalar_momentum'|'svf_vector_momentum'|'lddmm_shooting'|'lddmm_shooting_scalar_momentum'] all with suffix '_map' or '_image'")]
            self.params['model']['deformation'][('use_map', True, 'use a map for the solution or not True/False' )]
            self.set_model( model_name )

        if self.optimizer_name is None:
            self.optimizer_name = self.params['optimizer'][('name','lbfgs_ls','Optimizer (lbfgs|adam|sgd)')]

        self.optimizer_has_been_initialized = True

    def get_warped_image(self):
        """
        Returns the warped image
        :return: the warped image
        """

        p = dict()
        p['warped_images'] = []
        for current_batch in range(self.nr_of_batches):
            current_checkpoint_filename = self._get_checkpoint_filename(current_batch, self.iter_offset+self.nr_of_epochs - 1)
            dc = torch.load(current_checkpoint_filename)
            p['warped_images'].append(dc['res']['Iw'])

        return p


    def get_map(self):
        """
        Returns the deformation map
        :return: deformation map
        """

        p = dict()
        p['phi'] = []
        for current_batch in range(self.nr_of_batches):
            current_checkpoint_filename = self._get_checkpoint_filename(current_batch, self.iter_offset+self.nr_of_epochs - 1)
            dc = torch.load(current_checkpoint_filename)
            p['phi'].append(dc['res']['phi'])

        return p

    def get_model_parameters(self):
        """
        Returns the parameters of the model

        :return: model parameters
        """
        p = dict()
        p['consensus_state'] = self.current_consensus_state
        p['registration_pars'] = []
        for current_batch in range(self.nr_of_batches):
            current_checkpoint_filename = self._get_checkpoint_filename(current_batch,self.iter_offset+self.nr_of_epochs-1)
            dc = torch.load(current_checkpoint_filename)
            d = dict()
            d['model'] = dc['model']
            d['consensus_dual'] = dc['consensus_dual']
            p['registration_pars'].append(d)

        return p

    def set_model_parameters(self, p):
        raise ValueError('Setting model parameters not yet supported by consensus optimizer')

    def _get_checkpoint_filename(self,batch_nr,batch_iter):
        if self.save_intermediate_checkpoints:
            return os.path.join(self.checkpoint_output_directory,
                                "checkpoint_batch{:05d}_iter{:05d}.pt".format(batch_nr,batch_iter))
        else:
            return os.path.join(self.checkpoint_output_directory,
                                "checkpoint_batch{:05d}.pt".format(batch_nr))

    def _get_consensus_checkpoint_filename(self,batch_iter):
        return os.path.join(self.checkpoint_output_directory,
                            "consensus_state_iter{:05d}.pt".format(batch_iter))

    def _optimize_as_single_batch(self,resume_from_iter=None):
        """
        Does optimization where everything is represented as a single batch. This is essentially like an individual
        optimization, but supports checkpointing.

        :param resume_from_iter: resumes computations from this iteration (assumes the corresponding checkpoint exists here)
        :return: n/a
        """

        if resume_from_iter is not None:
            self.iter_offset = resume_from_iter+1
            print('Resuming from checkpoint iteration: ' + str(resume_from_iter))
        else:
            self.iter_offset = 0

        for iter_batch in range(self.iter_offset,self.nr_of_epochs+self.iter_offset):
            print('Computing epoch ' + str(iter_batch + 1) + ' of ' + str(self.iter_offset+self.nr_of_epochs))

            all_histories = []
            current_batch = 0 # there is only one batch, this one

            current_source_batch = self.ISource[:, ...].data
            current_target_batch = self.ITarget[:, ...].data
            current_batch_image_size = np.array(current_source_batch.size())

            # there is not consensus penalty here as this is technically not consensus optimization
            # todo: could ultimately replace the single scale optimizer; here used to write out checkpoints
            self.ssOpt = self._create_single_scale_optimizer(current_batch_image_size, consensus_penalty=False)

            # needs to be set before calling _set_all_still_missing_parameters
            self.ssOpt.set_source_image(current_source_batch)
            self.ssOpt.set_target_image(current_target_batch)

            # to make sure we have the model initialized, force parameter installation
            self.ssOpt._set_all_still_missing_parameters()

            # this loads the optimizer state and the model state, but here not the self.current_consensus_dual
            if iter_batch>0:
                previous_checkpoint_filename = self._get_checkpoint_filename(current_batch, iter_batch - 1)
                self._custom_single_batch_load_checkpoint(self.ssOpt, previous_checkpoint_filename)

            self.ssOpt.optimize()

            if (current_batch == self.nr_of_batches - 1) and (iter_batch == self.nr_of_epochs - 1):
                # the last time we run this
                all_histories.append(self.ssOpt.get_history())

            current_checkpoint_filename = self._get_checkpoint_filename(current_batch, iter_batch)
            self._custom_save_checkpoint(self.ssOpt, current_checkpoint_filename)

            self._add_to_history('batch_history', copy.deepcopy(all_histories))


    def _optimize_with_multiple_batches(self, resume_from_iter=None):
        """
        Does consensus optimization over multiple batches.

        :param resume_from_iter: resumes computations from this iteration (assumes the corresponding checkpoint exists here)
        :return: n/a
        """

        if resume_from_iter is not None:
            iter_offset = resume_from_iter+1
            print('Resuming from checkpoint iteration: ' + str(resume_from_iter))
        else:
            iter_offset = 0

        for iter_batch in range(iter_offset,self.nr_of_epochs+iter_offset):
            print('Computing epoch ' + str(iter_batch+1) + ' of ' + str(iter_offset+self.nr_of_epochs))

            next_consensus_initialized = False
            all_histories = []

            for current_batch in range(self.nr_of_batches):

                from_image = current_batch*self.batch_size
                to_image = min(self.nr_of_images,(current_batch+1)*self.batch_size)

                nr_of_images_in_batch = to_image-from_image

                current_source_batch = self.ISource[from_image:to_image, ...].data
                current_target_batch = self.ITarget[from_image:to_image, ...].data
                current_batch_image_size = np.array(current_source_batch.size())

                print('Computing image pair batch ' + str(current_batch+1) + ' of ' + str(self.nr_of_batches) +
                      ' of batch iteration ' + str(iter_batch+1) + ' of ' + str(iter_offset+self.nr_of_epochs))
                print('Image range: [' + str(from_image) + ',' + str(to_image) + ')')

                # create new optimizer
                if iter_batch==0:
                    # do not apply the penalty the first time around
                    self.ssOpt = self._create_single_scale_optimizer(current_batch_image_size,consensus_penalty=False)
                else:
                    self.ssOpt = self._create_single_scale_optimizer(current_batch_image_size,consensus_penalty=True)

                # to make sure we have the model initialized, force parameter installation
                self.ssOpt._set_all_still_missing_parameters()

                if iter_batch==0:
                    # in the first round just initialize the shared state with what was computed previously
                    if self.last_shared_state is not None:
                        self.ssOpt.set_shared_model_parameters(self.last_shared_state)

                self._initialize_consensus_variables_if_needed(self.ssOpt)

                if not next_consensus_initialized:
                    self._set_state_to_zero(self.next_consensus_state)
                    next_consensus_initialized = True

                if iter_batch==0:
                    # for the first time, just set the dual to zero
                    self._set_state_to_zero(self.current_consensus_dual)
                    # load the last
                else:
                    # this loads the optimizer state and the model state and also self.current_consensus_dual
                    previous_checkpoint_filename = self._get_checkpoint_filename(current_batch, iter_batch-1)
                    self._custom_load_checkpoint(self.ssOpt,previous_checkpoint_filename)

                    # first update the dual variable (we do this now that we have the consensus state still
                    self._add_scaled_difference_to_state(self.current_consensus_dual,
                                                         self.ssOpt.get_shared_model_parameters(),
                                                         self.current_consensus_state,-1.0)


                self.ssOpt.set_source_image(current_source_batch)
                self.ssOpt.set_target_image(current_target_batch)

                self.ssOpt.optimize()

                self._copy_state(self.last_shared_state,self.ssOpt.get_shared_model_parameters())

                if (current_batch==self.nr_of_batches-1) and (iter_batch==self.nr_of_epochs-1):
                    # the last time we run this
                    all_histories.append( self.ssOpt.get_history() )

                # update the consensus state (is done via next_consensus_state as
                # self.current_consensus_state is used as part of the optimization for all optimizations in the batch
                self._add_scaled_difference_to_state(self.next_consensus_state,
                                                     self.ssOpt.get_shared_model_parameters(),
                                                     self.current_consensus_dual,float(nr_of_images_in_batch)/float(self.nr_of_images))

                current_checkpoint_filename = self._get_checkpoint_filename(current_batch, iter_batch)
                self._custom_save_checkpoint(self.ssOpt,current_checkpoint_filename)

            self._add_to_history('batch_history', copy.deepcopy(all_histories))
            self._copy_state(self.current_consensus_state, self.next_consensus_state)

            if self.save_consensus_state_checkpoints:
                consensus_filename = self._get_consensus_checkpoint_filename(iter_batch)
                torch.save({'consensus_state':self.current_consensus_state},consensus_filename)


    def _get_checkpoint_iter_with_complete_batch(self,start_at_iter):

        if start_at_iter<0:
            print('Could NOT find a complete checkpoint batch.')
            return None

        is_complete_batch = True
        for current_batch in range(self.nr_of_batches):
            cfilename = self._get_checkpoint_filename(current_batch, start_at_iter)
            if os.path.isfile(cfilename):
                print('Checkpoint file: ' + cfilename + " exists.")
            else:
                print('Checkpoint file: ' + cfilename + " does NOT exist.")
                is_complete_batch = False
                break

        if is_complete_batch:
            print('Found complete batch for batch iteration ' + str(start_at_iter))
            return start_at_iter
        else:
            return self._get_checkpoint_iter_with_complete_batch(start_at_iter-1)


    def _get_last_checkpoint_iteration_from_checkpoint_files(self):
        """
        Looks through the checkpoint files and checks which ones were the last saved ones.
        This allows for picking up the iterations after a completed or terminated optimization.
        Also checks that the same number of batches are used, otherwise an optimization cannot be resumed
        from a checkpoint.

        :return: last iteration performed for complete batch
        """

        print('Attempting to resume optimization from checkpoint data.')
        print('Searching for existing checkpoint data ...')

        # first find all the computed iters
        largest_found_iter = None

        if self.save_intermediate_checkpoints:
            current_iter_batch = 0
            while os.path.isfile(self._get_checkpoint_filename(0,current_iter_batch)):
                print('Found checkpoint iteration: ' + str(current_iter_batch) + ' : ' + self._get_checkpoint_filename(0,current_iter_batch))
                largest_found_iter = current_iter_batch
                current_iter_batch +=1

        else:
            if os.path.isfile(self._get_checkpoint_filename(0,0)):
                print('Found checkpoint: ' + str(self._get_checkpoint_filename(0,0)))
                largest_found_iter = 0
                              
        if largest_found_iter is None:
            print('Could not find any checkpoint data from which to resume.')
            return None
        else:
            largest_iter_with_complete_batch = self._get_checkpoint_iter_with_complete_batch(largest_found_iter)
            return largest_iter_with_complete_batch

    def optimize(self):

        """
        This optimizer performs consensus optimization:

        1) (u_i_shared,u_i_individual)^{k+1} = argmin \sum_i f_i(u_i_shared,u_i_individual) + \sigma/2\|u_i_shared-u_consensus^k-z_i^k\|^2
        2) (u_consensus)^{k+1} = 1/n\sum_{i=1}^n ((u_i_shared)^{k+1}-z_i^k)
        3) z_i^{k+1} = z_i^k-((u_i_shared)^{k+1}-u_consensus_{k+1})

        :return: n/a
        """

        if self.optimizer is not None:
            raise ValueError('Custom optimizers are currently not supported for consensus optimization.\
                           Set the optimizer by name (e.g., in the json configuration) instead.')

        self._set_all_still_missing_parameters()

        # todo: support reading images from file
        self.nr_of_images = self.ISource.size()[0]
        self.nr_of_batches = np.ceil(float(self.nr_of_images)/float(self.batch_size)).astype('int')

        if self.continue_from_last_checkpoint:
            last_checkpoint_iteration = self._get_last_checkpoint_iteration_from_checkpoint_files()
        else:
            last_checkpoint_iteration = None

        if self.nr_of_batches==1:
            compute_as_single_batch = True
        else:
            compute_as_single_batch = False

        if not os.path.exists(self.checkpoint_output_directory):
            os.makedirs(self.checkpoint_output_directory)

        if compute_as_single_batch:
            self._optimize_as_single_batch(resume_from_iter=last_checkpoint_iteration)
        else:
            self._optimize_with_multiple_batches(resume_from_iter=last_checkpoint_iteration)


class MultiScaleRegistrationOptimizer(ImageRegistrationOptimizer):
    """
    Class to perform multi-scale optimization. Essentially puts a loop around multiple calls of the
    single scale optimizer and starts with the registration of downsampled images. When moving up
    the hierarchy, the registration parameters are upsampled from the solution at the previous lower resolution
    """

    def __init__(self, sz, spacing, useMap, mapLowResFactor, params, compute_inverse_map=False, default_learning_rate=None ):
        super(MultiScaleRegistrationOptimizer, self).__init__(sz, spacing, useMap, mapLowResFactor, params, compute_inverse_map=compute_inverse_map, default_learning_rate=default_learning_rate)
        self.scaleFactors = None
        """At what image scales optimization should be computed"""
        self.scaleIterations = None
        """number of iterations per scale"""

        self.addSimName = None
        """name of the similarity measure to be added"""
        self.addSimMeasure = None
        """similarity measure itself that should be added"""
        self.add_model_name = None
        """name of the model that should be added"""
        self.add_model_networkClass = None
        """network object of the model to be added"""
        self.add_model_lossClass = None
        """loss object of the model to be added"""
        self.model_name = None
        """name of the model to be added (if specified by name; gets dominated by specifying an optimizer directly"""
        self.ssOpt = None
        """Single scale optimizer"""
        self.params['optimizer'][('multi_scale', {}, 'multi scale settings')]

    def write_parameters_to_settings(self):
        if self.ssOpt is not None:
            self.ssOpt.write_parameters_to_settings()

    def add_similarity_measure(self, simName, simMeasure):
        """
        Adds a custom similarity measure

        :param simName: name of the similarity measure (string)
        :param simMeasure: the similarity measure itself (an object that can be instantiated)
        """
        self.addSimName = simName
        self.addSimMeasure = simMeasure

    def set_model(self, modelName):
        """
        Set the model to be optimized over by name

        :param modelName: the name of the model (string)
        """
        self.model_name = modelName

    def set_initial_map(self, map0, map0_inverse=None):
        """
        Sets the initial map (overwrites the default identity map)
        :param map0: intial map
        :return: n/a
        """
        if self.ssOpt is None:
            self.initialMap = map0
            self.initialInverseMap = map0_inverse

    def set_initial_weight_map(self,weight_map,freeze_weight=False):
        if self.ssOpt is None:
            self.weight_map = weight_map
            self.freeze_weight = freeze_weight

    def set_pair_name(self,pair_name):
        # f = lambda name: os.path.split(name)
        # get_in = lambda x: os.path.splitext(f(x)[1])[0]
        # get_fn = lambda x: f(f(x)[0])[1]
        # get_img_name = lambda x: get_fn(x)+'_'+get_in(x)
        # img_pair_name = [get_img_name(pair_name[0])+'_'+get_img_name(pair_name[1]) for pair_name in pair_names]
        self.pair_name = pair_name

    def set_save_fig_path(self, save_fig_path):
        """
        the path of saved figures, default is the ../data/expr_name
        :param save_fig_path:
        :return:
        """
        self.save_fig_path = os.path.join(save_fig_path, self.expr_name)



    def add_model(self, add_model_name, add_model_networkClass, add_model_lossClass, use_map):
        """
        Adds a custom model to be optimized over

        :param add_model_name: name of the model (string)
        :param add_model_networkClass: network model itself (as an object that can be instantiated)
        :param add_model_lossClass: loss of the model (as an object that can be instantiated)
        :param use_map: if set to true, model using a map, otherwise direcly works with the image
        """
        self.add_model_name = add_model_name
        self.add_model_networkClass = add_model_networkClass
        self.add_model_lossClass = add_model_lossClass
        self.add_model_use_map = use_map

    def set_scale_factors(self, scaleFactors):
        """
        Set the scale factors for the solution. Should be in decending order, e.g., [1.0, 0.5, 0.25]

        :param scaleFactors: scale factors for the multi-scale solution hierarchy
        """

        self.params['optimizer']['multi_scale']['scale_factors'] = (scaleFactors, 'how images are scaled')
        self.scaleFactors = scaleFactors

    def set_number_of_iterations_per_scale(self, scaleIterations):
        """
        Sets the number of iterations that will be performed per scale of the multi-resolution hierarchy. E.g, [50,100,200]

        :param scaleIterations: number of iterations per scale (array)
        """

        self.params['optimizer']['multi_scale']['scale_iterations'] = (scaleIterations, 'number of iterations per scale')
        self.scaleIterations = scaleIterations

    def _get_desired_size_from_scale(self, origSz, scale):

        osz = np.array(list(origSz))
        dsz = osz
        dsz[2::] = (np.round( scale*osz[2::] )).astype('int')

        return dsz

    def get_energy(self):
        """
        Returns the current energy
        :return: Returns a tuple (energy, similarity energy, regularization energy)
        """
        if self.ssOpt is not None:
            return self.ssOpt.get_energy()
        else:
            return None

    def get_warped_image(self):
        """
        Returns the warped image
        :return: the warped image
        """
        if self.ssOpt is not None:
            return self.ssOpt.get_warped_image()
        else:
            return None


    def get_warped_label(self):
        """
        Returns the warped label
        :return: the warped label
        """
        if self.ssOpt is not None:
            return self.ssOpt.get_warped_label()
        else:
            return None

    def get_map(self):
        """
        Returns the deformation map
        :return: deformation map
        """
        if self.ssOpt is not None:
            return self.ssOpt.get_map()
        else:
            return None

    def get_inverse_map(self):
        """
        Returns the inverse deformation map
        :return: deformation map
        """
        if self.ssOpt is not None:
            return self.ssOpt.get_inverse_map()
        else:
            return None

    def get_model_parameters(self):
        """
        Returns the parameters of the model

        :return: model parameters
        """
        if self.ssOpt is not None:
            return self.ssOpt.get_model_parameters()
        else:
            return None

    def set_model_parameters(self,p):
        raise ValueError('Setting model parameters not yet supported for multi-scale optimizer')

    def _set_all_still_missing_parameters(self):

        self.scaleFactors = self.params['optimizer']['multi_scale'][('scale_factors', [1.0, 0.5, 0.25], 'how images are scaled')]
        self.scaleIterations = self.params['optimizer']['multi_scale'][('scale_iterations', [10, 20, 20], 'number of iterations per scale')]

        if (self.optimizer is None) and (self.optimizer_name is None):
            self.optimizer_name = self.params['optimizer'][('name','lbfgs_ls','Optimizer (lbfgs|adam|sgd)')]

        if self.model_name is None:
            model_name = self.params['model']['registration_model'][('type', 'lddmm_shooting_map', "['svf'|'svf_quasi_momentum'|'svf_scalar_momentum'|'svf_vector_momentum'|'lddmm_shooting'|'lddmm_shooting_scalar_momentum'] all with suffix '_map' or '_image'")]
            self.params['model']['deformation'][('use_map', True, 'use a map for the solution or not True/False' )]
            self.set_model( model_name )

        self.optimizer_has_been_initialized = True

    def optimize(self):
        """
        Perform the actual multi-scale optimization
        """
        self._set_all_still_missing_parameters()

        if (self.ISource is None) or (self.ITarget is None):
            raise ValueError('Source and target images need to be set first')

        upsampledParameters = None
        upsampledParameterSpacing = None
        upsampledSz = None
        lastSuccessfulStepSizeTaken = None

        nrOfScales = len(self.scaleFactors)

        # check that we have the right number of iteration parameters
        assert (nrOfScales == len(self.scaleIterations))

        print('Performing multiscale optmization with scales: ' + str(self.scaleFactors))

        # go from lowest to highest scale
        reverseScales = self.scaleFactors[-1::-1]
        reverseIterations = self.scaleIterations[-1::-1]
        over_scale_iter_count = 0

        for en_scale in enumerate(reverseScales):
            print('Optimizing for scale = ' + str(en_scale[1]))

            # create the images
            currentScaleFactor = en_scale[1]
            currentScaleNumber = en_scale[0]

            currentDesiredSz = self._get_desired_size_from_scale(self.ISource.size(), currentScaleFactor)

            currentNrOfIteratons = reverseIterations[currentScaleNumber]

            ISourceC, spacingC = self.sampler.downsample_image_to_size(self.ISource, self.spacing, currentDesiredSz[2::],self.spline_order)
            ITargetC, spacingC = self.sampler.downsample_image_to_size(self.ITarget, self.spacing, currentDesiredSz[2::],self.spline_order)
            LSourceC = None
            LTargetC = None
            if self.LSource is not None and self.LTarget is not None:
                LSourceC, spacingC = self.sampler.downsample_image_to_size(self.LSource, self.spacing, currentDesiredSz[2::],0)
                LTargetC, spacingC = self.sampler.downsample_image_to_size(self.LTarget, self.spacing, currentDesiredSz[2::],0)
            initialMap = None
            initialInverseMap = None
            weight_map=None
            if self.initialMap is not None:
                initialMap,_ = self.sampler.downsample_image_to_size(self.initialMap,self.spacing, currentDesiredSz[2::],1,zero_boundary=False)
            if self.initialInverseMap is not None:
                initialInverseMap,_ = self.sampler.downsample_image_to_size(self.initialInverseMap,self.spacing, currentDesiredSz[2::],1,zero_boundary=False)
            if self.weight_map is not None:
                weight_map,_ =self.sampler.downsample_image_to_size(self.weight_map,self.spacing, currentDesiredSz[2::],1,zero_boundary=False)
            szC = np.array(ISourceC.size())  # this assumes the BxCxXxYxZ format
            mapLowResFactor = None if currentScaleNumber==0 else self.mapLowResFactor
            self.ssOpt = SingleScaleRegistrationOptimizer(szC, spacingC, self.useMap, mapLowResFactor, self.params, compute_inverse_map=self.compute_inverse_map,default_learning_rate=self.default_learning_rate)
            print('Setting learning rate to ' + str( lastSuccessfulStepSizeTaken ))
            self.ssOpt.set_last_successful_step_size_taken( lastSuccessfulStepSizeTaken )
            self.ssOpt.set_initial_map(initialMap,initialInverseMap)

            if ((self.add_model_name is not None) and
                    (self.add_model_networkClass is not None) and
                    (self.add_model_lossClass is not None)):
                self.ssOpt.add_model(self.add_model_name, self.add_model_networkClass, self.add_model_lossClass, use_map=self.add_model_use_map)

            # now set the actual model we want to solve
            self.ssOpt.set_model(self.model_name)
            if weight_map is not None:
                self.ssOpt.set_initial_weight_map(weight_map,self.freeze_weight)


            if (self.addSimName is not None) and (self.addSimMeasure is not None):
                self.ssOpt.add_similarity_measure(self.addSimName, self.addSimMeasure)

            # setting the optimizer
            if self.optimizer is not None:
                self.ssOpt.set_optimizer(self.optimizer)
                self.ssOpt.set_optimizer_params(self.optimizer_params)
            elif self.optimizer_name is not None:
                self.ssOpt.set_optimizer_by_name(self.optimizer_name)

            self.ssOpt.set_rel_ftol(self.get_rel_ftol())

            self.ssOpt.set_visualization(self.get_visualization())
            self.ssOpt.set_visualize_step(self.get_visualize_step())
            self.ssOpt.set_n_scale(en_scale[1])
            self.ssOpt.set_over_scale_iter_count(over_scale_iter_count)

            if self.get_save_fig():
                self.ssOpt.set_expr_name(self.get_expr_name())
                self.ssOpt.set_save_fig(self.get_save_fig())
                self.ssOpt.set_save_fig_path(self.get_save_fig_path())
                self.ssOpt.set_save_fig_num(self.get_save_fig_num())
                self.ssOpt.set_pair_name(self.get_pair_name())
                self.ssOpt.set_n_scale(en_scale[1])
                self.ssOpt.set_source_label(self.get_source_label())
                self.ssOpt.set_target_label(self.get_target_label())


            self.ssOpt.set_source_image(ISourceC)
            self.ssOpt.set_target_image(ITargetC)
            self.ssOpt.set_multi_scale_info(self.ISource,self.ITarget,self.spacing,self.LSource,self.LTarget)
            if self.LSource is not None and self.LTarget is not None:
                self.ssOpt.set_source_label(LSourceC)
                self.ssOpt.set_target_label(LTargetC)

            if upsampledParameters is not None:
                # check that the upsampled parameters are consistent with the downsampled images
                spacingError = False
                expectedSpacing = None

                if mapLowResFactor is not None:
                    expectedSpacing = utils._get_low_res_spacing_from_spacing(spacingC, szC, upsampledSz)
                    # the spacing of the upsampled parameters will be different
                    if not (abs(expectedSpacing - upsampledParameterSpacing) < 0.000001).all():
                        spacingError = True
                elif not (abs(spacingC - upsampledParameterSpacing) < 0.000001).all():
                    expectedSpacing = spacingC
                    spacingError = True

                if spacingError:
                    print(expectedSpacing)
                    print(upsampledParameterSpacing)
                    raise ValueError('Upsampled parameters and downsampled images are of inconsistent dimension')

                # now that everything is fine, we can use the upsampled parameters
                print('Explicitly setting the optimization parameters')
                self.ssOpt.set_model_parameters(upsampledParameters)

            # do the actual optimization
            print('Optimizing for at most ' + str(currentNrOfIteratons) + ' iterations')
            self.ssOpt._set_number_of_iterations_from_multi_scale(currentNrOfIteratons)
            self.ssOpt.optimize()

            self._add_to_history('scale_nr',currentScaleNumber)
            self._add_to_history('scale_factor',currentScaleFactor)
            self._add_to_history('ss_history',self.ssOpt.get_history())

            lastSuccessfulStepSizeTaken = self.ssOpt.get_last_successful_step_size_taken()
            over_scale_iter_count += currentNrOfIteratons

            # if we are not at the very last scale, then upsample the parameters
            if currentScaleNumber != nrOfScales - 1:
                # we need to revert the downsampling to the next higher level
                scaleTo = reverseScales[currentScaleNumber + 1]
                upsampledSz = self._get_desired_size_from_scale(self.ISource.size(), scaleTo)
                print('Before')
                print(upsampledSz)
                if self.useMap:
                    if self.mapLowResFactor is not None:
                        # parameters are upsampled differently here, because they are computed at low res
                        upsampledSz = utils._get_low_res_size_from_size(upsampledSz,self.mapLowResFactor)
                        print(self.mapLowResFactor)
                        print('After')
                        print(upsampledSz)
                upsampledParameters, upsampledParameterSpacing = self.ssOpt.upsample_model_parameters(upsampledSz[2::])
