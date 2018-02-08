"""
This package enables easy single-scale and multi-scale optimization support.
"""

from abc import ABCMeta, abstractmethod
import os
import time
import copy
import utils
import visualize_registration_results as vizReg
import custom_optimizers as CO
import numpy as np
import torch
from torch.autograd import Variable
from data_wrapper import USE_CUDA, AdaptVal, MyTensor
import model_factory as MF
import image_sampling as IS
from metrics import get_multi_metric
from res_recorder import XlsxRecorder

# add some convenience functionality
class SimpleRegistration(object):
    """
           Abstract optimizer base class.
    """
    __metaclass__ = ABCMeta

    def __init__(self,ISource,ITarget,spacing,params):
        """
        :param ISource: source image
        :param ITarget: target image
        :param spacing: image spacing
        :param params: parameters
        """
        self.params = params
        self.use_map = self.params['model']['deformation'][('use_map', True, '[True|False] either do computations via a map or directly using the image')]
        self.map_low_res_factor = self.params['model']['deformation'][('map_low_res_factor', 1.0, 'Set to a value in (0,1) if a map-based solution should be computed at a lower internal resolution (image matching is still at full resolution')]
        self.spacing = spacing
        self.ISource = ISource
        self.ITarget = ITarget
        self.sz = np.array( ISource.size() )
        self.optimizer = None
        self.light_analysis_on = None

    def get_history(self):
        """
        Returns the optimization history as a dictionary. Keeps track of energies, iterations counts, and additonal custom measures.

        :return: history dictionary
        """

        if self.optimizer is not None:
            return self.optimizer.get_history()
        else:
            return None

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

    def get_warped_image(self):
        """
        Returns the warped image
        :return: the warped image
        """
        if self.optimizer is not None:
            return self.optimizer.get_warped_image()
        else:
            return None

    def get_map(self):
        """
        Returns the deformation map
        :return: deformation map
        """
        if self.optimizer is not None:
            return self.optimizer.get_map()

    def get_model_parameters(self):
        """
        Returns the parameters of the model

        :return: model parameters 
        """
        return self.optimizer.get_model_parameters()

    def set_light_analysis_on(self, light_analysis_on):
        self.light_analysis_on = light_analysis_on


class SimpleSingleScaleRegistration(SimpleRegistration):
    """
    Simple single scale registration
    """
    def __init__(self,ISource,ITarget,spacing,params):
        super(SimpleSingleScaleRegistration, self).__init__(ISource,ITarget,spacing, params)
        self.optimizer = SingleScaleRegistrationOptimizer(self.sz,self.spacing,self.use_map,self.map_low_res_factor,self.params)

    def register(self):
        """
        Registers the source to the target image
        :return: n/a
        """
        self.optimizer.set_light_analysis_on(self.light_analysis_on)
        self.optimizer.register(self.ISource,self.ITarget)


class SimpleSingleScaleConsensusRegistration(SimpleRegistration):
    """
    Single scale registration making use of consensus optimization (to allow for multiple independent registration
    that can share parameters).
    """
    def __init__(self,ISource,ITarget,spacing,params):
        super(SimpleSingleScaleConsensusRegistration, self).__init__(ISource,ITarget,spacing, params)
        self.optimizer = SingleScaleConsensusRegistrationOptimizer(self.sz,self.spacing,self.use_map,self.map_low_res_factor,self.params)

    def register(self):
        """
        Registers the source to the target image
        :return: n/a
        """
        self.optimizer.set_light_analysis_on(self.light_analysis_on)
        self.optimizer.register(self.ISource,self.ITarget)


class SimpleMultiScaleRegistration(SimpleRegistration):
    """
    Simple multi scale registration
    """
    def __init__(self,ISource,ITarget,spacing,params):
        super(SimpleMultiScaleRegistration, self).__init__(ISource, ITarget, spacing, params)
        self.optimizer = MultiScaleRegistrationOptimizer(self.sz,self.spacing,self.use_map,self.map_low_res_factor,self.params)

    def register(self):
        """
        Registers the source to the target image
        :return: n/a
        """
        self.optimizer.set_light_analysis_on(self.light_analysis_on)
        self.optimizer.register(self.ISource,self.ITarget)



class Optimizer(object):
    """
       Abstract optimizer base class.
    """
    __metaclass__ = ABCMeta

    def __init__(self, sz, spacing, useMap, mapLowResFactor, params):
        """
        Constructor.
        
        :param sz: image size in BxCxXxYxZ format
        :param spacing: spatial spacing, e.g., [0.1,0.1,0.1] in 3D
        :param useMap: boolean, True if a coordinate map is evolved to warp images, False otherwise
        :param map_low_res_factor: if <1 evolutions happen at a lower resolution; >=1 ignored 
        :param params: ParametersDict() instance to hold parameters
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
        self.params = params
        """general parameters"""
        self.rel_ftol = 1e-4
        """relative termination tolerance for optimizer"""
        self.last_successful_step_size_taken = None
        """Records the last successful step size an optimizer took (possible use: propogate step size between multiscale levels"""
        self.batch_id = -1

        self.external_optimizer_parameter_loss = None

        if (self.mapLowResFactor is not None):
            self.lowResSize = self._get_low_res_size_from_size( sz, self.mapLowResFactor )
            self.lowResSpacing = self._get_low_res_spacing_from_spacing(self.spacing,sz,self.lowResSize)
        self.sampler = IS.ResampleImage()

        self.params[('optimizer', {}, 'optimizer settings')]
        self.params[('model', {}, 'general model settings')]
        self.params['model'][('deformation', {}, 'model describing the desired deformation model')]
        self.params['model'][('registration_model', {}, 'general settings for the registration model')]

        self.params['model']['deformation']['use_map']= (useMap, '[True|False] either do computations via a map or directly using the image')
        self.params['model']['deformation']['map_low_res_factor'] = (mapLowResFactor, 'Set to a value in (0,1) if a map-based solution should be computed at a lower internal resolution (image matching is still at full resolution')

        self.rel_ftol = self.params['optimizer']['single_scale'][('rel_ftol',self.rel_ftol,'relative termination tolerance for optimizer')]

        self.history = dict()

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
        if not self.history.has_key(key):
            self.history[key]=[value]
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

    def _get_low_res_spacing_from_spacing(self, spacing, sz, lowResSize):
        """
        Computes spacing for the low-res parameterization from image spacing
        :param spacing: image spacing
        :param sz: size of image
        :param lowResSize: size of low re parameterization
        :return: returns spacing of low res parameterization
        """
        return spacing * np.array(sz[2::]) / np.array(lowResSize[2::])

    def _get_low_res_size_from_size(self, sz, factor):
        """
        Returns the corresponding low-res size from a (high-res) sz
        :param sz: size (high-res)
        :param factor: low-res factor (needs to be <1)
        :return: low res size
        """
        if (factor is None) or (factor>=1):
            print('WARNING: Could not compute low_res_size as factor was ' + str( factor ))
            return sz
        else:
            lowResSize = np.array(sz)
            lowResSize[2::] = (np.ceil((np.array(sz[2::]) * factor))).astype('int16')
            return lowResSize

    def set_rel_ftol(self, rel_ftol):
        """
        Sets the relative termination tolerance: |f(x_i)-f(x_{i-1})|/f(x_i)<tol
        
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

    def set_batch_id(self, batch_id):
        self.batch_id = batch_id

    def get_batch_id(self):
        return self.batch_id

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
        return Variable(MyTensor(1).zero_(),requires_grad=False)

class ImageRegistrationOptimizer(Optimizer):
    """
    Optimization class for image registration.
    """

    def __init__(self, sz, spacing, useMap, mapLowResFactor, params):
        super(ImageRegistrationOptimizer, self).__init__(sz, spacing, useMap, mapLowResFactor, params)
        self.ISource = None
        """source image"""
        self.lowResISource = None
        """if mapLowResFactor <1, a lowres image needs to be created to parameterize some of the registration algorithms"""
        self.ITarget = None
        """target image"""
        self.LSource = None
        """ source label """
        self.LTarget = None
        """  target label """
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
        self.save_fig_path=None
        self.save_fig=None
        self.save_fig_num =None
        self.pair_path=None
        self.iter_count = 0
        self.recorder = None
        self.save_excel = False
        self.light_analysis_on = None
        self.limit_max_batch = -1


    def set_light_analysis_on(self, light_analysis_on):
        self.light_analysis_on = light_analysis_on

    def set_limit_max_batch(self, limit_max_batch):
        self.limit_max_batch= limit_max_batch

    def get_limit_max_batch(self):
        return self.limit_max_batch


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

    def set_recorder(self, recorder):
        self.recorder = recorder

    def get_recorder(self):
        return self.recorder

    def set_save_excel(self, save_excel):
        self.save_excel = save_excel

    def get_save_excel(self):
        return self.save_excel


    def get_save_fig_path(self):
        """
        the path of saved figures, default is the ../data/expr_name
        :param save_fig_path:
        :return:
        """
        return self.save_fig_path


    def set_save_fig_num(self, save_fig_num=1):
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

    def set_pair_path(self, pair_paths):
        self.pair_path = pair_paths


    def get_pair_path(self):
        return self.pair_path


    def register(self,ISource,ITarget):
        """
        Registers the source to the target image
        :param ISource: source image
        :param ITarget: target image
        :return: n/a
        """
        self.set_source_image(ISource)
        self.set_target_image(ITarget)
        self.optimize()

    def set_source_image(self, I):
        """
        Setting the source image which should be deformed to match the target image

        :param I: source image
        """
        self.ISource = I
        if self.mapLowResFactor is not None:
            self.lowResISource,_ = self.sampler.downsample_image_to_size(self.ISource,self.spacing,self.lowResSize[2::])

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

    def __init__(self, sz, spacing, useMap, mapLowResFactor, params):
        super(SingleScaleRegistrationOptimizer, self).__init__(sz, spacing, useMap, mapLowResFactor, params)

        if (self.mapLowResFactor is not None ):
            # computes model at a lower resolution than the image similarity
            self.mf = MF.ModelFactory(self.sz, self.spacing, self.lowResSize, self.lowResSpacing )
        else:
            # computes model and similarity at the same resolution
            self.mf = MF.ModelFactory(self.sz, self.spacing, self.sz, self.spacing)
        """model factory which will be used to create the model and its loss function"""

        self.model = None
        """the model itself"""
        self.criterion = None
        """the loss function"""

        self.identityMap = None
        """identity map, will be needed for map-based solutions"""
        self.lowResIdentityMap = None
        """low res identity map, will be needed for map-based solutions which are computed at lower resolution"""
        self.optimizer_instance = None
        """the optimizer instance to perform the actual optimization"""

        self.rec_energy = None
        self.rec_similarityEnergy = None
        self.rec_regEnergy = None
        self.rec_opt_par_loss_energy = None
        self.rec_phiWarped = None
        self.rec_IWarped = None
        self.last_energy = None
        self.rel_f = None
        self.rec_custom_optimizer_output_string = ''
        """the evaluation information"""
        self.rec_custom_optimizer_output_values = None

    def get_checkpoint_dict(self):
        if self.model is not None and self.optimizer_instance is not None:
            d = super(SingleScaleRegistrationOptimizer, self).get_checkpoint_dict()
            d['model'] = dict()
            d['model']['state'] = self.model.get_registration_parameters()
            d['model']['size'] = self.model.sz
            d['model']['spacing'] = self.model.spacing
            d['optimizer_state'] = self.optimizer_instance.state_dict()
            return d
        else:
            raise ValueError('Unable to create checkpoint, because either the model or the optimizer have not been initialized')

    def load_checkpoint_dict(self,d,load_optimizer_state=False):
        if self.model is not None and self.optimizer_instance is not None:
            self.model.set_registration_parameters(d['model']['state'],d['model']['size'],d['model']['spacing'])
            if load_optimizer_state:
                self.optimizer_instance.load_state_dict(d['optimizer_state'])
            else:
                print('WARNING: Turned off the loading of the optimizer state')
        else:
            raise ValueError('Cannot load checkpoint dictionary, because either the model or the optimizer have not been initialized')

    def get_energy(self):
        """
        Returns the current energy
        :return: Returns a tuple (energy, similarity energy, regularization energy)
        """
        return self.rec_energy.cpu().data.numpy(), self.rec_similarityEnergy.cpu().data.numpy(), self.rec_regEnergy.cpu().data.numpy()

    def get_warped_image(self):
        """
        Returns the warped image
        :return: the warped image
        """
        if self.useMap:
            cmap = self.get_map()
            # and now warp it
            return utils.compute_warped_image_multiNC(self.ISource, cmap, self.spacing)
        else:
            return self.rec_IWarped

    def get_map(self):
        """
        Returns the deformation map
        :return: deformation map
        """
        return self.rec_phiWarped


    def set_n_scale(self, n_scale):
        """
        the path of saved figures, default is the ../data/expr_name
        :param save_fig_path:
        :return:
        """
        self.n_scale = n_scale

    def set_model(self, modelName):
        """
        Sets the model that should be solved

        :param modelName: name of the model that should be solved (string)
        """

        self.params['model']['registration_model']['type'] = ( modelName, "['svf'|'svf_quasi_momentum'|'svf_scalar_momentum'|'svf_vector_momentum'|'lddmm_shooting'|'lddmm_shooting_scalar_momentum'] all with '_map' or '_image' suffix" )

        self.model, self.criterion = self.mf.create_registration_model(modelName, self.params['model'])
        print(self.model)

        if self.useMap:
            # create the identity map [-1,1]^d, since we will use a map-based implementation
            id = utils.identity_map_multiN(self.sz,self.spacing)
            self.identityMap = AdaptVal(Variable(torch.from_numpy(id), requires_grad=False))
            if self.mapLowResFactor is not None:
                # create a lower resolution map for the computations
                lowres_id = utils.identity_map_multiN(self.lowResSize,self.lowResSpacing)
                self.lowResIdentityMap = AdaptVal(Variable(torch.from_numpy(lowres_id), requires_grad=False))

    def add_similarity_measure(self, simName, simMeasure):
        """
        Adds a custom similarity measure.

        :param simName: name of the similarity measure (string)
        :param simMeasure: similarity measure itself (class object that can be instantiated)
        """
        self.criterion.add_similarity_measure(simName, simMeasure)
        self.params['model']['registration_model']['similarity_measure']['type'] = (simName, 'was customized; needs to be expplicitly instantiated, cannot be loaded')

    def add_model(self, modelName, modelNetworkClass, modelLossClass, useMap, modelDescription='custom model'):
        """
        Adds a custom model and its loss function

        :param modelName: name of the model to be added (string)
        :param modelNetworkClass: registration model itself (class object that can be instantiated)
        :param modelLossClass: registration loss (class object that can be instantiated)
        :param useMap: True/False: specifies if model uses a map or not
        :param modelDescription: optional model description
        """
        self.mf.add_model(modelName, modelNetworkClass, modelLossClass, useMap, modelDescription)
        self.params['model']['registration_model']['type'] = (modelName, 'was customized; needs to be explicitly instantiated, cannot be loaded')

    def set_model_parameters(self, p):
        """
        Set the parameters of the registration model

        :param p: parameters
        """
        if (self.useMap) and (self.mapLowResFactor is not None):
            self.model.set_registration_parameters(p, self.lowResSize, self.lowResSpacing)
        else:
            self.model.set_registration_parameters(p, self.sz, self.spacing)

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

    def get_shared_model_parameters(self):
        """
        Returns only the model parameters that are shared between models.

        :return: shared model parameters
        """
        return self.model.get_shared_registration_parameters()

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
        opt_variables = {'iter': self.iter_count}

        if self.useMap:
            if self.mapLowResFactor is not None:
                rec_tmp = self.model(self.lowResIdentityMap, self.lowResISource, opt_variables )
                # now upsample to correct resolution
                desiredSz = self.identityMap.size()[2::]
                self.rec_phiWarped, _ = self.sampler.upsample_image_to_size(rec_tmp, self.spacing, desiredSz)
            else:
                self.rec_phiWarped = self.model(self.identityMap, self.ISource, opt_variables )

            loss_overall_energy,sim_energy,reg_energy = self.criterion(self.identityMap, self.rec_phiWarped, self.ISource, self.ITarget, self.lowResISource,
                                  self.model.get_variables_to_transfer_to_loss_function(),
                                  opt_variables )
        else:
            self.rec_IWarped = self.model(self.ISource, opt_variables )
            loss_overall_energy,sim_energy,reg_energy = self.criterion(self.rec_IWarped, self.ISource, self.ITarget,
                                  self.model.get_variables_to_transfer_to_loss_function(),
                                  opt_variables )

        # to support consensus optimization we have the option of adding a penalty term
        # based on shared parameters
        opt_par_loss_energy = self.compute_optimizer_parameter_loss(self.model.get_shared_registration_parameters())
        loss_overall_energy = loss_overall_energy + opt_par_loss_energy
        loss_overall_energy.backward()
        #torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.5)

        self.rec_custom_optimizer_output_string = self.model.get_custom_optimizer_output_string()
        self.rec_custom_optimizer_output_values = self.model.get_custom_optimizer_output_values()

        self.rec_energy = loss_overall_energy
        self.rec_similarityEnergy = sim_energy
        self.rec_regEnergy = reg_energy
        self.rec_opt_par_loss_energy = opt_par_loss_energy

        #if self.useMap:
        #
        #    if self.iter_count % 1 == 0:
        #        self.rec_energy, self.rec_similarityEnergy, self.rec_regEnergy = self.criterion.get_energy(
        #            self.identityMap, self.rec_phiWarped, self.ISource, self.ITarget, self.lowResISource, self.model.get_variables_to_transfer_to_loss_function())
        #else:
        #    if self.iter_count % 1 == 0:
        #        self.rec_energy, self.rec_similarityEnergy, self.rec_regEnergy = self.criterion.get_energy(
        #            self.rec_IWarped, self.ISource, self.ITarget, self.model.get_variables_to_transfer_to_loss_function())

        return loss_overall_energy

    def analysis(self, energy, similarityEnergy, regEnergy, opt_par_energy, phi_or_warped_image, custom_optimizer_output_string ='', custom_optimizer_output_values=None):
        """
        print out the and visualize the result
        :param energy:
        :param similarityEnergy:
        :param regEnergy:
        :param opt_par_energy
        :param phi_or_warped_image:
        :return: returns True if termination tolerance was reached, otherwise returns False
        """

        cur_energy = utils.t2np(energy.float())
        # energy analysis

        self._add_to_history('iter',self.iter_count)
        self._add_to_history('energy',cur_energy[0])
        self._add_to_history('similarity_energy',utils.t2np(similarityEnergy.float())[0])
        self._add_to_history('regularization_energy',utils.t2np(regEnergy.float())[0])
        self._add_to_history('opt_par_energy',utils.t2np(opt_par_energy.float())[0])

        if custom_optimizer_output_values is not None:
            for key in custom_optimizer_output_values:
                self._add_to_history(key,custom_optimizer_output_values[key])

        if self.last_energy is not None:

            # relative function toleranc: |f(xi)-f(xi+1)|/(1+|f(xi)|)
            self.rel_f = abs(self.last_energy - cur_energy) / (1 + abs(cur_energy))
            self._add_to_history('relF',self.rel_f[0])

            print('Iter {iter}: E={energy}, similarityE={similarityE}, regE={regE}, optParE={optParE}, relF={relF} {cos}'
                  .format(iter=self.iter_count,
                          energy=cur_energy,
                          similarityE=utils.t2np(similarityEnergy.float()),
                          regE=utils.t2np(regEnergy.float()),
                          optParE=utils.t2np(opt_par_energy.float()),
                          relF=self.rel_f,
                          cos=custom_optimizer_output_string))

            # check if relative convergence tolerance is reached
            if self.rel_f < self.rel_ftol:
                print('Reached relative function tolerance of = ' + str(self.rel_ftol))
                return True

        else:
            self._add_to_history('relF',None)
            print('Iter {iter}: E={energy}, similarityE={similarityE}, regE={regE}, optParE={optParE}, relF=n/a {cos}'
                  .format(iter=self.iter_count,
                          energy=cur_energy,
                          similarityE=utils.t2np(similarityEnergy.float()),
                          regE=utils.t2np(regEnergy.float()),
                          optParE=utils.t2np(opt_par_energy.float()),
                          cos=custom_optimizer_output_string))

        self.last_energy = cur_energy
        iter = self.iter_count

        # performance analysis

        if self.useMap and not self.light_analysis_on and self.save_excel:
            if self.LSource is not None:
                if iter % 4 == 0:
                    LSource_warped = utils.get_warped_label_map(self.LSource, phi_or_warped_image, self.spacing)
                    LTarget = self.get_target_label()
                    metric_results_dic = get_multi_metric(LSource_warped, LTarget, eval_label_list=None, rm_bg=False)
                    self.recorder.set_batch_based_env(self.get_pair_path(),self.get_batch_id())
                    info = {}
                    info['label_info'] = metric_results_dic['label_list']
                    info['iter_info'] = 'scale_' + str(self.n_scale) + '_iter_' + str(self.iter_count)
                    self.recorder.saving_results(sched='batch', results=metric_results_dic['multi_metric_res'],  info=info,averaged_results=metric_results_dic['batch_avg_res'])
                    self.recorder.saving_results(sched='buffer', results=metric_results_dic['label_avg_res'],  info=info,averaged_results=None)

        # result visualization
        if self.visualize or self.save_fig:
            visual_param = {}
            visual_param['visualize'] = self.visualize
            if not self.light_analysis_on:
                visual_param['save_fig'] = self.save_fig
                visual_param['save_fig_path'] = self.save_fig_path
                visual_param['save_fig_path_byname'] = os.path.join(self.save_fig_path,'byname')
                visual_param['save_fig_path_byiter'] = os.path.join(self.save_fig_path,'byiter')
                visual_param['save_fig_num'] = self.save_fig_num
                visual_param['pair_path'] = self.pair_path
                visual_param['iter'] = 'scale_'+str(self.n_scale) + '_iter_' + str(self.iter_count)
            else:
                visual_param['save_fig'] = False

            if (iter % self.visualize_step == 0) or (iter==self.nrOfIterations-1):
                if self.useMap and self.mapLowResFactor is not None:
                    vizImage, vizName = self.model.get_parameter_image_and_name_to_visualize(self.lowResISource)
                else:
                    vizImage, vizName = self.model.get_parameter_image_and_name_to_visualize(self.ISource)
                if self.useMap:
                    I1Warped = utils.compute_warped_image_multiNC(self.ISource, phi_or_warped_image, self.spacing)
                    vizReg.show_current_images(iter, self.ISource, self.ITarget, I1Warped, vizImage, vizName, phi_or_warped_image, visual_param)
                else:
                    vizReg.show_current_images(iter, self.ISource, self.ITarget, phi_or_warped_image, vizImage, vizName, None, visual_param)

        return False

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
                if self.last_successful_step_size_taken is not None:
                    desired_lr = self.last_successful_step_size_taken
                else:
                    desired_lr = 0.25
                opt_instance = torch.optim.SGD(self.model.parameters(), lr=desired_lr, momentum=0.9, dampening=0, weight_decay=0, nesterov=True)
            elif self.optimizer_name == 'adam':
                if self.last_successful_step_size_taken is not None:
                    desired_lr = self.last_successful_step_size_taken
                else:
                    desired_lr = 0.0025
                opt_instance = torch.optim.Adam(self.model.parameters(), lr=desired_lr, betas=(0.9, 0.999), eps=self.rel_ftol, weight_decay=0)
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


    def optimize(self):
        """
        Do the single scale optimization
        """

        # obtain all missing parameters (i.e., only set the ones that were not explicitly set)
        self._set_all_still_missing_parameters()

        # optimize for a few steps
        start = time.time()

        self.last_energy = None
        could_not_find_successful_step = False

        for iter in range(self.nrOfIterations):

            # take a step of the optimizer
            # for p in self.optimizer_instance._params:
            #     p.data = p.data.float()
            self.optimizer_instance.step(self._closure)
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
                tolerance_reached = self.analysis(self.rec_energy, self.rec_similarityEnergy,
                                                  self.rec_regEnergy, self.rec_opt_par_loss_energy,
                                                  self.rec_phiWarped,
                                                  self.rec_custom_optimizer_output_string,
                                                  self.rec_custom_optimizer_output_values)
            else:
                tolerance_reached = self.analysis(self.rec_energy, self.rec_similarityEnergy,
                                                  self.rec_regEnergy, self.rec_opt_par_loss_energy,
                                                  self.rec_IWarped,
                                                  self.rec_custom_optimizer_output_string,
                                                  self.rec_custom_optimizer_output_values)

            if tolerance_reached or could_not_find_successful_step:
                if tolerance_reached:
                    print('Terminating optimization, because the desired tolerance was reached.')
                break

            self.iter_count = iter+1

        print('time:', time.time() - start)


class SingleScaleConsensusRegistrationOptimizer(ImageRegistrationOptimizer):

    def __init__(self, sz, spacing, useMap, mapLowResFactor, params):

        super(SingleScaleConsensusRegistrationOptimizer, self).__init__(sz, spacing, useMap, mapLowResFactor, params)

        self.params[('optimizer', {}, 'optimizer settings')]
        cparams = self.params['optimizer']
        cparams[('consensus_settings', {}, 'settings for the consensus optimizer')]
        cparams = cparams['consensus_settings']

        self.sigma = cparams[('sigma', 1.0, 'sigma/2 is multiplier for squared augmented Lagrangian penalty')]
        """Multiplier for squared augmented Lagrangian penalty"""

        self.nr_of_batch_iterations = cparams[('nr_of_batch_iterations', 5, 'how many iterations for consensus; i.e., how often to iterate over the entire dataset')]
        """how many iterations for consensus; i.e., how often to iterate over the entire dataset"""
        self.batch_size = cparams[('batch_size',1,'how many images per batch')]
        """how many images per batch"""
        self.save_intermediate_checkpoints = cparams[('save_intermediate_checkpoints',False,'when set to True checkpoints are retained for each batch iterations')]
        """when set to True checkpoints are retained for each batch iterations"""

        self.checkpoint_output_directory = cparams[('checkpoint_output_directory','checkpoints','directory where the checkpoints will be stored')]
        """output directory where the checkpoints will be saved"""

        self.save_consensus_state_checkpoints = cparams[('save_consensus_state_checkpoints',True,'saves the current consensus state; typically only the individual states are saved as checkpoints')]
        """saves the current consensus state; typically only the individual states are saved as checkpoints"""

        self.continue_from_last_checkpoint = cparams[('continue_from_last_checkpoint',True,'If true then iterations are resumed from last checkpoint. Allows restarting an optimization')]
        """allows restarting an optimization by continuing from the last checkpoint"""

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

    def _consensus_penalty_loss(self,shared_model_parameters):
        """
        This allows to define additional terms for the loss which are based on parameters that are shared
        between models (for example for the smoother). Can be used to define a form of consensus optimization.
        :param shared_model_parameters: parameters that have been declared shared in a model
        :return: 0 by default, otherwise the corresponding penalty
        """
        additional_loss = Variable(MyTensor(1).zero_(),requires_grad=False)
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

        ssOpt = SingleScaleRegistrationOptimizer(batch_size, self.spacing, self.useMap, self.mapLowResFactor, self.params)

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
        ssOpt.set_light_analysis_on(self.light_analysis_on)

        if consensus_penalty:
            ssOpt.set_external_optimizer_parameter_loss(self._consensus_penalty_loss)

        if not self.light_analysis_on:
            raise ValueError('not supported yet')

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
        if d.has_key('consensus_dual'):
            self.current_consensus_dual = d['consensus_dual']
        else:
            raise ValueError('checkpoint does not contain: consensus_dual')

    def _custom_load_checkpoint(self,ssOpt,filename):
        d = torch.load(filename)
        ssOpt.load_checkpoint_dict(d)
        self.load_checkpoint_dict(d)

    def _custom_single_batch_load_checkpoint(self,ssOpt,filename):
        d = torch.load(filename)
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
            if state_from.has_key(key):
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


    def get_warped_image(self):
        """
        Returns the warped image
        :return: the warped image
        """

        p = dict()
        p['warped_images'] = []
        for current_batch in range(self.nr_of_batches):
            current_checkpoint_filename = self._get_checkpoint_filename(current_batch, self.nr_of_batch_iterations - 1)
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
            current_checkpoint_filename = self._get_checkpoint_filename(current_batch, self.nr_of_batch_iterations - 1)
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
            current_checkpoint_filename = self._get_checkpoint_filename(current_batch,self.nr_of_batch_iterations-1)
            dc = torch.load(current_checkpoint_filename)
            d = dict()
            d['model'] = dc['model']
            d['consensus_dual'] = dc['consensus_dual']
            p['registration_pars'].append(d)

        return p

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
            iter_offset = resume_from_iter+1
            print('Resuming from checkpoint iteration: ' + str(resume_from_iter))
        else:
            iter_offset = 0

        for iter_batch in range(iter_offset,self.nr_of_batch_iterations+iter_offset):
            print('Computing batch iteration ' + str(iter_batch + 1) + ' of ' + str(iter_offset+self.nr_of_batch_iterations))

            all_histories = []
            current_batch = 0 # there is only one batch, this one

            current_source_batch = Variable(self.ISource[:, ...].data, requires_grad=False)
            current_target_batch = Variable(self.ITarget[:, ...].data, requires_grad=False)
            current_batch_image_size = np.array(current_source_batch.size())

            # there is not consensus penalty here as this is technically not consensus optimization
            # todo: could ultimately replace the single scale optimizer; here used to write out checkpoints
            ssOpt = self._create_single_scale_optimizer(current_batch_image_size, consensus_penalty=False)

            # to make sure we have the model initialized, force parameter installation
            ssOpt._set_all_still_missing_parameters()

            # this loads the optimizer state and the model state, but here not the self.current_consensus_dual
            if iter_batch>0:
                previous_checkpoint_filename = self._get_checkpoint_filename(current_batch, iter_batch - 1)
                self._custom_single_batch_load_checkpoint(ssOpt, previous_checkpoint_filename)

            ssOpt.set_source_image(current_source_batch)
            ssOpt.set_target_image(current_target_batch)

            ssOpt.optimize()

            if (current_batch == self.nr_of_batches - 1) and (iter_batch == self.nr_of_batch_iterations - 1):
                # the last time we run this
                all_histories.append(ssOpt.get_history())

            current_checkpoint_filename = self._get_checkpoint_filename(current_batch, iter_batch)
            self._custom_save_checkpoint(ssOpt, current_checkpoint_filename)

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

        for iter_batch in range(iter_offset,self.nr_of_batch_iterations+iter_offset):
            print('Computing batch iteration ' + str(iter_batch+1) + ' of ' + str(iter_offset+self.nr_of_batch_iterations))

            next_consensus_initialized = False
            all_histories = []

            for current_batch in range(self.nr_of_batches):

                from_image = current_batch*self.batch_size
                to_image = min(self.nr_of_images,(current_batch+1)*self.batch_size)

                nr_of_images_in_batch = to_image-from_image

                current_source_batch = Variable(self.ISource[from_image:to_image, ...].data, requires_grad=False)
                current_target_batch = Variable(self.ITarget[from_image:to_image, ...].data, requires_grad=False)
                current_batch_image_size = np.array(current_source_batch.size())

                print('Computing image pair batch ' + str(current_batch+1) + ' of ' + str(self.nr_of_batches) +
                      ' of batch iteration ' + str(iter_batch+1) + ' of ' + str(iter_offset+self.nr_of_batch_iterations))
                print('Image range: [' + str(from_image) + ',' + str(to_image) + ')')

                # create new optimizer
                if iter_batch==0:
                    # do not apply the penalty the first time around
                    ssOpt = self._create_single_scale_optimizer(current_batch_image_size,consensus_penalty=False)
                else:
                    ssOpt = self._create_single_scale_optimizer(current_batch_image_size,consensus_penalty=True)

                # to make sure we have the model initialized, force parameter installation
                ssOpt._set_all_still_missing_parameters()

                if iter_batch==0:
                    # in the first round just initialize the shared state with what was computed previously
                    if self.last_shared_state is not None:
                        ssOpt.set_shared_model_parameters(self.last_shared_state)

                self._initialize_consensus_variables_if_needed(ssOpt)

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
                    self._custom_load_checkpoint(ssOpt,previous_checkpoint_filename)

                    # first update the dual variable (we do this now that we have the consensus state still
                    self._add_scaled_difference_to_state(self.current_consensus_dual,
                                                         ssOpt.get_shared_model_parameters(),
                                                         self.current_consensus_state,-1.0)


                ssOpt.set_source_image(current_source_batch)
                ssOpt.set_target_image(current_target_batch)

                ssOpt.optimize()

                self._copy_state(self.last_shared_state,ssOpt.get_shared_model_parameters())

                if (current_batch==self.nr_of_batches-1) and (iter_batch==self.nr_of_batch_iterations-1):
                    # the last time we run this
                    all_histories.append( ssOpt.get_history() )

                # update the consensus state (is done via next_consensus_state as
                # self.current_consensus_state is used as part of the optimization for all optimizations in the batch
                self._add_scaled_difference_to_state(self.next_consensus_state,
                                                     ssOpt.get_shared_model_parameters(),
                                                     self.current_consensus_dual,float(nr_of_images_in_batch)/float(self.nr_of_images))

                current_checkpoint_filename = self._get_checkpoint_filename(current_batch, iter_batch)
                self._custom_save_checkpoint(ssOpt,current_checkpoint_filename)

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

        current_iter_batch = 0
        while os.path.isfile(self._get_checkpoint_filename(0,current_iter_batch)):
            print('Found checkpoint iteration: ' + str(current_iter_batch))
            largest_found_iter = current_iter_batch
            current_iter_batch +=1

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

    def __init__(self, sz, spacing, useMap, mapLowResFactor, params ):
        super(MultiScaleRegistrationOptimizer, self).__init__(sz, spacing, useMap, mapLowResFactor, params)
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


    def set_pair_path(self,pair_paths):
        # f = lambda name: os.path.split(name)
        # get_in = lambda x: os.path.splitext(f(x)[1])[0]
        # get_fn = lambda x: f(f(x)[0])[1]
        # get_img_name = lambda x: get_fn(x)+'_'+get_in(x)
        # img_pair_name = [get_img_name(pair_path[0])+'_'+get_img_name(pair_path[1]) for pair_path in pair_paths]
        self.pair_path = pair_paths

    def set_save_fig_path(self, save_fig_path):
        """
        the path of saved figures, default is the ../data/expr_name
        :param save_fig_path:
        :return:
        """
        self.save_fig_path = os.path.join(save_fig_path, self.expr_name)

    def init_recorder(self, task_name):
        self.recorder = XlsxRecorder(task_name, self.save_fig_path)
        return self.recorder


    def set_saving_env(self):
        if self.save_fig==True:
            # saving by files
            for file_name in self.pair_path[:self.save_fig_num]:
                save_folder = os.path.join(os.path.join(self.save_fig_path,'byname'),file_name)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

            # saving by iterations
            for idx, scale in enumerate(self.scaleFactors):
                for i in range(self.scaleIterations[idx]):
                    if i%self.visualize_step == 0:
                        save_folder = os.path.join(os.path.join(self.save_fig_path,'byiter'),'scale_'+str(scale) + '_iter_' + str(i))
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)

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

    def get_map(self):
        """
        Returns the deformation map
        :return: deformation map
        """
        if self.ssOpt is not None:
            return self.ssOpt.get_map()
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


    def _set_all_still_missing_parameters(self):

        self.scaleFactors = self.params['optimizer']['multi_scale'][('scale_factors', [1.0, 0.5, 0.25], 'how images are scaled')]
        self.scaleIterations = self.params['optimizer']['multi_scale'][('scale_iterations', [10, 20, 20], 'number of iterations per scale')]

        if (self.optimizer is None) and (self.optimizer_name is None):
            self.optimizer_name = self.params['optimizer'][('name','lbfgs_ls','Optimizer (lbfgs|adam|sgd)')]

        if self.model_name is None:
            model_name = self.params['model']['registration_model'][('type', 'lddmm_shooting_map', "['svf'|'svf_quasi_momentum'|'svf_scalar_momentum'|'svf_vector_momentum'|'lddmm_shooting'|'lddmm_shooting_scalar_momentum'] all with suffix '_map' or '_image'")]
            self.params['model']['deformation'][('use_map', True, 'use a map for the solution or not True/False' )]
            self.set_model( model_name )

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

        for en_scale in enumerate(reverseScales):
            print('Optimizing for scale = ' + str(en_scale[1]))

            # create the images
            currentScaleFactor = en_scale[1]
            currentScaleNumber = en_scale[0]

            currentDesiredSz = self._get_desired_size_from_scale(self.ISource.size(), currentScaleFactor)

            currentNrOfIteratons = reverseIterations[currentScaleNumber]

            ISourceC, spacingC = self.sampler.downsample_image_to_size(self.ISource, self.spacing, currentDesiredSz[2::])
            ITargetC, spacingC = self.sampler.downsample_image_to_size(self.ITarget, self.spacing, currentDesiredSz[2::])

            szC = ISourceC.size()  # this assumes the BxCxXxYxZ format

            self.ssOpt = SingleScaleRegistrationOptimizer(szC, spacingC, self.useMap, self.mapLowResFactor, self.params)
            print('Setting learning rate to ' + str( lastSuccessfulStepSizeTaken ))
            self.ssOpt.set_last_successful_step_size_taken( lastSuccessfulStepSizeTaken )

            if ((self.add_model_name is not None) and
                    (self.add_model_networkClass is not None) and
                    (self.add_model_lossClass is not None)):
                self.ssOpt.add_model(self.add_model_name, self.add_model_networkClass, self.add_model_lossClass)

            # now set the actual model we want to solve
            self.ssOpt.set_model(self.model_name)

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
            self.ssOpt.set_light_analysis_on(self.light_analysis_on)

            if not self.light_analysis_on:
                self.ssOpt.set_expr_name(self.get_expr_name())
                self.ssOpt.set_save_fig(self.get_save_fig())
                self.ssOpt.set_save_fig_path(self.get_save_fig_path())
                self.ssOpt.set_save_fig_num(self.get_save_fig_num())
                self.ssOpt.set_pair_path(self.get_pair_path())
                self.ssOpt.set_n_scale(en_scale[1])
                self.ssOpt.set_recorder(self.get_recorder())
                self.ssOpt.set_save_excel(self.get_save_excel())
                self.ssOpt.set_source_label(self.get_source_label())
                self.ssOpt.set_target_label(self.get_target_label())
                self.ssOpt.set_batch_id(self.get_batch_id())


            self.ssOpt.set_source_image(ISourceC)
            self.ssOpt.set_target_image(ITargetC)

            if upsampledParameters is not None:
                # check that the upsampled parameters are consistent with the downsampled images
                spacingError = False
                expectedSpacing = None
                if self.mapLowResFactor is not None:
                    expectedSpacing = self._get_low_res_spacing_from_spacing(spacingC, szC, upsampledSz)
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
                        upsampledSz = self._get_low_res_size_from_size(upsampledSz,self.mapLowResFactor)
                        print(self.mapLowResFactor)
                        print('After')
                        print(upsampledSz)
                upsampledParameters, upsampledParameterSpacing = self.ssOpt.upsample_model_parameters(upsampledSz[2::])
