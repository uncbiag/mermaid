"""
This package enables easy single-scale and multi-scale optimization support.
"""

from abc import ABCMeta, abstractmethod

import time
import utils
import visualize_registration_results as vizReg
import custom_optimizers as CO
import numpy as np

import torch
from torch.autograd import Variable
from dataWapper import USE_CUDA, AdpatVal
import model_factory as MF
import image_sampling as IS
from MyAdam import MyAdam
from configParsers import optimName, visualize


class Optimizer(object):
    """
       Abstract optimizer base class.
       """
    __metaclass__ = ABCMeta

    def __init__(self, sz, spacing, useMap, params):
        """
        Constructor.
        
        :param sz: image size in BxCxXxYxZ format
        :param spacing: spatial spacing, e.g., [0.1,0.1,0.1] in 3D
        :param useMap: boolean, True if a coordinate map is evolved to warp images, False otherwise
        :param params: ParametersDict() instance to hold parameters
        """
        self.sz = sz
        """image size"""
        self.spacing = spacing
        """image spacing"""
        self.useMap = useMap
        """makes use of map"""
        self.params = params
        """general parameters"""
        self.rel_ftol = 1e-4
        """relative termination tolerance for optimizer"""

    def set_rel_ftol(self, rel_ftol):
        """
        Sets the relative termination tolerance: |f(x_i)-f(x_{i-1})|/f(x_i)<tol
        
        :param rel_ftol: relative termination tolerance for optimizer
        """
        self.rel_ftol = rel_ftol

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


class ImageRegistrationOptimizer(Optimizer):
    """
    Optimization class for image registration.
    """

    def __init__(self, sz, spacing, useMap, params):
        super(ImageRegistrationOptimizer, self).__init__(sz, spacing, useMap, params)
        self.ISource = None
        """source image"""
        self.ITarget = None
        """target image"""
        self.optimizer_name = optimName  # 'lbfgs_ls'#''lbfgs_ls'
        """name of the optimizer to use"""
        self.optimizer_params = {}
        """parameters that should be passed to the optimizer"""
        self.optimizer = None
        """optimizer object itself (to be instantiated)"""

        self.visualize = visualize
        """if True figures are created during the run"""
        self.visualize_step = 10
        """how often the figures are updated; each self.visualize_step-th iteration"""

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

    def set_source_image(self, I):
        """
        Setting the source image which should be deformed to match the target image
        
        :param I: source image
        """
        self.ISource = I

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

    def __init__(self, sz, spacing, useMap, params):
        super(SingleScaleRegistrationOptimizer, self).__init__(sz, spacing, useMap, params)

        self.mf = MF.ModelFactory(self.sz, self.spacing)
        """model factory which will be used to create the model and its loss function"""

        self.model = None
        """the model itself"""
        self.criterion = None
        """the loss function"""

        self.identityMap = None
        """identity map, will be needed for map-based solutions"""
        self.optimizer_instance = None
        """the optimizer instance to perform the actual optimization"""

        self.nrOfIterations = 1
        """the maximum number of iterations for the optimizer"""
        self.iter_count = 0
        self.rec_energy = None
        self.rec_similarityEnergy = None
        self.rec_regEnergy = None
        self.rec_phiWarped = None
        self.rec_IWarped = None
        self.last_energy = None
        self.terminal_flag = 0
        """the evaluation information"""

    def set_model(self, modelName):
        """
        Sets the model that should be solved
        
        :param modelName: name of the model that should be solved (string) 
        """

        self.model, self.criterion = self.mf.create_registration_model(modelName, self.params)
        print(self.model)

        if self.useMap:
            # create the identity map [-1,1]^d, since we will use a map-based implementation
            id = utils.identity_map_multiN(self.sz)
            self.identityMap = AdpatVal(Variable(torch.from_numpy(id), requires_grad=False))

    def add_similarity_measure(self, simName, simMeasure):
        """
        Adds a custom similarity measure.
        
        :param simName: name of the similarity measure (string) 
        :param simMeasure: similarity measure itself (class object that can be instantiated)
        """
        self.criterion.add_similarity_measure(simName, simMeasure)

    def add_model(self, modelName, modelNetworkClass, modelLossClass):
        """
        Adds a custom model and its loss function
        
        :param modelName: name of the model to be added (string) 
        :param modelNetworkClass: registration model itself (class object that can be instantiated)
        :param modelLossClass: registration loss (class object that can be instantiated)
        """
        self.mf.add_model(modelName, modelNetworkClass, modelLossClass)

    def set_model_parameters(self, p):
        """
        Set the parameters of the registration model
        
        :param p: parameters 
        """
        self.model.set_registration_parameters(p, self.sz, self.spacing)

    def get_model_parameters(self):
        """
        Returns the parameters of the model
         
        :return: model parameters 
        """
        self.model.get_registration_parameters()

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

    def set_number_of_iterations(self, nrIter):
        """
        Set the number of iterations of the optimizer
        
        :param nrIter: number of iterations 
        """
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
        if self.useMap:
            self.rec_phiWarped = self.model(self.identityMap, self.ISource)
            loss = self.criterion(self.rec_phiWarped, self.ISource, self.ITarget)
        else:
            self.rec_IWarped = self.model(self.ISource)
            loss = self.criterion(self.rec_IWarped, self.ISource, self.ITarget)
        loss.backward()

        if self.useMap:
            if self.iter_count % 1 == 0:
                self.rec_energy, self.rec_similarityEnergy, self.rec_regEnergy = self.criterion.get_energy(
                    self.rec_phiWarped, self.ISource, self.ITarget)
        else:
            if self.iter_count % 1 == 0:
                self.rec_energy, self.rec_similarityEnergy, self.rec_regEnergy = self.criterion.get_energy(
                    self.rec_IWarped, self.ISource, self.ITarget)

        return loss

    def analysis(self, energy, similarityEnergy, regEnergy, Warped):
        """
        print out the and visualize the result
        :param energy:
        :param similarityEnergy:
        :param regEnergy:
        :param Warped:
        :return:
        """

        cur_energy = utils.t2np(energy.float())

        if self.last_energy is not None:

            # relative function toleranc: |f(xi)-f(xi+1)|/(1+|f(xi)|)
            rel_f = abs(self.last_energy - cur_energy) / (1 + abs(cur_energy))

            print('Iter {iter}: E={energy}, similarityE={similarityE}, regE={regE}, relF={relF}'
                  .format(iter=self.iter_count,
                          energy=cur_energy,
                          similarityE=utils.t2np(similarityEnergy.float()),
                          regE=utils.t2np(regEnergy.float()),
                          relF=rel_f))

            # check if relative convergence tolerance is reached
            if rel_f < self.rel_ftol:
                print('Reached relative function tolerance of = ' + str(self.rel_ftol))
                self.terminal_flag = 0
                exit(0)

        else:
            print('Iter {iter}: E={energy}, similarityE={similarityE}, regE={regE}, relF=n/a'
                  .format(iter=self.iter_count,
                          energy=cur_energy,
                          similarityE=utils.t2np(similarityEnergy.float()),
                          regE=utils.t2np(regEnergy.float())))

        self.last_energy = cur_energy
        iter = self.iter_count

        if self.visualize:
            if iter % self.visualize_step == 0:
                vizImage, vizName = self.model.get_parameter_image_and_name_to_visualize()
                if self.useMap:
                    I1Warped = utils.compute_warped_image_multiNC(self.ISource, Warped)
                    vizReg.show_current_images(iter, self.ISource, self.ITarget, I1Warped, vizImage, vizName, Warped)
                else:
                    vizReg.show_current_images(iter, self.ISource, self.ITarget, Warped, vizImage, vizName)

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
                opt_instance = CO.LBFGS_LS(self.model.parameters(),
                                           lr=1.0, max_iter=1, max_eval=5,
                                           tolerance_grad=self.rel_ftol * 10, tolerance_change=self.rel_ftol,
                                           history_size=5, line_search_fn='backtracking')
                return opt_instance
            elif self.optimizer_name == 'adam':
                opt_instance = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=self.rel_ftol,
                                      weight_decay=0)
                return opt_instance
            else:
                raise ValueError('Optimizer = ' + str(self.optimizer_name) + ' not yet supported')

    def optimize(self):
        """
        Do the single scale optimization
        """

        # do the actual optimization
        self.optimizer_instance = self._get_optimizer_instance()

        # optimize for a few steps
        start = time.time()
        if USE_CUDA:
            self.model = self.model.cuda()

        self.last_energy = None

        for iter in range(self.nrOfIterations):

            # take a step of the optimizer
            # for p in self.optimizer_instance._params:
            #     p.data = p.data.float()
            self.optimizer_instance.step(self._closure)

            if self.useMap:
                self.analysis(self.rec_energy, self.rec_similarityEnergy, self.rec_regEnergy, self.rec_phiWarped)
            else:
                self.analysis(self.rec_energy, self.rec_similarityEnergy, self.rec_regEnergy, self.rec_IWarped)
            self.rec_regEnergy = None
            self.rec_phiWarped = None
            if self.terminal_flag:
                break
            self.iter_count = iter+1

        print('time:', time.time() - start)


class MultiScaleRegistrationOptimizer(ImageRegistrationOptimizer):
    """
    Class to perform multi-scale optimization. Essentially puts a loop around multiple calls of the
    single scale optimizer and starts with the registration of downsampled images. When moving up
    the hierarchy, the registration parameters are upsampled from the solution at the previous lower resolution
    """

    def __init__(self, sz, spacing, useMap, params):
        super(MultiScaleRegistrationOptimizer, self).__init__(sz, spacing, useMap, params)
        self.scaleFactors = [1.]
        """At what image scales optimization should be computed"""
        self.scaleIterations = [100]
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
        self.scaleFactors = scaleFactors

    def set_number_of_iterations_per_scale(self, scaleIterations):
        """
        Sets the number of iterations that will be performed per scale of the multi-resolution hierarchy. E.g, [50,100,200]
        
        :param scaleIterations: number of iterations per scale (array)
        """
        self.scaleIterations = scaleIterations

    def _get_desired_size_from_scale(self, origSz, scale):

        osz = np.array(list(origSz))
        dsz = np.zeros(osz.shape, dtype='int')
        dim = len(osz)
        for d in range(dim):
            dsz[d] = round(scale * osz[d])

        return dsz

    def optimize(self):
        """
        Perform the actual multi-scale optimization
        """

        if (self.ISource is None) or (self.ITarget is None):
            raise ValueError('Source and target images need to be set first')

        upsampledParameters = None
        upsampledSpacing = None

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

            currentDesiredSz = self._get_desired_size_from_scale(self.ISource.size()[2::], currentScaleFactor)

            currentNrOfIteratons = reverseIterations[currentScaleNumber]

            sampler = IS.ResampleImage()

            ISourceC, spacingC = sampler.downsample_image_to_size(self.ISource, self.spacing, currentDesiredSz)
            ITargetC, spacingC = sampler.downsample_image_to_size(self.ITarget, self.spacing, currentDesiredSz)

            szC = ISourceC.size()  # this assumes the BxCxXxYxZ format

            ssOpt = SingleScaleRegistrationOptimizer(szC, spacingC, self.useMap, self.params)

            if ((self.add_model_name is not None) and
                    (self.add_model_networkClass is not None) and
                    (self.add_model_lossClass is not None)):
                ssOpt.add_model(self.add_model_name, self.add_model_networkClass, self.add_model_lossClass)

            # now set the actual model we want to solve
            ssOpt.set_model(self.model_name)

            if (self.addSimName is not None) and (self.addSimMeasure is not None):
                ssOpt.add_similarity_measure(self.addSimName, self.addSimMeasure)

            # setting the optimizer
            if self.optimizer is not None:
                ssOpt.set_optimizer(self.optimizer)
                ssOpt.set_optimizer_params(self.optimizer_params)
            elif self.optimizer_name is not None:
                ssOpt.set_optimizer_by_name(self.optimizer_name)

            ssOpt.set_rel_ftol(self.get_rel_ftol())
            ssOpt.set_visualization(self.get_visualization())
            ssOpt.set_visualize_step(self.get_visualize_step())

            ssOpt.set_source_image(ISourceC)
            ssOpt.set_target_image(ITargetC)

            if upsampledParameters is not None:
                # check that the upsampled parameters are consistent with the downsampled images
                if not (abs(spacingC - upsampledSpacing) < 0.000001).all():
                    print(spacingC)
                    print(upsampledSpacing)
                    raise ValueError('Upsampled parameters and downsampled images are of inconsistent dimension')
                # now that everything is fine, we can use the upsampled parameters
                print('Explicitly setting the optimization parameters')
                ssOpt.set_model_parameters(upsampledParameters)

            # do the actual optimization
            print('Optimizing for at most ' + str(currentNrOfIteratons) + ' iterations')
            ssOpt.set_number_of_iterations(currentNrOfIteratons)
            ssOpt.optimize()

            # if we are not at the very last scale, then upsample the parameters
            if currentScaleNumber != nrOfScales - 1:
                # we need to revert the downsampling to the next higher level
                scaleTo = reverseScales[currentScaleNumber + 1]
                desiredUpsampleSz = self.get_desired_size_from_scale(self.ISource.size()[2::], scaleTo)
                upsampledParameters, upsampledSpacing = ssOpt.upsample_model_parameters(desiredUpsampleSz)
