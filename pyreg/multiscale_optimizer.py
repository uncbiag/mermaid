"""
This package enables easy multi-scale optimization support
"""

from abc import ABCMeta, abstractmethod

import time
import utils
import visualize_registration_results as vizReg
import custom_optimizers as CO
import numpy as np

import torch
from torch.autograd import Variable

import model_factory as MF
import image_sampling as IS

class Optimizer(object):
    """
       Abstract base class.
       """
    __metaclass__ = ABCMeta

    def __init__(self,sz,spacing,useMap,params):
        self.sz = sz
        self.spacing = spacing
        self.useMap = useMap
        self.params = params

    @abstractmethod
    def set_model(self,modelName):
        pass

    @abstractmethod
    def optimize(self):
        pass

class ImageRegistrationOptimizer(Optimizer):

    def __init__(self,sz,spacing,useMap,params):
        super(ImageRegistrationOptimizer,self).__init__(sz,spacing,useMap,params)
        self.ISource = None
        self.ITarget = None

    def set_source_image(self, I):
        self.ISource = I

    def set_target_image(self, I):
        self.ITarget = I

class SingleScaleRegistrationOptimizer(ImageRegistrationOptimizer):
    def __init__(self,sz,spacing,useMap,params):
        super(SingleScaleRegistrationOptimizer,self).__init__(sz,spacing,useMap,params)

        self.mf = MF.ModelFactory(self.sz, self.spacing)

        self.model = None
        self.criterion = None

        self.identityMap = None
        self.optimizer = None

        self.nrOfIterations = 1
        self.visualize = True

    def set_model(self,modelName):

        self.model, self.criterion = self.mf.create_registration_model(modelName, self.params)
        print(self.model)

        if self.useMap:
            # create the identity map [-1,1]^d, since we will use a map-based implementation
            id = utils.identity_map_multiN(sz)
            self.identityMap = Variable(torch.from_numpy(id), requires_grad=False)


    def add_similarity_measure(self, simName, simMeasure):
        self.criterion.add_similarity_measure(simName, simMeasure)

    def add_model(self, modelName, modelNetworkClass, modelLossClass):
        self.mf.add_model(modelName,modelNetworkClass,modelLossClass)

    def set_optimization_parameters(self, p):
        self.model.set_registration_parameters(p, self.sz, self.spacing)

    def get_optimization_parameters(self, p):
        self.model.get_registration_parameters()

    def upsample_optimization_parameters(self, desiredSize):
        return self.model.upsample_registration_parameters(desiredSize)

    def downsample_optimization_parameters(self, desiredSize):
        return self.model.downsample_registration_parameters(desiredSize)

    def set_number_of_iterations(self, nrIter):
        self.nrOfIterations = nrIter

    def closure(self):
        self.optimizer.zero_grad()
        # 1) Forward pass: Compute predicted y by passing x to the model
        # 2) Compute loss
        if self.useMap:
            phiWarped = self.model(self.identityMap, self.ISource)
            loss = self.criterion(phiWarped, self.ISource, self.ITarget)
        else:
            IWarped = self.model(self.ISource)
            loss = self.criterion(IWarped, self.ISource, self.ITarget)

        loss.backward()
        return loss

    def optimize(self):
        # do the actual optimization

        if (self.model is None) or (self.criterion is None):
            raise ValueError('Please specify a model to solve with set_model first')

        self.optimizer = CO.LBFGS_LS(self.model.parameters(),
                                lr=1.0, max_iter=1, max_eval=5,
                                tolerance_grad=1e-3, tolerance_change=1e-4,
                                history_size=5, line_search_fn='backtracking')

        # optimize for a few steps
        start = time.time()

        for iter in range(self.nrOfIterations):

            # take a step of the optimizer
            self.optimizer.step(self.closure)

            # apply the current state to either get the warped map or directly the warped source image
            if self.useMap:
                phiWarped = self.model(self.identityMap, self.ISource)
            else:
                cIWarped = self.model(self.ISource)

            if iter%1==0:
                if self.useMap:
                    energy, similarityEnergy, regEnergy = self.criterion.get_energy(phiWarped, self.ISource, self.ITarget)
                else:
                    energy, similarityEnergy, regEnergy = self.criterion.get_energy(cIWarped, self.ISource, self.ITarget)

                print('Iter {iter}: E={energy}, similarityE={similarityE}, regE={regE}'
                      .format(iter=iter,
                              energy=utils.t2np(energy),
                              similarityE=utils.t2np(similarityEnergy),
                              regE=utils.t2np(regEnergy)))

            if self.visualize:
                if iter%5==0:
                    if self.useMap:
                        I1Warped = utils.compute_warped_image_multiNC(self.ISource, phiWarped)
                        vizReg.show_current_images(iter, self.ISource, self.ITarget, I1Warped, phiWarped)
                    else:
                        vizReg.show_current_images(iter, self.ISource, self.ITarget, cIWarped)

        print('time:', time.time() - start)



class MultiScaleRegistrationOptimizer(ImageRegistrationOptimizer):
    def __init__(self,sz,spacing,useMap,params):
        super(MultiScaleRegistrationOptimizer,self).__init__(sz,spacing,useMap,params)
        self.scaleFactors = [1.]
        self.scaleIterations = [100]

        self.addSimName = None
        self.addSimMeasure = None
        self.add_model_name = None
        self.add_model_networkClass = None
        self.add_model_lossClass = None

        self.model_name = None

    def add_similarity_measure(self, simName, simMeasure):
        self.addSimName = simName
        self.addSimMeasure = simMeasure

    def set_model(self,modelName):
        self.model_name = modelName

    def add_model(self, add_model_name, add_model_networkClass, add_model_lossClass):
        self.add_model_name = add_model_name
        self.add_model_networkClass = add_model_networkClass
        self.add_model_lossClass = add_model_lossClass

    def set_scale_factors(self, scaleFactors):
        self.scaleFactors = scaleFactors

    def set_number_of_iterations_per_scale(self, scaleIterations):
        self.scaleIterations = scaleIterations

    def get_desired_size_from_scale(self, origSz, scale):

        osz = np.array(list(origSz))
        dsz = np.zeros(osz.shape,dtype='int')
        dim = len(osz)
        for d in range(dim):
            dsz[d]=round(scale*osz[d])

        return dsz


    def optimize(self):

        if (self.ISource is None) or (self.ITarget is None):
            raise ValueError('Source and target images need to be set first')

        upsampledParameters = None
        upsampledSpacing = None

        nrOfScales = len(self.scaleFactors)

        # check that we have the right number of iteration parameters
        assert( nrOfScales==len(self.scaleIterations) )

        print('Performing multiscale optmization with scales: ' + str(self.scaleFactors))

        # go from lowest to highest scale
        reverseScales = self.scaleFactors[-1::-1]
        reverseIterations = self.scaleIterations[-1::-1]

        for en_scale in enumerate(reverseScales):
            print('Optimizing for scale = ' + str(en_scale[1]))

            # create the images
            currentScaleFactor = en_scale[1]
            currentScaleNumber = en_scale[0]

            currentDesiredSz = self.get_desired_size_from_scale(self.ISource.size()[2::], currentScaleFactor)

            currentNrOfIteratons = reverseIterations[currentScaleNumber]

            sampler = IS.ResampleImage()

            ISourceC, spacingC = sampler.downsample_image_to_size(self.ISource, self.spacing,currentDesiredSz)
            ITargetC, spacingC = sampler.downsample_image_to_size(self.ITarget, self.spacing,currentDesiredSz)

            szC = ISourceC.size() # this assumes the BxCxXxYxZ format

            ssOpt = SingleScaleRegistrationOptimizer(szC,spacingC,self.useMap,self.params)

            if (self.addSimName is not None) and (self.addSimMeasure is not None):
                ssOpt.add_similarity_measure(self.addSimName, self.addSimMeasure)

            if ( (self.add_model_name is not None) and
                     (self.add_model_networkClass is not None) and
                     (self.add_model_lossClass is not None) ):
                ssOpt.add_model(self.add_model_name,self.add_model_networkClass,self.add_model_lossClass)

            # now set the actual model we want to solve
            ssOpt.set_model(self.model_name)

            ssOpt.set_source_image(ISourceC)
            ssOpt.set_target_image(ITargetC)

            if upsampledParameters is not None:
                # check that the upsampled parameters are consistent with the downsampled images
                if not (abs(spacingC-upsampledSpacing)<0.000001).all():
                    print(spacingC)
                    print(upsampledSpacing)
                    raise ValueError('Upsampled parameters and downsampled images are of inconsistent dimension')
                # now that everything is fine, we can use the upsampled parameters
                print('Explicitly setting the optimization parameters')
                ssOpt.set_optimization_parameters(upsampledParameters)

            # do the actual optimization
            print( 'Optimizing for at most ' + str(currentNrOfIteratons) + ' iterations')
            ssOpt.set_number_of_iterations(currentNrOfIteratons)
            ssOpt.optimize()

            # if we are not at the very last scale, then upsample the parameters
            if currentScaleNumber!=nrOfScales-1:
                # we need to revert the downsampling to the next higher level
                scaleTo = reverseScales[currentScaleNumber+1]
                desiredUpsampleSz = self.get_desired_size_from_scale(self.ISource.size()[2::], scaleTo)
                upsampledParameters,upsampledSpacing = ssOpt.upsample_optimization_parameters(desiredUpsampleSz)