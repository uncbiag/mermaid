"""
Defines different registration methods as pyTorch networks
Currently implemented:
    * SVFNet: image-based stationary velocity field
"""

import torch
import torch.nn as nn

import rungekutta_integrators as RK
import forward_models as FM

import regularizer_factory as RF
import similarity_measure_factory as SM

import smoother_factory as SF

import utils

from abc import ABCMeta, abstractmethod

class RegistrationNet(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, sz, spacing, params):
        super(RegistrationNet,self).__init__()
        self.sz = sz
        self.spacing = spacing
        self.nrOfTimeSteps = utils.getpar(params, 'numberOfTimeSteps', 10)
        self.tFrom = 0.
        self.tTo = 1.
        self.nrOfImages = sz[0]
        self.nrOfChannels = sz[1]

    @abstractmethod
    def createRegistrationParameters(self):
        pass

    @abstractmethod
    def getRegistrationParameters(self):
        pass

    @abstractmethod
    def createIntegrator(self):
        pass


class SVFNet(RegistrationNet):
    def __init__(self,sz,spacing,params):
        super(SVFNet, self).__init__(sz,spacing,params)
        self.v = self.createRegistrationParameters()
        self.integrator = self.createIntegrator()

    def createRegistrationParameters(self):
        return utils.createNDVectorFieldParameter_multiN( self.sz[2::],self.nrOfImages )

    def getRegistrationParameters(self):
        return self.v


class SVFImageNet(SVFNet):
    def __init__(self, sz, spacing, params):
        super(SVFImageNet, self).__init__(sz,spacing,params)

    def createIntegrator(self):
        advection = FM.AdvectImage(self.sz, self.spacing)
        return RK.RK4(advection.f, advection.u, self.v)

    def forward(self, I):
        I1 = self.integrator.solve([I], self.tFrom, self.tTo, self.nrOfTimeSteps)
        return I1[0]


class RegistrationLoss(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self,sz,spacing,params):
        super(RegistrationLoss, self).__init__()
        self.spacing = spacing
        self.sz = sz
        self.similarityMeasure = (SM.SimilarityMeasureFactory(self.spacing).
                                  createSimilarityMeasure(utils.getpar(params, 'similarityMeasure', 'ssd'), params))

    @abstractmethod
    def computeRegularizationEnergy(self, I0_source):
        pass


class RegistrationImageLoss(RegistrationLoss):

    def __init__(self,sz,spacing,params):
        super(RegistrationImageLoss, self).__init__(sz,spacing,params)

    def getEnergy(self, I1_warped, I0_source, I1_target):
        sim = self.similarityMeasure.computeSimilarity(I1_warped, I1_target)
        reg = self.computeRegularizationEnergy(I0_source)
        energy = sim + reg
        return energy, sim, reg

    def forward(self, I1_warped, I0_source, I1_target):
        energy, sim, reg = self.getEnergy( I1_warped, I0_source, I1_target)
        return energy


class RegistrationMapLoss(RegistrationLoss):
    def __init__(self, sz, spacing, params):
        super(RegistrationMapLoss, self).__init__(sz, spacing, params)

    def getEnergy(self, phi1, I0_source, I1_target):
        I1_warped = utils.computeWarpedImage_multiNC(I0_source, phi1)
        sim = self.similarityMeasure.computeSimilarity(I1_warped, I1_target)
        reg = self.computeRegularizationEnergy(I0_source)
        energy = sim + reg
        return energy, sim, reg

    def forward(self, phi1, I0_source, I1_target):
        energy, sim, reg = self.getEnergy(phi1, I0_source, I1_target)
        return energy


class SVFImageLoss(RegistrationImageLoss):
    def __init__(self,v,sz,spacing,params):
        super(SVFImageLoss, self).__init__(sz,spacing,params)
        self.v = v
        self.regularizer = (RF.RegularizerFactory(self.spacing).
                            createRegularizer(utils.getpar(params,'regularizer','helmholtz'),params))

    def computeRegularizationEnergy(self,I0_source):
        return self.regularizer.computeRegularizer_multiN(self.v)


class SVFMapNet(SVFNet):
    def __init__(self,sz,spacing,params):
        super(SVFMapNet, self).__init__(sz,spacing,params)

    def createIntegrator(self):
        advectionMap = FM.AdvectMap( self.sz, self.spacing )
        return RK.RK4(advectionMap.f,advectionMap.u,self.v)

    def forward(self, phi, I0_source):
        phi1 = self.integrator.solve([phi], self.tFrom, self.tTo, self.nrOfTimeSteps)
        return phi1[0]


class SVFMapLoss(RegistrationMapLoss):
    def __init__(self,v,sz,spacing,params):
        super(SVFMapLoss, self).__init__(sz,spacing,params)
        self.v = v
        self.regularizer = (RF.RegularizerFactory(self.spacing).
                            createRegularizer(utils.getpar(params, 'regularizer', 'helmholtz'), params))

    def computeRegularizationEnergy(self,I0_source):
        return self.regularizer.computeRegularizer_multiN(self.v)


class LDDMMShootingVectorMomentumNet(RegistrationNet):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingVectorMomentumNet, self).__init__(sz,spacing,params)
        self.m = self.createRegistrationParameters()
        self.integrator = self.createIntegrator()

    def createRegistrationParameters(self):
        return utils.createNDVectorFieldParameter_multiN( self.sz[2::], self.nrOfImages )

    def getRegistrationParameters(self):
        return self.m


class LDDMMShootingVectorMomentumImageNet(LDDMMShootingVectorMomentumNet):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingVectorMomentumImageNet, self).__init__(sz,spacing,params)

    def createIntegrator(self):
        epdiffImage = FM.EPDiffImage( self.sz, self.spacing )
        return RK.RK4(epdiffImage.f)

    def forward(self, I):
        mI1 = self.integrator.solve([self.m,I], self.tFrom, self.tTo, self.nrOfTimeSteps)
        return mI1[1]


class LDDMMShootingVectorMomentumImageLoss(RegistrationImageLoss):
    def __init__(self,m,sz,spacing,params):
        super(LDDMMShootingVectorMomentumImageLoss, self).__init__(sz,spacing,params)
        self.m = m
        self.smoother = SF.SmootherFactory(self.sz[2::],self.spacing).createSmoother('gaussian')

    def computeRegularizationEnergy(self,I0_source):
        v = self.smoother.computeSmootherVectorField_multiN(self.m)
        reg = (v * self.m).sum() * self.spacing.prod()
        return reg


class LDDMMShootingVectorMomentumMapNet(LDDMMShootingVectorMomentumNet):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingVectorMomentumMapNet, self).__init__(sz,spacing,params)

    def createIntegrator(self):
        epdiffMap = FM.EPDiffMap( self.sz, self.spacing )
        return RK.RK4(epdiffMap.f)

    def forward(self, phi, I0_source):
        mphi1 = self.integrator.solve([self.m,phi], self.tFrom, self.tTo, self.nrOfTimeSteps)
        return mphi1[1]


class LDDMMShootingVectorMomentumMapLoss(RegistrationMapLoss):
    def __init__(self,m,sz,spacing,params):
        super(LDDMMShootingVectorMomentumMapLoss, self).__init__(sz,spacing,params)
        self.m = m
        self.smoother = SF.SmootherFactory(self.sz[2::],self.spacing).createSmoother('gaussian')

    def computeRegularizationEnergy(self,I0_source):
        v = self.smoother.computeSmootherVectorField_multiN(self.m)
        reg = (v * self.m).sum() * self.spacing.prod()
        return reg


class LDDMMShootingScalarMomentumNet(RegistrationNet):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingScalarMomentumNet, self).__init__(sz,spacing,params)
        self.lam = self.createRegistrationParameters()
        self.integrator = self.createIntegrator()

    def createRegistrationParameters(self):
        return utils.createNDScalarFieldParameter_multiNC( self.sz[2::], self.nrOfImages, self.nrOfChannels )

    def getRegistrationParameters(self):
        return self.lam


class LDDMMShootingScalarMomentumImageNet(LDDMMShootingScalarMomentumNet):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingScalarMomentumImageNet, self).__init__(sz,spacing,params)

    def createIntegrator(self):
        epdiffScalarMomentumImage = FM.EPDiffScalarMomentumImage( self.sz, self.spacing )
        return RK.RK4(epdiffScalarMomentumImage.f)

    def forward(self, I):
        lamI1 = self.integrator.solve([self.lam,I], self.tFrom, self.tTo, self.nrOfTimeSteps)
        return lamI1[1]


class LDDMMShootingScalarMomentumImageLoss(RegistrationImageLoss):
    def __init__(self,lam,sz,spacing,params):
        super(LDDMMShootingScalarMomentumImageLoss, self).__init__(sz,spacing,params)
        self.lam = lam
        self.smoother = SF.SmootherFactory(self.sz[2::],self.spacing).createSmoother('gaussian')

    def computeRegularizationEnergy(self, I0_source):
        m = utils.computeVectorMomentumFromScalarMomentum_multiNC(self.lam, I0_source, self.sz, self.spacing)
        v = self.smoother.computeSmootherVectorField_multiN(m)
        reg = (v * m).sum() * self.spacing.prod()
        return reg


class LDDMMShootingScalarMomentumMapNet(LDDMMShootingScalarMomentumNet):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingScalarMomentumMapNet, self).__init__(sz,spacing,params)

    def createIntegrator(self):
        epdiffScalarMomentumMap = FM.EPDiffScalarMomentumMap( self.sz, self.spacing )
        return RK.RK4(epdiffScalarMomentumMap.f)

    def forward(self, phi, I0_source):
        lamIphi1 = self.integrator.solve([self.lam,I0_source, phi], self.tFrom, self.tTo, self.nrOfTimeSteps)
        return lamIphi1[2]


class LDDMMShootingScalarMomentumMapLoss(RegistrationMapLoss):
    def __init__(self,lam,sz,spacing,params):
        super(LDDMMShootingScalarMomentumMapLoss, self).__init__(sz,spacing,params)
        self.lam = lam
        self.smoother = SF.SmootherFactory(self.sz[2::],self.spacing).createSmoother('gaussian')

    def computeRegularizationEnergy(self, I0_source):
        m = utils.computeVectorMomentumFromScalarMomentum_multiNC(self.lam, I0_source, self.sz, self.spacing)
        v = self.smoother.computeSmootherVectorField_multiN(m)
        reg = (v * m).sum() * self.spacing.prod()
        return reg