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

class SVFNet(nn.Module):
    def __init__(self,sz,spacing,params):
        super(SVFNet, self).__init__()
        self.spacing = spacing
        self.nrOfTimeSteps = utils.getpar(params,'numberOfTimeSteps',10)
        self.tFrom = 0.
        self.tTo = 1.
        self.v = utils.createNDVectorFieldParameter( sz )
        self.advection = FM.AdvectImage( sz, self.spacing )
        self.integrator = RK.RK4(self.advection.f,self.advection.u,self.v)

    def forward(self, I):
        I1 = self.integrator.solve([I], self.tFrom, self.tTo, self.nrOfTimeSteps)
        return I1[0]


class SVFLoss(nn.Module):
    def __init__(self,v,sz,spacing,params):
        super(SVFLoss, self).__init__()
        self.v = v
        self.spacing = spacing
        self.sz = sz
        self.regularizer = (RF.RegularizerFactory(self.spacing).
                            createRegularizer(utils.getpar(params,'regularizer','helmholtz'),params))
        self.similarityMeasure = (SM.SimilarityMeasureFactory(self.spacing).
                                  createSimilarityMeasure(utils.getpar(params,'similarityMeasure','ssd'), params))

    def getEnergy(self, I1_warped, I0_source, I1_target):
        sim = self.similarityMeasure.computeSimilarity(I1_warped, I1_target)
        reg = self.regularizer.computeRegularizer(self.v)
        energy = sim + reg
        return energy, sim, reg

    def forward(self, I1_warped, I0_source, I1_target):
        energy, sim, reg = self.getEnergy( I1_warped, I0_source, I1_target)
        return energy

class SVFNetMap(nn.Module):
    def __init__(self,sz,spacing,params):
        super(SVFNetMap, self).__init__()
        self.spacing = spacing
        self.nrOfTimeSteps = utils.getpar(params,'numberOfTimeSteps',10)
        self.tFrom = 0.
        self.tTo = 1.
        self.v = utils.createNDVectorFieldParameter( sz )
        self.advectionMap = FM.AdvectMap( sz, self.spacing )
        self.integrator = RK.RK4(self.advectionMap.f,self.advectionMap.u,self.v)

    def forward(self, phi, I0_source):
        phi1 = self.integrator.solve([phi], self.tFrom, self.tTo, self.nrOfTimeSteps)
        return phi1[0]


class SVFLossMap(nn.Module):
    def __init__(self,v,sz,spacing,params):
        super(SVFLossMap, self).__init__()
        self.v = v
        self.spacing = spacing
        self.sz = sz
        self.regularizer = (RF.RegularizerFactory(self.spacing).
                            createRegularizer(utils.getpar(params, 'regularizer', 'helmholtz'), params))
        self.similarityMeasure = (SM.SimilarityMeasureFactory(self.spacing).
                                  createSimilarityMeasure(utils.getpar(params, 'similarityMeasure', 'ssd'), params))

    def getEnergy(self, phi1, I0_source, I1_target):
        I1_warped = utils.computeWarpedImage(I0_source, phi1)
        sim = self.similarityMeasure.computeSimilarity(I1_warped, I1_target)
        reg = self.regularizer.computeRegularizer(self.v)
        energy = sim + reg
        return energy, sim, reg

    def forward(self, phi1, I0_source, I1_target):
        energy, sim, reg = self.getEnergy(phi1, I0_source, I1_target)
        return energy

class LDDMMShootingNet(nn.Module):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingNet, self).__init__()
        self.spacing = spacing
        self.nrOfTimeSteps = utils.getpar(params,'numberOfTimeSteps',10)
        self.tFrom = 0.
        self.tTo = 1.
        self.m = utils.createNDVectorFieldParameter( sz )
        self.epdiffImage = FM.EPDiffImage( sz, self.spacing )
        self.integrator = RK.RK4(self.epdiffImage.f)

    def forward(self, I):
        mI1 = self.integrator.solve([self.m,I], self.tFrom, self.tTo, self.nrOfTimeSteps)
        return mI1[1]


class LDDMMShootingLoss(nn.Module):
    def __init__(self,m,sz,spacing,params):
        super(LDDMMShootingLoss, self).__init__()
        self.m = m
        self.spacing = spacing
        self.sz = sz
        self.smoother = SF.SmootherFactory(self.sz,self.spacing).createSmoother('gaussian')
        self.similarityMeasure = (SM.SimilarityMeasureFactory(self.spacing).
                                  createSimilarityMeasure(utils.getpar(params,'similarityMeasure','ssd'), params))

    def getEnergy(self, I1_warped, I0_source, I1_target):
        sim = self.similarityMeasure.computeSimilarity(I1_warped,I1_target)
        v = self.smoother.computeSmootherVectorField(self.m)
        reg = (v * self.m).sum() * self.spacing.prod()
        energy = sim + reg
        return energy, sim, reg

    def forward(self, I1_warped, I0_source, I1_target):
        energy, sim, reg = self.getEnergy(I1_warped, I0_source, I1_target)
        return energy


class LDDMMShootingNetMap(nn.Module):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingNetMap, self).__init__()
        self.spacing = spacing
        self.nrOfTimeSteps = utils.getpar(params,'numberOfTimeSteps',10)
        self.tFrom = 0.
        self.tTo = 1.
        self.m = utils.createNDVectorFieldParameter( sz )
        self.epdiffMap = FM.EPDiffMap( sz, self.spacing )
        self.integrator = RK.RK4(self.epdiffMap.f)

    def forward(self, phi, I0_source):
        mphi1 = self.integrator.solve([self.m,phi], self.tFrom, self.tTo, self.nrOfTimeSteps)
        return mphi1[1]


class LDDMMShootingLossMap(nn.Module):
    def __init__(self,m,sz,spacing,params):
        super(LDDMMShootingLossMap, self).__init__()
        self.m = m
        self.spacing = spacing
        self.sz = sz
        self.smoother = SF.SmootherFactory(self.sz,self.spacing).createSmoother('gaussian')
        self.similarityMeasure = (SM.SimilarityMeasureFactory(self.spacing).
                                  createSimilarityMeasure(utils.getpar(params, 'similarityMeasure', 'ssd'), params))

    def getEnergy(self, phi1, I0_source, I1_target):
        I1_warped = utils.computeWarpedImage(I0_source, phi1)
        sim = self.similarityMeasure.computeSimilarity(I1_warped, I1_target)
        v = self.smoother.computeSmootherVectorField(self.m)
        reg = (v * self.m).sum() * self.spacing.prod()
        energy = sim + reg
        return energy, sim, reg

    def forward(self, phi1, I0_source, I1_target):
        energy, sim, reg = self.getEnergy(phi1, I0_source, I1_target)
        return energy

class LDDMMShootingScalarMomentumNet(nn.Module):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingScalarMomentumNet, self).__init__()
        self.spacing = spacing
        self.nrOfTimeSteps = utils.getpar(params,'numberOfTimeSteps',10)
        self.tFrom = 0.
        self.tTo = 1.
        self.lam = utils.createNDScalarFieldParameter( sz )
        self.epdiffScalarMomentumImage = FM.EPDiffScalarMomentumImage( sz, self.spacing )
        self.integrator = RK.RK4(self.epdiffScalarMomentumImage.f)

    def forward(self, I):
        lamI1 = self.integrator.solve([self.lam,I], self.tFrom, self.tTo, self.nrOfTimeSteps)
        return lamI1[1]


class LDDMMShootingScalarMomentumLoss(nn.Module):
    def __init__(self,lam,sz,spacing,params):
        super(LDDMMShootingScalarMomentumLoss, self).__init__()
        self.lam = lam
        self.spacing = spacing
        self.sz = sz
        self.smoother = SF.SmootherFactory(self.sz,self.spacing).createSmoother('gaussian')
        self.similarityMeasure = (SM.SimilarityMeasureFactory(self.spacing).
                                  createSimilarityMeasure(utils.getpar(params,'similarityMeasure','ssd'), params))

    def getEnergy(self, I1_warped, I0_source, I1_target):
        sim = self.similarityMeasure.computeSimilarity(I1_warped,I1_target)
        m = utils.computeVectorMomentumFromScalarMomentum(self.lam,I0_source,self.sz,self.spacing)
        v = self.smoother.computeSmootherVectorField(m)
        reg = (v * m).sum() * self.spacing.prod()
        energy = sim + reg
        return energy, sim, reg

    def forward(self, I1_warped, I0_source, I1_target):
        energy, sim, reg = self.getEnergy(I1_warped, I0_source, I1_target)
        return energy

class LDDMMShootingScalarMomentumNetMap(nn.Module):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingScalarMomentumNetMap, self).__init__()
        self.spacing = spacing
        self.nrOfTimeSteps = utils.getpar(params,'numberOfTimeSteps',10)
        self.tFrom = 0.
        self.tTo = 1.
        self.lam = utils.createNDScalarFieldParameter( sz )
        self.epdiffScalarMomentumMap = FM.EPDiffScalarMomentumMap( sz, self.spacing )
        self.integrator = RK.RK4(self.epdiffScalarMomentumMap.f)

    def forward(self, phi, I0_source):
        lamIphi1 = self.integrator.solve([self.lam,I0_source, phi], self.tFrom, self.tTo, self.nrOfTimeSteps)
        return lamIphi1[2]


class LDDMMShootingScalarMomentumLossMap(nn.Module):
    def __init__(self,lam,sz,spacing,params):
        super(LDDMMShootingScalarMomentumLossMap, self).__init__()
        self.lam = lam
        self.spacing = spacing
        self.sz = sz
        self.smoother = SF.SmootherFactory(self.sz,self.spacing).createSmoother('gaussian')
        self.similarityMeasure = (SM.SimilarityMeasureFactory(self.spacing).
                                  createSimilarityMeasure(utils.getpar(params, 'similarityMeasure', 'ssd'), params))

    def getEnergy(self, phi1, I0_source, I1_target):
        I1_warped = utils.computeWarpedImage(I0_source, phi1)
        sim = self.similarityMeasure.computeSimilarity(I1_warped, I1_target)
        m = utils.computeVectorMomentumFromScalarMomentum(self.lam,I0_source,self.sz,self.spacing)
        v = self.smoother.computeSmootherVectorField(m)
        reg = (v * m).sum() * self.spacing.prod()
        energy = sim + reg
        return energy, sim, reg

    def forward(self, phi1, I0_source, I1_target):
        energy, sim, reg = self.getEnergy(phi1, I0_source, I1_target)
        return energy