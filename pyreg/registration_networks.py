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
import image_sampling as IS

import utils

from abc import ABCMeta, abstractmethod

class RegistrationNet(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, sz, spacing, params):
        super(RegistrationNet,self).__init__()
        self.sz = sz
        self.spacing = spacing
        self.params = params
        self.tFrom = 0.
        self.tTo = 1.
        self.nrOfImages = sz[0]
        self.nrOfChannels = sz[1]

    @abstractmethod
    def create_registration_parameters(self):
        pass

    @abstractmethod
    def get_registration_parameters(self):
        pass

    @abstractmethod
    def set_registration_parameters(self, p, sz, spacing):
        pass

    @abstractmethod
    def create_integrator(self):
        pass

    @abstractmethod
    def downsample_registration_parameters(self, desiredSz):
        pass

    @abstractmethod
    def upsample_registration_parameters(self, desiredSz):
        pass

class SVFNet(RegistrationNet):
    def __init__(self,sz,spacing,params):
        super(SVFNet, self).__init__(sz,spacing,params)
        self.v = self.create_registration_parameters()
        self.integrator = self.create_integrator()

    def create_registration_parameters(self):
        return utils.create_ND_vector_field_parameter_multiN(self.sz[2::], self.nrOfImages)

    def get_registration_parameters(self):
        return self.v

    # TODO: improve interface. Do we need sz as input to the constructors?
    def set_registration_parameters(self, p, sz, spacing):
        self.v.data = p.data
        self.sz = sz
        self.spacing = spacing

    def upsample_registration_parameters(self, desiredSz):
        sampler = IS.ResampleImage()
        vUpsampled,upsampled_spacing=sampler.upsample_image_to_size(self.v,self.spacing,desiredSz)
        return vUpsampled,upsampled_spacing

    def downsample_registration_parameters(self, desiredSz):
        sampler = IS.ResampleImage()
        vDownsampled,downsampled_spacing=sampler.downsample_image_to_size(self.v,self.spacing,desiredSz)
        return vDownsampled,downsampled_spacing

class SVFImageNet(SVFNet):
    def __init__(self, sz, spacing, params):
        super(SVFImageNet, self).__init__(sz,spacing,params)

    def create_integrator(self):
        cparams = params[('forward_model',{},'settings for the forward model')]
        advection = FM.AdvectImage(self.sz, self.spacing)
        return RK.RK4(advection.f, advection.u, self.v, cparams)

    def forward(self, I):
        I1 = self.integrator.solve([I], self.tFrom, self.tTo)
        return I1[0]


class RegistrationLoss(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self,sz,spacing,params):
        super(RegistrationLoss, self).__init__()
        self.spacing = spacing
        self.sz = sz
        self.params = params

        self.smFactory = SM.SimilarityMeasureFactory(self.spacing)
        self.similarityMeasure = None

    def add_similarity_measure(self, simName, simMeasure):
        self.smFactory.add_similarity_measure(simName,simMeasure)

    def compute_similarity_energy(self, I1_warped, I1_target):
        if self.similarityMeasure is None:
            self.similarityMeasure = self.smFactory.create_similarity_measure(self.params)
        sim = self.similarityMeasure.compute_similarity_multiNC(I1_warped, I1_target)
        return sim

    @abstractmethod
    def compute_regularization_energy(self, I0_source):
        pass


class RegistrationImageLoss(RegistrationLoss):

    def __init__(self,sz,spacing,params):
        super(RegistrationImageLoss, self).__init__(sz,spacing,params)

    def get_energy(self, I1_warped, I0_source, I1_target):
        sim = self.compute_similarity_energy(I1_warped, I1_target)
        reg = self.compute_regularization_energy(I0_source)
        energy = sim + reg
        return energy, sim, reg

    def forward(self, I1_warped, I0_source, I1_target):
        energy, sim, reg = self.get_energy(I1_warped, I0_source, I1_target)
        return energy


class RegistrationMapLoss(RegistrationLoss):
    def __init__(self, sz, spacing, params):
        super(RegistrationMapLoss, self).__init__(sz, spacing, params)

    def get_energy(self, phi1, I0_source, I1_target):
        I1_warped = utils.compute_warped_image_multiNC(I0_source, phi1)
        sim = self.compute_similarity_energy(I1_warped, I1_target)
        reg = self.compute_regularization_energy(I0_source)
        energy = sim + reg
        return energy, sim, reg

    def forward(self, phi1, I0_source, I1_target):
        energy, sim, reg = self.get_energy(phi1, I0_source, I1_target)
        return energy


class SVFImageLoss(RegistrationImageLoss):
    def __init__(self,v,sz,spacing,params):
        super(SVFImageLoss, self).__init__(sz,spacing,params)
        self.v = v

        cparams = params[('loss',{},'settings for the loss function')]

        self.regularizer = (RF.RegularizerFactory(self.spacing).
                            create_regularizer(cparams))

    def compute_regularization_energy(self, I0_source):
        return self.regularizer.compute_regularizer_multiN(self.v)


class SVFMapNet(SVFNet):
    def __init__(self,sz,spacing,params):
        super(SVFMapNet, self).__init__(sz,spacing,params)

    def create_integrator(self):
        cparams = self.params[('forward_model',{},'settings for the forward model')]
        advectionMap = FM.AdvectMap( self.sz, self.spacing )
        return RK.RK4(advectionMap.f,advectionMap.u,self.v,cparams)

    def forward(self, phi, I0_source):
        phi1 = self.integrator.solve([phi], self.tFrom, self.tTo)
        return phi1[0]


class SVFMapLoss(RegistrationMapLoss):
    def __init__(self,v,sz,spacing,params):
        super(SVFMapLoss, self).__init__(sz,spacing,params)
        self.v = v

        cparams = params[('loss',{},'settings for the loss function')]

        self.regularizer = (RF.RegularizerFactory(self.spacing).
                            create_regularizer(cparams))

    def compute_regularization_energy(self, I0_source):
        return self.regularizer.compute_regularizer_multiN(self.v)


class LDDMMShootingVectorMomentumNet(RegistrationNet):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingVectorMomentumNet, self).__init__(sz,spacing,params)
        self.m = self.create_registration_parameters()
        self.integrator = self.create_integrator()

    def create_registration_parameters(self):
        return utils.create_ND_vector_field_parameter_multiN(self.sz[2::], self.nrOfImages)

    def get_registration_parameters(self):
        return self.m

    def set_registration_parameters(self, p, sz, spacing):
        self.m.data = p.data
        self.sz = sz
        self.spacing = spacing

    def upsample_registration_parameters(self, desiredSz):
        # 1) convert momentum to velocity
        # 2) upsample velocity
        # 3) convert upsampled velocity to momentum

        raise ValueError('Not yet properly implemented. Run at a single scale instead for now')

        cparams = self.params[('forward_model',{},'settings for the forward model')]
        smoother = SF.SmootherFactory(self.sz[2::],self.spacing).create_smoother(cparams)
        v = smoother.smooth_vector_field_multiN(self.m)
        sampler = IS.ResampleImage()
        vUpsampled,upsampled_spacing=sampler.upsample_image_to_size(v,self.spacing,desiredSz)
        smootherInverse = SF.SmootherFactory(desiredSz,upsampled_spacing).create_smoother(cparams)
        mUpsampled = smootherInverse.inverse_smooth_vector_field_multiN(vUpsampled)
        return mUpsampled,upsampled_spacing

    def downsample_registration_parameters(self, desiredSz):
        # 1) convert momentum to velocity
        # 2) downsample velocity
        # 3) convert downsampled velocity to momentum

        raise ValueError('Not yet properly implemented. Run at a single scale instead for now')

        cparams = self.params[('forward_model', {}, 'settings for the forward model')]
        smoother = SF.SmootherFactory(self.sz[2::], self.spacing).create_smoother(cparams)
        v = smoother.smooth_vector_field_multiN(self.m)
        sampler = IS.ResampleImage()
        vDownsampled,downsampled_spacing=sampler.downsample_image_to_size(self.v,self.spacing,desiredSz)
        smootherInverse = SF.SmootherFactory(desiredSz, downsampled_spacing).create_smoother(cparams)
        mDownsampled = smootherInverse.inverse_smooth_vector_field_multiN(vDownsampled)
        return mDownsampled, downsampled_spacing

class LDDMMShootingVectorMomentumImageNet(LDDMMShootingVectorMomentumNet):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingVectorMomentumImageNet, self).__init__(sz,spacing,params)

    def create_integrator(self):
        cparams = self.params[('forward_model',{},'settings for the forward model')]
        epdiffImage = FM.EPDiffImage( self.sz, self.spacing, cparams )
        return RK.RK4(epdiffImage.f,None,None,cparams)

    def forward(self, I):
        mI1 = self.integrator.solve([self.m,I], self.tFrom, self.tTo)
        return mI1[1]


class LDDMMShootingVectorMomentumImageLoss(RegistrationImageLoss):
    def __init__(self,m,sz,spacing,params):
        super(LDDMMShootingVectorMomentumImageLoss, self).__init__(sz,spacing,params)
        self.m = m

        cparams = params[('forward_model',{},'settings for the forward model')]
        self.smoother = SF.SmootherFactory(self.sz[2::],self.spacing).create_smoother(cparams)

    def compute_regularization_energy(self, I0_source):
        v = self.smoother.smooth_vector_field_multiN(self.m)
        reg = (v * self.m).sum() * self.spacing.prod()
        return reg


class LDDMMShootingVectorMomentumMapNet(LDDMMShootingVectorMomentumNet):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingVectorMomentumMapNet, self).__init__(sz,spacing,params)

    def create_integrator(self):
        cparams = self.params[('forward_model',{},'settings for the forward model')]
        epdiffMap = FM.EPDiffMap( self.sz, self.spacing, cparams )
        return RK.RK4(epdiffMap.f,None,None,cparams)

    def forward(self, phi, I0_source):
        mphi1 = self.integrator.solve([self.m,phi], self.tFrom, self.tTo)
        return mphi1[1]


class LDDMMShootingVectorMomentumMapLoss(RegistrationMapLoss):
    def __init__(self,m,sz,spacing,params):
        super(LDDMMShootingVectorMomentumMapLoss, self).__init__(sz,spacing,params)
        self.m = m

        cparams = params[('forward_model',{},'settings for the forward model')]
        self.smoother = SF.SmootherFactory(self.sz[2::],self.spacing).create_smoother(cparams)

    def compute_regularization_energy(self, I0_source):
        v = self.smoother.smooth_vector_field_multiN(self.m)
        reg = (v * self.m).sum() * self.spacing.prod()
        return reg


class LDDMMShootingScalarMomentumNet(RegistrationNet):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingScalarMomentumNet, self).__init__(sz,spacing,params)
        self.lam = self.create_registration_parameters()
        self.integrator = self.create_integrator()

    def create_registration_parameters(self):
        return utils.create_ND_scalar_field_parameter_multiNC(self.sz[2::], self.nrOfImages, self.nrOfChannels)

    def get_registration_parameters(self):
        return self.lam

    def set_registration_parameters(self, p, sz, spacing):
        self.lam.data = p.data
        self.sz = sz
        self.spacing = spacing

    def upsample_registration_parameters(self, desiredSz):
        # 1) convert scalar momentum to velocity
        # 2) upsample velocity
        # 3) convert upsampled velocity to scalar momentum
        # (for this find the scalar momentum which can generate the smoothed velocity)

        raise ValueError('Not yet properly implemented. Run at a single scale instead for now')

        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(self.lam, I0_source, self.sz, self.spacing)

        cparams = self.params[('forward_model',{},'settings for the forward model')]
        smoother = SF.SmootherFactory(self.sz[2::],self.spacing).create_smoother(cparams)
        v = smoother.smooth_vector_field_multiN(m)

        sampler = IS.ResampleImage()
        vUpsampled,upsampled_spacing=sampler.upsample_image_to_size(v,self.spacing,desiredSz)

        mUpsampled = None
        raise ValueError('TODO: implement velocity field to scalar momentum')

        return mUpsampled,upsampled_spacing

    def downsample_registration_parameters(self, desiredSz):
        # 1) convert scalar momentum to velocity
        # 2) downsample velocity
        # 3) convert downsampled velocity to scalar momentum
        # (for this find the scalar momentum which can generate the smoothed velocity)

        raise ValueError('Not yet properly implemented. Run at a single scale instead for now')

        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(self.lam, I0_source, self.sz, self.spacing)

        cparams = self.params[('forward_model', {}, 'settings for the forward model')]
        smoother = SF.SmootherFactory(self.sz[2::], self.spacing).create_smoother(cparams)
        v = smoother.smooth_vector_field_multiN(m)

        sampler = IS.ResampleImage()
        vDownsampled,downsampled_spacing=sampler.downsample_image_to_size(self.v,self.spacing,desiredSz)

        mUpsampled = None
        raise ValueError('TODO: implement velocity field to scalar momentum')

        return mDownsampled, downsampled_spacing

class LDDMMShootingScalarMomentumImageNet(LDDMMShootingScalarMomentumNet):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingScalarMomentumImageNet, self).__init__(sz,spacing,params)

    def create_integrator(self):
        cparams = self.params[('forward_model',{},'settings for the forward model')]
        epdiffScalarMomentumImage = FM.EPDiffScalarMomentumImage( self.sz, self.spacing, cparams )
        return RK.RK4(epdiffScalarMomentumImage.f,None,None,cparams)

    def forward(self, I):
        lamI1 = self.integrator.solve([self.lam,I], self.tFrom, self.tTo, self.nrOfTimeSteps)
        return lamI1[1]


class LDDMMShootingScalarMomentumImageLoss(RegistrationImageLoss):
    def __init__(self,lam,sz,spacing,params):
        super(LDDMMShootingScalarMomentumImageLoss, self).__init__(sz,spacing,params)
        self.lam = lam

        cparams = params[('forward_model',{},'settings for the forward model')]
        self.smoother = SF.SmootherFactory(self.sz[2::],self.spacing).create_smoother(cparams)

    def compute_regularization_energy(self, I0_source):
        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(self.lam, I0_source, self.sz, self.spacing)
        v = self.smoother.smooth_vector_field_multiN(m)
        reg = (v * m).sum() * self.spacing.prod()
        return reg


class LDDMMShootingScalarMomentumMapNet(LDDMMShootingScalarMomentumNet):
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingScalarMomentumMapNet, self).__init__(sz,spacing,params)

    def create_integrator(self):
        cparams = self.params[('forward_model',{},'settings for the forward model')]
        epdiffScalarMomentumMap = FM.EPDiffScalarMomentumMap( self.sz, self.spacing, cparams )
        return RK.RK4(epdiffScalarMomentumMap.f,None,None,cparams)

    def forward(self, phi, I0_source):
        lamIphi1 = self.integrator.solve([self.lam,I0_source, phi], self.tFrom, self.tTo)
        return lamIphi1[2]


class LDDMMShootingScalarMomentumMapLoss(RegistrationMapLoss):
    def __init__(self,lam,sz,spacing,params):
        super(LDDMMShootingScalarMomentumMapLoss, self).__init__(sz,spacing,params)
        self.lam = lam

        cparams = params[('forward_model',{},'settings for the forward model')]
        self.smoother = SF.SmootherFactory(self.sz[2::],self.spacing).create_smoother(cparams)

    def compute_regularization_energy(self, I0_source):
        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(self.lam, I0_source, self.sz, self.spacing)
        v = self.smoother.smooth_vector_field_multiN(m)
        reg = (v * m).sum() * self.spacing.prod()
        return reg