"""
Defines different registration methods as pyTorch networks.
Currently implemented:
    * SVFImageNet: image-based stationary velocity field
    * SVFMapNet: map-based stationary velocity field
    * SVFQuasiMomentumImageNet: EXPERIMENTAL (not working yet): SVF which is parameterized by a momentum
    * SVFScalarMomentumImageNet: image-based SVF using the scalar-momentum parameterization
    * SVFScalarMomentumMapNet: map-based SVF using the scalar-momentum parameterization
    * SVFVectorMomentumImageNet: image-based SVF using the vector-momentum parameterization
    * SVFVectorMomentumMapNet: map-based SVF using the vector-momentum parameterization
    * LDDMMShootingVectorMomentumImageNet: image-based LDDMM using the vector-momentum parameterization
    * LDDMMShootingVectorMomentumImageNet: map-based LDDMM using the vector-momentum parameterization
    * LDDMMShootingScalarMomentumImageNet: image-based LDDMM using the scalar-momentum parameterization
    * LDDMMShootingScalarMomentumImageNet: map-based LDDMM using the scalar-momentum parameterization
"""

import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from torch.nn.parameter import Parameter

import rungekutta_integrators as RK
import forward_models as FM
from data_wrapper import AdaptVal
import regularizer_factory as RF
import similarity_measure_factory as SM

import smoother_factory as SF
import image_sampling as IS

from data_wrapper import MyTensor

import utils

from abc import ABCMeta, abstractmethod

class RegistrationNet(nn.Module):
    """
    Abstract base-class for all the registration networks
    """
    __metaclass__ = ABCMeta

    def __init__(self, sz, spacing, params):
        """
        Constructor
        
        :param sz: image size (BxCxXxYxZ format) 
        :param spacing: spatial spacing, e.g., [0.1,0.1,0.2]
        :param params: ParameterDict() object to hold general parameters
        """
        super(RegistrationNet,self).__init__()
        self.sz = sz
        """image size"""
        self.spacing = spacing
        """image spacing"""
        self.params = params
        """ParameterDict() object for the parameters"""
        self.nrOfImages = sz[0]
        """the number of images, i.e., the batch size B"""
        self.nrOfChannels = sz[1]
        """the number of image channels, i.e., C"""

    def get_variables_to_transfer_to_loss_function(self):
        """
        This is a function that can be overwritten by models to allow to return variables which are also 
        needed for the computation of the loss function. Returns None by default, but can for example be used
        to pass parameters or smoothers which are needed for the model itself and its loss. By convention
        these variables should be returned as a dictionary.
        :return: 
        """
        return None

    def get_custom_optimizer_output_string(self):
        """
        Can be overwritten by a method to allow for additional optimizer output (on top of the energy values)
        :return: 
        """
        return ''

    @abstractmethod
    def create_registration_parameters(self):
        """
        Abstract method to create the registration parameters over which should be optimized. They need to be of type torch Parameter() 
        """
        pass

    @abstractmethod
    def get_registration_parameters(self):
        """
        Abstract method to return the registration parameters
        
        :return: returns the registration parameters 
        """
        pass

    @abstractmethod
    def set_registration_parameters(self, p, sz, spacing):
        """
        Abstract method to set the registration parameters externally. This can for example be useful when the optimizer should be initialized at a specific value
        
        :param p: parameter to be set 
        :param sz: size of the image the parameter corresponds to 
        :param spacing: spacing of the image the parameter corresponds to 
        """
        pass

    def downsample_registration_parameters(self, desiredSz):
        """
        Method to downsample the registration parameters spatially to a desired size. Should be overwritten by a derived class. 
        
        :param desiredSz: desired size in XxZxZ format, e.g., [50,100,40]
        :return: should return a tuple (downsampled_image,downsampled_spacing) 
        """
        raise NotImplementedError

    def upsample_registration_parameters(self, desiredSz):
        """
        Method to upsample the registration parameters spatially to a desired size. Should be overwritten by a derived class. 

        :param desiredSz: desired size in XxZxZ format, e.g., [50,100,40]
        :return: should return a tuple (upsampled_image,upsampled_spacing) 
        """
        raise NotImplementedError

    def get_parameter_image_and_name_to_visualize(self):
        """
        Convenience function to specify an image that should be visualized including its caption. 
        This will typically be related to the parameter of a model. This method should be overwritten by a derived class
         
        :return: should return a tuple (image,desired_caption)
        """
        # not defined yet
        return None,None


class RegistrationNetDisplacement(RegistrationNet):
    """
        Abstract base-class for all the registration networks without time-integration
        which directly estimate a deformation field.
        """

    def __init__(self, sz, spacing, params):
        """
        Constructor

        :param sz: image size (BxCxXxYxZ format) 
        :param spacing: spatial spacing, e.g., [0.1,0.1,0.2]
        :param params: ParameterDict() object to hold general parameters
        """
        super(RegistrationNetDisplacement, self).__init__(sz,spacing,params)

        self.d = self.create_registration_parameters()
        """displacement field that will be optimized over"""

    def create_registration_parameters(self):
        """
        Creates the displacement field that is being optimized over
    
        :return: displacement field parameter 
        """
        return utils.create_ND_vector_field_parameter_multiN(self.sz[2::], self.nrOfImages)


    def get_registration_parameters(self):
        """
        Returns the displacement field parameter
    
        :return: dispalcement field parameter 
        """
        return self.d


    def set_registration_parameters(self, p, sz, spacing):
        """
        Sets the displacement field registration parameter
    
        :param p: displacement field 
        :param sz: size of the corresponding image
        :param spacing: spacing of the corresponding image
        """
        self.d.data = p.data
        self.sz = sz
        self.spacing = spacing


    def get_parameter_image_and_name_to_visualize(self):
        """
        Returns the displacement field parameter magnitude image and a name
    
        :return: Returns the tuple (displacement_magnitude_image,name) 
        """
        name = '|d|'
        par_image = ((self.d[:, ...] ** 2).sum(1)) ** 0.5  # assume BxCxXxYxZ format
        return par_image, name


    def upsample_registration_parameters(self, desiredSz):
        """
        Upsamples the displacement field to a desired size
    
        :param desiredSz: desired size of the upsampled displacement field 
        :return: returns a tuple (upsampled_image,upsampled_spacing)
        """
        sampler = IS.ResampleImage()
        dUpsampled, upsampled_spacing = sampler.upsample_image_to_size(self.d, self.spacing, desiredSz)
        return dUpsampled, upsampled_spacing


    def downsample_registration_parameters(self, desiredSz):
        """
        Downsamples the displacemebt field to a desired size
    
        :param desiredSz: desired size of the downsampled displacement field 
        :return: returns a tuple (downsampled_image,downsampled_spacing)
        """
        sampler = IS.ResampleImage()
        dDownsampled, downsampled_spacing = sampler.downsample_image_to_size(self.d, self.spacing, desiredSz)
        return dDownsampled, downsampled_spacing

    def forward(self, phi, I0_source):
        """
        Solved the map-based equation forward

        :param phi: initial condition for the map
        :param I0_source: not used
        :return: returns the map with the displacement subtracted
        """
        return (phi-self.d)


class RegistrationNetTimeIntegration(RegistrationNet):
    """
        Abstract base-class for all the registration networks with time-integration
        """
    __metaclass__ = ABCMeta

    def __init__(self, sz, spacing, params):
        """
        Constructor

        :param sz: image size (BxCxXxYxZ format) 
        :param spacing: spatial spacing, e.g., [0.1,0.1,0.2]
        :param params: ParameterDict() object to hold general parameters
        """
        super(RegistrationNetTimeIntegration, self).__init__(sz,spacing,params)

        self.tFrom = 0.
        """time to solve a model from"""
        self.tTo = 1.
        """time to solve a model to"""

    @abstractmethod
    def create_integrator(self):
        """
        Abstract method to create an integrator for time-integration of a model
        """
        pass

class SVFNet(RegistrationNetTimeIntegration):
    """
    Base class for SVF-type registrations. Provides a velocity field (as a parameter) and an integrator
    """
    def __init__(self,sz,spacing,params):
        super(SVFNet, self).__init__(sz,spacing,params)
        self.v = self.create_registration_parameters()
        """velocity field that will be optimized over"""
        self.integrator = self.create_integrator()
        """integrator to do the time-integration"""

    def create_registration_parameters(self):
        """
        Creates the velocity field that is being optimized over
        
        :return: velocity field parameter 
        """
        return utils.create_ND_vector_field_parameter_multiN(self.sz[2::], self.nrOfImages)

    def get_registration_parameters(self):
        """
        Returns the velocity field parameter
        
        :return: velocity field parameter 
        """
        return self.v

    def set_registration_parameters(self, p, sz, spacing):
        """
        Sets the velocity field registration parameter
        
        :param p: velocity field 
        :param sz: size of the corresponding image
        :param spacing: spacing of the corresponding image
        """
        self.v.data = p.data
        self.sz = sz
        self.spacing = spacing

    def get_parameter_image_and_name_to_visualize(self):
        """
        Returns the velocity field parameter magnitude image and a name
        
        :return: Returns the tuple (velocity_magnitude_image,name) 
        """
        name = '|v|'
        par_image = ((self.v[:,...]**2).sum(1))**0.5 # assume BxCxXxYxZ format
        return par_image,name

    def upsample_registration_parameters(self, desiredSz):
        """
        Upsamples the velocity field to a desired size
        
        :param desiredSz: desired size of the upsampled velocity field 
        :return: returns a tuple (upsampled_image,upsampled_spacing)
        """
        sampler = IS.ResampleImage()
        vUpsampled,upsampled_spacing=sampler.upsample_image_to_size(self.v,self.spacing,desiredSz)
        return vUpsampled,upsampled_spacing

    def downsample_registration_parameters(self, desiredSz):
        """
        Downsamples the velocity field to a desired size

        :param desiredSz: desired size of the downsampled velocity field 
        :return: returns a tuple (downsampled_image,downsampled_spacing)
        """
        sampler = IS.ResampleImage()
        vDownsampled,downsampled_spacing=sampler.downsample_image_to_size(self.v,self.spacing,desiredSz)
        return vDownsampled,downsampled_spacing

class SVFImageNet(SVFNet):
    """
    Specialization for SVF-based image registration
    """
    def __init__(self, sz, spacing, params):
        super(SVFImageNet, self).__init__(sz,spacing,params)

    def create_integrator(self):
        """
        Creates an integrator for the advection equation of the image
        
        :return: returns this integrator 
        """
        cparams = self.params[('forward_model',{},'settings for the forward model')]
        advection = FM.AdvectImage(self.sz, self.spacing)
        return RK.RK4(advection.f, advection.u, self.v, cparams)

    def forward(self, I):
        """
        Solves the image-based advection equation
        
        :param I: initial condition for the image 
        :return: returns the image at the final time (tTo)
        """
        I1 = self.integrator.solve([I], self.tFrom, self.tTo)
        return I1[0]


class SVFQuasiMomentumNet(RegistrationNetTimeIntegration):
    """
    Attempt at parameterizing SVF with a momentum-like vector field (EXPERIMENTAL, not working yet)
    """
    def __init__(self,sz,spacing,params):
        super(SVFQuasiMomentumNet, self).__init__(sz,spacing,params)
        self.m = self.create_registration_parameters()
        """momentum parameter"""
        cparams = params[('forward_model', {}, 'settings for the forward model')]
        self.smoother = SF.SmootherFactory(self.sz[2::], self.spacing).create_smoother(cparams)
        """smoother to go from momentum to velocity"""
        self.smoother_params = self.smoother.get_optimization_parameters()
        """smoother parameters to be optimized over if supported by smoother"""
        self.v = self.smoother.smooth(self.m)
        """corresponding velocity field"""

        self.integrator = self.create_integrator()
        """integrator to solve the forward model"""

    def get_custom_optimizer_output_string(self):
        return self.smoother.get_custom_optimizer_output_string()

    def create_registration_parameters(self):
        """
        Creates the registration parameters (the momentum field) and returns them
        
        :return: momentum field 
        """
        return utils.create_ND_vector_field_parameter_multiN(self.sz[2::], self.nrOfImages)

    def get_registration_parameters(self):
        """
        Returns the registration parameters (the momentum field)
        
        :return: momentum field 
        """
        return self.m

    def set_registration_parameters(self, p, sz, spacing):
        """
        Sets the registration parameters (the momentum field)
        
        :param p: momentum field 
        :param sz: corresponding image size
        :param spacing: corresponding image spacing
        """
        self.m.data = p.data
        self.sz = sz
        self.spacing = spacing

    def get_parameter_image_and_name_to_visualize(self):
        """
        Returns the momentum magnitude image and :math:`|m|` as the image caption
        
        :return: Returns a tuple (magnitude_m,name) 
        """
        name = '|m|'
        par_image = ((self.m[:,...]**2).sum(1))**0.5 # assume BxCxXxYxZ format
        return par_image,name

    def upsample_registration_parameters(self, desiredSz):
        sampler = IS.ResampleImage()
        mUpsampled,upsampled_spacing=sampler.upsample_image_to_size(self.m,self.spacing,desiredSz)
        return mUpsampled,upsampled_spacing

    def downsample_registration_parameters(self, desiredSz):
        sampler = IS.ResampleImage()
        mDownsampled,downsampled_spacing=sampler.downsample_image_to_size(self.m,self.spacing,desiredSz)
        return mDownsampled,downsampled_spacing

class SVFQuasiMomentumImageNet(SVFQuasiMomentumNet):
    """
    Specialization for image registation
    """
    def __init__(self, sz, spacing, params):
        super(SVFQuasiMomentumImageNet, self).__init__(sz,spacing,params)

    def create_integrator(self):
        """
        Creates the integrator that solve the advection equation (based on the smoothed momentum)
        :return: returns this integrator
        """
        cparams = self.params[('forward_model',{},'settings for the forward model')]
        advection = FM.AdvectImage(self.sz, self.spacing)
        return RK.RK4(advection.f, advection.u, self.v, cparams)

    def forward(self, I):
        """
        Solves the model by first smoothing the momentum field and then using it as the velocity for the advection equation
        
        :param I: initial condition for the image 
        :return: returns the image at the final time point (tTo)
        """
        self.smoother.smooth(self.m,self.v)
        I1 = self.integrator.solve([I], self.tFrom, self.tTo)
        return I1[0]

class RegistrationLoss(nn.Module):
    """
    Abstract base class to define a loss function for image registration
    """
    __metaclass__ = ABCMeta

    def __init__(self,sz,spacing,params):
        """
        Constructor
        
        :param sz: image size 
        :param spacing: image spacing
        :param params: ParameterDict() object to hold and keep track of general parameters
        """
        super(RegistrationLoss, self).__init__()
        self.spacing = spacing
        """image spacing"""
        self.sz = sz
        """image size"""
        self.params = params
        """ParameterDict() paramters"""

        self.smFactory = SM.SimilarityMeasureFactory(self.spacing)
        """factory to create similarity measures on the fly"""
        self.similarityMeasure = None
        """the similarity measure itself"""


    def add_similarity_measure(self, simName, simMeasure):
        """
        To add a custom similarity measure to the similarity measure factory
        
        :param simName: desired name of the similarity measure (string) 
        :param simMeasure: similarity measure itself (to instantiate an object)
        """
        self.smFactory.add_similarity_measure(simName,simMeasure)

    def compute_similarity_energy(self, I1_warped, I1_target, variables_from_forward_model=None):
        """
        Computing the image matching energy based on the selected similarity measure
        
        :param I1_warped: warped image at time tTo 
        :param I1_target: target image to register to 
        :param variables_from_forward_model: allows passing in additional variables (intended to pass variables between the forward modell and the loss function)
        :return: returns the value for image similarity energy
        """
        if self.similarityMeasure is None:
            self.similarityMeasure = self.smFactory.create_similarity_measure(self.params)
        sim = self.similarityMeasure.compute_similarity_multiNC(I1_warped, I1_target)
        return sim

    @abstractmethod
    def compute_regularization_energy(self, I0_source, variables_from_forward_model=None):
        """
        Abstract method computing the regularization energy based on the registration parameters and (if desired) the initial image
        
        :param I0_source: Initial image 
        :param variables_from_forward_model: allows passing in additional variables (intended to pass variables between the forward modell and the loss function)
        :return: should return the valie for the regularization energy
        """
        pass


class RegistrationImageLoss(RegistrationLoss):
    """
    Specialization for image-based registration losses
    """

    def __init__(self,sz,spacing,params):
        super(RegistrationImageLoss, self).__init__(sz,spacing,params)

    def get_energy(self, I1_warped, I0_source, I1_target, variables_from_forward_model=None):
        """
        Computes the overall registration energy as E = E_sim + E_reg
        
        :param I1_warped: warped image 
        :param I0_source: source image
        :param I1_target: target image
        :param variables_from_forward_model: allows passing in additional variables (intended to pass variables between the forward modell and the loss function)
        :return: return the energy value
        """
        sim = self.compute_similarity_energy(I1_warped, I1_target, variables_from_forward_model)
        reg = self.compute_regularization_energy(I0_source, variables_from_forward_model)
        energy = sim + reg
        return energy, sim, reg

    def forward(self, I1_warped, I0_source, I1_target, variables_from_forward_model=None):
        """
        Computes the loss by evaluating the energy
        :param I1_warped: warped image
        :param I0_source: source image
        :param I1_target: target image
        :param variables_from_forward_model: allows passing in additional variables (intended to pass variables between the forward modell and the loss function)
        :return: registration energy
        """
        energy, sim, reg = self.get_energy(I1_warped, I0_source, I1_target, variables_from_forward_model)
        return energy


class RegistrationMapLoss(RegistrationLoss):
    """
    Specialization for map-based registration losses
    """
    def __init__(self, sz, spacing, params):
        super(RegistrationMapLoss, self).__init__(sz, spacing, params)
        cparams = params[('loss', {}, 'settings for the loss function')]
        self.display_max_displacement = cparams[('display_max_displacement',False,'displays the current maximal displacement')]
        self.limit_displacement = cparams[('limit_displacement',False,'[True/False] if set to true limits the maximal displacement based on the max_displacement_setting')]
        max_displacement = cparams[('max_displacement',0.05,'Max displacement penalty added to loss function of limit_displacement set to True')]
        self.max_displacement_sqr = max_displacement**2

    def get_energy(self, phi0, phi1, I0_source, I1_target, lowres_I0, variables_from_forward_model=None ):
        """
        Compute the energy by warping the source image via the map and then comparing it to the target image
        
        :param phi0: map (initial map from which phi1 is computed by integration; likely the identity map) 
        :param phi1: map (mapping the source image to the target image, defined in the space of the target image) 
        :param I0_source: source image
        :param I1_target: target image
        :param lowres_I0: for map with reduced resolution this is the downsampled source image, may be needed to compute the regularization energy
        :param variables_from_forward_model: allows passing in additional variables (intended to pass variables between the forward modell and the loss function)
        :return: registration energy
        """
        I1_warped = utils.compute_warped_image_multiNC(I0_source, phi1)
        sim = self.compute_similarity_energy(I1_warped, I1_target, variables_from_forward_model)
        if lowres_I0 is not None:
            reg = self.compute_regularization_energy(lowres_I0, variables_from_forward_model)
        else:
            reg = self.compute_regularization_energy(I0_source, variables_from_forward_model)


        if self.limit_displacement:
            # first compute squared displacement
            dispSqr = ((phi1-phi0)**2).sum(1)
            if self.display_max_displacement==True:
                dispMax = ( torch.sqrt( dispSqr ) ).max()
                print( 'Max disp = ' + str( utils.t2np( dispMax )))
            sz = dispSqr.size()
            dispPenalty = ( torch.max( ( dispSqr - self.max_displacement_sqr ), Variable( MyTensor(sz).zero_(), requires_grad=False ) )).sum()
            reg = reg + dispPenalty
        else:
            if self.display_max_displacement==True:
                dispMax = ( torch.sqrt( ((phi1-phi0)**2).sum(1) ) ).max()
                print( 'Max disp = ' + str( utils.t2np( dispMax )))
        factor=1
        reg = reg *factor
        energy = sim + reg
        return energy, sim, reg

    def forward(self, phi0, phi1, I0_source, I1_target, lowres_I0, variables_from_forward_model=None ):
        """
        Compute the loss function value by evaluating the registration energy
        
        :param phi0: map (initial map from which phi1 is computed by integration; likely the identity map) 
        :param phi1:  map (mapping the source image to the target image, defined in the space of the target image) 
        :param I0_source: source image
        :param I1_target: target image
        :param lowres_I0: for map with reduced resolution this is the downsampled source image, may be needed to compute the regularization energy
        :param variables_from_forward_model: allows passing in additional variables (intended to pass variables between the forward modell and the loss function)
        :return: returns the value of the loss function (i.e., the registration energy)
        """
        energy, sim, reg = self.get_energy(phi0, phi1, I0_source, I1_target, lowres_I0, variables_from_forward_model)
        return energy


class SVFImageLoss(RegistrationImageLoss):
    """
    Loss specialization for image-based SVF 
    """
    def __init__(self,v,sz,spacing,params):
        """
        Constructor
        
        :param v: velocity field parameter 
        :param sz: size of image
        :param spacing: spacing of image
        :param params: general parameters via ParameterDict()
        """
        super(SVFImageLoss, self).__init__(sz,spacing,params)
        self.v = v
        """veclocity field parameter"""

        cparams = params[('loss',{},'settings for the loss function')]

        self.regularizer = (RF.RegularizerFactory(self.spacing).
                            create_regularizer(cparams))
        """regularizer to compute the regularization energy"""

    def compute_regularization_energy(self, I0_source, variables_from_forward_model=None):
        """
        Computing the regularization energy
        
        :param I0_source: source image (not used)
        :param variables_from_forward_model: (not used)
        :return: returns the regularization energy
        """
        return self.regularizer.compute_regularizer_multiN(self.v)


class SVFQuasiMomentumImageLoss(RegistrationImageLoss):
    """
    Loss function specialization for the image-based quasi-momentum SVF implementation.
    Essentially the same as for SVF but has to smooth the momentum field first to obtain the velocity field.
    """
    def __init__(self,m,sz,spacing,params):
        """
        Constructor
        
        :param m: momentum field 
        :param sz: image size
        :param spacing: image spacing
        :param params: ParameterDict() parameters
        """
        super(SVFQuasiMomentumImageLoss, self).__init__(sz,spacing,params)
        self.m = m
        """vector momentum"""

        cparams = params[('loss',{},'settings for the loss function')]

        self.regularizer = (RF.RegularizerFactory(self.spacing).
                            create_regularizer(cparams))
        """regularizer to compute the regularization energy"""
        if params['similarity_measure'][('develop_mod_on',False,'developing mode')]:
            cparams = params[('similarity_measure',{},'settings for the similarity ')]
        else:
            cparams = self.params[('forward_model', {}, 'settings for the forward model')]

        #TODO: support smoother optimization here -> move smoother to model instead of loss function
        self.smoother = SF.SmootherFactory(self.sz[2::], self.spacing).create_smoother(cparams)
        """smoother to convert from momentum to velocity"""

    def compute_regularization_energy(self, I0_source, variables_from_forward_model=None):
        """
        Compute the regularization energy from the momentum
        
        :param I0_source: not used
        :param variables_from_forward_model: (not used)
        :return: returns the regularization energy
        """
        m = self.m
        v = self.smoother.smooth(m)
        return self.regularizer.compute_regularizer_multiN(v)

class SVFMapNet(SVFNet):
    """
    Network specialization to a map-based SVF 
    """
    def __init__(self,sz,spacing,params):
        super(SVFMapNet, self).__init__(sz,spacing,params)

    def create_integrator(self):
        """
        Creates an integrator to solve a map-based advection equation
        
        :return: returns this integrator
        """
        cparams = self.params[('forward_model',{},'settings for the forward model')]
        advectionMap = FM.AdvectMap( self.sz, self.spacing )
        return RK.RK4(advectionMap.f,advectionMap.u,self.v,cparams)

    def forward(self, phi, I0_source):
        """
        Solved the map-based equation forward
        
        :param phi: initial condition for the map
        :param I0_source: not used
        :return: returns the map at time tTo
        """
        phi1 = self.integrator.solve([phi], self.tFrom, self.tTo)
        return phi1[0]


class SVFMapLoss(RegistrationMapLoss):
    """
    Specialization of the loss function for SVF to a map-based solution
    """
    def __init__(self,v,sz,spacing,params):
        super(SVFMapLoss, self).__init__(sz,spacing,params)
        self.v = v
        """velocity field parameter"""

        cparams = params[('loss',{},'settings for the loss function')]

        self.regularizer = (RF.RegularizerFactory(self.spacing).
                            create_regularizer(cparams))
        """regularizer to compute the regularization energy"""

    def compute_regularization_energy(self, I0_source, variables_from_forward_model=None):
        """
        Computes the regularizaton energy from the velocity field parameter
        
        :param I0_source: not used 
        :param variables_from_forward_model: (not used)
        :return: returns the regularization energy
        """
        return self.regularizer.compute_regularizer_multiN(self.v)

class DiffusionMapLoss(RegistrationMapLoss):
    """
    Specialization of the loss function for displacement-based registration to diffusion registration
    """

    def __init__(self, d, sz, spacing, params):
        super(DiffusionMapLoss, self).__init__(sz, spacing, params)
        self.d = d
        """displacement field parameter"""

        cparams = params[('loss', {}, 'settings for the loss function')]

        self.regularizer = (RF.RegularizerFactory(self.spacing).
                            create_regularizer_by_name('diffusion',cparams))
        """regularizer to compute the regularization energy"""

    def compute_regularization_energy(self, I0_source, variables_from_forward_model=None):
        """
        Computes the regularizaton energy from the velocity field parameter

        :param I0_source: not used 
        :param variables_from_forward_model: (not used)
        :return: returns the regularization energy
        """
        return self.regularizer.compute_regularizer_multiN(self.d)

class TotalVariationMapLoss(RegistrationMapLoss):
    """
    Specialization of the loss function for displacement-based registration to diffusion registration
    """

    def __init__(self, d, sz, spacing, params):
        super(TotalVariationMapLoss, self).__init__(sz, spacing, params)
        self.d = d
        """displacement field parameter"""

        cparams = params[('loss', {}, 'settings for the loss function')]

        self.regularizer = (RF.RegularizerFactory(self.spacing).
                            create_regularizer_by_name('totalVariation',cparams))
        """regularizer to compute the regularization energy"""

    def compute_regularization_energy(self, I0_source, variables_from_forward_model=None):
        """
        Computes the regularizaton energy from the velocity field parameter

        :param I0_source: not used 
        :param variables_from_forward_model: (not used)
        :return: returns the regularization energy
        """
        return self.regularizer.compute_regularizer_multiN(self.d)

class CurvatureMapLoss(RegistrationMapLoss):
    """
    Specialization of the loss function for displacement-based registration to diffusion registration
    """

    def __init__(self, d, sz, spacing, params):
        super(CurvatureMapLoss, self).__init__(sz, spacing, params)
        self.d = d
        """displacement field parameter"""

        cparams = params[('loss', {}, 'settings for the loss function')]

        self.regularizer = (RF.RegularizerFactory(self.spacing).
                            create_regularizer_by_name('curvature',cparams))
        """regularizer to compute the regularization energy"""

    def compute_regularization_energy(self, I0_source, variables_from_forward_model=None):
        """
        Computes the regularizaton energy from the velocity field parameter

        :param I0_source: not used 
        :param variables_from_forward_model: (not used)
        :return: returns the regularization energy
        """
        return self.regularizer.compute_regularizer_multiN(self.d)

class AffineMapNet(RegistrationNet):
    """
    Registration network for affine transformation
    """
    def __init__(self,sz,spacing,params):
        super(AffineMapNet, self).__init__(sz,spacing,params)
        self.dim = len(self.sz) - 2
        self.Ab = self.create_registration_parameters()

    def create_registration_parameters(self):
        pars = Parameter(AdaptVal(torch.zeros(self.nrOfImages,self.dim*self.dim+self.dim)))
        utils.set_affine_transform_to_identity_multiN(pars.data)
        return pars

    def get_registration_parameters(self):
        """
        Returns the affine parameters as a vector

        :return: affine parameter vector 
        """
        return self.Ab

    def set_registration_parameters(self, p, sz, spacing):
        """
        Sets the affine parameters

        :param p: affine parameter vector 
        :param sz: size of the corresponding image
        :param spacing: spacing of the corresponding image
        """
        self.Ab.data = p.data
        self.sz = sz
        self.spacing = spacing

    def get_parameter_image_and_name_to_visualize(self):
        """
        Returns the velocity field parameter magnitude image and a name

        :return: Returns the tuple (velocity_magnitude_image,name) 
        """
        name = 'Ab'
        par_image = self.Ab
        return par_image, name

    def upsample_registration_parameters(self, desiredSz):
        """
        Upsamples the afffine parameters to a desired size (ie., just returns them)

        :param desiredSz: desired size of the upsampled image
        :return: returns a tuple (upsampled_pars,upsampled_spacing)
        """
        upsampled_spacing = self.spacing*(self.sz[2::].astype('float')/desiredSz[2::].astype('float'))
        return self.Ab, upsampled_spacing

    def downsample_registration_parameters(self, desiredSz):
        """
        Downsamples the affine parameters to a desired size (ie., just returns them)

        :param desiredSz: desired size of the downsampled image 
        :return: returns a tuple (downsampled_pars,downsampled_spacing)
        """
        downsampled_spacing = self.spacing*(self.sz[2::].astype('float')/desiredSz[2::].astype('float'))
        return self.Ab, downsampled_spacing

    def forward(self, phi, I0_source):
        """
        Solved the map-based equation forward

        :param phi: initial condition for the map
        :param I0_source: not used
        :return: returns the map at time tTo
        """
        phi1 = utils.apply_affine_transform_to_map_multiNC(self.Ab,phi)
        return phi1


class AffineMapLoss(RegistrationMapLoss):
    """
    Specialization of the loss function for a map-based affine transformation
    """

    def __init__(self, Ab, sz, spacing, params):
        super(AffineMapLoss, self).__init__(sz, spacing, params)
        self.Ab = Ab
        """affine parameters"""

    def compute_regularization_energy(self, I0_source, variables_from_forward_model=None):
        """
        Computes the regularizaton energy from the affine parameter

        :param I0_source: not used 
        :param variables_from_forward_model: (not used)
        :return: returns the regularization energy
        """
        regE = Variable(MyTensor(1).zero_(), requires_grad=False)
        return regE # so far there is no regularization


class ShootingVectorMomentumNet(RegistrationNetTimeIntegration):
    """
    Specialization to vector-momentum-based shooting for LDDMM
    """
    def __init__(self,sz,spacing,params):
        super(ShootingVectorMomentumNet, self).__init__(sz, spacing, params)
        self.m = self.create_registration_parameters()
        cparams = params[('forward_model', {}, 'settings for the forward model')]
        self.smoother = SF.SmootherFactory(self.sz[2::], self.spacing).create_smoother(cparams)
        """smoother"""
        self.smoother_params = self.smoother.get_optimization_parameters()
        """smoother parameters to be optimized over if supported by smoother"""

        if params['forward_model']['smoother']['type'] == 'adaptiveNet':
            self.add_module('mod_smoother', self.smoother.smoother)
        """vector momentum"""
        self.integrator = self.create_integrator()
        """integrator to solve EPDiff variant"""

    def get_custom_optimizer_output_string(self):
        return self.smoother.get_custom_optimizer_output_string()

    def get_variables_to_transfer_to_loss_function(self):
        d = dict()
        d['smoother'] = self.smoother
        return d

    def create_registration_parameters(self):
        """
        Creates the vector momentum parameter
        
        :return: Returns the vector momentum parameter 
        """
        return utils.create_ND_vector_field_parameter_multiN(self.sz[2::], self.nrOfImages)

    def get_registration_parameters(self):
        """
        Returns the vector momentum parameter
        
        :return: vector momentum 
        """
        return self.m

    def set_registration_parameters(self, p, sz, spacing):
        """
        Sets the vector momentum registration parameter
        
        :param p: vector momentum 
        :param sz: size of image
        :param spacing: spacing of image
        """
        self.m.data = p.data
        self.sz = sz
        self.spacing = spacing

    def get_parameter_image_and_name_to_visualize(self):
        """
        Creates a magnitude image for the momentum and returns it with name :math:`|m|`
        
        :return: Returns tuple (m_magnitude_image,name) 
        """
        name = '|m|'
        par_image = ((self.m[:,...]**2).sum(1))**0.5 # assume BxCxXxYxZ format
        return par_image,name

    def upsample_registration_parameters(self, desiredSz):
        """
        Upsamples the vector-momentum parameter
        
        :param desiredSz: desired size of the upsampled momentum 
        :return: Returns tuple (upsampled_momentum,upsampled_spacing)
        """

        sampler = IS.ResampleImage()
        mUpsampled, upsampled_spacing = sampler.upsample_image_to_size(self.m, self.spacing, desiredSz)

        return mUpsampled,upsampled_spacing

    def downsample_registration_parameters(self, desiredSz):
        """
        Downsamples the vector-momentum parameter

        :param desiredSz: desired size of the downsampled momentum 
        :return: Returns tuple (downsampled_momentum,downsampled_spacing)
        """

        sampler = IS.ResampleImage()
        mDownsampled,downsampled_spacing=sampler.downsample_image_to_size(self.m,self.spacing,desiredSz)

        return mDownsampled, downsampled_spacing

class LDDMMShootingVectorMomentumImageNet(ShootingVectorMomentumNet):
    """
    Specialization of vector-momentum LDDMM for direct image matching.
    """
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingVectorMomentumImageNet, self).__init__(sz,spacing,params)

    def create_integrator(self):
        """
        Creates integrator to solve EPDiff together with an advevtion equation for the image
        
        :return: returns this integrator 
        """

        cparams = self.params[('forward_model',{},'settings for the forward model')]
        epdiffImage = FM.EPDiffImage( self.sz, self.spacing, self.smoother, cparams )
        return RK.RK4(epdiffImage.f,None,None,cparams)

    def forward(self, I):
        """
        Integrates EPDiff plus advection equation for image forward
        
        :param I: Initial condition for image 
        :return: returns the image at time tTo
        """

        mI1 = self.integrator.solve([self.m,I], self.tFrom, self.tTo)
        return mI1[1]


class LDDMMShootingVectorMomentumImageLoss(RegistrationImageLoss):
    """
    Specialization of the image loss to vector-momentum LDDMM
    """
    def __init__(self,m,sz,spacing,params):
        super(LDDMMShootingVectorMomentumImageLoss, self).__init__(sz,spacing,params)
        self.m = m
        """momentum"""
        if params['similarity_measure'][('develop_mod_on',False,'developing mode')]:
            cparams = params[('similarity_measure',{},'settings for the similarity ')]
            self.develop_smoother = SF.SmootherFactory(self.sz[2::],self.spacing).create_smoother(cparams)
            """smoother to convert from momentum to velocity"""
        else:
            self.develop_smoother = None

    def compute_regularization_energy(self, I0_source, variables_from_forward_model):
        """
        Computes the regularzation energy based on the inital momentum
        :param I0_source: not used
        :param variables_from_forward_model: (not used)
        :return: regularization energy
        """
        m = self.m
        if self.develop_smoother is not None:
            v = self.develop_smoother.smooth(m)
        else:
            v = variables_from_forward_model['smoother'].smooth(m)

        reg = (v * m).sum() * self.spacing.prod()
        return reg

class SVFVectorMomentumImageNet(ShootingVectorMomentumNet):
    """
    Specialization of scalar-momentum LDDMM to SVF image-based matching
    """

    def __init__(self, sz, spacing, params):
        super(SVFVectorMomentumImageNet, self).__init__(sz, spacing, params)

    def create_integrator(self):
        """
        Creates an integrator integrating the scalar momentum conservation law and an advection equation for the image

        :return: returns this integrator
        """
        cparams = self.params[('forward_model', {}, 'settings for the forward model')]

        advection = FM.AdvectImage(self.sz, self.spacing)
        return RK.RK4(advection.f, advection.u, None, cparams)

    def forward(self, I):
        """
        Solved the scalar momentum forward equation and returns the image at time tTo

        :param I: initial image
        :return: image at time tTo
        """
        v = self.smoother.smooth(self.m)
        self.integrator.set_pars(v)  # to use this as external parameter
        I1 = self.integrator.solve([I], self.tFrom, self.tTo)
        return I1[0]

class SVFVectorMomentumImageLoss(RegistrationImageLoss):
    """
    Specialization of the loss to scalar-momentum LDDMM on images
    """

    def __init__(self, m, sz, spacing, params):
        super(SVFVectorMomentumImageLoss, self).__init__(sz, spacing, params)
        self.m = m
        """vector momentum"""
        if params['similarity_measure'][('develop_mod_on',False,'developing mode')]:
            cparams = params[('similarity_measure',{},'settings for the similarity ')]
            self.develop_smoother = SF.SmootherFactory(self.sz[2::], self.spacing).create_smoother(cparams)
            """smoother to go from momentum to velocity"""
        else:
            self.develop_smoother = None

    def compute_regularization_energy(self, I0_source, variables_from_forward_model):
        """
        Computes the regularization energy from the initial vector momentum as obtained from the scalar momentum

        :param I0_source: source image
        :param variables_from_forward_model: (not used)
        :return: returns the regularization energy
        """
        m = self.m

        if self.develop_smoother is not None:
            v = self.develop_smoother.smooth(m)
        else:
            v = variables_from_forward_model['smoother'].smooth(m)
        reg = (v * m).sum() * self.spacing.prod()
        return reg


class LDDMMShootingVectorMomentumMapNet(ShootingVectorMomentumNet):
    """
    Specialization for map-based vector-momentum where the map itself is advected
    """
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingVectorMomentumMapNet, self).__init__(sz,spacing,params)


    def create_integrator(self):
        """
        Creates an integrator for EPDiff + advection equation for the map
        
        :return: returns this integrator 
        """
        cparams = self.params[('forward_model',{},'settings for the forward model')]
        epdiffMap = FM.EPDiffMap( self.sz, self.spacing, self.smoother, cparams )
        return RK.RK4(epdiffMap.f,None,None,self.params)

    def forward(self, phi, I0_source):
        """
        Solves EPDiff + advection equation forward and returns the map at time tTo
        
        :param phi: initial condition for the map 
        :param I0_source: not used
        :return: returns the map at time tTo
        """

        mphi1 = self.integrator.solve([self.m,phi], self.tFrom, self.tTo)
        return mphi1[1]


class LDDMMShootingVectorMomentumMapLoss(RegistrationMapLoss):
    """
    Specialization of the loss for map-based vector momumentum. Image similarity is computed based on warping the source
    image with the advected map.
    """

    def __init__(self,m,sz,spacing,params):
        super(LDDMMShootingVectorMomentumMapLoss, self).__init__(sz,spacing,params)
        self.m = m
        """vector momentum"""

        if params['similarity_measure'][('develop_mod_on',False,'developing mode')]:
            cparams = params[('similarity_measure',{},'settings for the similarity ')]
            self.develop_smoother = SF.SmootherFactory(self.sz[2::], self.spacing).create_smoother(cparams)
            """smoother to obtain the velocity field from the momentum field"""
            self.use_net = True if cparams['smoother']['type'] == 'adaptiveNet' else False
        else:
            self.develop_smoother = None
            self.use_net = False

    def compute_regularization_energy(self, I0_source, variables_from_forward_model):
        """
        Commputes the regularization energy from the initial vector momentum
        
        :param I0_source: not used 
        :param variables_from_forward_model: (not used)
        :return: returns the regularization energy
        """
        m = self.m

        if self.develop_smoother is not None:
            v = self.develop_smoother.smooth(m)
        else:
            v = variables_from_forward_model['smoother'].smooth(m)

        reg = (v * m).sum() * self.spacing.prod()
        return reg

class SVFVectorMomentumMapNet(ShootingVectorMomentumNet):
    """
    Specialization of scalar-momentum LDDMM to SVF image-based matching
    """

    def __init__(self, sz, spacing, params):
        super(SVFVectorMomentumMapNet, self).__init__(sz, spacing, params)

    def create_integrator(self):
        """
        Creates an integrator integrating the scalar momentum conservation law and an advection equation for the image

        :return: returns this integrator
        """
        cparams = self.params[('forward_model', {}, 'settings for the forward model')]

        advectionMap = FM.AdvectMap(self.sz, self.spacing)
        return RK.RK4(advectionMap.f, advectionMap.u, None, cparams)

    def forward(self, phi, I_source):
        """
        Solved the scalar momentum forward equation and returns the map at time tTo

        :param phi: initial map
        :param I_source: not used
        :return: image at time tTo
        """
        v = self.smoother.smooth(self.m)
        self.integrator.set_pars(v)  # to use this as external parameter
        phi1 = self.integrator.solve([phi], self.tFrom, self.tTo)
        return phi1[0]

class SVFVectorMomentumMapLoss(RegistrationMapLoss):
    """
    Specialization of the loss to scalar-momentum LDDMM on images
    """

    def __init__(self, m, sz, spacing, params):
        super(SVFVectorMomentumMapLoss, self).__init__(sz, spacing, params)
        self.m = m
        """vector momentum"""
        if params['similarity_measure'][('develop_mod_on',False,'developing mode')]:
            cparams = params[('similarity_measure',{},'settings for the similarity ')]
            self.develop_smoother = SF.SmootherFactory(self.sz[2::], self.spacing).create_smoother(cparams)
            """smoother to go from momentum to velocity"""
        else:
            self.develop_smoother = None

    def compute_regularization_energy(self, I0_source,variables_from_forward_model):
        """
        Computes the regularization energy from the initial vector momentum as obtained from the scalar momentum

        :param I0_source: source image
        :param variables_from_forward_model: (not used)
        :return: returns the regularization energy
        """
        m = self.m

        if self.develop_smoother is not None:
            v = self.develop_smoother.smooth(m)
        else:
            v = variables_from_forward_model['smoother'].smooth(m)

        reg = (v * m).sum() * self.spacing.prod()
        return reg

class ShootingScalarMomentumNet(RegistrationNetTimeIntegration):
    """
    Specialization of the registration network to registrations with scalar momentum. Provides an integrator
    and the scalar momentum parameter.
    """
    def __init__(self,sz,spacing,params):
        super(ShootingScalarMomentumNet, self).__init__(sz, spacing, params)
        self.lam = self.create_registration_parameters()
        """scalar momentum"""
        cparams = params[('forward_model', {}, 'settings for the forward model')]
        self.smoother = SF.SmootherFactory(self.sz[2::], self.spacing).create_smoother(cparams)
        """smoother"""
        self.smoother_params = self.smoother.get_optimization_parameters()
        """smoother parameters to be optimized over if supported by smoother"""

        if params['forward_model']['smoother']['type'] == 'adaptiveNet':
            self.add_module('mod_smoother', self.smoother.smoother)
        self.integrator = self.create_integrator()
        """integrator to integrate EPDiff and associated equations (for image or map)"""

    def get_custom_optimizer_output_string(self):
        return self.smoother.get_custom_optimizer_output_string()

    def get_variables_to_transfer_to_loss_function(self):
        d = dict()
        d['smoother'] = self.smoother
        return d

    def create_registration_parameters(self):
        """
        Creates the scalar momentum registration parameter
        
        :return: Returns this scalar momentum parameter 
        """
        return utils.create_ND_scalar_field_parameter_multiNC(self.sz[2::], self.nrOfImages, self.nrOfChannels)

    def get_registration_parameters(self):
        """
        Returns the scalar momentum registration parameter
        
        :return: scalar momentum 
        """
        return self.lam

    def set_registration_parameters(self, p, sz, spacing):
        """
        Sets the scalar momentum registration parameter
        
        :param p: scalar momentum 
        :param sz: image size
        :param spacing: image spacing 
        """
        self.lam.data = p.data
        self.sz = sz
        self.spacing = spacing

    def get_parameter_image_and_name_to_visualize(self):
        """
        Returns an image of the scalar momentum (magnitude over all channels) and 'lambda' as name
        
        :return: Returns tuple (lamda_magnitude,lambda_name) 
        """
        name = 'lambda'
        par_image = ((self.lam[:,...]**2).sum(1))**0.5 # assume BxCxXxYxZ format
        return par_image,name

    def upsample_registration_parameters(self, desiredSz):
        """
        Upsample the scalar momentum
        
        :param desiredSz: desired size to be upsampled to, e.g., [100,50,40] 
        :return: returns a tuple (upsampled_scalar_momentum,upsampled_spacing)
        """

        sampler = IS.ResampleImage()
        lamUpsampled, upsampled_spacing = sampler.upsample_image_to_size(self.lam, self.spacing, desiredSz)

        return lamUpsampled,upsampled_spacing

    def downsample_registration_parameters(self, desiredSz):
        """
        Downsample the scalar momentum

        :param desiredSz: desired size to be downsampled to, e.g., [40,20,10] 
        :return: returns a tuple (downsampled_scalar_momentum,downsampled_spacing)
        """

        sampler = IS.ResampleImage()
        lamDownsampled,downsampled_spacing=sampler.downsample_image_to_size(self.lam,self.spacing,desiredSz)

        return lamDownsampled, downsampled_spacing


class SVFScalarMomentumImageNet(ShootingScalarMomentumNet):
    """
    Specialization of scalar-momentum LDDMM to SVF image-based matching
    """

    def __init__(self, sz, spacing, params):
        super(SVFScalarMomentumImageNet, self).__init__(sz, spacing, params)

    def create_integrator(self):
        """
        Creates an integrator integrating the scalar momentum conservation law and an advection equation for the image

        :return: returns this integrator 
        """
        cparams = self.params[('forward_model', {}, 'settings for the forward model')]

        advection = FM.AdvectImage(self.sz, self.spacing)
        return RK.RK4(advection.f, advection.u, None, cparams)

    def forward(self, I):
        """
        Solved the scalar momentum forward equation and returns the image at time tTo

        :param I: initial image 
        :return: image at time tTo
        """
        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(self.lam, I, self.sz, self.spacing)
        v = self.smoother.smooth(m)
        self.integrator.set_pars(v)  # to use this as external parameter
        I1 = self.integrator.solve([I], self.tFrom, self.tTo)
        return I1[0]

class SVFScalarMomentumImageLoss(RegistrationImageLoss):
    """
    Specialization of the loss to scalar-momentum LDDMM on images
    """

    def __init__(self, lam, sz, spacing, params):
        super(SVFScalarMomentumImageLoss, self).__init__(sz, spacing, params)
        self.lam = lam
        """scalar momentum"""
        if params['similarity_measure'][('develop_mod_on', False, 'developing mode')]:
            cparams = params[('similarity_measure', {}, 'settings for the similarity ')]
            self.develop_smoother = SF.SmootherFactory(self.sz[2::], self.spacing).create_smoother(cparams)
            """smoother to go from momentum to velocity"""
        else:
            self.develop_smoother = None

    def compute_regularization_energy(self, I0_source,variables_from_forward_model):
        """
        Computes the regularization energy from the initial vector momentum as obtained from the scalar momentum

        :param I0_source: source image 
        :param variables_from_forward_model: (not used)
        :return: returns the regularization energy
        """
        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(self.lam, I0_source, self.sz, self.spacing)

        if self.develop_smoother is not None:
            v = self.develop_smoother.smooth(m)
        else:
            v = variables_from_forward_model['smoother'].smooth(m)

        reg = (v * m).sum() * self.spacing.prod()
        return reg


class LDDMMShootingScalarMomentumImageNet(ShootingScalarMomentumNet):
    """
    Specialization of scalar-momentum LDDMM to image-based matching
    """
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingScalarMomentumImageNet, self).__init__(sz,spacing,params)

    def create_integrator(self):
        """
        Creates an integrator integrating the scalar momentum conservation law and an advection equation for the image
        
        :return: returns this integrator 
        """
        cparams = self.params[('forward_model',{},'settings for the forward model')]
        epdiffScalarMomentumImage = FM.EPDiffScalarMomentumImage( self.sz, self.spacing, self.smoother, cparams )
        return RK.RK4(epdiffScalarMomentumImage.f,None,None,cparams)

    def forward(self, I):
        """
        Solved the scalar momentum forward equation and returns the image at time tTo
        
        :param I: initial image 
        :return: image at time tTo
        """
        lamI1 = self.integrator.solve([self.lam,I], self.tFrom, self.tTo)
        return lamI1[1]


class LDDMMShootingScalarMomentumImageLoss(RegistrationImageLoss):
    """
    Specialization of the loss to scalar-momentum LDDMM on images
    """
    def __init__(self,lam,sz,spacing,params):
        super(LDDMMShootingScalarMomentumImageLoss, self).__init__(sz,spacing,params)
        self.lam = lam
        """scalar momentum"""
        if params['similarity_measure'][('develop_mod_on', False, 'developing mode')]:
            cparams = params[('similarity_measure', {}, 'settings for the similarity ')]
            self.develop_smoother = SF.SmootherFactory(self.sz[2::], self.spacing).create_smoother(cparams)
            """smoother to go from momentum to velocity"""
        else:
            self.develop_smoother = None

    def compute_regularization_energy(self, I0_source,variables_from_forward_model):
        """
        Computes the regularization energy from the initial vector momentum as obtained from the scalar momentum
        
        :param I0_source: source image 
        :param variables_from_forward_model: (not used)
        :return: returns the regularization energy
        """
        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(self.lam, I0_source, self.sz, self.spacing)

        if self.develop_smoother is not None:
            v = self.develop_smoother.smooth(m)
        else:
            v = variables_from_forward_model['smoother'].smooth(m)

        reg = (v * m).sum() * self.spacing.prod()
        return reg


class LDDMMShootingScalarMomentumMapNet(ShootingScalarMomentumNet):
    """
    Specialization of scalar-momentum LDDMM registration to map-based image matching
    """
    def __init__(self,sz,spacing,params):
        super(LDDMMShootingScalarMomentumMapNet, self).__init__(sz,spacing,params)

    def create_integrator(self):
        """
        Creates an integrator integrating the scalar conservation law for the scalar momentum,
        the advection equation for the image and the advection equation for the map,
        
        :return: returns this integrator 
        """
        cparams = self.params[('forward_model',{},'settings for the forward model')]
        epdiffScalarMomentumMap = FM.EPDiffScalarMomentumMap( self.sz, self.spacing, self.smoother, cparams )
        return RK.RK4(epdiffScalarMomentumMap.f,None,None,cparams)

    def forward(self, phi, I0_source):
        """
        Solves the scalar conservation law and the two advection equations forward in time.
        
        :param phi: initial condition for the map 
        :param I0_source: initial condition for the image
        :return: returns the map at time tTo
        """
        lamIphi1 = self.integrator.solve([self.lam,I0_source, phi], self.tFrom, self.tTo)
        return lamIphi1[2]


class LDDMMShootingScalarMomentumMapLoss(RegistrationMapLoss):
    """
    Specialization of the loss function to scalar-momentum LDDMM for maps. 
    """
    def __init__(self,lam,sz,spacing,params):
        super(LDDMMShootingScalarMomentumMapLoss, self).__init__(sz,spacing,params)
        self.lam = lam
        """scalar momentum"""

        if params['similarity_measure'][('develop_mod_on',False,'developing mode')]:
            cparams = params[('similarity_measure',{},'settings for the similarity ')]
            self.develop_smoother = SF.SmootherFactory(self.sz[2::],self.spacing).create_smoother(cparams)
            """smoother to go from momentum to velocity for development configuration"""
        else:
            self.develop_smoother = None

    def compute_regularization_energy(self, I0_source,variables_from_forward_model):
        """
        Computes the regularizaton energy from the initial vector momentum as computed from the scalar momentum
        
        :param I0_source: initial image
        :param variables_from_forward_model: (not used)
        :return: returns the regularization energy
        """
        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(self.lam, I0_source, self.sz, self.spacing)
        if self.develop_smoother is not None:
            v = self.develop_smoother.smooth(m)
        else:
            v = variables_from_forward_model['smoother'].smooth(m)

        reg = (v * m).sum() * self.spacing.prod()
        return reg

class SVFScalarMomentumMapNet(ShootingScalarMomentumNet):
    """
    Specialization of scalar-momentum LDDMM to SVF image-based matching
    """

    def __init__(self, sz, spacing, params):
        super(SVFScalarMomentumMapNet, self).__init__(sz, spacing, params)

    def create_integrator(self):
        """
        Creates an integrator integrating the scalar momentum conservation law and an advection equation for the image

        :return: returns this integrator
        """
        cparams = self.params[('forward_model', {}, 'settings for the forward model')]

        advectionMap = FM.AdvectMap(self.sz, self.spacing)
        return RK.RK4(advectionMap.f, advectionMap.u, None, cparams)

    def forward(self, phi, I_source):
        """
        Solved the scalar momentum forward equation and returns the map at time tTo

        :param I: initial image
        :return: image at time tTo
        """
        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(self.lam, I_source, self.sz, self.spacing)
        v = self.smoother.smooth(m)
        self.integrator.set_pars(v)  # to use this as external parameter
        phi1 = self.integrator.solve([phi], self.tFrom, self.tTo)
        return phi1[0]

class SVFScalarMomentumMapLoss(RegistrationMapLoss):
    """
    Specialization of the loss to scalar-momentum LDDMM on images
    """

    def __init__(self, lam, sz, spacing, params):
        super(SVFScalarMomentumMapLoss, self).__init__(sz, spacing, params)
        self.lam = lam
        """scalar momentum"""

        if params['similarity_measure'][('develop_mod_on',False,'developing mode')]:
            cparams = params[('similarity_measure',{},'settings for the similarity ')]
            self.develop_smoother = SF.SmootherFactory(self.sz[2::], self.spacing).create_smoother(cparams)
            """smoother to go from momentum to velocity"""
        else:
            self.develop_smoother = None

    def compute_regularization_energy(self, I0_source,variables_from_forward_model):
        """
        Computes the regularization energy from the initial vector momentum as obtained from the scalar momentum

        :param I0_source: source image
        :param variables_from_forward_model: (not used)
        :return: returns the regularization energy
        """
        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(self.lam, I0_source, self.sz, self.spacing)
        if self.develop_smoother is not None:
            v = self.develop_smoother.smooth(m)
        else:
            v = variables_from_forward_model['smoother'].smooth(m)

        reg = (v * m).sum() * self.spacing.prod()
        return reg
