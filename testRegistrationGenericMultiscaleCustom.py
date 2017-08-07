"""
This is a test implementation to see if we can use pyTorch to solve
LDDMM-style registration problems via automatic differentiation.
In particular, it demonstrates how to add a custom class and similarity measure.

Contributors:
  Marc Niethammer: mn@cs.unc.edu
"""

# Note: all images have to be in the format BxCxXxYxZ (BxCxX in 1D and BxCxXxY in 2D)
# I.e., in 1D, 2D, 3D we are dealing with 3D, 4D, 5D tensors. B is the batchsize, and C are the channels
# (for example to support color-images or general multi-modal registration scenarios)

# first do the torch imports
from __future__ import print_function
import torch
from torch.autograd import Variable

import numpy as np

import set_pyreg_paths

import pyreg.example_generation as eg
import pyreg.module_parameters as pars
import pyreg.smoother_factory as SF

import pyreg.multiscale_optimizer as MO

# load settings from file
loadSettingsFromFile = False
saveSettingsToFile = True

# select the desired dimension of the registration
useMap = False # set to true if a map-based implementation should be used
visualize = True # set to true if intermediate visualizations are desired
smoothImages = True
useRealImages = False

modelName = 'my_svf'
multi_scale_scale_factors = [1.0, 0.5, 0.25]
multi_scale_iterations_per_scale = [50, 100, 100]
dim = 2

# general parameters
params = pars.ParameterDict()

if loadSettingsFromFile:
    settingFile = modelName + '_settings.json'
    params.load_JSON(settingFile)

if useRealImages:
    I0,I1= eg.CreateRealExampleImages(dim).create_image_pair()

else:
    szEx = np.tile( 50, dim )         # size of the desired images: (sz)^dim

    params['square_example_images']=({},'Settings for example image generation')
    params['square_example_images']['len_s'] = szEx.min()/6
    params['square_example_images']['len_l'] = szEx.max()/4

    # create a default image size with two sample squares
    I0,I1= eg.CreateSquares(dim).create_image_pair(szEx,params)

sz = np.array(I0.shape)

assert( len(sz)==dim+2 )

# spacing so that everything is in [0,1]^2 for now
spacing = 1./(sz[2::]-1) # the first two dimensions are batch size and number of image channels
print ('Spacing = ' + str( spacing ) )

# create the source and target image as pyTorch variables
ISource = Variable( torch.from_numpy( I0.copy() ), requires_grad=False )
ITarget = Variable( torch.from_numpy( I1 ), requires_grad=False )

if smoothImages:
    # smooth both a little bit
    cparams = params[('image_smoothing',{},'general settings to pre-smooth images')]
    cparams[('smoother',{})]
    cparams['smoother']['type']='gaussian'
    cparams['smoother']['gaussianStd']=0.05
    s = SF.SmootherFactory( sz[2::], spacing ).create_smoother(cparams)
    ISource = s.smooth_scalar_field(ISource)
    ITarget = s.smooth_scalar_field(ITarget)

mo = MO.MultiScaleRegistrationOptimizer(sz,spacing,useMap,params)

# now customize everything

params['registration_model']['similarity_measure']['type'] = 'mySSD'

import similarity_measure_factory as SM

class MySSD(SM.SimilarityMeasure):
    def compute_similarity(self,I0,I1):
        print('Computing my SSD')
        return ((I0 - I1) ** 2).sum() / (0.1**2) * self.volumeElement

mo.add_similarity_measure('mySSD', MySSD)

import registration_networks as RN
import utils
import image_sampling as IS
import rungekutta_integrators as RK
import forward_models as FM
import regularizer_factory as RF

class MySVFNet(RN.RegistrationNet):
    def __init__(self,sz,spacing,params):
        super(MySVFNet, self).__init__(sz,spacing,params)
        self.v = self.create_registration_parameters()
        self.integrator = self.create_integrator()

    def create_registration_parameters(self):
        return utils.create_ND_vector_field_parameter_multiN(self.sz[2::], self.nrOfImages)

    def get_registration_parameters(self):
        return self.v

    def set_registration_parameters(self, p, sz, spacing):
        self.v.data = p.data
        self.sz = sz
        self.spacing = spacing

    def create_integrator(self):
        cparams = self.params[('forward_model',{},'settings for the forward model')]
        advection = FM.AdvectImage(self.sz, self.spacing)
        return RK.RK4(advection.f, advection.u, self.v, cparams)

    def forward(self, I):
        I1 = self.integrator.solve([I], self.tFrom, self.tTo)
        return I1[0]

    def upsample_registration_parameters(self, desiredSz):
        sampler = IS.ResampleImage()
        vUpsampled,upsampled_spacing=sampler.upsample_image_to_size(self.v,self.spacing,desiredSz)
        return vUpsampled,upsampled_spacing

    def downsample_registration_parameters(self, desiredSz):
        sampler = IS.ResampleImage()
        vDownsampled,downsampled_spacing=sampler.downsample_image_to_size(self.v,self.spacing,desiredSz)
        return vDownsampled,downsampled_spacing

class MySVFImageLoss(RN.RegistrationImageLoss):
    def __init__(self,v,sz,spacing,params):
        super(MySVFImageLoss, self).__init__(sz,spacing,params)
        self.v = v
        cparams = params[('loss',{},'settings for the loss function')]
        self.regularizer = (RF.RegularizerFactory(self.spacing).
                            create_regularizer(cparams))

    def compute_regularization_energy(self, I0_source):
        return self.regularizer.compute_regularizer_multiN(self.v)

mo.add_model(modelName,MySVFNet,MySVFImageLoss)
mo.set_model(modelName)

mo.set_model(modelName)

mo.set_source_image(ISource)
mo.set_target_image(ITarget)

mo.set_scale_factors(multi_scale_scale_factors)
mo.set_number_of_iterations_per_scale(multi_scale_iterations_per_scale)

# and now do the optimization
mo.optimize()

if saveSettingsToFile:
    params.write_JSON(modelName + '_settings_clean.json')
    params.write_JSON_comments(modelName + '_settings_comments.json')