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

from __future__ import print_function
import set_pyreg_paths

# first do the torch imports
import torch
from torch.autograd import Variable
import numpy as np

import pyreg.example_generation as eg
import pyreg.module_parameters as pars
import pyreg.smoother_factory as SF
from pyreg.data_wrapper import AdaptVal
import pyreg.multiscale_optimizer as MO

import pyreg.load_default_settings as ds

# general parameters
params = pars.ParameterDict()
params['registration_model'] = ds.par_algconf['model']['registration_model']

model_name = 'mySVFNet'

if ds.load_settings_from_file:
    settingFile = model_name + '_settings.json'
    params.load_JSON(settingFile)

if ds.use_real_images:
    I0,I1= eg.CreateRealExampleImages(ds.dim).create_image_pair()

else:
    szEx = np.tile( 50, ds.dim )         # size of the desired images: (sz)^dim

    params['square_example_images']=({},'Settings for example image generation')
    params['square_example_images']['len_s'] = szEx.min()/6
    params['square_example_images']['len_l'] = szEx.max()/4

    # create a default image size with two sample squares
    I0,I1= eg.CreateSquares(ds.dim).create_image_pair(szEx,params)

sz = np.array(I0.shape)

assert( len(sz)==ds.dim+2 )

# spacing so that everything is in [0,1]^2 for now
spacing = 1./(sz[2::]-1) # the first two dimensions are batch size and number of image channels
print ('Spacing = ' + str( spacing ) )

# create the source and target image as pyTorch variables
ISource = AdaptVal(Variable( torch.from_numpy( I0.copy() ), requires_grad=False ))
ITarget = AdaptVal(Variable( torch.from_numpy( I1 ), requires_grad=False ))

if ds.smooth_images:
    # smooth both a little bit
    params['image_smoothing'] = ds.par_algconf['image_smoothing']
    cparams = params['image_smoothing']
    s = SF.SmootherFactory(sz[2::], spacing).create_smoother(cparams)
    ISource = s.smooth_scalar_field(ISource)
    ITarget = s.smooth_scalar_field(ITarget)

use_map = False # this custom registration algorithm does not use a map, so force it to False
mo = MO.MultiScaleRegistrationOptimizer(sz,spacing,use_map,params)

# now customize everything

params['registration_model']['similarity_measure']['type'] = 'mySSD'

import pyreg.similarity_measure_factory as SM

class MySSD(SM.SimilarityMeasure):
    def compute_similarity(self,I0,I1):
        print('Computing my SSD')
        return ((I0 - I1) ** 2).sum() / (0.1**2) * self.volumeElement

mo.add_similarity_measure('mySSD', MySSD)

import pyreg.registration_networks as RN
import pyreg.utils as utils
import pyreg.image_sampling as IS
import pyreg.rungekutta_integrators as RK
import pyreg.forward_models as FM
import pyreg.regularizer_factory as RF

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

mo.add_model(model_name,MySVFNet,MySVFImageLoss)
mo.set_model(model_name)

mo.set_visualization( ds.visualize )
mo.set_visualize_step( ds.visualize_step )

mo.set_source_image(ISource)
mo.set_target_image(ITarget)

mo.set_scale_factors( ds.multi_scale_scale_factors )
mo.set_number_of_iterations_per_scale( ds.multi_scale_iterations_per_scale )

# now we also pick a custom optimizer
mo.set_optimizer(torch.optim.Adam)
mo.set_optimizer_params(dict(lr=0.01))

# and now do the optimization
mo.optimize()

if ds.save_settings_to_file:
    params.write_JSON(model_name + '_settings_clean.json')
    params.write_JSON_comments(model_name + '_settings_comments.json')
