"""
This is a test implementation to see if we can use pyTorch to solve
LDDMM-style registration problems via automatic differentiation.

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
from time import time
import numpy as np

import set_pyreg_paths

import pyreg.example_generation as eg
import pyreg.module_parameters as pars
import pyreg.smoother_factory as SF

import pyreg.multiscale_optimizer as MO
from configParsers import *
from pyreg.data_wrapper import AdaptVal

since = time()
if use_map:
    model_name = model_name + '_map'
else:
    model_name = model_name + '_image'

# general parameters
params = pars.ParameterDict()
params['registration_model']['forward_model'] = base_default_params['model']['deformation']['forward_model']

if load_settings_from_file:
    settingFile = model_name + '_settings.json'
    params.load_JSON(settingFile)

torch.set_num_threads( nr_of_threads )
print('Number of pytorch threads set to: ' + str(torch.get_num_threads()))

if use_real_images:
    I0,I1= eg.CreateRealExampleImages(dim).create_image_pair()

else:
    szEx = np.tile( example_img_len, dim )         # size of the desired images: (sz)^dim

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
ISource = AdaptVal(Variable(torch.from_numpy(I0.copy()), requires_grad=False))
ITarget = AdaptVal(Variable(torch.from_numpy(I1), requires_grad=False))

if smooth_images:
    # smooth both a little bit
    cparams = params[('image_smoothing',{},'general settings to pre-smooth images')]
    cparams[('smoother',{})]
    cparams['smoother']['type']= image_smoothing_type
    cparams['smoother']['gaussian_std']= smooth_images_gaussian_std
    s = SF.SmootherFactory( sz[2::], spacing ).create_smoother(cparams)
    ISource = s.smooth_scalar_field(ISource)
    ITarget = s.smooth_scalar_field(ITarget)

mo = MO.MultiScaleRegistrationOptimizer(sz,spacing,use_map,params)

mo.set_optimizer_by_name( optimizer_name )
mo.set_visualization( visualize )
mo.set_visualize_step( visualize_step )

mo.set_model(model_name)

mo.set_source_image(ISource)
mo.set_target_image(ITarget)

mo.set_scale_factors(multi_scale_scale_factors)
mo.set_number_of_iterations_per_scale(multi_scale_iterations_per_scale)

# and now do the optimization
mo.optimize()

if save_settings_to_file:
    params.write_JSON(model_name + '_settings_clean.json')
    params.write_JSON_comments(model_name + '_settings_comments.json')


print("time{}".format(time()-since))
