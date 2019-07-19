"""
This is a test implementation to see if we can use pyTorch to solve
LDDMM-style registration problems via automatic differentiation.

Contributors:
  Marc Niethammer: mn@cs.unc.edu
"""

# Note: all images have to be in the format BxCxXxYxZ (BxCxX in 1D and BxCxXxY in 2D)
# I.e., in 1D, 2D, 3D we are dealing with 3D, 4D, 5D tensors. B is the batchsize, and C are the channels
# (for example to support color-images or general multi-modal registration scenarios)

from __future__ import print_function
from builtins import str
import set_pyreg_paths

# first do the torch imports
import torch
from mermaid.data_wrapper import CUDA_ON, MyTensor, AdaptVal
import numpy as np

from time import time
import mermaid.example_generation as eg
import mermaid.module_parameters as pars
import mermaid.smoother_factory as SF
import mermaid.multiscale_optimizer as MO

import mermaid.load_default_settings as ds

# load settings from file
since = time()

if ds.use_map:
    model_name = ds.model_name + '_map'
else:
    model_name = ds.model_name + '_image'

# general parameters
params = pars.ParameterDict(ds.par_algconf)

if ds.load_settings_from_file:
    settingFile = 'testRegistrationGeneric_' + model_name + '_settings.json'
    params.load_JSON(settingFile)

torch.set_num_threads( ds.nr_of_threads ) # not sure if this actually affects anything
print('Number of pytorch threads set to: ' + str(torch.get_num_threads()))

if ds.use_real_images:
    I0,I1,spacing= eg.CreateRealExampleImages(ds.dim).create_image_pair()

else:
    szEx = np.tile(ds.example_img_len,ds.dim)         # size of the desired images: (sz)^dim

    params['square_example_images']=({},'Settings for example image generation')
    params['square_example_images']['len_s'] = szEx.min()//6
    params['square_example_images']['len_l'] = szEx.max()//4

    # create a default image size with two sample squares
    I0,I1,spacing= eg.CreateSquares(ds.dim).create_image_pair(szEx,params)

sz = np.array(I0.shape)

assert( len(sz)==ds.dim+2 )

print ('Spacing = ' + str( spacing ) )

# create the source and target image as pyTorch variables

ISource = AdaptVal(torch.from_numpy(I0.copy()))
ITarget = AdaptVal(torch.from_numpy(I1))

if ds.smooth_images:
    # smooth both a little bit
    params['image_smoothing'] = ds.par_algconf['image_smoothing']
    cparams = params['image_smoothing']
    s = SF.SmootherFactory( sz[2::], spacing ).create_smoother(cparams)
    ISource = s.smooth(ISource)
    ITarget = s.smooth(ITarget)

so = MO.SingleScaleRegistrationOptimizer(sz,spacing,ds.use_map,ds.map_low_res_factor,params)
so.set_model(model_name)
so.set_optimizer_by_name( ds.optimizer_name )
so.set_visualization( ds.visualize )
so.set_visualize_step( ds.visualize_step )

so.set_number_of_iterations( ds.nr_of_iterations)
so.set_rel_ftol(1e-10)

so.set_source_image(ISource)
so.set_target_image(ITarget)

# and now do the optimization
so.optimize()

if ds.save_settings_to_file:
    params.write_JSON( 'testRegistrationGeneric_' + model_name + '_settings_clean.json')
    params.write_JSON_comments( 'testRegistrationGeneric_' + model_name + '_settings_comments.json')

time_elapsed = time() - since

print('time: {}'.format(time_elapsed))
