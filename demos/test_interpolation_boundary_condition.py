from __future__ import print_function
#import set_pyreg_paths

import numpy as np
import matplotlib.pyplot as plt
import torch
import pyreg.simple_interface as SI
import pyreg.example_generation as EG
import pyreg.module_parameters as pars
from pyreg.data_wrapper import  AdaptVal
from pyreg.utils import *
# keep track of general parameters
params = pars.ParameterDict()

use_synthetic_test_case = True
dim = 3
try_all_models = False
add_noise_to_bg = True
if use_synthetic_test_case:
    len = 64
    szEx = np.tile( len, dim )         # size of the desired images: (sz)^dim
    I0,I1,spacing= EG.CreateSquares(dim,add_noise_to_bg).create_image_pair(szEx,params) # create a default image size with two sample squares
else:
    I0,I1,spacing = EG.CreateRealExampleImages(dim).create_image_pair() # create a default image size with two sample squares

if dim ==2:
    Ab = AdaptVal(torch.zeros(1,6))
    Ab[0,0]=1.2#0.8
    Ab[0,3]=1.2 #0.8
elif dim==3:
    Ab = AdaptVal(torch.zeros(1, 12))
    Ab[0, 0] = 1.2
    Ab[0, 4] = 1.2
    Ab[0, 8] = 1.2



phi = identity_map_multiN(I0.shape, spacing, dtype='float32')
phi = AdaptVal(torch.Tensor(phi))
affine_map = apply_affine_transform_to_map_multiNC(Ab,phi)
si = SI.RegisterImagePair()
si.opt = None
si.set_initial_map(affine_map.detach())

si.register_images(I0, I0, spacing,
                        model_name='affine_map',
                        use_multi_scale=True,
                        rel_ftol=1e-7,
                        json_config_out_filename='test_for_boundary.json',
                        compute_inverse_map=True,
                        params='cur_settings.json')
