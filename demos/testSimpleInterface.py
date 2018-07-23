from __future__ import print_function
import set_pyreg_paths
import torch
import numpy as np
import matplotlib.pyplot as plt
import itk as itk

import pyreg.simple_interface as SI
import pyreg.example_generation as EG
import pyreg.module_parameters as pars
import pyreg.smoother_factory as SF
import pyreg.fileio as FIO
from torch.autograd import Variable
from pyreg.data_wrapper import AdaptVal,MyTensor
import time

start = time.time()
# keep track of general parameters
params = pars.ParameterDict()

use_synthetic_test_case = False
dim = 2
try_all_models = False
smooth_before_reg = False

if use_synthetic_test_case:
    len = 64
    szEx = np.tile( len, dim )         # size of the desired images: (sz)^dim
    I0,I1,spacing= EG.CreateSquares(dim).create_image_pair(szEx,params) # create a default image size with two sample squares
else:
    I0,I1,spacing = EG.CreateRealExampleImages(dim).create_image_pair() # create a default image size with two sample squares



if smooth_before_reg:
    print('---------------------smoothing image pair before registration-------------------')
    sz = list(np.array(list(I0.shape[-dim:])))
    I0smooth = I0.copy()
    I0smooth = AdaptVal(Variable(MyTensor(I0smooth), requires_grad=False))
    I1smooth = I1.copy()
    I1smooth = AdaptVal(Variable(MyTensor(I1smooth), requires_grad=False))
    sf = SF.AdaptiveSingleGaussianFourierSmoother(sz,spacing,params)
    sf.set_gaussian_std(0.01)
    print (params)
    I0smooth=sf.smooth(I0smooth)
    I1smooth = sf.smooth(I1smooth)
    I0smooth = I0smooth.data.numpy()
    I0 = I0 + I0smooth*(I0==0)
    I1smooth = I1smooth.data.numpy()
    I1 = I1 + I1smooth * (I1 == 0)




"""
Known registration models are:
------------------------------
                     total_variation_map: displacement-based total variation registration
                 svf_vector_momentum_map: map-based stationary velocity field using the vector momentum
               svf_vector_momentum_image: image-based stationary velocity field using the vector momentum
                              affine_map: map-based affine registration
                 svf_scalar_momentum_map: map-based stationary velocity field using the scalar momentum
                           curvature_map: displacement-based curvature registration
                    lddmm_shooting_image: image-based shooting-based LDDMM using the vector momentum
                           diffusion_map: displacement-based diffusion registration
                               svf_image: image-based stationary velocity field
                                 svf_map: map-based stationary velocity field
               svf_scalar_momentum_image: image-based stationary velocity field using the scalar momentum
                svf_quasi_momentum_image: EXPERIMENTAL: Image-based SVF version which estimates velcocity based on a smoothed vetor field
    lddmm_shooting_scalar_momentum_image: image-based shooting-based LDDMM using the scalar momentum
      lddmm_shooting_scalar_momentum_map: map-based shooting-based LDDMM using the scalar momentum
                      lddmm_shooting_map: map-based shooting-based LDDMM using the vector momentum
"""

# print possible model names
si = SI.RegisterImagePair()

si.print_available_models()
all_models = si.get_available_models()

I0_filename = './findTheBug_I0_readIn_testSI.nrrd'
print("Writing I0")
FIO.ImageIO().write(I0_filename, I0[0,0, :])
I1_filename = './findTheBug_I1_readIn_testSI.nrrd'
print("Writing I1")
FIO.ImageIO().write(I1_filename, I1[0,0, :])

if try_all_models:
    # try all the models
    for model_name in all_models:
        print('Registering with model: ' + model_name )
        si.register_images(I0, I1, spacing, model_name=model_name)
else:
    # just try one of them, explicitly specified
    #si.register_images(I0, I1, spacing, model_name='total_variation_map')
    #si.register_images(I0, I1, spacing, model_name='svf_vector_momentum_map',nr_of_iterations=30,optimizer_name='sgd',compute_inverse_map=True)

    si.register_images(I0, I1, spacing, model_name='svf_scalar_momentum_map',
                       nr_of_iterations=50,
                       optimizer_name='lbfgs_ls',
                       json_config_out_filename = 'test2d_tst.json',
                       params = 'test2d_tst.json')

    #si.register_images(I0, I1, spacing, model_name='svf_vector_momentum_image')
    # si.register_images(I0, I1, spacing, model_name='affine_map', nr_of_iterations=10,
    #                    visualize_step=1, rel_ftol=1e-4,similarity_measure_sigma=0.1,
    #                            similarity_measure_type="ssd"
    #                    ,json_config_out_filename='affine.json', params='affine.json')
    si.register_images(I0, I1, spacing, model_name='svf_scalar_momentum_map',
                                          #optimize_over_smoother_parameters=True,
                                           smoother_type='multiGaussian',
                                           compute_similarity_measure_at_low_res=False,
                                           map_low_res_factor=1.0,
                                           visualize_step=15,
                                           nr_of_iterations=15,
                                           rel_ftol=1e-8,
                                           similarity_measure_type="ssd",
                                           params='aaa.json',
                                           similarity_measure_sigma=0.1)
                                          #json_config_out_filename='aaa.json')
                                          #,params='aaa.json')
    #si.register_images(I0, I1, spacing, model_name='curvature_map',rel_ftol=1e-12, similarity_measure_sigma=0.005,nr_of_iterations=100)
    #si.register_images(I0, I1, spacing, model_name='lddmm_shooting_image',nr_of_iterations=100)
    #si.register_images(I0, I1, spacing, model_name='diffusion_map')
    #si.register_images(I0, I1, spacing, model_name='svf_image')
    #si.register_images(I0, I1, spacing, model_name='svf_map')
    #si.register_images(I0, I1, spacing, model_name='svf_scalar_momentum_image')
    #si.register_images(I0, I1, spacing, model_name='svf_quasi_momentum_image',nr_of_iterations=100)
    #si.register_images(I0, I1, spacing, model_name='lddmm_shooting_scalar_momentum_image',nr_of_iterations=10,use_multi_scale=False)
    #si.register_images(I0, I1, spacing, model_name='lddmm_shooting_scalar_momentum_map')
    #si.register_images(I0, I1, spacing, model_name='lddmm_shooting_map')

    #si.register_images(I0, I1, spacing, model_name='svf_scalar_momentum_map',
    #                    visualize_step=5,
    #                    nr_of_iterations=20,
    #                    similarity_measure_type='omt')

    h = si.get_history()

    e_p, = plt.plot(h['energy'], label='energy')
    s_p, = plt.plot(h['similarity_energy'], label='similarity_energy')
    r_p, = plt.plot(h['regularization_energy'], label='regularization_energy')
    plt.legend(handles=[e_p,s_p,r_p])
    plt.show()

print('total computing time:', time.time() - start)


