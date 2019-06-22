from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import mermaid.simple_interface as SI
import mermaid.example_generation as EG
import mermaid.module_parameters as pars

# keep track of general parameters
params = pars.ParameterDict()

use_synthetic_test_case = True
dim = 2
try_all_models = False
add_noise_to_bg = True
if use_synthetic_test_case:
    len = 64
    szEx = np.tile( len, dim )         # size of the desired images: (sz)^dim
    I0,I1,spacing= EG.CreateSquares(dim,add_noise_to_bg).create_image_pair(szEx,params) # create a default image size with two sample squares
else:
    I0,I1,spacing = EG.CreateRealExampleImages(dim).create_image_pair() # create a default image size with two sample squares

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

if try_all_models:
    # try all the models
    for model_name in all_models:
        print('Registering with model: ' + model_name )
        si.register_images(I0, I1, spacing, model_name=model_name)
else:
    # just try one of them, explicitly specified
    #si.register_images(I0, I1, spacing, model_name='total_variation_map')
    #si.register_images(I0, I1, spacing, model_name='svf_vector_momentum_map',nr_of_iterations=30,optimizer_name='sgd',compute_inverse_map=True)

    # si.register_images(I0, I1, spacing, model_name='svf_vector_momentum_map',
    #                    nr_of_iterations=50,
    #                    visualize_step=10,
    #                    optimizer_name='sgd',
    #                    rel_ftol=1e-7,
    #
    #                    json_config_out_filename = 'test2d_tst.json',
    #                    params = 'test2d_tst.json')

    si.register_images(I0, I1, spacing, model_name='lddmm_shooting_map',
                       nr_of_iterations=50,
                       use_multi_scale=False,
                       visualize_step=10,
                       optimizer_name='sgd',
                       learning_rate=0.02,
                       rel_ftol=1e-7,
                       json_config_out_filename=('test2d_tst.json','test2d_tst_with_comments.json'),
                       params='test2d_tst.json')

    #si.register_images(I0, I1, spacing, model_name='svf_vector_momentum_image')
    #si.register_images(I0, I1, spacing, model_name='affine_map')
    #si.register_images(I0, I1, spacing, model_name='svf_scalar_momentum_map',
    #                                       smoother_type='adaptive_multiGaussian',
    #                                       optimize_over_smoother_parameters=True,
    #                                       map_low_res_factor=1.0,
    #                                       visualize_step=10,
    #                                       nr_of_iterations=10,
    #                                       rel_ftol=1e-8,
    #                                       similarity_measure_sigma=0.01)
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



