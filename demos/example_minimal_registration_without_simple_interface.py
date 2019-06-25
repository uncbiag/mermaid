"""

Minimal registration example
============================

This example script is a very minimialistic example for using mermaid.
"""

# first do the torch imports
import torch
from mermaid.data_wrapper import AdaptVal
import numpy as np

import mermaid.example_generation as eg
import mermaid.module_parameters as pars
import mermaid.multiscale_optimizer as MO

use_map = True
model_name = 'lddmm_shooting'
dim = 2
example_img_len = 64
map_low_res_factor = 0.5
optimizer_name = 'sgd'
visualize = True
visualize_step = 10
nr_of_iterations = 50

if use_map:
    model_name = model_name + '_map'
else:
    model_name = model_name + '_image'

# keep track of general parameters
params = pars.ParameterDict()

szEx = np.tile( example_img_len, dim )         # size of the desired images: (sz)^dim
I0,I1,spacing= eg.CreateSquares(dim).create_image_pair(szEx,params) # create a default image size with two sample squares
sz = np.array(I0.shape)

# create the source and target image as pyTorch variables
ISource = AdaptVal(torch.from_numpy(I0.copy()))
ITarget = AdaptVal(torch.from_numpy(I1))

so = MO.SingleScaleRegistrationOptimizer(sz,spacing,use_map,map_low_res_factor,params)
so.set_model(model_name)
so.set_optimizer_by_name( optimizer_name )
so.set_visualization( visualize )
so.set_visualize_step( visualize_step )

so.set_number_of_iterations(nr_of_iterations)

so.set_source_image(ISource)
so.set_target_image(ITarget)
so.set_light_analysis_on(True)

# and now do the optimization
so.optimize()

params.write_JSON( 'test_minimal_registration_' + model_name + '_settings_clean.json')
params.write_JSON_comments( 'test_minimal_registration_' + model_name + '_settings_comments.json')

