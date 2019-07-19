import set_pyreg_paths

# first do the torch imports
import torch
from mermaid.data_wrapper import AdaptVal
import numpy as np

import mermaid.example_generation as eg
import mermaid.module_parameters as pars
import mermaid.multiscale_optimizer as MO

import mermaid.load_default_settings as ds

if ds.use_map:
    model_name = ds.model_name + '_map'
else:
    model_name = ds.model_name + '_image'

# keep track of general parameters
params = pars.ParameterDict( ds.par_algconf )

szEx = np.tile( ds.example_img_len, ds.dim )         # size of the desired images: (sz)^dim
I0,I1,spacing= eg.CreateSquares(ds.dim).create_image_pair(szEx,params) # create a default image size with two sample squares
sz = np.array(I0.shape)

# create the source and target image as pyTorch variables
ISource = AdaptVal(torch.from_numpy(I0.copy()))
ITarget = AdaptVal(torch.from_numpy(I1))

so = MO.SingleScaleRegistrationOptimizer(sz,spacing,ds.use_map,ds.map_low_res_factor,params)
so.set_model(model_name)
so.set_optimizer_by_name( ds.optimizer_name )
so.set_visualization( ds.visualize )
so.set_visualize_step( ds.visualize_step )

so.set_number_of_iterations(ds.nr_of_iterations)

so.set_source_image(ISource)
so.set_target_image(ITarget)

# and now do the optimization
so.optimize()

params.write_JSON( 'testMinimalRegistration_' + model_name + '_settings_clean.json')
params.write_JSON_comments( 'testMinimalRegistration_' + model_name + '_settings_comments.json')

