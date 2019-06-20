from __future__ import print_function
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

if ds.load_settings_from_file:
    settingFile = 'testMinimalSimpleRegistration_' + model_name + '_settings_clean.json'
    print('Attempting to load settings from file: ' + settingFile )
    params.load_JSON(settingFile)

szEx = np.tile( ds.example_img_len, ds.dim )         # size of the desired images: (sz)^dim
I0,I1,spacing= eg.CreateSquares(ds.dim).create_image_pair(szEx,params) # create a default image size with two sample squares
#I0,I1,spacing = eg.CreateRealExampleImages(ds.dim).create_image_pair() # create a default image size with two sample squares

# create the source and target image as pyTorch variables
ISource = AdaptVal(torch.from_numpy(I0.copy()))
ITarget = AdaptVal(torch.from_numpy(I1))

params['model']['deformation']['map_low_res_factor'] = 0.5
params['optimizer']['single_scale']['nr_of_iterations'] = 25
so = MO.SimpleSingleScaleRegistration(ISource,ITarget,spacing,np.array(ISource.size()),params)
so.get_optimizer().set_visualization( ds.visualize )
so.get_optimizer().set_visualize_step( ds.visualize_step )
so.set_light_analysis_on(True)
so.register()

params.write_JSON( 'testMinimalSimpleRegistration_' + model_name + '_settings_clean.json')
params.write_JSON_comments( 'testMinimalSimpleRegistration_' + model_name + '_settings_comments.json')
