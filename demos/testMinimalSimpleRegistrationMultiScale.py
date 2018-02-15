import set_pyreg_paths

# first do the torch imports
import torch
from torch.autograd import Variable
from pyreg.data_wrapper import AdaptVal
import numpy as np

import pyreg.example_generation as eg
import pyreg.module_parameters as pars
import pyreg.multiscale_optimizer as MO

import pyreg.load_default_settings as ds

if ds.use_map:
    model_name = ds.model_name + '_map'
else:
    model_name = ds.model_name + '_image'

# keep track of general parameters
params = pars.ParameterDict( ds.par_algconf )

szEx = np.tile( ds.example_img_len, ds.dim )         # size of the desired images: (sz)^dim
I0,I1,spacing= eg.CreateSquares(ds.dim).create_image_pair(szEx,params) # create a default image size with two sample squares

# create the source and target image as pyTorch variables
ISource = AdaptVal(Variable(torch.from_numpy(I0.copy()), requires_grad=False))
ITarget = AdaptVal(Variable(torch.from_numpy(I1), requires_grad=False))

params['model']['deformation']['map_low_res_factor'] = 0.25
params['optimizer']['single_scale']['nr_of_iterations'] = 50
so = MO.SimpleMultiScaleRegistration(ISource,ITarget,spacing,np.array(ISource.size()),params)
so.get_optimizer().set_visualization( ds.visualize )
so.get_optimizer().set_visualize_step( ds.visualize_step )
so.set_light_analysis_on(True)

so.register()

params.write_JSON( 'testMinimalSimpleRegistrationMultiScale_' + model_name + '_settings_clean.json')
params.write_JSON_comments( 'testMinimalSimpleRegistrationMultiScale_' + model_name + '_settings_comments.json')

