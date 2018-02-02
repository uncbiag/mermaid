import set_pyreg_paths

# first do the torch imports
import torch
from torch.autograd import Variable
from pyreg.data_wrapper import AdaptVal
import numpy as np

import pyreg.example_generation as eg
import pyreg.module_parameters as pars
import pyreg.multiscale_optimizer as MO
import pyreg.simple_interface as si

import pyreg.load_default_settings as ds

import pyreg.fileio as FIO

if ds.use_map:
    model_name = ds.model_name + '_map'
else:
    model_name = ds.model_name + '_image'

# keep track of general parameters
params = pars.ParameterDict( ds.par_algconf )

if ds.load_settings_from_file:
    settingFile = 'testBatchSimpleRegistration_' + model_name + '_settings_clean.json'
    print('Attempting to load settings from file: ' + settingFile )
    params.load_JSON(settingFile)

im_io = FIO.ImageIO()

def get_image_range(im_from,im_to):
    f = []
    for i in range(im_from,im_to):
        current_filename = '../test_data/oasis_2d/oasis2d_' + str(i).zfill(4) + '.nrrd'
        f.append( current_filename )
    return f

# load a bunch of images as source
I0,hdr,spacing0,_ = im_io.read_batch_to_nc_format(get_image_range(0,20))
sz = np.array(I0.shape)
# and a bunch of images as target images
I1,hdr,spacing1,_ = im_io.read_batch_to_nc_format(get_image_range(20,40))

assert( np.all(spacing0==spacing1) )

torch.set_num_threads(2)

reg = si.RegisterImagePair()

if True:
    reg.register_images(I0,I1,spacing0,
                    model_name='svf_scalar_momentum_map',
                    nr_of_iterations=200,
                    visualize_step=None,
                    map_low_res_factor=1.0,
                    rel_ftol=1e-10,
                    json_config_out_filename='testBatchNewerSmoother.json',
                    params='testBatchNewerSmoother.json')

if False:
    reg.register_images(I0,I1,spacing0,
                    model_name='lddmm_shooting_scalar_momentum_map',
                    nr_of_iterations=5,
                    visualize_step=5,
                    map_low_res_factor=0.5,
                    json_config_out_filename='testBatchNewerSmoother.json',
                    params='testBatchNewerSmoother.json')

h = reg.get_history()

pars = reg.get_model_parameters()

Iw = reg.get_warped_image()
phi = reg.get_map()

vars_to_save = dict()
vars_to_save['registration_pars'] = pars
vars_to_save['I0'] = I0
vars_to_save['I1'] = I1
vars_to_save['sz'] = sz
vars_to_save['Iw'] = Iw
vars_to_save['phi'] = phi
vars_to_save['spacing'] = spacing0
vars_to_save['params'] = reg.get_params()
vars_to_save['history'] = h

#torch.save(vars_to_save,'testBatchGlobalWeightRegularizedOpt_with_NCC_lddmm.pt')
torch.save(vars_to_save,'testBatchGlobalWeightRegularizedOpt_tst.pt')


