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
I0,hdr,spacing0,_ = im_io.read_batch_to_nc_format(get_image_range(0,5))
sz = np.array(I0.shape)
# and a bunch of images as target images
I1,hdr,spacing1,_ = im_io.read_batch_to_nc_format(get_image_range(5,10))

assert( np.all(spacing0==spacing1) )

si.RegisterImagePair().register_images(I0,I1,spacing0,
                                       model_name='svf_scalar_momentum_map',
                                       nr_of_iterations=50,
                                       visualize_step=5,
                                       map_low_res_factor=0.5,
                                       rel_ftol=1e-10,
                                       json_config_out_filename='testBatch.json',
                                       params='testBatch.json')
