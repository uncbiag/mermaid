import set_pyreg_paths

# first do the torch imports
import numpy as np

import torch
import pyreg.simple_interface as si
import pyreg.fileio as FIO

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

torch.set_num_threads(8)

reg = si.RegisterImagePair()

reg.register_images(I0,I1,spacing0,
                        model_name='svf_scalar_momentum_map',
                        nr_of_iterations=100,
                        visualize_step=5,
                        map_low_res_factor=0.5,
                        rel_ftol=1e-10,
                        json_config_out_filename='testInitial.json',
                        params='testInitial.json')

pars = reg.get_model_parameters()

vars_to_save = dict()
vars_to_save['registration_pars'] = pars
vars_to_save['I0'] = I0
vars_to_save['I1'] = I1
vars_to_save['sz'] = sz
vars_to_save['spacing'] = spacing0
vars_to_save['params'] = reg.get_params()

torch.save(vars_to_save,'testInitialPars.pt')

print('Hello')