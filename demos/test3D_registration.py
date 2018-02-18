import set_pyreg_paths

# first do the torch imports
import torch
import multiprocessing as mp
import pyreg.simple_interface as si

import pyreg.load_default_settings

import pyreg.fileio as FIO

I0_filenames = ['./data/m1.nii']
I1_filenames = ['./data/m2.nii']

im_io = FIO.ImageIO()
map_io = FIO.MapIO()

# load a bunch of images as source
I0,hdr,spacing0,_ = im_io.read_batch_to_nc_format(I0_filenames,intensity_normalize=True)
# and a bunch of images as target images
I1,hdr,spacing1,_ = im_io.read_batch_to_nc_format(I1_filenames,intensity_normalize=True)

spacing = spacing0

torch.set_num_threads(mp.cpu_count())

reg = si.RegisterImagePair()

reg.register_images(I0, I1, spacing,
                    model_name='svf_scalar_momentum_map',
                    nr_of_iterations=1,
                    visualize_step=None,
                    map_low_res_factor=0.25,
                    rel_ftol=1e-10,
                    json_config_out_filename='test3d.json',
                    params='test3d.json')

h = reg.get_history()

pars = reg.get_model_parameters()

Iw = reg.get_warped_image()
phi = reg.get_map()

map_io.write('phi3d.nrrd',phi,hdr)
im_io.write('I0_warped.nrrd',Iw,hdr)

torch.save(pars,'model_3d_pars.pt')


