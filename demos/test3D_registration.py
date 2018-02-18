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
# load a bunch of images as source
I0,hdr,spacing0,_ = im_io.read_batch_to_nc_format(I0_filenames,intensity_normalize=True)
# and a bunch of images as target images
I1,hdr,spacing1,_ = im_io.read_batch_to_nc_format(I1_filenames,intensity_normalize=True)

spacing = spacing0

torch.set_num_threads(mp.cpu_count())

reg = si.RegisterImagePair()

reg.register_images(I0, I1, spacing,
                    model_name='svf_scalar_momentum_map',
                    nr_of_iterations=50,
                    visualize_step=None,
                    map_low_res_factor=0.25,
                    rel_ftol=1e-10,
                    use_consensus_optimization=True,
                    json_config_out_filename='test3d.json',
                    params='test3d.json')

h = reg.get_history()

pars = reg.get_model_parameters()

Iw = reg.get_warped_image()
phi = reg.get_map()

im_io.MapIO().write('phi3d.nrrd',phi,hdr)
im_io.ImageIO().write('I0_warped.nrrd',Iw,hdr)

vars_to_save = dict()
vars_to_save['registration_pars'] = pars
vars_to_save['I0'] = I0
vars_to_save['I1'] = I1
vars_to_save['Iw'] = Iw
vars_to_save['phi'] = phi
vars_to_save['spacing'] = spacing
vars_to_save['params'] = reg.get_params()
vars_to_save['history'] = h

torch.save(vars_to_save,'test3d.pt')


