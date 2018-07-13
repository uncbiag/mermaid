import set_pyreg_paths

# first do the torch imports
import torch
import multiprocessing as mp
import pyreg.simple_interface as si
import nrrd

import pyreg.load_default_settings

import pyreg.fileio as FIO

I0_filenames = ['./data/m1.nii']
I1_filenames = ['./data/m2.nii']

im_io = FIO.ImageIO()
map_io = FIO.MapIO()

use_batch_optimization = True
if use_batch_optimization:
    I0 = I0_filenames
    I1 = I1_filenames
    spacing = None
else:
    # load a bunch of images as source
    I0,hdr,spacing0,_ = im_io.read_batch_to_nc_format(I0_filenames,intensity_normalize=False)
    # and a bunch of images as target images
    I1,hdr,spacing1,_ = im_io.read_batch_to_nc_format(I1_filenames,intensity_normalize=False)

    spacing = spacing0

torch.set_num_threads(mp.cpu_count())

reg = si.RegisterImagePair()

if use_batch_optimization:
    reg.register_images(I0, I1, spacing,
                        model_name='svf_scalar_momentum_map',
                        nr_of_iterations=1,
                        visualize_step=25,
                        map_low_res_factor=0.25,
                        rel_ftol=1e-10,
                        use_batch_optimization=True,
                        json_config_out_filename='test3d_batch.json',
                        params='test3d.json')
else:
    reg.register_images(I0, I1, spacing,
                        model_name='svf_scalar_momentum_map',
                        nr_of_iterations=1,
                        visualize_step=10,
                        map_low_res_factor=0.25,
                        rel_ftol=1e-10,
                        use_consensus_optimization=False,
                        json_config_out_filename='test3d_consensus.json',
                        params='test3d.json')

h = reg.get_history()

pars = reg.get_model_parameters()

Iw = reg.get_warped_image()
phis = reg.get_map()

#map_io.write('phi3d.nrrd',phis['phi'][0],hdr)
#im_io.write('I0_warped.nrrd',Iw,hdr)

save_me = False
if save_me:
    d = dict()
    d['pars'] = pars
    d['I0'] = I0
    d['I1'] = I1
    d['spacing'] = spacing
    d['Iw'] = Iw
    d['phi'] = phis['phi'][0]
    torch.save(d,'reg_results_3d.pt')

    nrrd.write('./res/Iw_tst_3d.nrrd', (d['Iw']['warped_images'][0].detach().cpu().numpy()[0, 0, ...]))
    nrrd.write('./res/I0_tst_3d.nrrd', d['I0'][0, 0, ...])
    nrrd.write('./res/I1_tst_3d.nrrd', d['I1'][0, 0, ...])

