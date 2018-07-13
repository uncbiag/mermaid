from __future__ import print_function
from builtins import str
from builtins import range
import set_pyreg_paths

# first do the torch imports
import torch
import numpy as np

import pyreg.simple_interface as si
import pyreg.fileio as FIO
import pyreg.visualize_registration_results as vizreg

im_io = FIO.ImageIO()

def get_image_range(im_from,im_to):
    f = []
    for i in range(im_from,im_to):
        current_filename = '../test_data/oasis_2d/oasis2d_' + str(i).zfill(4) + '.nrrd'
        f.append( current_filename )
    return f

I0_filenames = get_image_range(0,2)
I1_filenames = get_image_range(2,4)

results_filename = 'ind_versus_batch_small.pt'
nr_of_iterations = 100
viz_interval = 24
read_results_from_file = False

if not read_results_from_file:

    # first do the independent registration
    results = dict()
    ind_energies = []


    nr_of_filenames = len(I0_filenames)

    for i in range(nr_of_filenames):

        I0, hdr, spacing0, _ = im_io.read_batch_to_nc_format(I0_filenames[i:i+1])
        sz = np.array(I0.shape)
        # and a bunch of images as target images
        I1, hdr, spacing1, _ = im_io.read_batch_to_nc_format(I1_filenames[i:i+1])

        assert (np.all(spacing0 == spacing1))

        reg = si.RegisterImagePair()
        reg.register_images(I0,I1,spacing0,
                            model_name='svf_scalar_momentum_map',
                            nr_of_iterations=nr_of_iterations,
                            visualize_step=viz_interval,
                            map_low_res_factor=0.5,
                            rel_ftol=1e-10,
                            params='test_ind_versus_batch.json')

        I0warped = reg.get_warped_image()
        phi = reg.get_map()
        pars = reg.get_model_parameters()

        ind_energies.append( reg.get_energy() )

        # create the data structures
        if i==0:
            sz_new = sz.copy()
            sz_new[0] = nr_of_filenames

            sz_new_map = list(phi.size())
            sz_new_map[0] = nr_of_filenames

            sz_new_lam = list(pars['lam'].size())
            sz_new_lam[0] = nr_of_filenames

            res_warped_images = torch.FloatTensor(*sz_new)
            res_I0 = torch.FloatTensor(*sz_new)
            res_I1 = torch.FloatTensor(*sz_new)
            res_maps = torch.FloatTensor(*sz_new_map)
            res_lam = torch.FloatTensor(*sz_new_lam)

            results['ind_I0_warped'] = res_warped_images
            results['ind_phi'] = res_maps
            results['ind_I0'] = res_I0
            results['ind_I1'] = res_I1
            results['ind_lam'] = res_lam

        # now store the current results
        results['ind_I0_warped'][i,...] = I0warped.data
        results['ind_phi'][i,...] = phi.data
        results['ind_I0'][i,...] = torch.from_numpy(I0[0,...])
        results['ind_I1'][i,...] = torch.from_numpy(I1[0,...])
        results['ind_lam'][i,...] = pars['lam']

    results['ind_energies'] = ind_energies

    # now register the entire batch

    # load a bunch of images as source
    I0,hdr,spacing0,_ = im_io.read_batch_to_nc_format(I0_filenames)
    sz = np.array(I0.shape)
    # and a bunch of images as target images
    I1,hdr,spacing1,_ = im_io.read_batch_to_nc_format(I1_filenames)

    assert( np.all(spacing0==spacing1) )

    reg = si.RegisterImagePair()
    reg.register_images(I0,I1,spacing0,
                        model_name='svf_scalar_momentum_map',
                        nr_of_iterations=nr_of_iterations,
                        visualize_step=viz_interval,
                        map_low_res_factor=0.5,
                        rel_ftol=1e-10,
                        params='test_ind_versus_batch.json',
                        json_config_out_filename='test_ind_versus_batch.json')

    I0warped = reg.get_warped_image()
    phi = reg.get_map()
    pars = reg.get_model_parameters()

    results['batch_I0_warped'] = I0warped.data
    results['batch_phi'] = phi.data
    results['batch_I0'] = torch.from_numpy(I0)
    results['batch_I1'] = torch.from_numpy(I1)
    results['batch_lam'] = pars['lam']
    results['batch_energy'] = reg.get_energy()
    results['sz'] = sz
    results['spacing'] = spacing0

    torch.save(results,results_filename)

else: # read results from file
    results = torch.load(results_filename)

nr_of_results = results['ind_I0'].size()[0]

for i in range(nr_of_results):
    vizreg.show_current_images(iter=i, iS=results['ind_I0'][i:i+1,...],
                               iT=results['ind_I1'][i:i+1,...],
                               iW=results['ind_I0_warped'][i:i+1,...],
                               vizImages=results['ind_lam'][i,...], vizName='i-lambda', phiWarped=results['ind_phi'][i:i+1,...])

    vizreg.show_current_images(iter=i, iS=results['batch_I0'][i:i+1,...],
                               iT=results['batch_I1'][i:i+1,...],
                               iW=results['batch_I0_warped'][i:i+1,...],
                               vizImages=results['batch_lam'][i,...], vizName='b-lambda', phiWarped=results['batch_phi'][i:i+1,...])

print()
print('Hello')