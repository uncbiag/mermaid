import set_pyreg_paths
import pyreg.smoother_factory as SF

import torch
import pyreg.simple_interface as si
import pyreg.fileio as FIO
import pyreg.utils as utils

im_io = FIO.ImageIO()

def get_image_range(im_from,im_to):
    f = []
    for i in range(im_from,im_to):
        current_filename = '../test_data/oasis_2d/oasis2d_' + str(i).zfill(4) + '.nrrd'
        f.append( current_filename )
    return f

source_images = get_image_range(0,20)
target_images = get_image_range(20,40)

# load a bunch of images as source
I0,hdr,spacing0,_ = im_io.read_batch_to_nc_format(get_image_range(0,20))
sz = np.array(I0.shape)
# and a bunch of images as target images
I1,hdr,spacing1,_ = im_io.read_batch_to_nc_format(get_image_range(20,40))

assert( np.all(spacing0==spacing1) )

#reg = si.RegisterImagePair()
#
#reg.register_images(I0,I1,spacing0,
#                        model_name='svf_scalar_momentum_map',
#                        nr_of_iterations=100,
#                        visualize_step=5,
#                        map_low_res_factor=0.5,
#                        rel_ftol=1e-10,
#                        json_config_out_filename='testInitial.json',
#                        params='testInitial.json')

pars = reg.get_model_parameters()
#torch.save(pars,'testInitialPars.pt')


d = torch.load('testInitialPars.pt')

I0 = d['I0']
I1 = d['I1']
lam = d['registration_pars']['lam']
sz = d['sz']
spacing = d['spacing']

smoother = SF.SmootherFactory(sz,spacing).create_smoother_by_name('adaptive_multiGaussian')


m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(lam,I0,sz,spacing)
v = smoother.smooth(m)
