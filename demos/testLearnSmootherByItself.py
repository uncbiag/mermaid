import set_pyreg_paths
import pyreg.smoother_factory as SF

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

import pyreg.simple_interface as si
import pyreg.fileio as FIO
import pyreg.utils as utils
import pyreg.image_sampling as IS

im_io = FIO.ImageIO()

def get_image_range(im_from,im_to):
    f = []
    for i in range(im_from,im_to):
        current_filename = '../test_data/oasis_2d/oasis2d_' + str(i).zfill(4) + '.nrrd'
        f.append( current_filename )
    return f

I0_filenames = get_image_range(0,20)
I1_filenames = get_image_range(20,40)

results_filename = 'testInitialPars.pt'
read_results_from_file = False
visualize_smooth_vector_fields = True

d = torch.load(results_filename)

I0 = Variable( torch.from_numpy(d['I0']), requires_grad=False)
I1 = Variable( torch.from_numpy(d['I1']), requires_grad=False)
lam = Variable( d['registration_pars']['lam'], requires_grad=False)
sz = d['sz']
lowResSize = lam.size()
spacing = d['spacing']

lowResI0, lowResSpacing = IS.ResampleImage().downsample_image_to_size(I0, spacing, lowResSize[2:],spline_order=1)

smoother_not_learned = SF.SmootherFactory(lowResSize[2:],lowResSpacing).create_smoother_by_name('adaptive_multiGaussian')
smoother = SF.SmootherFactory(lowResSize[2:],lowResSpacing).create_smoother_by_name('learned_multiGaussianCombination')
m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(lam,lowResI0,lowResSize,lowResSpacing)
#v = smoother.smooth(m)
v = smoother.smooth(m,None,[lowResI0,False])
v_nl = smoother_not_learned.smooth(m)

if visualize_smooth_vector_fields:

    nr_of_images = sz[0]
    for n in range(nr_of_images):

        plt.subplot(3,2,1)
        plt.imshow(m[n,0,...].data.numpy())

        plt.subplot(3,2,2)
        plt.imshow(m[n,1,...].data.numpy())

        plt.subplot(3, 2, 3)
        plt.imshow(v[n, 0, ...].data.numpy())

        plt.subplot(3, 2, 4)
        plt.imshow(v[n, 1, ...].data.numpy())

        plt.subplot(3, 2, 5)
        plt.imshow(v_nl[n, 0, ...].data.numpy())

        plt.subplot(3, 2, 6)
        plt.imshow(v_nl[n, 1, ...].data.numpy())

        plt.title( str(n) )

        plt.show()

