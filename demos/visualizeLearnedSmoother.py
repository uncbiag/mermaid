from builtins import str
from builtins import range
import set_pyreg_paths
import torch
from torch.autograd import Variable

import pyreg.smoother_factory as SF
import pyreg.deep_smoothers as DS
import pyreg.utils as utils
import pyreg.image_sampling as IS

import numpy as np

import matplotlib.pyplot as plt

def visualize_filter_grid(filter,title=None,print_figures=False):
    nr_of_channels = filter.size()[1]
    nr_of_features_1 = filter.size()[0]

    assert( nr_of_channels==1 )

    # determine grid size
    nr_x = np.ceil(np.sqrt(nr_of_features_1)).astype('int')
    nr_y = nr_x

    for f in range(nr_of_features_1):
        plt.subplot(nr_y, nr_x, f+1)
        plt.imshow(filter[f, 0, ...], cmap='gray')
        plt.colorbar()
        plt.axis('off')

    plt.subplots_adjust(wspace=0.5, hspace=1)

    if title is not None:
        plt.suptitle( title )

    if print_figures:
        plt.savefig('filters_w1.pdf')
    else:
        plt.show()


def visualize_filter(filter,title=None,print_figures=False):
    nr_of_gaussians = filter.size()[1]
    nr_of_features_1 = filter.size()[0]

    for c in range(nr_of_gaussians):
        for r in range(nr_of_features_1):
            cp = 1 + c * nr_of_features_1 + r
            plt.subplot(nr_of_gaussians, nr_of_features_1, cp)
            plt.imshow(filter[r, c, ...], cmap='gray')
            plt.axis('off')

    plt.subplots_adjust(wspace=0.5, hspace=1)

    if title is not None:
        plt.suptitle( title )

    if print_figures:
        plt.savefig('filters_w2.pdf')
    else:
        plt.show()

def compute_overall_std(weights,stds):
    szw = weights.size()
    ret = torch.zeros(szw[1:])

    for i,s in enumerate(stds):
        ret += (weights[i,...])*(s**2)

    # now we have the variances, so take the sqrt
    return torch.sqrt(ret)

def get_array_from_set_of_lists(dat, nr):
    res = []
    for n in range(len(dat)):
        res.append(dat[n][nr])
    return res

def compute_mask(im):
    '''
    computes a mask by finding all the voxels where the image is exactly zero

    :param im:
    :return:
    '''
    mask = np.zeros_like(im)
    mask[im!=0] = 1
    mask[im==0] = np.nan

    return mask

#d = torch.load('testBatchPars.pt')
#d = torch.load('testBatchParsMoreIterations.pt')
#d = torch.load('testBatchParsNewSmoother.pt')
#d = torch.load('testBatchParsNewSmootherMoreImages.pt')
#d = torch.load('testBatchGlobalWeightOpt.pt')
#d = torch.load('testBatchGlobalWeightRegularizedOpt.pt')
#d = torch.load('testBatchGlobalWeightRegularizedOpt_with_lNCC.pt')
#d = torch.load('testBatchGlobalWeightRegularizedOpt_with_NCC_lddmm.pt')
d = torch.load('testBatchGlobalWeightRegularizedOpt_tst.pt')

I0 =  torch.from_numpy(d['I0'])
I1 =  torch.from_numpy(d['I1'])
Iw = d['Iw']
phi = d['phi']
lam =  d['registration_pars']['lam']
sz = I0.size()
history = d['history']
lowResSize = lam.size()
spacing = d['spacing']
params = d['params']

stds = params['model']['registration_model']['forward_model']['smoother']['multi_gaussian_stds']
max_std = max(stds)

spline_order = params['model']['registration_model'][('spline_order', 1, 'Spline interpolation order; 1 is linear interpolation (default); 3 is cubic spline')]

visualize_filters = True
visualize_smooth_vector_fields = False
visualize_weights = True
visualize_energies = True
nr_of_gaussians = len(stds)
nr_of_images = sz[0]
# nr_of_images = 5 # only show a few of them
print_figures = True


if visualize_filters:
    w1 = d['registration_pars']['weighted_smoothing_net.conv_layers.0.weight']
    b1 = d['registration_pars']['weighted_smoothing_net.conv_layers.0.bias']
    w2 = d['registration_pars']['weighted_smoothing_net.conv_layers.1.weight']
    b2 = d['registration_pars']['weighted_smoothing_net.conv_layers.1.bias']

    visualize_filter_grid(w1,'w1',print_figures)
    visualize_filter(w2,'w2',print_figures)


lowResI0, lowResSpacing = IS.ResampleImage().downsample_image_to_size(I0, spacing, lowResSize[2:], spline_order)

# smoother needs to be in the same state as before, so we need to set the parameters correctly
smoother_params = params['model']['registration_model']['forward_model']

smoother = SF.SmootherFactory(lowResSize[2:],lowResSpacing).create_smoother_by_name('learned_multiGaussianCombination',smoother_params)
smoother.set_state_dict(d['registration_pars'])
smoother.set_debug_retain_computed_local_weights(True)

m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(lam,lowResI0,lowResSize,lowResSpacing)

v = smoother.smooth(m,None,[lowResI0,False])

local_weights = smoother.get_debug_computed_local_weights()
default_multi_gaussian_weights = smoother.get_default_multi_gaussian_weights()

if visualize_smooth_vector_fields:

    smoother_not_learned = SF.SmootherFactory(lowResSize[2:], lowResSpacing).create_smoother_by_name('adaptive_multiGaussian')
    v_nl = smoother_not_learned.smooth(m)

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

if visualize_energies:
    plt.clf()
    e_p, = plt.plot(history['energy'], label='energy')
    s_p, = plt.plot(history['similarity_energy'], label='similarity_energy')
    r_p, = plt.plot(history['regularization_energy'], label='regularization_energy')
    plt.legend(handles=[e_p, s_p, r_p])
    if print_figures:
        plt.savefig('energy.pdf')
    else:
        plt.show()

    plt.clf()
    for nw in range(len(history['smoother_weights'][0])):
        cd = get_array_from_set_of_lists(history['smoother_weights'],nw)
        plt.plot(cd)
    plt.title('Smoother weights')
    if print_figures:
        plt.savefig('weights.pdf')
    else:
        plt.show()

if visualize_weights:

    for n in range(nr_of_images):
        os = compute_overall_std(local_weights[n,...], stds )

        plt.clf()

        source_mask = compute_mask(I0[n:n+1,0:1,...].data.numpy())
        lowRes_source_mask_v, _ = IS.ResampleImage().downsample_image_to_size( torch.from_numpy(source_mask), spacing, lowResSize[2:],spline_order)
        lowRes_source_mask = lowRes_source_mask_v.data.numpy()[0,0,...]

        plt.subplot(2,3,1)
        plt.imshow(I0[n,0,...].data.numpy(),cmap='gray')
        plt.title('source')

        plt.subplot(2, 3, 2)
        plt.imshow(I1[n, 0, ...].data.numpy(), cmap='gray')
        plt.title('target')

        plt.subplot(2, 3, 3)
        plt.imshow(Iw[n, 0, ...].data.numpy(), cmap='gray')
        plt.title('warped')

        plt.subplot(2, 3, 4)
        plt.imshow(Iw[n, 0, ...].data.numpy(), cmap='gray')
        plt.contour(phi[n,0,...].data.numpy(),np.linspace(-1, 1, 20), colors='r', linestyles='solid')
        plt.contour(phi[n,1,...].data.numpy(),np.linspace(-1, 1, 20), colors='r', linestyles='solid')
        plt.title('warped+grid')

        plt.subplot(2,3,5)
        plt.imshow(lam[n,0,...].data.numpy(),cmap='gray')
        plt.title('lambda')

        plt.subplot(2,3,6)
        plt.imshow(os.numpy(),cmap='gray')
        plt.title('std')

        plt.suptitle('Registration: ' + str(n))

        if print_figures:
            plt.savefig('{:0>3d}'.format(n) + '_regresult.pdf')
        else:
            plt.show()

        plt.clf()

        for g in range(nr_of_gaussians):
            plt.subplot(2, 4, g + 1)
            clw = local_weights[n,g,...].numpy()
            cmin = clw[lowRes_source_mask==1].min()
            cmax = clw[lowRes_source_mask==1].max()
            plt.imshow((local_weights[n,g,...]).numpy()*lowRes_source_mask,vmin=cmin,vmax=cmax)
            plt.title("{:.2f}".format(stds[g]))
            plt.colorbar()

        plt.subplot(2, 4, 8)
        os = compute_overall_std(local_weights[n,...], stds )

        cmin = os.numpy()[lowRes_source_mask==1].min()
        cmax = os.numpy()[lowRes_source_mask==1].max()
        plt.imshow(os.numpy()*lowRes_source_mask,vmin=cmin,vmax=cmax)
        plt.colorbar()
        plt.suptitle('Registration: ' + str(n))

        if print_figures:
            plt.savefig('{:0>3d}'.format(n) + '_weights.pdf')
        else:
            plt.show()


