import set_pyreg_paths
import torch
from torch.autograd import Variable

import pyreg.smoother_factory as SF
import pyreg.deep_smoothers as DS
import pyreg.utils as utils
import pyreg.image_sampling as IS

import numpy as np

import matplotlib.pyplot as plt

def visualize_filter_grid(filter,title=None,print_figures=False,fname=None):
    nr_of_channels = filter.size()[1]
    nr_of_features_1 = filter.size()[0]

    assert( nr_of_channels==1 )

    # determine grid size
    nr_x = np.ceil(np.sqrt(nr_of_features_1)).astype('int')
    nr_y = nr_x

    plt.clf()

    for f in range(nr_of_features_1):
        plt.subplot(nr_y, nr_x, f+1)
        plt.imshow(filter[f, 0, ...], cmap='gray')
        plt.colorbar()
        plt.axis('off')

    plt.subplots_adjust(wspace=0.5, hspace=1)

    if title is not None:
        plt.suptitle( title )

    if print_figures:
        if fname is None:
            fname = 'filters_w1.pdf'
        plt.savefig(fname)
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

def get_checkpoint_filename(batch_nr, batch_iter):
    return "./checkpoints/checkpoint_batch{:05d}_iter{:05d}.pt".format(batch_nr, batch_iter)


#d = torch.load('testBatchPars.pt')
#d = torch.load('testBatchParsMoreIterations.pt')
#d = torch.load('testBatchParsNewSmoother.pt')
#d = torch.load('testBatchParsNewSmootherMoreImages.pt')
#d = torch.load('testBatchGlobalWeightOpt.pt')
#d = torch.load('testBatchGlobalWeightRegularizedOpt.pt')
#d = torch.load('testBatchGlobalWeightRegularizedOpt_with_lNCC.pt')
#d = torch.load('testBatchGlobalWeightRegularizedOpt_with_NCC_lddmm.pt')

d = torch.load('testBatchGlobalWeightRegularizedOpt_tst.pt')

I0 = Variable( torch.from_numpy(d['I0']), requires_grad=False)
I1 = Variable( torch.from_numpy(d['I1']), requires_grad=False)
sz = I0.size()
history = d['history']
spacing = d['spacing']
params = d['params']

nr_of_batches = 1
nr_of_batch_iters = 1

c_filename = get_checkpoint_filename(0, nr_of_batch_iters-1)
q = torch.load(c_filename)
#lam_one = Variable( q['model']['state']['lam'], requires_grad=False)
lam_one = q['model']['parameters']['lam']
lam_one_sz = lam_one.size()
batch_size = lam_one_sz[0]

new_size_lam = list(lam_one_sz)
new_size_lam[0] = sz[0]

import copy
new_size_phi = list(sz)
new_size_phi[1] = 2

# get all the lambdas
lam = torch.FloatTensor(*new_size_lam).zero_()
phi = Variable(torch.FloatTensor(*new_size_phi).zero_(),requires_grad=False)
Iw = Variable(torch.FloatTensor(*sz).zero_(), requires_grad=False)

for b in range(nr_of_batches):
    c_filename = get_checkpoint_filename(b, nr_of_batch_iters - 1)
    q = torch.load(c_filename)
    lam[b*batch_size:min((b+1)*batch_size,sz[0]),...] = q['model']['parameters']['lam']
    Iw[b*batch_size:min((b+1)*batch_size,sz[0]),...] = q['res']['Iw']
    phi[b*batch_size:min((b+1)*batch_size,sz[0]),...] = q['res']['phi']

lam = Variable(lam,requires_grad=False)
lowResSize = lam.size()

stds = params['model']['registration_model']['forward_model']['smoother']['multi_gaussian_stds']
max_std = max(stds)

single_batch = True
visualize_filters = False
visualize_smooth_vector_fields = False
visualize_weights = True
visualize_energies = False
nr_of_gaussians = len(stds)
nr_of_images = sz[0]
# nr_of_images = 5 # only show a few of them
print_figures = True

lowResI0, lowResSpacing = IS.ResampleImage().downsample_image_to_size(I0, spacing, lowResSize[2:])
lowResI1, lowResSpacing = IS.ResampleImage().downsample_image_to_size(I1, spacing, lowResSize[2:])

# smoother needs to be in the same state as before, so we need to set the parameters correctly
smoother_params = params['model']['registration_model']['forward_model']

smoother = SF.SmootherFactory(lowResSize[2:],lowResSpacing).create_smoother_by_name('learned_multiGaussianCombination',smoother_params)
if single_batch:
    smoother.set_state_dict(d['registration_pars']['registration_pars'][0]['model']['parameters'])
else:
    smoother.set_state_dict(d['registration_pars']['consensus_state'])
smoother.set_debug_retain_computed_local_weights(True)

m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(lam,lowResI0,lowResSize,lowResSpacing)

v = smoother.smooth(m,None,{'I':lowResI0,'I0':lowResI0,'I1':lowResI1})

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

#if visualize_energies:
if False:
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
        lowRes_source_mask_v, _ = IS.ResampleImage().downsample_image_to_size(Variable( torch.from_numpy(source_mask), requires_grad=False), spacing, lowResSize[2:])
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
        cmin = os.numpy()[lowRes_source_mask == 1].min()
        cmax = os.numpy()[lowRes_source_mask == 1].max()
        plt.imshow(os.numpy()*lowRes_source_mask,cmap='gray',vmin=cmin,vmax=cmax)
        plt.title('std')

        plt.suptitle('Registration: ' + str(n))

        if print_figures:
            plt.savefig('{:0>3d}'.format(n) + '_regresult.pdf')
        else:
            plt.show()

        plt.clf()

        for g in range(nr_of_gaussians):
            plt.subplot(2, 4, g + 1)
            clw = local_weights[n,g, ...].numpy()
            cmin = clw[lowRes_source_mask==1].min()
            cmax = clw[lowRes_source_mask==1].max()
            plt.imshow((local_weights[n,g, ...]).numpy()*lowRes_source_mask,vmin=cmin,vmax=cmax)
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


if visualize_filters:
    if single_batch:
        w1 = d['registration_pars']['registration_pars'][0]['model']['parameters']['weighted_smoothing_net.conv_layers.0.weight']
    else:
        w1 = d['registration_pars']['consensus_state']['weighted_smoothing_net.conv_layers.0.weight']
        b1 = d['registration_pars']['consensus_state']['weighted_smoothing_net.conv_layers.0.bias']
        w2 = d['registration_pars']['consensus_state']['weighted_smoothing_net.conv_layers.1.weight']
        b2 = d['registration_pars']['consensus_state']['weighted_smoothing_net.conv_layers.1.bias']

    visualize_filter_grid(w1,'w1',print_figures,'filters_w1_consensus.pdf')
    #visualize_filter(w2,'w2',print_figures)

    for batch in range(1):
        #for iter in range(4):
        for iter in range(0,1):
            c_filename = get_checkpoint_filename(batch,iter)
            cd = torch.load(c_filename)
            cw1 = cd['model']['parameters']['weighted_smoothing_net.conv_layers.0.weight']

            cfname = 'filters_w1_batch_{:05d}.pdf'.format(batch)

            visualize_filter_grid(cw1,c_filename,print_figures,cfname)