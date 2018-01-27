import set_pyreg_paths
import torch
from torch.autograd import Variable

import pyreg.smoother_factory as SF
import pyreg.deep_smoothers as DS
import pyreg.utils as utils
import pyreg.image_sampling as IS

import matplotlib.pyplot as plt

def visualize_filter(filter,title=None):
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

    plt.show()

d = torch.load('testBatchPars.pt')

visualize_filters = False

if visualize_filters:
    w1 = d['registration_pars']['weighted_smoothing_net.conv1.weight']
    b1 = d['registration_pars']['weighted_smoothing_net.conv1.bias']
    w2 = d['registration_pars']['weighted_smoothing_net.conv2.weight']
    b2 = d['registration_pars']['weighted_smoothing_net.conv2.bias']

    visualize_filter(w1,'w1')
    visualize_filter(w2,'w2')

import collections

def get_state_dict_for_module(state_dict,module_name):

    res_dict = collections.OrderedDict()
    for k in state_dict.keys():
        if k.startswith(module_name + '.'):
            adapted_key = k[len(module_name)+1:]
            res_dict[adapted_key] = state_dict[k]
    return res_dict

I0 = Variable( torch.from_numpy(d['I0']), requires_grad=False)
I1 = Variable( torch.from_numpy(d['I1']), requires_grad=False)
lam = Variable( d['registration_pars']['lam'], requires_grad=False)
sz = d['sz']
lowResSize = lam.size()
spacing = d['spacing']

lowResI0, lowResSpacing = IS.ResampleImage().downsample_image_to_size(I0, spacing, lowResSize[2:])

smoother = SF.SmootherFactory(lowResSize[2:],lowResSpacing).create_smoother_by_name('learned_multiGaussianCombination')
smoother_not_learned = SF.SmootherFactory(lowResSize[2:],lowResSpacing).create_smoother_by_name('adaptive_multiGaussian')

m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(lam,lowResI0,lowResSize,lowResSpacing)

v = smoother.smooth(m,None,[lowResI0,False])
v_nl = smoother_not_learned.smooth(m)

visualize_smooth_vector_fields = True

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





