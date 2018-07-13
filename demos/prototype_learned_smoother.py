from __future__ import print_function
from builtins import range
import set_pyreg_paths

import pyreg.custom_pytorch_extensions as ce
from pyreg.data_wrapper import AdaptVal

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import pyreg.deep_smoothers as DS

d = torch.load('vsmooth_tst_data.pt')

m = d['v']
sz = d['sz']
spacing = d['spacing']
I_or_phi = d['I_or_phi']
nr_of_gaussians = d['nr_of_gaussians']
gaussian_stds = d['stds']
gaussian_weights = d['weights']

compute_std_gradients = True

gaussian_fourier_filter_generator = ce.GaussianFourierFilterGenerator(sz, spacing, nr_of_gaussians)

vcollection = ce.fourier_set_of_gaussian_convolutions(m, gaussian_fourier_filter_generator,
                                                      torch.from_numpy(gaussian_stds), compute_std_gradients)

I = I_or_phi[0]

#ws = DS.WeightedSmoothingModel(nr_of_gaussians)
ws = DS.ConsistentWeightedSmoothingModel(nr_of_gaussians,gaussian_weights)
smoothed_v = ws(vcollection,I,retain_weights=True)

plt.subplot(7,2,1)
plt.imshow(m[0,0,...].detach().cpu().numpy())
#plt.colorbar()

plt.subplot(7,2,2)
plt.imshow(m[0,1,...].detach().cpu().numpy())
#plt.colorbar()

for i in range(5):

    plt.subplot(7,2,3+2*i)
    plt.imshow(vcollection[i,0,0,...].detach().cpu().numpy())
    #plt.colorbar()

    plt.subplot(7,2,4+2*i)
    plt.imshow(vcollection[i,0,1,...].detach().cpu().numpy())
    #plt.colorbar()

plt.subplot(7,2,13)
plt.imshow(smoothed_v[0,0,...].detach().cpu().numpy())
#plt.colorbar()

plt.subplot(7,2,14)
plt.imshow(smoothed_v[0,1,...].detach().cpu().numpy())
#plt.colorbar()

plt.show()

sz = vcollection.size()
#torch.Size([5, 5, 2, 64, 64])

print(spacing)
