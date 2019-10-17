# first do the torch imports
from __future__ import print_function
from builtins import str
import torch
from torch.autograd import Variable
import time

import utils

import numpy as np
import example_generation as eg

import matplotlib.pyplot as plt

import custom_pytorch_extensions as ce

nrPoints = 100
nrPointsKernel = 100

dim = 2
sz = np.tile( nrPoints, dim )         # size of the desired images: (sz)^dim

params = dict()
params['len_s'] = sz.min()//3
params['len_l'] = sz.min()//5

# create a default image size with two sample squares
cs = eg.CreateSquares(sz)
I0,I1 = cs.create_image_pair(params)

# spacing so that everything is in [0,1]^2 for now
spacing = 1./(sz-1)
print ('Spacing = ' + str( spacing ) )

# 2D
X = utils.identityMap([nrPointsKernel,nrPointsKernel])
gus = utils.computeNormalizedGaussian(3*X,[0,0],[1,1])
g = gus

fconvFilter = ce.FourierConvolution(g, sz)
Ib = fconvFilter(torch.from_numpy(I0))
Ib.requires_grad = True

plt.subplot(221)
plt.imshow(g)
plt.subplot(222)
plt.imshow(I0)
plt.subplot(223)
plt.imshow(Ib.data.numpy())
plt.subplot(224)
plt.imshow(I0-Ib.data.numpy())
plt.show()
