from __future__ import print_function
from builtins import str
import matplotlib.pyplot as plt
import numpy as np
import smoother_factory as sf
import example_generation as eg

import torch
from torch.autograd import Variable

import utils

dim = 1
sz = np.tile( 30, dim )         # size of the desired images: (sz)^dim

params = dict()
params['len_s'] = sz.min()//6
params['len_l'] = sz.min()//3

# create a default image size with two sample squares
cs = eg.CreateSquares(sz)
I0,I1 = cs.create_image_pair(params)

# create the source and target image as pyTorch variables
ISource =  torch.from_numpy( I0 )
ITarget =  torch.from_numpy( I1 )

# spacing so that everything is in [0,1]^2 for now
spacing = 1./(sz-1)
print ('Spacing = ' + str( spacing ) )

s = sf.SmootherFactory( sz, spacing ).create_smoother('diffusion', {'iter':10})
r = s.smooth_scalar_field(ISource)

plt.figure(1)
plt.plot(utils.t2np(ISource))
plt.plot(utils.t2np(r))
plt.plot(utils.t2np(ISource))

plt.show()

