# first do the torch imports
import set_pyreg_paths

import torch
from torch.autograd import Variable
from pyreg.data_wrapper import AdaptVal
import numpy as np
import pyreg.module_parameters as MP
import pyreg.example_generation as eg
import pyreg.utils as utils

import pyreg.smoother_factory as SF

import matplotlib.pyplot as plt

params = MP.ParameterDict()

dim = 2

szEx = np.tile( 64, dim )
I0,I1,spacing= eg.CreateSquares(dim).create_image_pair(szEx,params) # create a default image size with two sample squares
sz = np.array(I0.shape)

# create the source and target image as pyTorch variables
ISource = AdaptVal(Variable(torch.from_numpy(I0.copy()), requires_grad=False))

smoother = SF.MySingleGaussianFourierSmoother(sz[2:],spacing,params)

g_std = smoother.get_gaussian_std()

ISmooth = smoother.smooth_scalar_field(ISource)
ISmooth.backward(torch.ones_like(ISmooth))
#ISmooth.backward(torch.zeros_like(ISmooth))

print('g_std.grad')
print(g_std.grad)


plt.subplot(121)
plt.imshow(utils.t2np(ISource[0,0,...]))

plt.subplot(122)
plt.imshow(utils.t2np(ISmooth[0,0,...]))
plt.show()

#ITarget = AdaptVal(Variable(torch.from_numpy(I1), requires_grad=False))


