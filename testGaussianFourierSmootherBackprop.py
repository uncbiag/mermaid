# first do the torch imports
from __future__ import print_function
import torch
from torch.autograd import Variable
import time

import numpy as np

import example_generation as eg
import custom_pytorch_extensions as ce
import utils

# select the desired dimension of the registration
nrOfIterations = 5 # number of iterations for the optimizer
#modelName = 'SVF'
modelName = 'LDDMMShooting'
dim = 1
sz = np.tile( 10, dim )         # size of the desired images: (sz)^dim

params = dict()
params['len_s'] = sz.min()/6
params['len_l'] = sz.min()/5

# create a default image size with two sample squares
cs = eg.CreateSquares(sz)
I0,I1 = cs.create_image_pair(params)

# spacing so that everything is in [0,1]^2 for now
spacing = 1./(sz-1)
print ('Spacing = ' + str( spacing ) )

# create the source and target image as pyTorch variables
ISource = Variable( torch.from_numpy( I0.copy() ), requires_grad=True )
ITarget = Variable( torch.from_numpy( I1 ), requires_grad=False )

# smooth both a little bit
#s = SF.SmootherFactory( spacing ).createSmoother('diffusion',{'iter':10})

mus = np.zeros(dim)
stds = np.ones(dim)
id = utils.identityMap(sz)
g = utils.computeNormalizedGaussian(id,mus,stds)
FFilter = ce.createComplexFourierFilter(g,sz)

I1 = ce.fourierConvolution(ISource,FFilter)
I2 = ce.fourierConvolution(I1,FFilter)

start = time.time()

#l1 = s.computeSmootherScalarField(ISource)
#l2 = s.computeSmootherScalarField(l1)
#l3 = s.computeSmootherScalarField(l2)
#l4 = s.computeSmootherScalarField(l3)

#loss = l4.sum()
loss = I2.sum()
print(loss)
loss.backward()
print(ISource.grad)

print('time:', time.time() - start)
