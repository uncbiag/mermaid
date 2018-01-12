# first do the torch imports
from __future__ import print_function
import os
import sys
os.chdir('../')

sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('./pyreg'))
sys.path.insert(0,os.path.abspath('./pyreg/libraries'))
import torch
from torch.autograd import Variable
import time
import pyreg.module_parameters as pars

import numpy as np
import pyreg.smoother_factory as SF

import pyreg.example_generation as eg
import pyreg.custom_pytorch_extensions as ce
import pyreg.utils as utils

# select the desired dimension of the registration

ce.check_fourier_conv()


nrOfIterations = 5 # number of iterations for the optimizer

modelName = 'SVF'
modelName = 'LDDMMShooting'
dim = 2
gaussianStd =0.05
params = pars.ParameterDict()
szEx = np.tile(120, 2)         # size of the desired images: (sz)^dim

params['square_example_images']=({},'Settings for example image generation')
params['square_example_images']['len_s'] = szEx.min()/6
params['square_example_images']['len_l'] = szEx.max()/4

# create a default image size with two sample squares
I0,I1,spacing= eg.CreateSquares(dim).create_image_pair(szEx,params)
sz = np.array(I0.shape)

assert( len(sz)==dim+2 )

print ('Spacing = ' + str( spacing ) )

# create the source and target image as pyTorch variables
ISource = Variable( torch.from_numpy( I0.copy() ), requires_grad=True )
ITarget = Variable( torch.from_numpy( I1 ), requires_grad=False )

# smooth both a little bit
#s = SF.SmootherFactory( spacing ).createSmoother('diffusion',{'iter':10})
cparams = params[('image_smoothing', {}, 'general settings to pre-smooth images')]
cparams[('smoother', {})]
cparams['smoother']['type'] = 'gaussianSpatial'
cparams['smoother']['gaussianStd'] =  0.005
s = SF.SmootherFactory(sz[2::], spacing).create_smoother(cparams)
ISource = s.smooth_scalar_field(ISource)

mus = np.zeros(dim)
stds = gaussianStd * np.ones(dim)
id = utils.identity_map(szEx,spacing)
g = utils.compute_normalized_gaussian(id, mus, stds)

FFilter,_ = ce.create_complex_fourier_filter(g, szEx)
fc1 = ce.FourierConvolution(FFilter)
I1 = fc1(ISource)

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
