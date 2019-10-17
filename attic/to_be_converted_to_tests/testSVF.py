"""
This is a test implementation to see if we can use pyTorch to solve
LDDMM-style registration problems via automatic differentiation.

Contributors:
  Marc Niethammer: mn@cs.unc.edu
"""

# first do the torch imports
from __future__ import print_function
from builtins import str
from builtins import range
import torch
from torch.autograd import Variable

import numpy as np
import time

import registration_networks as RN
import visualize_registration_results as vizReg

import example_generation as eg
import utils

# select the desired dimension of the registration
dim = 2
sz = np.tile( 30, dim )         # size of the desired images: (sz)^dim

params = dict()
params['len_s'] = sz.min()//4
params['len_l'] = sz.min()//3

# create a default image size with two sample squares
cs = eg.CreateSquares(sz)
I0,I1 = cs.create_image_pair(params)

# spacing so that everything is in [0,1]^2 for now
spacing = 1./(sz-1)
print ('Spacing = ' + str( spacing ) )

# some debugging output to show image gradients
# compute gradients
#vizReg.debugOutput( I0, I1, spacing )

# some settings for the registration energy
# Reg[\Phi,\alpha,\gamma] + 1/\sigma^2 Sim[I(1),I_1]

params['sigma']=0.1
params['gamma']=1.
params['alpha']=0.2

params['similarityMeasure'] = 'ssd'
params['regularizer'] = 'helmholtz'

params['numberOfTimeSteps'] = 10

model = RN.SVFNet(sz,spacing,params)    # instantiate a stationary velocity field model
print(model)

# create the source and target image as pyTorch variables
ISource =torch.from_numpy( I0.copy() )
ITarget =torch.from_numpy( I1 )

# use the standard SVFLoss
criterion = RN.SVFLoss(list(model.parameters())[0],sz,spacing,params)
# use LBFGS as optimizer; this is essential for convergence when not using the Hilbert gradient
optimizer = torch.optim.LBFGS(model.parameters(),
                              lr=1,max_iter=5,max_eval=10,
                              tolerance_grad=1e-3,tolerance_change=1e-4,
                              history_size=5)

# optimize for a few steps
start = time.time()

for iter in range(100):

    def closure():
        optimizer.zero_grad()
        # Forward pass: Compute predicted y by passing x to the model
        IWarped = model(ISource)
        # Compute loss
        loss = criterion(IWarped, ITarget)
        loss.backward()
        return loss

    optimizer.step(closure)
    cIWarped = model(ISource)

    if iter%1==0:
        energy, similarityEnergy, regEnergy = criterion.get_energy(cIWarped, ITarget)
        print('Iter {iter}: E={energy}, similarityE={similarityE}, regE={regE}'
              .format(iter=iter,
                      energy=utils.t2np(energy),
                      similarityE=utils.t2np(similarityEnergy),
                      regE=utils.t2np(regEnergy)))

    if iter%10==0:
        vizReg.showCurrentImages(iter,ISource,ITarget,cIWarped)

print('time:', time.time() - start)
