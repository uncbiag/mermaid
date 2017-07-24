"""
This is a test implementation to see if we can use pyTorch to solve
LDDMM-style registration problems via automatic differentiation.

Contributors:
  Marc Niethammer: mn@cs.unc.edu
"""

# first do the torch imports
from __future__ import print_function
import torch
from torch.autograd import Variable
import time

import utils

import numpy as np

import visualize_registration_results as vizReg
import example_generation as eg

import model_factory as MF

# select the desired dimension of the registration
useMap = True # set to true if a map-based implementation should be used
modelName = 'SVF'
dim = 2
sz = np.tile( 30, dim )         # size of the desired images: (sz)^dim

params = dict()
params['len_s'] = sz.min()/6
params['len_l'] = sz.min()/3

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

mf = MF.ModelFactory( sz, spacing )

model,criterion = mf.createRegistrationModel( modelName, useMap, params)
print(model)

# create the source and target image as pyTorch variables
ISource = Variable( torch.from_numpy( I0.copy() ), requires_grad=False )
ITarget = Variable( torch.from_numpy( I1 ), requires_grad=False )

if useMap:
    # create the identity map [-1,1]^d, since we will use a map-based implementation
    id = utils.identityMap(sz)
    identityMap = Variable( torch.from_numpy( id ), requires_grad=False )

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
        # 1) Forward pass: Compute predicted y by passing x to the model
        # 2) Compute loss
        if useMap:
            phiWarped = model(identityMap)
            loss = criterion(phiWarped, ISource, ITarget)
        else:
            IWarped = model(ISource)
            loss = criterion(IWarped, ITarget)

        loss.backward()
        return loss

    # take a step of the optimizer
    optimizer.step(closure)

    # apply the current state to either get the warped map or directly the warped source image
    if useMap:
        phiWarped = model(identityMap)
    else:
        cIWarped = model(ISource)

    if iter%1==0:
        if useMap:
            energy, similarityEnergy, regEnergy = criterion.getEnergy(phiWarped, ISource, ITarget)
        else:
            energy, similarityEnergy, regEnergy = criterion.getEnergy(cIWarped, ITarget)

        print('Iter {iter}: E={energy}, similarityE={similarityE}, regE={regE}'
              .format(iter=iter, energy=energy, similarityE=similarityEnergy, regE=regEnergy))

    if iter%10==0:
        if useMap:
            I1Warped = utils.computeWarpedImage(ISource,phiWarped)
            vizReg.showCurrentImages(iter, ISource, ITarget, I1Warped, phiWarped)
        else:
            vizReg.showCurrentImages(iter, ISource, ITarget, cIWarped)

print('time:', time.time() - start)
