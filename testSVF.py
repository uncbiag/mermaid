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

import numpy as np

import registration_networks as RN
import visualize_registration_results as vizReg

import example_generation as eg

# select the desired dimension of the registration
dim = 3
sz = 30         # size of the desired images: (sz)^dim

# create a default image size with two sample squares
cs = eg.CreateSquares(dim,sz)
I0,I1 = cs.create_image_pair()

# spacing so that everything is in [0,1]^2 for now
# TODO: change to support arbitrary spacing
hx = 1./(sz-1)
spacing = np.tile( hx, dim )
print ('Spacing = ' + str( spacing ) )

# some debugging output to show image gradients
# compute gradients
vizReg.debugOutput( I0, I1, spacing )

# some settings for the registration energy
# Reg[\Phi,\alpha,\gamma] + 1/\sigma^2 Sim[I(1),I_1]
params = dict()
params['sigma']=0.1
params['gamma']=1.
params['alpha']=0.2

model = RN.SVFNet(sz,spacing,params)    # instantiate a stationary velocity field model
print(model)

# create the source and target image as pyTorch variables
ISource = Variable( torch.from_numpy( I0.copy() ), requires_grad=False )
ITarget = Variable( torch.from_numpy( I1 ), requires_grad=False )

# use the standard SVFLoss
criterion = RN.SVFLoss(list(model.parameters())[0],sz,spacing,params)
# use LBFGS as optimizer; this is essential for convergence when not using the Hilbert gradient
optimizer = torch.optim.LBFGS(model.parameters(),lr=1)

# optimize for a few steps
for iter in range(100):

    def closure():
        optimizer.zero_grad()
        # Forward pass: Compute predicted y by passing x to the model
        IWarped = model(ISource)
        # Compute and print loss
        loss = criterion(IWarped, ITarget)
        #print(iter, loss.data[0])
        loss.backward()
        return loss

    optimizer.step(closure)
    p = list(model.parameters())
    #print( 'v norm = ' + str( (p[0]**2).sum()[0] ) )
    cIWarped = model(ISource)

    if iter%1==0:
        energy, ssdEnergy, regEnergy = criterion.getEnergy(cIWarped, ITarget)
        print('Iter {iter}: E={energy}, ssdE={ssdE}, regE={regE}'
              .format(iter=iter, energy=energy, ssdE=ssdEnergy, regE=regEnergy))

    if iter%10==0:
        vizReg.showCurrentImages(iter,ISource,ITarget,cIWarped)
