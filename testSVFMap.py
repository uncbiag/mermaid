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

import utils
import registration_networks as RN

from modules.stn_nd import STN_ND

import numpy as np

import visualize_registration_results as vizReg
import example_generation as eg

# select the desired dimension of the registration
dim = 2
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

model = RN.SVFNetMap(sz,spacing,params)    # instantiate a stationary velocity field model
print(model)

# create the source and target image as pyTorch variables
ISource = Variable( torch.from_numpy( I0.copy() ), requires_grad=False )
ITarget = Variable( torch.from_numpy( I1 ), requires_grad=False )

# create the identity map
if dim==1:
    id1 = utils.identityMap([sz])
    id = np.zeros([sz], dtype='float32')
    id = id1
elif dim==2:
    id2 = utils.identityMap([sz, sz])
    id = np.zeros([sz, sz, 2], dtype='float32')
    id[:, :, 0] = id2[0]
    id[:, :, 1] = id2[1]
elif dim==3:
    id3 = utils.identityMap([sz, sz, sz])
    id = np.zeros([sz, sz, sz, 3], dtype='float32')
    id[:, :, :, 0] = id3[0]
    id[:, :, :, 1] = id3[1]
    id[:, :, :, 2] = id3[2]
else:
    raise ValueError('Can only deal with images of dimension 1-3 so far')

identityMap = Variable( torch.from_numpy( id ), requires_grad=False )

criterion = RN.SVFLossMap(list(model.parameters())[0],sz,spacing,params) # stationaty velocity field with maps
# use LBFGS as optimizer; this is essential for convergence when not using the Hilbert gradient
optimizer = torch.optim.LBFGS(model.parameters(),lr=1)

stn = STN_ND( dim ) # spatial transformer so we can apply the map

# optimize for a few steps
for iter in range(100):

    def closure():
        optimizer.zero_grad()
        # Forward pass: Compute predicted y by passing x to the model
        phiWarped = model(identityMap)
        # Compute and print loss
        loss = criterion(phiWarped, ISource, ITarget)
        #print(iter, loss.data[0])
        loss.backward()
        return loss

    optimizer.step(closure)
    #p = list(model.parameters())
    #print( 'v norm = ' + str( (p[0]**2).sum()[0] ) )
    phiWarped = model(identityMap)

    if iter%1==0:
        energy, ssdEnergy, regEnergy = criterion.getEnergy(phiWarped, ISource, ITarget)
        print('Iter {iter}: E={energy}, ssdE={ssdE}, regE={regE}'
              .format(iter=iter, energy=energy, ssdE=ssdEnergy, regE=regEnergy))

    if iter%10==0:
        if dim==1:
            phiWarped_stn = phiWarped.view(torch.Size([1, sz]))
            ISource_stn = ISource.view(torch.Size([1, sz, 1]))
            I1Warped = stn(ISource_stn, phiWarped_stn)
            vizReg.showCurrentImages(iter, ISource, ITarget, I1Warped[0, :, 0])
        elif dim==2:
            phiWarped_stn = phiWarped.view(torch.Size([1, sz, sz, 2]))
            ISource_stn = ISource.view(torch.Size([1,sz,sz,1]))
            I1Warped = stn(ISource_stn, phiWarped_stn)
            vizReg.showCurrentImages(iter,ISource,ITarget,I1Warped[0,:,:,0])
        elif dim==3:
            phiWarped_stn = phiWarped.view(torch.Size([1, sz, sz, sz, 3]))
            ISource_stn = ISource.view(torch.Size([1, sz, sz, sz, 1]))
            I1Warped = stn(ISource_stn, phiWarped_stn)
            vizReg.showCurrentImages(iter, ISource, ITarget, I1Warped[0, :, :, :, 0])