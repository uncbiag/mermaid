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
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

import finite_differences as fd
import rungekutta_integrators as RK
import forward_models as FM
import utils
import registration_networks as RN

from modules.stn_nd import STN_ND

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

dim = 2 # dimension

# create a default image size with two sample squares
sz = 30         # size of the 2D image: sz x sz
c = sz/2        # center coordinates
len_s = sz/6    # half of side-length for small square
len_l = sz/3    # half of side-length for large square

# create two example square images
I0 = np.zeros([sz,sz], dtype='float32' )
I1 = np.zeros([sz,sz], dtype='float32' )

# create small and large squares
I0[c-len_s:c+len_s,c-len_s:c+len_s] = 1
I1[c-len_l:c+len_l,c-len_l:c+len_l] = 1

# spacing so that everything is in [0,1]^2 for now
# TODO: change to support arbitrary spacing
hx = 1./(sz-1)
print ('Spacing = ' + str( hx ) )

fdnp = fd.FD_np()       # numpy finite differencing
fdnp.setspacing(hx,hx)  # set spacing

# some debugging output to show image gradients
# compute gradients
dx0 = fdnp.dXc( I0 )
dy0 = fdnp.dYc( I0 )

dx1 = fdnp.dXc( I1 )
dy1 = fdnp.dYc( I1 )

plt.figure(1)
plt.setp( plt.gcf(),'facecolor','white')
plt.style.use('bmh')

plt.subplot(321)
plt.imshow( I0 )
plt.title( 'I0' )
plt.subplot(323)
plt.imshow( dx0 )
plt.subplot(325)
plt.imshow( dy0 )

plt.subplot(322)
plt.imshow( I1 )
plt.title( 'I1' )
plt.subplot(324)
plt.imshow( dx1 )
plt.subplot(326)
plt.imshow( dy1 )
#plt.axis('tight')
plt.show( block=False )


def showCurrentImages(iter,iS,iT,iW):
    """
    Show current 2D registration results in relation to the source and target images
    :param iter: iteration number
    :param iS: source image
    :param iT: target image
    :param iW: current warped image
    :return: no return arguments
    """
    plt.figure(1)
    plt.clf()

    plt.suptitle( 'Iteration = ' + str(iter))
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.subplot(131)
    plt.imshow(utils.t2np(iS))
    plt.colorbar()
    plt.title('source image')

    plt.subplot(132)
    plt.imshow(utils.t2np(iT))
    plt.colorbar()
    plt.title('target image')

    plt.subplot(133)
    plt.imshow(utils.t2np(iW))
    plt.colorbar()
    plt.title('warped image')

    plt.show(block=False)
    plt.draw_all(force=True)
    plt.waitforbuttonpress()

# some settings for the registration energy
# Reg[\Phi,\alpha,\gamma] + 1/\sigma^2 Sim[I(1),I_1]
params = dict()
params['sigma']=0.1
params['gamma']=1.
params['alpha']=0.2

model = RN.SVFNetMap(sz,params)    # instantiate a stationary velocity field model
print(model)

# create the source and target image as pyTorch variables
ISource = Variable( torch.from_numpy( I0.copy() ), requires_grad=False )
ITarget = Variable( torch.from_numpy( I1 ), requires_grad=False )

# create the identity map
id2 = utils.identityMap([sz,sz])

id = np.zeros([sz,sz,2], dtype='float32')
id[:,:,0] = id2[0]
id[:,:,1] = id2[1]

identityMap = Variable( torch.from_numpy( id ), requires_grad=False )

# use the standard SVFLoss
criterion = RN.SVFLossMap(list(model.parameters())[0],sz,params)
# use LBFGS as optimizer; this is essential for convergence when not using the Hilbert gradient
optimizer = torch.optim.LBFGS(model.parameters(),lr=1)

stn = STN_ND( dim )
# optimize for a few steps
for iter in range(100):
    print( 'Iteration = ' + str( iter ) )

    def closure():
        optimizer.zero_grad()
        # Forward pass: Compute predicted y by passing x to the model
        phiWarped = model(identityMap)
        # Compute and print loss
        loss = criterion(phiWarped, ISource, ITarget)
        print(iter, loss.data[0])
        loss.backward()
        return loss

    optimizer.step(closure)
    p = list(model.parameters())
    print( 'v norm = ' + str( (p[0]**2).sum()[0] ) )
    phiWarped = model(identityMap)
    if iter%10==0:
        phiWarped_stn = phiWarped.view(torch.Size([1, sz, sz, 2]))
        ISource_stn = ISource.view(torch.Size([1,sz,sz,1]))
        I1Warped = stn(ISource_stn, phiWarped_stn)
        showCurrentImages(iter,ISource,ITarget,I1Warped[0,:,:,0])
