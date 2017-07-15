'''
This is a test implementation to see if we can use pyTorch to solve
LDDMM-style registration problems via automatic differentiation.

Contributors:
  Marc Niethammer: mn@cs.unc.edu
'''

# first do the torch imports
from __future__ import print_function
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

import finitedifferences as fd
import rungekutta_integrators as RK
import forward_models as FM

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

# create a default image size
sz = 20
c = sz/2
len_s = sz/6
len_l = sz/3

# create an example image
I0 = np.zeros([sz,sz], dtype='float32' )
I1 = np.zeros([sz,sz], dtype='float32' )

# create small and large squares and show them
I0[c-len_s:c+len_s,c-len_s:c+len_s] = 1
I1[c-len_l:c+len_l,c-len_l:c+len_l] = 1

hx = 1./(sz-1)
print ('Spacing = ' + str( hx ) )
fdnp = fd.FD_np()
fdnp.setspacing(hx,hx)

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


def t2np( v ):
    return (v.data).cpu().numpy()

def showI( I ):
    plt.figure(2)
    plt.imshow(t2np(I))
    plt.colorbar()
    plt.show()

def showV( v ):
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(t2np(v[:, :, 0]))
    plt.colorbar()
    plt.title('v[0]')

    plt.subplot(122)
    plt.imshow(t2np(v[:, :, 1]))
    plt.colorbar()
    plt.title('v[1]')
    plt.show()


def showI(I):
    plt.figure(2)
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.imshow(t2np(I))
    plt.colorbar()
    plt.show()
    plt.title('I')

def showVandI(v,I):
    plt.figure(1)
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.subplot(131)
    plt.imshow(t2np(I))
    plt.colorbar()
    plt.title('I')

    plt.subplot(132)
    plt.imshow(t2np(v[:, :, 0]))
    plt.colorbar()
    plt.title('v[0]')

    plt.subplot(133)
    plt.imshow(t2np(v[:, :, 1]))
    plt.colorbar()
    plt.title('v[1]')
    plt.show()

def showCurrentImages(iter,iS,iT,iW):
    plt.figure(1)
    plt.clf()

    plt.suptitle( 'Iteration = ' + str(iter))
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.subplot(131)
    plt.imshow(t2np(iS))
    plt.colorbar()
    plt.title('source image')

    plt.subplot(132)
    plt.imshow(t2np(iT))
    plt.colorbar()
    plt.title('target image')

    plt.subplot(133)
    plt.imshow(t2np(iW))
    plt.colorbar()
    plt.title('warped image')

    plt.show(block=False)
    plt.draw_all(force=True)
    plt.waitforbuttonpress()

# some settings for the registration energy
params = dict()
params['sigma']=0.1
params['gamma']=1.
params['alpha']=0.2

sigma = params['sigma']
gamma = params['gamma']
alpha = params['alpha']

class SVFNet(nn.Module):
# TODO: do proper spacing
    def __init__(self,sz,params):
        super(SVFNet, self).__init__()
        self.sigma = params['sigma']
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.hx = hx = 1./(sz-1)
        self.nrOfTimeSteps = 10
        self.tFrom = 0.
        self.tTo = 1.
        self.v = Parameter(torch.zeros(sz, sz, 2))
        self.advection = FM.AdvectImage( self.hx, self.hx )
        self.integrator = RK.RK4(self.advection.f,self.advection.u,self.v)

    def forward(self, I):
        I1 = self.integrator.solve([I], self.tFrom, self.tTo, self.nrOfTimeSteps)
        return I1[0]


class SVFLoss(nn.Module):
# make velocity field an input here
    def __init__(self,v,sz,params):
        super(SVFLoss, self).__init__()
        self.sigma = params['sigma']
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.v = v
        self.fdt = fd.FD_torch()
        self.hx = hx = 1. / (sz - 1)
        self.fdt.setspacing( self.hx, self.hx )

    def computeRegularizer(self, v, alpha, gamma):
        # just do the standard component-wise gamma id -\alpha \Delta
        ndim = v.ndimension()
        if ndim != 3:
            raise ValueError('Regularizer only supported in 2D at the moment')

        Lv = Variable(torch.zeros(v.size()), requires_grad=False)
        for i in [0, 1]:
            Lv[:, :, i] = v[:, :, i] * gamma - self.fdt.lap(v[:, :, i]) * alpha

        # now compute the norm
        return (Lv[:, :, 0] ** 2 + Lv[:, :, 1] ** 2).sum()

    def forward(self, I1_warped, I1_target):
        ssd = ((I1_target - I1_warped) ** 2).sum() / (self.sigma ** 2)
        reg = self.computeRegularizer( self.v, self.alpha, self.gamma)
        energy = ssd + reg
        print('Energy = ' + str(t2np(energy)[0]) + '; ssd = ' + str(t2np(ssd)[0]) + '; reg = ' + str(t2np(reg)[0]))
        # compute max velocity
        maxVelocity = ( ( self.v[0]**2 + self.v[1]**2 ).max() ).sqrt()
        print('Max velocity = ' + str(t2np(maxVelocity)[0]) )
        return energy

model = SVFNet(sz,params)
print(model)

ISource = Variable( torch.from_numpy( I0.copy() ), requires_grad=False )
ITarget = Variable( torch.from_numpy( I1 ), requires_grad=False )

criterion = SVFLoss(list(model.parameters())[0],sz,params)
optimizer = torch.optim.LBFGS(model.parameters(),lr=1)
for iter in range(100):
    print( 'Iteration = ' + str( iter ) )

    def closure():
        optimizer.zero_grad()
        # Forward pass: Compute predicted y by passing x to the model
        IWarped = model(ISource)
        # Compute and print loss
        loss = criterion(IWarped, ITarget)
        print(iter, loss.data[0])
        loss.backward()
        return loss

    optimizer.step(closure)
    p = list(model.parameters())
    print( 'v norm = ' + str( (p[0]**2).sum()[0] ) )
    cIWarped = model(ISource)
    if iter%10==0:
        showCurrentImages(iter,ISource,ITarget,cIWarped)
