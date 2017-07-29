"""
This is a test implementation to see if we can use pyTorch to solve
LDDMM-style registration problems via automatic differentiation.

Contributors:
  Marc Niethammer: mn@cs.unc.edu
"""

#TODO: Change all the images to be of the format 1x1x(...)

# first do the torch imports
from __future__ import print_function
import torch
from torch.autograd import Variable
import time

import numpy as np

from pyreg import utils
import pyreg.visualize_registration_results as vizReg
import pyreg.example_generation as eg

import pyreg.model_factory as MF
import pyreg.smoother_factory as SF

import pyreg.custom_optimizers as CO

# select the desired dimension of the registration
useMap = True # set to true if a map-based implementation should be used
visualize = True # set to true if intermediate visualizations are desired
nrOfIterations = 50 # number of iterations for the optimizer
modelName = 'SVF'
#modelName = 'LDDMMShooting'
#modelName = 'LDDMMShootingScalarMomentum'
dim = 2
sz = np.tile( 50, dim )         # size of the desired images: (sz)^dim

torch.set_num_threads(4) # not sure if this actually affects anything
print('Number of pytorch threads set to: ' + str(torch.get_num_threads()))

params = dict()
params['len_s'] = sz.min()/6
params['len_l'] = sz.min()/4

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

# smooth both a little bit
#s = SF.SmootherFactory( spacing ).createSmoother('diffusion',{'iter':10})
s = SF.SmootherFactory( sz, spacing ).createSmoother('gaussian', {'gaussianStd':0.05}) #,{'k_sz_h':np.tile(20,dim)})
ISource = s.computeSmootherScalarField(ISource)
ITarget = s.computeSmootherScalarField(ITarget)

if useMap:
    # create the identity map [-1,1]^d, since we will use a map-based implementation
    id = utils.identityMap(sz)
    identityMap = Variable( torch.from_numpy( id ), requires_grad=False )

# use LBFGS as optimizer; this is essential for convergence when not using the Hilbert gradient
#optimizer = torch.optim.LBFGS(model.parameters(),
#                              lr=1,max_iter=1,max_eval=5,
#                              tolerance_grad=1e-3,tolerance_change=1e-4,
#                              history_size=5)

optimizer = CO.LBFGS_LS(model.parameters(),
                              lr=1.0,max_iter=1,max_eval=5,
                              tolerance_grad=1e-3,tolerance_change=1e-4,
                              history_size=5,line_search_fn='backtracking')

#optimizer = torch.optim.LBFGS(model.parameters(),
#                              lr=1,max_iter=1,max_eval=5,
#                              tolerance_grad=1e-3,tolerance_change=1e-4,
#                              history_size=5)

# optimize for a few steps
start = time.time()

for iter in range(nrOfIterations):

    def closure():
        optimizer.zero_grad()
        # 1) Forward pass: Compute predicted y by passing x to the model
        # 2) Compute loss
        if useMap:
            phiWarped = model(identityMap, ISource)
            loss = criterion(phiWarped, ISource, ITarget)
        else:
            IWarped = model(ISource)
            loss = criterion(IWarped, ISource, ITarget)

        loss.backward()
        return loss

    # take a step of the optimizer
    optimizer.step(closure)

    # apply the current state to either get the warped map or directly the warped source image
    if useMap:
        phiWarped = model(identityMap, ISource)
    else:
        cIWarped = model(ISource)

    if iter%1==0:
        if useMap:
            energy, similarityEnergy, regEnergy = criterion.getEnergy(phiWarped, ISource, ITarget)
        else:
            energy, similarityEnergy, regEnergy = criterion.getEnergy(cIWarped, ISource, ITarget)

        print('Iter {iter}: E={energy}, similarityE={similarityE}, regE={regE}'
              .format(iter=iter,
                      energy=utils.t2np(energy),
                      similarityE=utils.t2np(similarityEnergy),
                      regE=utils.t2np(regEnergy)))

    if visualize:
        if iter%5==0:
            if useMap:
                I1Warped = utils.computeWarpedImage(ISource,phiWarped)
                vizReg.showCurrentImages(iter, ISource, ITarget, I1Warped, phiWarped)
            else:
                vizReg.showCurrentImages(iter, ISource, ITarget, cIWarped)

print('time:', time.time() - start)
