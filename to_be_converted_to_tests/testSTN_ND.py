"""
Tests the spatial transformer network code
"""

from __future__ import print_function

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

import utils
import math

import time

from modules.stn_nd import STN_ND, STN

import matplotlib.pyplot as plt
import matplotlib.image as img

dim = 3 # desired dimension

# create a default image which we want to transform
sz = 30         # size of the 2D image: sz x sz
coorCenter = sz//2        # center coordinates
len_s = sz//6    # half of side-length for small square

# create small square
if dim==1:
    # create the images
    I0 = np.zeros([1,sz,1], dtype='float32' )
    I0[0,coorCenter-len_s:coorCenter+len_s,0] = 1
    IC = np.ones([1,sz,sz,1], dtype='float32' )
    # create the map
    id1 = utils.identityMap([sz])
    XX = id1
    grid = np.zeros([1, sz], dtype='float32')
    grid[0, :] = XX + 0.2
elif dim==2:
    # create the images
    I0 = np.zeros([1,sz,sz,1], dtype='float32' )
    I0[0,coorCenter-2*len_s:coorCenter+2*len_s,coorCenter-len_s:coorCenter+len_s,0] = 1
    IC = np.ones([1,sz,sz,1], dtype='float32' )
    # create the map
    id2 = utils.identityMap([sz,sz])
    # create the grid
    XX = id2[0]
    YY = id2[1]
    grid = np.zeros([1, sz, sz, 2], dtype='float32')
    grid[0, :, :, 0] = XX + 0.2
    grid[0, :, :, 1] = YY
elif dim==3:
    # create the images
    I0 = np.zeros([1, sz, sz, sz, 1], dtype='float32')
    I0[0, coorCenter - 2 * len_s:coorCenter + 2 * len_s, coorCenter - len_s:coorCenter + len_s,
        int( math.floor(coorCenter-1.5*len_s ) ): int( math.floor(coorCenter+1.5*len_s ) )] = 1
    IC = np.ones([1, sz, sz, sz, 1], dtype='float32')
    # create the map
    id3 = utils.identityMap([sz,sz,sz])
    XX = id3[0]
    YY = id3[1]
    ZZ = id3[2]
    grid = np.zeros([1, sz, sz, sz, 3], dtype='float32')
    grid[0, :, :, :, 0] = XX + 0.2
    grid[0, :, :, :, 1] = YY
    grid[0, :, :, :, 2] = ZZ
else:
    raise ValueError('Only dimensions 1 to 3 are currently supported.')

ICV = torch.from_numpy(IC)
ISource =  torch.from_numpy( I0 )
gridV = torch.from_numpy( grid )
gridV.requires_grad = True
# now run it through the spatial transformer network
s = STN_ND( dim )
start = time.time()
out = s(ISource, gridV)
print(out.size(), ' model evaluation time time:', time.time() - start)
# compute the gradient to check that this works
start = time.time()
#out.backward(gridV.data)
out.backward( ICV )
grad = gridV.grad
print(gridV.grad.size(), ' gradient evaluation time:', time.time() - start)

'''
# If we want to compare to the old 2D implementation, this can be done here
so = STN()
outo = s(ISource, gridV)
outo.backward( ICV )
gradO = gridV.grad
'''

# now visualize the results depending on dimension
def visualize1D():
    plt.figure(1)
    plt.subplot(121)
    plt.plot(I0[0, :, 0])
    plt.subplot(122)
    plt.plot(utils.t2np(out[0, :, 0]))
    plt.show()

def visualize2D():
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(I0[0, :, :, 0], clim=(-1, 1))
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(utils.t2np(out[0, :, :, 0]), clim=(-1, 1))
    plt.colorbar()
    plt.show()

def visualize3D():
    plt.figure(1)
    plt.subplot(231)
    plt.imshow(I0[0, coorCenter, :, :, 0], clim=(-1, 1))
    plt.colorbar()
    plt.subplot(232)
    plt.imshow(I0[0, :, coorCenter, :, 0], clim=(-1, 1))
    plt.colorbar()
    plt.subplot(233)
    plt.imshow(I0[0, :, :, coorCenter, 0], clim=(-1, 1))
    plt.colorbar()

    plt.subplot(234)
    plt.imshow(utils.t2np(out[0, coorCenter, :, :, 0]), clim=(-1, 1))
    plt.colorbar()
    plt.subplot(235)
    plt.imshow(utils.t2np(out[0, :, coorCenter, :, 0]), clim=(-1, 1))
    plt.colorbar()
    plt.subplot(236)
    plt.imshow(utils.t2np(out[0, :, :, coorCenter, 0]), clim=(-1, 1))
    plt.colorbar()

    plt.show()

def visualize1DGrad():
    plt.figure(1)
    plt.plot(utils.t2np(grad[0,:]))
    plt.title('grad')
    plt.show()

def visualize2DGrad():
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(utils.t2np(grad[0, :, :, 0])) #, clim=(-1, 1))
    plt.title('gradX')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(utils.t2np(grad[0, :, :, 1])) #, clim=(-1, 1))
    plt.title('gradY')
    plt.colorbar()
    plt.show()

def visualize3DGrad():
    plt.figure(1)

    plt.subplot(331)
    plt.imshow(utils.t2np(grad[0, coorCenter, :, :, 0])) #, clim=(-1, 1))
    plt.title('gradX - xslice')
    plt.colorbar()
    plt.subplot(332)
    plt.imshow(utils.t2np(grad[0, :, coorCenter, :, 0])) #, clim=(-1, 1))
    plt.title('gradX - yslice')
    plt.colorbar()
    plt.subplot(333)
    plt.imshow(utils.t2np(grad[0, :, :, coorCenter, 0])) #, clim=(-1, 1))
    plt.title('gradX - zslice')
    plt.colorbar()

    plt.subplot(334)
    plt.imshow(utils.t2np(grad[0, coorCenter, :, :, 1]))  # , clim=(-1, 1))
    plt.title('gradY - xslice')
    plt.colorbar()
    plt.subplot(335)
    plt.imshow(utils.t2np(grad[0, :, coorCenter, :, 1]))  # , clim=(-1, 1))
    plt.title('gradY - yslice')
    plt.colorbar()
    plt.subplot(336)
    plt.imshow(utils.t2np(grad[0, :, :, coorCenter, 1]))  # , clim=(-1, 1))
    plt.title('gradY - zslice')
    plt.colorbar()

    plt.subplot(337)
    plt.imshow(utils.t2np(grad[0, coorCenter, :, :, 2]))  # , clim=(-1, 1))
    plt.title('gradZ - xslice')
    plt.colorbar()
    plt.subplot(338)
    plt.imshow(utils.t2np(grad[0, :, coorCenter, :, 2]))  # , clim=(-1, 1))
    plt.title('gradZ - yslice')
    plt.colorbar()
    plt.subplot(339)
    plt.imshow(utils.t2np(grad[0, :, :, coorCenter, 2]))  # , clim=(-1, 1))
    plt.title('gradZ - zslice')
    plt.colorbar()

    plt.show()

if dim==1:
    visualize1D()
    visualize1DGrad()
elif dim==2:
    visualize2D()
    visualize2DGrad()
elif dim==3:
    visualize3D()
    visualize3DGrad()
else:
    raise ValueError('Can only visualize dimensions 1-3')


