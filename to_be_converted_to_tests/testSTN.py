"""
Tests the spatial transformer network code
"""

from __future__ import print_function

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

import utils

import time

from modules.stn import STN
from modules.gridgen import AffineGridGen, CylinderGridGen, CylinderGridGenV2, DenseAffine3DGridGen, DenseAffine3DGridGen_rotate

import matplotlib.pyplot as plt
import matplotlib.image as img

# create a default image which we want to transform
sz = 100         # size of the 2D image: sz x sz
c = sz//2        # center coordinates
len_s = sz//6    # half of side-length for small square

# create small square
I0 = np.zeros([1,sz,sz,1], dtype='float32' )
I0[0,c-2*len_s:c+2*len_s,c-len_s:c+len_s,0] = 1

# now create the grids
xvals = np.array(np.linspace(-1,1,sz))
yvals = np.array(np.linspace(-1,1,sz))

YY,XX = np.meshgrid(xvals,yvals)

grid = np.zeros([1,sz,sz,2], dtype='float32')
grid[0,:,:,0] = XX + 0.2
grid[0,:,:,1] = YY

ISource = torch.from_numpy( I0 )
gridV =  torch.from_numpy( grid )

#print input2.data
#s = STN(layout = 'BCHW')
s = STN()
start = time.time()
out = s(ISource, gridV)

plt.figure(1)
plt.subplot(121)
plt.imshow( I0[0,:,:,0] )
plt.subplot(122)
plt.imshow( utils.t2np(out[0,:,:,0]) )

print(out.size(), 'time:', time.time() - start)
start = time.time()
out.backward(gridV.data)
print(gridV.grad.size(), 'time:', time.time() - start)

