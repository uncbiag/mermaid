# first do the torch imports
from __future__ import print_function
from builtins import str
from builtins import range
import torch
from torch.autograd import Variable
import time

import utils
import numpy as np
import example_generation as eg

# select the desired dimension of the registration
dim = 3
sz = np.tile( 100, dim )         # size of the desired images: (sz)^dim

# create a default image size with two sample squares
cs = eg.CreateSquares(sz)
I0,I1 = cs.create_image_pair()

# spacing so that everything is in [0,1]^2 for now
spacing = 1./(sz-1)
print ('Spacing = ' + str( spacing ) )

# create the source and target image as pyTorch variables
ISource =  torch.from_numpy( I0.copy() )
ITarget =  torch.from_numpy( I1 )

# create the identity map [-1,1]^d
id = utils.identityMap(sz)
identityMap =  torch.from_numpy( id )

# just do basic interpolation with the identity map, to time the STN
start = time.time()
for iter in range(1000):
    print( 'Iteration: ' + str(iter))
    I1Warped = utils.computeWarpedImage(ISource,identityMap)
print('time:', time.time() - start)
