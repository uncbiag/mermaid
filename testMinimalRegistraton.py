# first do the torch imports
import torch
from torch.autograd import Variable

import numpy as np

import set_pyreg_paths

import pyreg.example_generation as eg
import pyreg.module_parameters as pars
import pyreg.multiscale_optimizer as MO

modelName = 'lddmm_shooting_map'
useMap = True
dim = 2
nrOfIterations = 500 # number of iterations for the optimizer

# general parameters
params = pars.ParameterDict()

szEx = np.tile( 50, dim )         # size of the desired images: (sz)^dim
I0,I1= eg.CreateSquares(dim).create_image_pair(szEx,params) # create a default image size with two sample squares
sz = np.array(I0.shape)
spacing = 1./(sz[2::]-1) # the first two dimensions are batch size and number of image channels

# create the source and target image as pyTorch variables
ISource = Variable( torch.from_numpy( I0.copy() ), requires_grad=False )
ITarget = Variable( torch.from_numpy( I1 ), requires_grad=False )

so = MO.SingleScaleRegistrationOptimizer(sz,spacing,useMap,params)
so.set_model(modelName)

so.set_number_of_iterations(nrOfIterations)

so.set_source_image(ISource)
so.set_target_image(ITarget)

# and now do the optimization
so.optimize()

