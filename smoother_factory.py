'''
General purpose regularizers which can be used
'''

from abc import ABCMeta, abstractmethod

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import finite_differences as fd
import utils

import custom_pytorch_extensions as ce

class Smoother(object):
    __metaclass__ = ABCMeta

    def __init__(self, sz, spacing, params):
        self.sz = sz
        self.spacing = spacing
        self.fdt = fd.FD_torch( self.spacing )
        self.volumeElement = self.spacing.prod()
        self.dim = len(spacing)
        self.params = params

    @abstractmethod
    def computeSmootherScalarField(self, v):
        pass

    def computeSmootherVectorField(self, v):
        if self.dim==1:
            return self.computeSmootherScalarField(v) # if one dimensional, default to scalar-field smoothing
        else:
            Sv = Variable( torch.FloatTensor( v.size() ) )
            # smooth every dimension individually
            for d in range(0, self.dim):
                Sv[d,...] = self.computeSmootherScalarField(v[d,...])
            return Sv

class DiffusionSmoother(Smoother):

    def __init__(self, sz, spacing, params):
        super(DiffusionSmoother,self).__init__(sz,spacing,params)
        self.iter = utils.getpar(params, 'iter', 5)

    def computeSmootherScalarField(self,v):
        # basically just solving the heat equation for a few steps
        Sv = v.clone()
        # now iterate and average based on the neighbors
        for i in range(0,self.iter*2**self.dim): # so that we smooth the same indepdenent of dimension
            # multiply with smallest h^2 and divide by 2^dim to assure stability
            Sv = Sv + 0.5/(2**self.dim)*self.fdt.lap(Sv)*self.spacing.min()**2 # multiply with smallest h^2 to assure stability
        return Sv

# TODO: clean up the two Gaussian smoothers
class GaussianSmoother(Smoother):

    def __init__(self, sz, spacing, params):
        super(GaussianSmoother,self).__init__(sz,spacing,params)

class GaussianSpatialSmoother(GaussianSmoother):

    def __init__(self, sz, spacing, params):
        super(GaussianSpatialSmoother,self).__init__(sz,spacing,params)

        k_sz_h = utils.getpar(params, 'k_sz_h', None)
        if k_sz_h is None:
            self.k_sz = (2 * 5 + 1) * np.ones(self.dim, dtype='int')  # default kernel size
        else:
            self.k_sz = k_sz_h * 2 + 1  # this is to assure that the kernel is odd size

        self.smoothingKernel = self._createSmoothingKernel(self.k_sz)

        self.required_padding = (self.k_sz-1)/2

        if self.dim==1:
            self.filter = Variable(torch.from_numpy(self.smoothingKernel).view([1,1,k_sz[0]]))
        elif self.dim==2:
            self.filter = Variable(torch.from_numpy(self.smoothingKernel).view([1,1,k_sz[0],k_sz[1]]))
        elif self.dim==3:
            self.filter = Variable(torch.from_numpy(self.smoothingKernel).view([1,1,k_sz[0],k_sz[1],k_sz[2]]))
        else:
            raise ValueError('Can only create the smoothing kernel in dimensions 1-3')

        # TODO: Potentially do all of the computations in physical coordinates (for now just [-1,1]^d)
    def _createSmoothingKernel(self, k_sz):
        mus = np.zeros(self.dim)
        stds = np.ones(self.dim)
        id = utils.identityMap(k_sz)
        g = utils.computeNormalizedGaussian(id, mus, stds)

        return g

    # TODO: See if we can avoid the clone calls somehow
    # This is likely due to the slicing along the dimension for vector-valued field
    def _filterInputWithPadding(self,I):
        if self.dim==1:
            I_4d = I.view([1,1,1]+list(I.size()))
            I_pad = F.pad(I_4d,(self.required_padding[0],self.required_padding[0],0,0),mode='replicate').view(1,1,-1)
            return F.conv1d(I_pad,self.filter).view(I.size())
        elif self.dim==2:
            I_4d = I.view([1,1]+list(I.size()))
            I_pad = F.pad(I_4d,(self.required_padding[0],self.required_padding[0],
                                self.required_padding[1],self.required_padding[1]),mode='replicate')
            return F.conv2d(I_pad,self.filter).view(I.size())
        elif self.dim==3:
            I_5d = I.view([1, 1] + list(I.size()))
            I_pad = F.pad(I_5d, (self.required_padding[0], self.required_padding[0],
                                 self.required_padding[1], self.required_padding[1],
                                 self.required_padding[2], self.required_padding[2]), mode='replicate')
            return F.conv3d(I_pad, self.filter).view(I.size())
        else:
            raise ValueError('Can only perform padding in dimensions 1-3')

    def computeSmootherScalarField(self,v):
        # just doing a Gaussian smoothing
        return self._filterInputWithPadding(v)

class GaussianFourierSmoother(GaussianSmoother):

    def __init__(self, sz, spacing, params):
        super(GaussianFourierSmoother,self).__init__(sz,spacing,params)
        gaussianStd = utils.getpar(params, 'gaussianStd', 0.15)

        mus = np.zeros(self.dim)
        stds = gaussianStd*np.ones(self.dim)
        id = utils.identityMap(sz)
        g = utils.computeNormalizedGaussian(id, mus, stds)

        self.FFilter = ce.createComplexFourierFilter(g, self.sz)

    def computeSmootherScalarField(self,v):
        # just doing a Gaussian smoothing
        # we need to instantiate a new filter function here every time for the autograd to work
        return ce.fourierConvolution(v, self.FFilter)

class SmootherFactory(object):

    __metaclass__ = ABCMeta

    def __init__(self,sz,spacing):
        self.spacing = spacing
        self.sz = sz
        self.dim = len( spacing )

    def createSmoother(self,smootherName='gaussian',params=None):
        if smootherName=='diffusion':
            return DiffusionSmoother(self.sz,self.spacing,params)
        elif smootherName=='gaussian':
            return GaussianFourierSmoother(self.sz,self.spacing,params)
        elif smootherName=='gaussianSpatial':
            return GaussianSpatialSmoother(self.sz,self.spacing,params)
        else:
            raise ValueError( 'Smoother: ' + smootherName + ' not known')
