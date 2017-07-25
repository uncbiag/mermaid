'''
General purpose regularizers which can be used
'''

from abc import ABCMeta, abstractmethod

import torch
from torch.autograd import Variable

import finite_differences as fd
import utils


class Smoother(object):
    __metaclass__ = ABCMeta

    def __init__(self, spacing, params):
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
            Sv = v.clone()
            # smooth every dimension individually
            for d in range(0, self.dim):
                Sv[..., d] = self.computeSmootherScalarField(Sv[..., d])
            return Sv

class DiffusionSmoother(Smoother):

    def __init__(self, spacing, params):
        super(DiffusionSmoother,self).__init__(spacing,params)
        self.iter = utils.getpar(params, 'iter', 5)

    def computeSmootherScalarField(self,v):
        # basically just solving the heat equation for a few steps
        Sv = v.clone()
        # now iterate and average based on the neighbors
        for i in range(0,self.iter*2**self.dim): # so that we smooth the same indepdenent of dimension
            # multiply with smallest h^2 and divide by 2^dim to assure stability
            Sv = Sv + 0.5/(2**self.dim)*self.fdt.lap(Sv)*self.spacing.min()**2 # multiply with smallest h^2 to assure stability
                # now compute the norm
        return Sv

class SmootherFactory(object):

    __metaclass__ = ABCMeta

    def __init__(self,spacing):
        self.spacing = spacing
        self.dim = len( spacing )

    def createSmoother(self,smootherName='diffusion',params=None):
        if smootherName=='diffusion':
            return DiffusionSmoother(self.spacing,params)
        else:
            raise ValueError( 'Smoother: ' + smootherName + ' not known')
