'''
General purpose regularizers which can be used
'''

from abc import ABCMeta, abstractmethod

import torch
from torch.autograd import Variable

import finite_differences as fd

class Regularizer(object):
    __metaclass__ = ABCMeta

    def __init__(self, spacing, params):
        self.spacing = spacing
        self.fdt = fd.FD_torch( self.spacing )
        self.volumeElement = self.spacing.prod()
        self.dim = len(spacing)
        self.params = params

    @abstractmethod
    def _computeRegularizer(self, v):
        pass

    def computeRegularizer_multiN(self,v):
        szv = v.size()
        reg = Variable(torch.zeros(1), requires_grad=False)
        for nrI in range(szv[0]): # loop over number of images
            reg = reg + self._computeRegularizer(v[nrI,...])
        return reg

class HelmholtzRegularizer(Regularizer):

    def __init__(self, spacing, params):
        super(HelmholtzRegularizer,self).__init__(spacing,params)

        self.alpha = params[('alpha', 0.2, 'penalty for 2nd derivative' )]
        self.gamma = params[('gamma', 1.0, 'penaly for magnitude' )]


    def _computeRegularizer(self,v):
        # just do the standard component-wise gamma id -\alpha \Delta

        if self.dim == 1:
            return self._computeRegularizer_1d(v, self.alpha, self.gamma)
        elif self.dim == 2:
            return self._computeRegularizer_2d(v, self.alpha, self.gamma)
        elif self.dim == 3:
            return self._computeRegularizer_3d(v, self.alpha, self.gamma)
        else:
            raise ValueError('Regularizer is currently only supported in dimensions 1 to 3')

    def _computeRegularizer_1d(self, v, alpha, gamma):
        Lv = Variable(torch.zeros(v.size()), requires_grad=False)
        Lv[0,:] = v[0,:] * gamma - self.fdt.lap(v[0,:]) * alpha
        # now compute the norm
        return (Lv[0,:] ** 2).sum()*self.volumeElement

    def _computeRegularizer_2d(self, v, alpha, gamma):
        Lv = Variable(torch.zeros(v.size()), requires_grad=False)
        for i in [0, 1]:
            Lv[i,:, :] = v[i,:, :] * gamma - self.fdt.lap(v[i,:, :]) * alpha

        # now compute the norm
        return (Lv[0,:, :] ** 2 + Lv[1,:, :] ** 2).sum()*self.volumeElement

    def _computeRegularizer_3d(self, v, alpha, gamma):
        Lv = Variable(torch.zeros(v.size()), requires_grad=False)
        for i in [0, 1, 2]:
            Lv[i,:, :, :] = v[i,:, :, :] * gamma - self.fdt.lap(v[i,:, :, :]) * alpha

        # now compute the norm
        return (Lv[0,:, :, :] ** 2 + Lv[1,:, :, :] ** 2 + Lv[2,:, :, :] ** 2).sum()*self.volumeElement


class RegularizerFactory(object):

    __metaclass__ = ABCMeta

    def __init__(self,spacing):
        self.spacing = spacing
        self.dim = len( spacing )

    def createRegularizer(self,params):

        cparams = params[('regularizer',{},'Parameters for the regularizer')]
        regularizerType = cparams[('type','helmholtz',
                                             'type of regularizer (only helmholtz at the moment)')]

        if regularizerType=='helmholtz':
            return HelmholtzRegularizer(self.spacing,cparams)
        else:
            raise ValueError( 'Regularizer: ' + regularizerType + ' not known')





