"""
Defines different registration methods as pyTorch networks
Currently implemented:
    * SVFNet: image-based stationary velocity field
"""

# first do the torch imports
#from __future__ import print_function
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn

import finite_differences as fd
import rungekutta_integrators as RK
import forward_models as FM

from modules.stn import STN

import utils

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
        print('Energy = ' + str(utils.t2np(energy)[0]) + '; ssd = ' + str(utils.t2np(ssd)[0]) + '; reg = ' + str(utils.t2np(reg)[0]))
        # compute max velocity
        maxVelocity = ( ( self.v[0]**2 + self.v[1]**2 ).max() ).sqrt()
        print('Max velocity = ' + str(utils.t2np(maxVelocity)[0]) )
        return energy



class SVFNetMap(nn.Module):
# TODO: do proper spacing
    def __init__(self,sz,params):
        super(SVFNetMap, self).__init__()
        self.sigma = params['sigma']
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.hx = hx = 1./(sz-1)
        self.nrOfTimeSteps = 10
        self.tFrom = 0.
        self.tTo = 1.
        self.v = Parameter(torch.zeros(sz, sz, 2))
        self.advectionMap = FM.AdvectMap( self.hx, self.hx )
        self.integrator = RK.RK4(self.advectionMap.f,self.advectionMap.u,self.v)

    def forward(self, phi):
        phi1 = self.integrator.solve([phi], self.tFrom, self.tTo, self.nrOfTimeSteps)
        return phi1[0]


class SVFLossMap(nn.Module):
# make velocity field an input here
    def __init__(self,v,sz,params):
        super(SVFLossMap, self).__init__()
        self.sigma = params['sigma']
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.v = v
        self.fdt = fd.FD_torch()
        self.hx = hx = 1. / (sz - 1)
        self.fdt.setspacing( self.hx, self.hx )
        self.stn = STN()
        self.sz = sz

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

    def forward(self, phi1, I0_source, I1_target):

        I0_source_stn = I0_source.view(torch.Size([1,self.sz,self.sz,1]))
        phi1_stn = phi1.view(torch.Size([1,self.sz,self.sz,2]))
        I1_warped = self.stn(I0_source_stn, phi1_stn)

        ssd = ((I1_target - I1_warped[0,:,:,0]) ** 2).sum() / (self.sigma ** 2)
        reg = self.computeRegularizer( self.v, self.alpha, self.gamma)
        energy = ssd + reg
        print('Energy = ' + str(utils.t2np(energy)[0]) + '; ssd = ' + str(utils.t2np(ssd)[0]) + '; reg = ' + str(utils.t2np(reg)[0]))
        # compute max velocity
        maxVelocity = ( ( self.v[0]**2 + self.v[1]**2 ).max() ).sqrt()
        print('Max velocity = ' + str(utils.t2np(maxVelocity)[0]) )
        return energy