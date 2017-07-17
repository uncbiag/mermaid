"""
Defines dynamic forward models, for example transport equations
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import finite_differences as fd

import torch
from torch.autograd import Variable

class ForwardModel(object):
    """
    Abstract forward model class. Should never be instantiated.
    Derived classes require the definition of f(self,t,x,u,pars) and u(self,t,pars).
    These functions will be used for integration: x'(t) = f(t,x(t),u(t))
    """
    __metaclass__ = ABCMeta

    spacing = np.ones(3)
    fdt = fd.FD_torch()

    def __init__(self, hx=None, hy=None, hz=None):
        """
        Constructor of abstract forward model class
        :param self: 
        :param hx: spacing in x direction
        :param hy: spacing in y direction
        :param hz: spacing in z direction
        :return: 
        """
        if hx is None:
            self.spacing[0] = 1.
        else:
            self.spacing[0] = hx
        if hy is None:
            self.spacing[1] = 1.
        else:
            self.spacing[1] = hy
        if hz is None:
            self.spacing[2] = 1.
        else:
            self.spacing[2] = hz
        self.fdt.setspacing( hx, hy, hz )

    @abstractmethod
    def f(self,t,x,u,pars):
        """
        Function to be integrated
        :param self: 
        :param t: time
        :param x: state
        :param u: input
        :param pars: optional parameters
        :return: 
        """
        pass

    def u(self,t,pars):
        """
        External input
        :param self: 
        :param t: time
        :param pars: parameters
        :return: 
        """
        return []


class AdvectMap(ForwardModel):
    """
    Forward model to advect a 2D map using a transport equation: \Phi_t + D\Phi v = 0.
    v is treated as an external argument and \Phi is the state
    """
    def u(self,t, pars):
        """
        External input, to hold the velocity field
        :param t: time (ignored; not time-dependent) 
        :param pars: assumes a 2D velocity field is passed as the only input argument
        :return: Simply returns this velocity field
        """
        return pars

    def f(self,t, x, u, pars):
        """
        Function to be integrated, i.e., right hand side of transport equation: -\nabla I^T v
        :param t: time (ignored; not time-dependent) 
        :param x: state, here the map, \Phi, itself (assumes 3D array; [:,:,0] x-coors; [:,:,1] y-coors
        :param u: external input, will be the velocity field here
        :param pars: ignored (does not expect any additional inputs)
        :return: 
        """
        # TODO: make this work for generic dimensions
        xp = Variable( torch.zeros(x[0].size()), requires_grad=False )
        xp[:,:,0] = -(u[:, :, 0] * self.fdt.dXc(x[0][:,:,0]) + u[:, :, 1] * self.fdt.dYc(x[0][:,:,0]))
        xp[:,:,1] = -(u[:, :, 0] * self.fdt.dXc(x[0][:,:,1]) + u[:, :, 1] * self.fdt.dYc(x[0][:,:,1]))
        return [xp]

class AdvectImage(ForwardModel):
    """
    Forward model to advect a 2D image using a transport equation: I_t + \nabla I^Tv = 0.
    v is treated as an external argument and I is the state
    """
    def u(self,t, pars):
        """
        External input, to hold the velocity field
        :param t: time (ignored; not time-dependent) 
        :param pars: assumes a 2D velocity field is passed as the only input argument
        :return: Simply returns this velocity field
        """
        return pars

    def f(self,t, x, u, pars):
        """
        Function to be integrated, i.e., right hand side of transport equation: -\nabla I^T v
        :param t: time (ignored; not time-dependent) 
        :param x: state, here the image, I, itself
        :param u: external input, will be the velocity field here
        :param pars: ignored (does not expect any additional inputs)
        :return: 
        """
        return [-(u[:, :, 0] * self.fdt.dXc(x[0]) + u[:, :, 1] * self.fdt.dYc(x[0]))]