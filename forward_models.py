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

    def __init__(self, spacing):
        """
        Constructor of abstract forward model class
        :param self: 
        :param spacing: numpy array for spacing in x,y,z directions
        :return: 
        """

        self.dim = spacing.size # spatial dimension of the problem
        self.spacing = np.ones(self.dim)

        if spacing.size == 1:
            self.spacing[0] = spacing[0]
        elif spacing.size == 2:
            self.spacing[0] = spacing[0]
            self.spacing[1] = spacing[1]
        elif spacing.size == 3:
            self.spacing = spacing
        else:
            raise ValueError('Forward models are currently only supported in dimensions 1 to 3')

        self.fdt = fd.FD_torch( self.spacing )

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

    def __init__(self, spacing):
        super(AdvectMap,self).__init__(spacing)

    """
    Forward model to advect an n-D map using a transport equation: \Phi_t + D\Phi v = 0.
    v is treated as an external argument and \Phi is the state
    """
    def u(self,t, pars):
        """
        External input, to hold the velocity field
        :param t: time (ignored; not time-dependent) 
        :param pars: assumes an n-D velocity field is passed as the only input argument
        :return: Simply returns this velocity field
        """
        return pars

    def f(self,t, x, u, pars):
        """
        Function to be integrated, i.e., right hand side of transport equation: -\nabla I^T v
        :param t: time (ignored; not time-dependent) 
        :param x: state, here the map, \Phi, itself (assumes 3D array; [:,:,0] x-coors; [:,:,1] y-coors; ...
        :param u: external input, will be the velocity field here
        :param pars: ignored (does not expect any additional inputs)
        :return: 
        """
        if self.dim==1:
            return self._f1d(t,x,u,pars)
        elif self.dim==2:
            return self._f2d(t,x,u,pars)
        elif self.dim==3:
            return self._f3d(t,x,u,pars)
        else:
            raise ValueError('Forward models are currently only supported in dimensions 1 to 3')

    def _f1d(self, t, x, u, pars):
        xp = Variable(torch.zeros(x[0].size()), requires_grad=False)
        xp = -(u * self.fdt.dXc(x[0]))
        return [xp]

    def _f2d(self, t, x, u, pars):
        xp = Variable(torch.zeros(x[0].size()), requires_grad=False)
        xp[:, :, 0] = -(u[:, :, 0] * self.fdt.dXc(x[0][:, :, 0]) + u[:, :, 1] * self.fdt.dYc(x[0][:, :, 0]))
        xp[:, :, 1] = -(u[:, :, 0] * self.fdt.dXc(x[0][:, :, 1]) + u[:, :, 1] * self.fdt.dYc(x[0][:, :, 1]))
        return [xp]

    def _f3d(self, t, x, u, pars):
        xp = Variable(torch.zeros(x[0].size()), requires_grad=False)

        xp[:, :, :, 0] = -(u[:, :, :, 0] * self.fdt.dXc(x[0][:, :, :, 0]) +
                           u[:, :, :, 1] * self.fdt.dYc(x[0][:, :, :, 0]) +
                           u[:, :, :, 2] * self.fdt.dZc(x[0][:, :, :, 0]) )

        xp[:, :, :, 1] = -(u[:, :, :, 0] * self.fdt.dXc(x[0][:, :, :, 1]) +
                           u[:, :, :, 1] * self.fdt.dYc(x[0][:, :, :, 1]) +
                           u[:, :, :, 2] * self.fdt.dZc(x[0][:, :, :, 1]) )

        xp[:, :, :, 2] = -(u[:, :, :, 0] * self.fdt.dXc(x[0][:, :, :, 2]) +
                           u[:, :, :, 1] * self.fdt.dYc(x[0][:, :, :, 2]) +
                           u[:, :, :, 2] * self.fdt.dZc(x[0][:, :, :, 2]) )

        return [xp]


class AdvectImage(ForwardModel):

    def __init__(self, spacing):
        super(AdvectImage, self).__init__(spacing)

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
        if self.dim==1:
            return self._f1d(t,x,u,pars)
        elif self.dim==2:
            return self._f2d(t,x,u,pars)
        elif self.dim==3:
            return self._f3d(t,x,u,pars)
        else:
            raise ValueError('Forward models are currently only supported in dimensions 1 to 3')

    def _f1d(self,t,x,u,pars):
        return [-(u* self.fdt.dXc(x[0]))]

    def _f2d(self,t,x,u,pars):
        return [-(u[:, :, 0] * self.fdt.dXc(x[0]) + u[:, :, 1] * self.fdt.dYc(x[0]))]

    def _f3d(self,t,x,u,pars):
        return [-(u[:, :, :, 0] * self.fdt.dXc(x[0]) +
                  u[:, :, :, 1] * self.fdt.dYc(x[0]) +
                  u[:,:,:,2]*self.fdt.dZc(x[0]))]
