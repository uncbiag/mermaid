"""
Defines dynamic forward models, for example transport equations
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import finite_differences as fd
import smoother_factory as sf

import torch
from torch.autograd import Variable

class RHSLibrary(object):

    def __init__(self, spacing):
        self.spacing = spacing
        self.fdt = fd.FD_torch( self.spacing )
        self.dim = len(self.spacing)

    def rhs_advect_image(self,I,v):
        if self.dim==1:
            return -self.fdt.dXc(I) * v
        elif self.dim==2:
            return -self.fdt.dXc(I) * v[0,:,:] -self.fdt.dYc(I)*v[1,:,:]
        elif self.dim==3:
            return -self.fdt.dXc(I) * v[0,:,:,:] -self.fdt.dYc(I)*v[1,:,:,:]-self.fdt.dZc(I)*v[2,:,:,:]
        else:
            raise ValueError('Only supported up to dimension 3')

    def rhs_advect_map(self,phi,v):
        if self.dim==1:
            return -self.fdt.dXc(phi) * v
        elif self.dim==2:
            rhsphi = Variable(torch.zeros(phi.size()), requires_grad=False)
            rhsphi[0,:, :] = -(v[0,:, :] * self.fdt.dXc(phi[0,:, :]) + v[1,:, :] * self.fdt.dYc(phi[0,:, :]))
            rhsphi[1,:, :] = -(v[0,:, :] * self.fdt.dXc(phi[1,:, :]) + v[1,:, :] * self.fdt.dYc(phi[1,:, :]))
            return rhsphi
        elif self.dim==3:
            rhsphi = Variable(torch.zeros(phi.size()), requires_grad=False)
            rhsphi[0,:, :, :] = -(v[0,:, :, :] * self.fdt.dXc(phi[0,:, :, :]) +
                                   v[1,:, :, :] * self.fdt.dYc(phi[0,:, :, :]) +
                                   v[2,:, :, :] * self.fdt.dZc(phi[0,:, :, :]))

            rhsphi[1,:, :, :] = -(v[0,:, :, :] * self.fdt.dXc(phi[1,:, :, :]) +
                                   v[1,:, :, :] * self.fdt.dYc(phi[1,:, :, :]) +
                                   v[2,:, :, :] * self.fdt.dZc(phi[1,:, :, :]))

            rhsphi[2,:, :, :] = -(v[0,:, :, :] * self.fdt.dXc(phi[2,:, :, :]) +
                                   v[1,:, :, :] * self.fdt.dYc(phi[2,:, :, :]) +
                                   v[2,:, :, :] * self.fdt.dZc(phi[2,:, :, :]))
            return rhsphi
        else:
            raise ValueError('Only supported up to dimension 3')

    def rhs_epdiff(self,m,v):
        if self.dim==1:
            return -self.fdt.dXc(m * v) - self.fdt.dXc(v) * m
        elif self.dim==2:
            rhsm = Variable(torch.zeros(m.size()), requires_grad=False)
             # (m_1,...,m_d)^T_t = -(div(m_1v),...,div(m_dv))^T-(Dv)^Tm  (EPDiff equation)
            rhsm[0,:, :] = (-self.fdt.dXc(m[0,:, :] * v[0,:, :])
                             - self.fdt.dYc(m[0,:, :] * v[1,:, :])
                             - self.fdt.dXc(v[0,:, :]) * m[0,:, :]
                             - self.fdt.dXc(v[1,:, :]) * m[1,:, :])

            rhsm[1,:, :] = (-self.fdt.dXc(m[1,:, :] * v[0,:, :])
                             - self.fdt.dYc(m[1,:, :] * v[1,:, :])
                             - self.fdt.dYc(v[0,:, :]) * m[0,:, :]
                             - self.fdt.dYc(v[1,:, :]) * m[1,:, :])
            return rhsm
        elif self.dim==3:
            rhsm = Variable(torch.zeros(m.size()), requires_grad=False)
            # (m_1,...,m_d)^T_t = -(div(m_1v),...,div(m_dv))^T-(Dv)^Tm  (EPDiff equation)
            rhsm[0,:, :, :] = (-self.fdt.dXc(m[0,:, :, :] * v[0,:, :, :])
                                - self.fdt.dYc(m[0,:, :, :] * v[1,:, :, :])
                                - self.fdt.dZc(m[0,:, :, :] * v[2,:, :, :])
                                - self.fdt.dXc(v[0,:, :, :]) * m[0,:, :, :]
                                - self.fdt.dXc(v[1,:, :, :]) * m[1,:, :, :]
                                - self.fdt.dXc(v[2,:, :, :]) * m[2,:, :, :])

            rhsm[1,:, :, :] = (-self.fdt.dXc(m[1,:, :, :] * v[0,:, :, :])
                                - self.fdt.dYc(m[1,:, :, :] * v[1,:, :, :])
                                - self.fdt.dZc(m[1,:, :, :] * v[2,:, :, :])
                                - self.fdt.dYc(v[0,:, :, :]) * m[0,:, :, :]
                                - self.fdt.dYc(v[1,:, :, :]) * m[1,:, :, :]
                                - self.fdt.dYc(v[2,:, :, :]) * m[2,:, :, :])

            rhsm[2,:, :, :] = (-self.fdt.dXc(m[2,:, :, :] * v[0,:, :, :])
                                - self.fdt.dYc(m[2,:, :, :] * v[1,:, :, :])
                                - self.fdt.dZc(m[2,:, :, :] * v[2,:, :, :])
                                - self.fdt.dZc(v[0,:, :, :]) * m[0,:, :, :]
                                - self.fdt.dZc(v[1,:, :, :]) * m[1,:, :, :]
                                - self.fdt.dZc(v[2,:, :, :]) * m[2,:, :, :])
            return rhsm
        else:
            raise ValueError('Only supported up to dimension 3')

class ForwardModel(object):
    """
    Abstract forward model class. Should never be instantiated.
    Derived classes require the definition of f(self,t,x,u,pars) and u(self,t,pars).
    These functions will be used for integration: x'(t) = f(t,x(t),u(t))
    """
    __metaclass__ = ABCMeta

    def __init__(self, sz, spacing, params=None):
        """
        Constructor of abstract forward model class
        :param self: 
        :params sz: size of images
        :param spacing: numpy array for spacing in x,y,z directions
        :return: 
        """

        self.dim = spacing.size # spatial dimension of the problem
        self.spacing = spacing
        self.sz = sz
        self.params = params
        self.rhs = RHSLibrary(self.spacing)

        if self.dim>3 or self.dim<1:
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

    def __init__(self, sz, spacing, params=None):
        super(AdvectMap,self).__init__(sz,spacing,params)

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
        return [self.rhs.rhs_advect_map(x[0],u)]

class AdvectImage(ForwardModel):

    def __init__(self, sz, spacing, params=None):
        super(AdvectImage, self).__init__(sz, spacing,params)

    """
    Forward model to advect an image using a transport equation: I_t + \nabla I^Tv = 0.
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
        return [self.rhs.rhs_advect_image(x[0],u)]

class EPDiffImage(ForwardModel):

    def __init__(self, sz, spacing, params=None):
        super(EPDiffImage, self).__init__(sz, spacing,params)
        self.smoother = sf.SmootherFactory(self.sz,self.spacing).createSmoother('gaussian')

    """
    Forward model for the EPdiff equation. State is the momentum, m, and the image I
    """

    def f(self,t, x, u, pars):
        """
        Function to be integrated, i.e., right hand side of transport equation: -\nabla I^T v
        :param t: time (ignored; not time-dependent) 
        :param x: state, here the image, I, itself
        :param u: external input, will be the velocity field here
        :param pars: ignored (does not expect any additional inputs)
        :return: 
        """
        # assume x[0] is m and x[1] is I for the state
        m = x[0]
        I = x[1]
        v = self.smoother.computeSmootherVectorField(m)
        # print('max(|v|) = ' + str( v.abs().max() ))
        return [self.rhs.rhs_epdiff(m,v), self.rhs.rhs_advect_image(I,v)]


class EPDiffMap(ForwardModel):

    def __init__(self, sz, spacing, params=None):
        super(EPDiffMap, self).__init__(sz,spacing,params)
        self.smoother = sf.SmootherFactory(self.sz,self.spacing).createSmoother('gaussian')

    """
    Forward model for the EPDiff equation. State is the momentum, m, and the transform, phi.
    """

    def f(self,t, x, u, pars):
        """
        Function to be integrated, i.e., right hand side of transport equation: -\nabla I^T v
        :param t: time (ignored; not time-dependent) 
        :param x: state, here the image, I, itself
        :param u: external input, will be the velocity field here
        :param pars: ignored (does not expect any additional inputs)
        :return: 
        """

        # assume x[0] is m and x[1] is phi for the state
        m = x[0]
        phi = x[1]
        v = self.smoother.computeSmootherVectorField(m)
        # print('max(|v|) = ' + str( v.abs().max() ))
        return [self.rhs.rhs_epdiff(m,v),self.rhs.rhs_advect_map(phi,v)]


