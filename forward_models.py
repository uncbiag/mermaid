from abc import ABCMeta, abstractmethod
import numpy as np
import finitedifferences as fd

class ForwardModel(object):

    __metaclass__ = ABCMeta

    spacing = np.ones(3)
    fdt = fd.FD_torch()

    def __init__(self, hx=None, hy=None, hz=None):
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
    def f(t,x,u,pars):
        pass

    def u(self,t,pars):
        return []


class AdvectImage(ForwardModel):

    def u(self,t, pars):
        return pars

    def f(self,t, x, u, pars):
        return [-(u[:, :, 0] * self.fdt.dXc(x[0]) + u[:, :, 1] * self.fdt.dYc(x[0]))]