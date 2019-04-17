from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
from .  import torchdiffeq


class ODEBlock(nn.Module):
    """
    SOLVERS = {
    'explicit_adams': AdamsBashforth,
    'fixed_adams': AdamsBashforthMoulton,
    'adams': VariableCoefficientAdamsBashforth,
    'tsit5': Tsit5Solver,
    'dopri5': Dopri5Solver,
    'euler': Euler,
    'midpoint': Midpoint,
    'rk4': RK4,
}

    """

    def __init__(self, param=None):
        super(ODEBlock, self).__init__()
        self.odefunc = None
        self.integration_time = torch.Tensor([0, 1]).float()
        self.method = param[('solver', 'rk4','ode solver')]
        self.adjoin_on = param[('adjoin_on',True,'use adjoint optimization')]
        self.rtol = param[('rtol', 1e-5,'relative error torlance for dopri5')]
        self.atol = param[('atol', 1e-5,'absolute error torlance for dopri5')]
        self.n_step = param[('number_of_time_steps', 20,'Number of time-steps to per unit time-interval integrate the PDE')]
        self.dt = 1./self.n_step
    def solve(self,x):
        return self.forward(x)
    
    def set_func(self, func):
        self.odefunc = func

    def get_dt(self):
        return self.dt

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        odesolver = torchdiffeq.odeint_adjoint if self.adjoin_on else torchdiffeq.odeint
        #out = odeint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol)
        try:
            out = odesolver(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol,method=self.method, options={'step_size':self.dt})
        except:
            print("the {} solver failed, now move into the debug mode".format(self.method))
            self.odefunc.set_debug_mode_on()
            out = odesolver(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol,method=self.method, options={'step_size':self.dt})

        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


# test

#ODEBlock(ODEfunc(64))