import torch
import torch.nn as nn
adjoint = True
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

class ODEBlock(nn.Module):

    def __init__(self, odefunc,rtol = 1e-3,atol=1e-3):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.Tensor([0, 1]).float()
        self.rtol = rtol
        self.atol = atol
        self.dt =0.1
    def solve(self,x):
        return self.forward(x)

    def get_dt(self):
        return self.dt

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        #out = odeint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol,method='rk4',options={'step_size':self.dt})
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


# test

#ODEBlock(ODEfunc(64))