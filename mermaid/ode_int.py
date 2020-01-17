from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
from .  import torchdiffeq
from . import rungekutta_integrators as RK
from . import forward_models_wrap as FMW


class ODEBlock(nn.Module):
    """
    A interface class for torchdiffeq, https://github.com/rtqichen/torchdiffeq
    we add some constrains in torchdiffeq package to avoid collapse or traps, so this local version is recommended
    the solvers supported by the torchdiffeq are listed as following
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
        """the ode problem to be solved"""
        tFrom = param[('tFrom', 0.0, 'time to solve a model from')]
        """time to solve a model from"""
        tTo = param[('tTo', 1.0, 'time to solve a model to')]
        """time to solve a model to"""
        self.integration_time = torch.Tensor([tFrom, tTo]).float()
        """intergration time, list, typically set as [0,1]"""
        self.method = param[('solver', 'rk4','ode solver')]
        """ solver,rk4 as default, supported list: explicit_adams,fixed_adams,tsit5,dopri5,euler,midpoint, rk4 """
        self.adjoin_on = param[('adjoin_on',True,'use adjoint optimization')]
        """ adjoint method, benefits from memory consistency, which can be refer to "Neural Ordinary Differential Equations" """
        self.rtol = param[('rtol', 1e-5,'relative error tolerance for dopri5')]
        """ relative error tolerance for dopri5"""
        self.atol = param[('atol', 1e-5,'absolute error tolerance for dopri5')]
        """ absolute error tolerance for dopri5"""
        self.n_step = param[('number_of_time_steps', 20,'Number of time-steps to per unit time-interval integrate the ode')]
        """ Number of time-steps to per unit time-interval integrate the PDE, for fixed time-step solver, i.e. rk4"""
        self.dt = 1./self.n_step
        """time step, we assume integration time is from 0,1 so the step is 1/n_step"""
    def solve(self,x):
        return self.forward(x)
    
    def set_func(self, func):
        self.odefunc = func

    def get_dt(self):
        return self.dt

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x) if type(x) is not tuple else self.integration_time.type_as(x[0])
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


class ODEWrapBlock(nn.Module):
    """
    A warp on ODE method, providing interface for embedded rungekutta sovler and for torchdiffeq solver
    """
    def __init__(self, model, cparams=None, use_odeint=True,use_ode_tuple=False,tFrom=0., tTo=1.):
        """

        :param model: the ode/pde model to be solved
        :param cparams: ParameterDict, the model settings
        :param use_odeint: if true, use torchdiffeq, else, use embedded rungekutta (rk4)
        :param use_ode_tuple: assume torchdiffeq is used, if use_ode_tuple, take the tuple as the solver input, else take a tensor as the solver input
        :param tFrom: start time point, typically 0
        :param tTo: end time point, typically 1
        """
        super(ODEWrapBlock, self).__init__()
        self.model = model
        """ the ode/pde model to be solved"""
        self.cparams = cparams
        """ParameterDict, the model settings"""
        self.use_odeint=use_odeint
        """if true, use torchdiffeq, else, use embedded rungekutta (rk4)"""
        self.use_ode_tuple=use_ode_tuple
        """assume torchdiffeq is used, if use_ode_tuple, take the tuple as the solver input, else take a tensor as the solver input """
        self.integrator=None
        """if use_odeint, then intergrator from torchdiffeq is used else use the embedded rk4 intergrator"""
        self.tFrom=tFrom
        """start time point, typically 0"""
        self.tTo =tTo
        """ end time point, typically 1"""

    def get_dt(self):
        self.n_step = self.cparams[('number_of_time_steps', 20, 'Number of time-steps to per unit time-interval integrate the PDE')]
        self.dt = 1. / self.n_step
        return self.dt

    def init_solver(self,pars_to_pass_i,variables_from_optimizer,has_combined_input=False):
        if self.use_odeint:
            self.integrator = ODEBlock(self.cparams)
            wraped_func = FMW.ODEWrapFunc_tuple if self.use_ode_tuple else FMW.ODEWrapFunc
            func = wraped_func(self.model, has_combined_input=has_combined_input, pars=pars_to_pass_i,
                                           variables_from_optimizer=variables_from_optimizer)
            self.integrator.set_func(func)
        else:
            self.integrator = RK.RK4(self.model.f, self.model.u, pars_to_pass_i, self.cparams)
            self.integrator.set_pars(pars_to_pass_i)

    def solve_odeint(self,input_list):
        if self.use_ode_tuple:
            return self.solve_odeint_tuple(input_list)
        else:
            return self.solve_odeint_tensor(input_list)

    def solve_odeint_tensor(self, input_list):
        input_list_dim = [item.shape[1] for item in input_list]
        self.integrator.odefunc.set_dim_info(input_list_dim)
        input_tensor = torch.cat(tuple(input_list), 1)
        output_tensor = self.integrator.solve(input_tensor)
        dim_info = [0] + list(np.cumsum(input_list_dim))
        output_list = [output_tensor[:, dim_info[ind]:dim_info[ind + 1], ...] for ind in range(len(dim_info) - 1)]
        return output_list

    def solve_odeint_tuple(self, input_list):
        output_tuple = self.integrator.solve(tuple(input_list))
        return list(output_tuple)

    def solve_embedded_ode(self, input_list, variables_from_optimizer):
        return self.integrator.solve(input_list, self.tFrom, self.tTo, variables_from_optimizer)

    def solve(self,input_list,  variables_from_optimizer):
        if self.use_odeint:
            return self.solve_odeint(input_list)
        else:
            return self.solve_embedded_ode(input_list, variables_from_optimizer)





