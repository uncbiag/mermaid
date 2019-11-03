
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np


class ODEWrapFunc(nn.Module):
    """
    a wrap on tensor based torchdiffeq input
    """
    def __init__(self, nested_class, has_combined_input=False, pars=None, variables_from_optimizer=None, extra_var=None, dim_info=None):
        """

        :param nested_class: the model to be integrated
        :param has_combined_input: the model has combined input in x e.g. EPDiff* equation, otherwise, model has individual input e.g. advect* , has x,u two inputs
        :param pars: ParameterDict, settings passed to integrator
        :param variables_from_optimizer: allows passing variables (as a dict from the optimizer; e.g., the current iteration)
        :param extra_var: extra variable
        :param dim_info: the input x can be a tensor concatenated by several variables along channel, dim_info is a list indicates the dim of each variable,
        """
        super(ODEWrapFunc, self).__init__()
        self.nested_class = nested_class
        """the model to be integrated"""
        self.pars = pars
        """ParameterDict, settings passed to integrator"""
        self.variables_from_optimizer = variables_from_optimizer
        """allows passing variables (as a dict from the optimizer; e.g., the current iteration)"""
        self.extra_var = extra_var
        """extra variable"""
        self.has_combined_input = has_combined_input
        """the model has combined input in x e.g. EPDiff* equation, otherwise, model has individual input e.g. advect* , has x,u two inputs"""
        self.dim_info = dim_info
        """the input x can be a tensor concatenated by several variables along channel, dim_info is a list indicates the dim of each variable"""
        self.opt_param = None

    def set_dim_info(self, dim_info):
        self.dim_info = [0] + list(np.cumsum(dim_info))

    def set_opt_param(self, opt_param):
        self.opt_param = opt_param

    def set_debug_mode_on(self):
        self.nested_class.debug_mode_on = True

    def factor_input(self, y):
        x = [y[:, self.dim_info[ind]:self.dim_info[ind + 1], ...] for ind in range(len(self.dim_info)-1)]
        if not self.has_combined_input:
            u = x[0]
            x = x[1:]
        else:
            u = None
        return u, x

    @staticmethod
    def factor_res(u, res):
        if u is not None:
            res = torch.cat((torch.zeros_like(u), *res), 1)
        else:
            res = torch.cat(res, 1)
        return res

    def forward(self,t,y):
        u, x = self.factor_input(y)
        res = self.nested_class.f(t, x, u, pars=self.pars, variables_from_optimizer=self.variables_from_optimizer)
        res = self.factor_res(u, res)
        return res


class ODEWrapFunc_tuple(nn.Module):
    """
    a warp on tuple based torchdiffeq input
    """
    def __init__(self, nested_class, has_combined_input=False, pars=None, variables_from_optimizer=None, extra_var=None, dim_info=None):
        """

        :param nested_class: the model to be integrated
        :param has_combined_input: the model has combined input in x e.g. EPDiff* equation, otherwise, model has individual input e.g. advect* , has x,u two inputs
        :param pars: ParameterDict, settings passed to integrator
        :param variables_from_optimizer: allows passing variables (as a dict from the optimizer; e.g., the current iteration)
        :param extra_var: extra variable
        :param dim_info: not use in tuple version
        """
        super(ODEWrapFunc_tuple, self).__init__()
        self.nested_class = nested_class
        """ the model to be integrated"""
        self.pars = pars
        """ParameterDict, settings passed to integrator"""
        self.variables_from_optimizer = variables_from_optimizer
        """ allows passing variables (as a dict from the optimizer; e.g., the current iteration)"""
        self.extra_var = extra_var
        """extra variable"""
        self.has_combined_input = has_combined_input
        """ the model has combined input in x e.g. EPDiff* equation, otherwise, model has individual input e.g. advect* , has x,u two inputs"""
        self.dim_info = dim_info
        """not use in tuple version"""
        self.opt_param = None

    def set_dim_info(self, dim_info):
        self.dim_info = [0] + list(np.cumsum(dim_info))

    def set_opt_param(self, opt_param):
        self.opt_param = opt_param

    def set_debug_mode_on(self):
        self.nested_class.debug_mode_on = True

    def factor_input(self, y):
        if not self.has_combined_input:
            u = y[0]
            x=list(y[1:])
        else:
            x=list(y)
            u=None
        return u, x

    @staticmethod
    def factor_res(u, res):
        if u is not None:
            zero_grad=torch.zeros_like(u)
            zero_grad.requires_grad=res[0].requires_grad
            return (zero_grad, *res)
        else:
            return tuple(res)

    def forward(self,t,y):
        u, x = self.factor_input(y)
        res = self.nested_class.f(t, x, u, pars=self.pars, variables_from_optimizer=self.variables_from_optimizer)
        res = self.factor_res(u, res)
        return res