
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np


class ODEWarpedFunc(nn.Module):
    def __init__(self, nested_class, single_param=False, pars=None, variables_from_optimizer=None, extra_var=None, dim_info=None):
        super(ODEWarpedFunc, self).__init__()
        self.nested_class = nested_class
        self.pars = pars
        self.variables_from_optimizer = variables_from_optimizer
        self.extra_var = extra_var
        self.single_param = single_param
        self.dim_info = dim_info
        self.opt_param = None

    def set_dim_info(self, dim_info):
        self.dim_info = [0] + list(np.cumsum(dim_info))

    def set_opt_param(self, opt_param):
        self.opt_param = opt_param

    def set_debug_mode_on(self):
        self.nested_class.debug_mode_on = True

    def factor_y(self, y):
        x = [y[:, self.dim_info[ind]:self.dim_info[ind + 1], ...] for ind in range(len(self.dim_info)-1)]
        if not self.single_param:
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
        u, x = self.factor_y(y)
        res = self.nested_class.f(t, x, u, pars=self.pars, variables_from_optimizer=self.variables_from_optimizer)
        res = self.factor_res(u, res)
        return res


class ODEWarpedFunc_tuple(nn.Module):
    def __init__(self, nested_class, single_param=False, pars=None, variables_from_optimizer=None, extra_var=None, dim_info=None):
        super(ODEWarpedFunc_tuple, self).__init__()
        self.nested_class = nested_class
        self.pars = pars
        self.variables_from_optimizer = variables_from_optimizer
        self.extra_var = extra_var
        self.single_param = single_param
        self.dim_info = dim_info
        self.opt_param = None

    def set_dim_info(self, dim_info):
        self.dim_info = [0] + list(np.cumsum(dim_info))

    def set_opt_param(self, opt_param):
        self.opt_param = opt_param

    def set_debug_mode_on(self):
        self.nested_class.debug_mode_on = True

    def factor_y(self, y):
        if not self.single_param:
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
        u, x = self.factor_y(y)
        res = self.nested_class.f(t, x, u, pars=self.pars, variables_from_optimizer=self.variables_from_optimizer)
        res = self.factor_res(u, res)
        return res