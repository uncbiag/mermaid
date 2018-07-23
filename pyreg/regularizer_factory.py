'''
Package implementing general purpose regularizers.
'''
from __future__ import absolute_import

from builtins import range
from builtins import object
from abc import ABCMeta, abstractmethod

import torch

from . import finite_differences as fd
from .data_wrapper import MyTensor
from future.utils import with_metaclass

class Regularizer(with_metaclass(ABCMeta, object)):
    """
    Abstract regularizer base class
    """

    def __init__(self, spacing, params):
        """
        Constructor.
        
        :param spacing: Spatial spacing (BxCxXxYxZ format) 
        :param params: ParameterDict dictionary instance to pass parameters around
        """
        self.spacing = spacing
        """spacing"""
        self.fdt = fd.FD_torch( self.spacing )
        """finite differencing support"""
        self.volumeElement = self.spacing.prod()
        """volume element, i.e., volume of a pixel/voxel"""
        self.dim = len(spacing)
        """spatial dimension"""
        self.params = params
        """parameters"""

    @abstractmethod
    def _compute_regularizer(self, v):
        pass

    def compute_regularizer_multiN(self, v):
        """
        Compute a regularized vector field
        
        :param v: Input vector field
        :return: Regularizer energy
        """
        szv = v.size()
        reg = MyTensor(1).zero_()
        for nrI in range(szv[0]): # loop over number of images
            reg = reg + self._compute_regularizer(v[nrI, ...])
        return reg


class DiffusionRegularizer(Regularizer):
    """
    Implements a diffusion regularizer sum of squared gradients of vector field components
    """

    def __init__(self, spacing, params):
        """
        Constructor

        :param spacing: spatial spacing 
        :param params: ParameterDict dictionary instance
        """
        super(DiffusionRegularizer, self).__init__(spacing, params)

    def _compute_regularizer(self, d):
        # just do the standard component-wise norm of gradient squared

        if self.dim == 1:
            return self._compute_regularizer_1d(d)
        elif self.dim == 2:
            return self._compute_regularizer_2d(d)
        elif self.dim == 3:
            return self._compute_regularizer_3d(d)
        else:
            raise ValueError('Regularizer is currently only supported in dimensions 1 to 3')

    # None in the following refers to batch, which is added here for compatibility, the following [0] is used for this reason
    # now compute the norm

    def _compute_regularizer_1d(self, d):
        return (self.fdt.dXc(d[None, 0, :])[0]**2).sum() * self.volumeElement

    def _compute_regularizer_2d(self, d):
        return ( (self.fdt.dXc(d[None, 0, :, :])[0] ** 2) +
                 (self.fdt.dYc(d[None, 0, :, :])[0] ** 2) +
                 (self.fdt.dXc(d[None, 1, :, :])[0] ** 2) +
                 (self.fdt.dYc(d[None, 1, :, :])[0] ** 2)).sum() * self.volumeElement

    def _compute_regularizer_3d(self, d):
        return ( (self.fdt.dXc(d[None, 0, :, :, :])[0] ** 2) +
                 (self.fdt.dYc(d[None, 0, :, :, :])[0] ** 2) +
                 (self.fdt.dZc(d[None, 0, :, :, :])[0] ** 2) +
                 (self.fdt.dXc(d[None, 1, :, :, :])[0] ** 2) +
                 (self.fdt.dYc(d[None, 1, :, :, :])[0] ** 2) +
                 (self.fdt.dZc(d[None, 1, :, :, :])[0] ** 2) +
                 (self.fdt.dXc(d[None, 2, :, :, :])[0] ** 2) +
                 (self.fdt.dYc(d[None, 2, :, :, :])[0] ** 2) +
                 (self.fdt.dZc(d[None, 2, :, :, :])[0] ** 2) ).sum() * self.volumeElement


class CurvatureRegularizer(Regularizer):
    """
    Implements a curvature regularizer sum of squared Laplacians of the vector field components
    """

    def __init__(self, spacing, params):
        """
        Constructor

        :param spacing: spatial spacing 
        :param params: ParameterDict dictionary instance
        """
        super(CurvatureRegularizer, self).__init__(spacing, params)

    def _compute_regularizer(self, d):
        # just do the standard component-wise norm of gradient squared

        if self.dim == 1:
            return self._compute_regularizer_1d(d)
        elif self.dim == 2:
            return self._compute_regularizer_2d(d)
        elif self.dim == 3:
            return self._compute_regularizer_3d(d)
        else:
            raise ValueError('Regularizer is currently only supported in dimensions 1 to 3')

    # None in the following refers to batch, which is added here for compatibility, the following [0] is used for this reason
    # now compute the norm

    def _compute_regularizer_1d(self, d):
        return (self.fdt.lap(d[None, 0, :])[0]**2).sum() * self.volumeElement

    def _compute_regularizer_2d(self, d):
        return ( (self.fdt.lap(d[None, 0, :, :])[0] ** 2) +
                 (self.fdt.lap(d[None, 1, :, :])[0] ** 2)).sum() * self.volumeElement

    def _compute_regularizer_3d(self, d):
        return ( (self.fdt.lap(d[None, 0, :, :, :])[0] ** 2) +
                 (self.fdt.lap(d[None, 1, :, :, :])[0] ** 2) +
                 (self.fdt.lap(d[None, 2, :, :, :])[0] ** 2) +
                 (self.fdt.dYc(d[None, 2, :, :, :])[0] ** 2) ).sum() * self.volumeElement


class TotalVariationRegularizer(Regularizer):
    """
    Implements a total variation regularizer sum of Euclidean norms of gradient of vector field components
    """

    def __init__(self, spacing, params):
        """
        Constructor

        :param spacing: spatial spacing 
        :param params: ParameterDict dictionary instance
        """
        super(TotalVariationRegularizer, self).__init__(spacing, params)

        self.pnorm = params[('pnorm', 2, 'p-norm type: 2 is Euclidean')]

    def set_pnorm(self, pnorm):
        """
        Sets the norm type

        :param pnorm: norm type
        """
        self.pnorm = pnorm
        self.params['pnorm']  = pnorm

    def get_pnorm(self):
        """
        Gets the norm type

        :return: Returns the norm type
        """
        return self.pnorm

    def _compute_regularizer(self, d):
        # just do the standard component-wise Euclidean norm of the gradient

        if self.dim == 1:
            return self._compute_regularizer_1d(d)
        elif self.dim == 2:
            return self._compute_regularizer_2d(d)
        elif self.dim == 3:
            return self._compute_regularizer_3d(d)
        else:
            raise ValueError('Regularizer is currently only supported in dimensions 1 to 3')

    # None in the following refers to batch, which is added here for compatibility, the following [0] is used for this reason
    # now compute the norm

    def _compute_regularizer_1d(self, d):

        # need to use torch.abs here to make sure the proper subgradient is computed at zero
        v0 = torch.abs(self.fdt.dXc(d[None, 0, :])[0])

        return (v0).sum() * self.volumeElement

    def _compute_regularizer_2d(self, d):

        # need to use torch.norm here to make sure the proper subgradient is computed at zero
        v0 = torch.norm(torch.stack((self.fdt.dXc(d[None, 0, :, :])[0],self.fdt.dYc(d[None, 0, :, :])[0])),self.pnorm,0)
        v1 = torch.norm(torch.stack((self.fdt.dXc(d[None, 1, :, :])[0],self.fdt.dYc(d[None, 1, :, :])[0])),self.pnorm,0)

        return (v0+v1).sum()*self.volumeElement

    def _compute_regularizer_3d(self, d):

        # need to use torch.norm here to make sure the proper subgradient is computed at zero
        v0 = torch.norm(torch.stack((self.fdt.dXc(d[None, 0, :, :, :])[0],
                                     self.fdt.dYc(d[None, 0, :, :, :])[0],
                                     self.fdt.dZc(d[None, 0, :, :, :])[0])), self.pnorm, 0)
        v1 = torch.norm(torch.stack((self.fdt.dXc(d[None, 1, :, :, :])[0],
                                     self.fdt.dYc(d[None, 1, :, :, :])[0],
                                     self.fdt.dZc(d[None, 1, :, :, :])[0])), self.pnorm, 0)
        v2 = torch.norm(torch.stack((self.fdt.dXc(d[None, 2, :, :, :])[0],
                                     self.fdt.dYc(d[None, 2, :, :, :])[0],
                                     self.fdt.dZc(d[None, 2, :, :, :])[0])), self.pnorm, 0)

        return (v0+v1+v2).sum()*self.volumeElement


class HelmholtzRegularizer(Regularizer):
    """
    Implements a Helmholtz regularizer
    :math:`Reg[v] = \\langle\\gamma v -\\alpha \\Delta v,  \\gamma v -\\alpha \\Delta v\\rangle`
    """

    def __init__(self, spacing, params):
        """
        Constructor
        
        :param spacing: spatial spacing 
        :param params: ParameterDict dictionary instance
        """
        super(HelmholtzRegularizer,self).__init__(spacing,params)

        self.alpha = params[('alpha', 0.2, 'penalty for 2nd derivative' )]
        """penalty for second derivative"""
        self.gamma = params[('gamma', 1.0, 'penalty for magnitude' )]
        """penalty for magnitude"""

    def set_alpha(self,alpha):
        """
        Sets the penalty for the second derivative
        
        :param alpha: penalty  
        """
        self.alpha = alpha
        self.params['alpha'] = alpha

    def get_alpha(self):
        """
        Gets the penalty for the second derivative
        
        :return: Returns the penalty for the second derivative 
        """
        return self.alpha

    def set_gamma(self,gamma):
        """
        Sets the penalty for the magnitude
        
        :param gamma: penalty 
        """
        self.gamma = gamma
        self.params['gamma'] = gamma

    def get_gamma(self):
        """
        Gest the penalty for the magnitude
        
        :return: Returns the penalty for the magnitude 
        """
        return self.gamma

    def _compute_regularizer(self, v):
        # just do the standard component-wise gamma id -\alpha \Delta

        if self.dim == 1:
            return self._compute_regularizer_1d(v, self.alpha, self.gamma)
        elif self.dim == 2:
            return self._compute_regularizer_2d(v, self.alpha, self.gamma)
        elif self.dim == 3:
            return self._compute_regularizer_3d(v, self.alpha, self.gamma)
        else:
            raise ValueError('Regularizer is currently only supported in dimensions 1 to 3')

    def _compute_regularizer_1d(self, v, alpha, gamma):
        Lv = MyTensor(v.size()).zero_()
        # None is refer to batch, which is added here for compatibility, the following [0] is used for this reason
        Lv[0,:] = v[0,:] * gamma - self.fdt.lap(v[None,0,:])[0] * alpha
        # now compute the norm
        return (Lv[0,:] ** 2).sum()*self.volumeElement

    def _compute_regularizer_2d(self, v, alpha, gamma):
        Lv = MyTensor(v.size()).zero_()
        for i in [0, 1]:
            # None is refer to batch, which is added here for compatibility, the following [0] is used for this reason
            Lv[i,:, :] = v[i,:, :] * gamma - self.fdt.lap(v[None, i,:, :])[0] * alpha

        # now compute the norm
        return (Lv[0,:, :] ** 2 + Lv[1,:, :] ** 2).sum()*self.volumeElement

    def _compute_regularizer_3d(self, v, alpha, gamma):
        Lv = MyTensor(v.size()).zero_()
        for i in [0, 1, 2]:
            # None is refer to batch, which is added here for compatibility, the following [0] is used for this reason
            Lv[i,:, :, :] = v[i,:, :, :] * gamma - self.fdt.lap(v[None,i,:, :, :])[0] * alpha

        # now compute the norm
        return (Lv[0,:, :, :] ** 2 + Lv[1,:, :, :] ** 2 + Lv[2,:, :, :] ** 2).sum()*self.volumeElement


class RegularizerFactory(with_metaclass(ABCMeta, object)):
    """
    Regularizer factory to instantiate a regularizer by name.
    """

    def __init__(self,spacing):
        """
        Constructor 
        
        :param spacing: spatial spacing 
        """
        self.spacing = spacing
        """spacing"""
        self.dim = len( spacing )
        """spatial dimension"""
        self.default_regularizer_type = 'helmholtz'
        """type of the regularizer used by default"""

    def set_default_regularizer_type_to_helmholtz(self):
        """
        Sets the default regularizer type to helmholtz 
        """
        self.default_regularizer_type = 'helmholtz'

    def set_default_regularizer_type_to_diffusion(self):
        """
        Sets the default regularizer type to diffusion 
        """
        self.default_regularizer_type = 'diffusion'

    def set_default_regularizer_type_to_total_variation(self):
        """
        Sets the default regularizer type to totalVariation 
        """
        self.default_regularizer_type = 'totalVariation'

    def set_default_regularizer_type_to_curvature(self):
        """
        Sets the default regularizer type to curvature 
        """
        self.default_regularizer_type = 'curvature'

    def _get_regularizer_instance(self, regularizerType, cparams):
        if regularizerType == 'helmholtz':
            return HelmholtzRegularizer(self.spacing, cparams)
        elif regularizerType == 'totalVariation':
            return TotalVariationRegularizer(self.spacing, cparams)
        elif regularizerType == 'diffusion':
            return DiffusionRegularizer(self.spacing, cparams)
        elif regularizerType == 'curvature':
            return CurvatureRegularizer(self.spacing, cparams)
        else:
            raise ValueError('Regularizer: ' + regularizerType + ' not known')

    def create_regularizer_by_name(self, regularizerType, params):
        """
        Create a regularizer by name. This is a convenience function in the case where
        there should be no free choice of regularizer (because a particular one is required for a model)
        :param regularizerType: name of the regularizer: helmholtz|totalVariation|diffusion|curvature
        :param params: ParameterDict instance
        :return: returns a regularizer which can compute the regularization energy
        """
        cparams = params[('regularizer', {}, 'Parameters for the regularizer')]
        cparams['type'] = regularizerType

        return self._get_regularizer_instance(regularizerType,cparams)

    def create_regularizer(self, params):
        """
        Create the regularizer
        
        :param params: ParameterDict instance, expecting category 'regularizer', with variables 'type' and any settings the regularizer may require
          
        :return: returns the regularizer which can commpute the regularization energy
        """

        cparams = params[('regularizer',{},'Parameters for the regularizer')]
        regularizerType = cparams[('type',self.default_regularizer_type,
                                             'type of regularizer (only helmholtz at the moment)')]

        return self._get_regularizer_instance(regularizerType,cparams)






