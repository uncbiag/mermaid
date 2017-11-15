'''
Package implementing general purpose regularizers.
'''

from abc import ABCMeta, abstractmethod

import torch
from torch.autograd import Variable

import finite_differences as fd
from dataWarper import AdaptVal

class Regularizer(object):
    """
    Abstract regularizer base class
    """
    __metaclass__ = ABCMeta

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
        reg = AdaptVal(Variable(torch.zeros(1), requires_grad=False))
        for nrI in range(szv[0]): # loop over number of images
            reg = reg + self._compute_regularizer(v[nrI, ...])
        return reg

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
        Lv = AdaptVal(Variable(torch.zeros(v.size()), requires_grad=False))
        # None is refer to batch, which is added here for compatibility, the following [0] is used for this reason
        Lv[0,:] = v[0,:] * gamma - self.fdt.lap(v[None,0,:])[0] * alpha
        # now compute the norm
        return (Lv[0,:] ** 2).sum()*self.volumeElement

    def _compute_regularizer_2d(self, v, alpha, gamma):
        Lv = AdaptVal(Variable(torch.zeros(v.size()), requires_grad=False))
        for i in [0, 1]:
            # None is refer to batch, which is added here for compatibility, the following [0] is used for this reason
            Lv[i,:, :] = v[i,:, :] * gamma - self.fdt.lap(v[None, i,:, :])[0] * alpha

        # now compute the norm
        return (Lv[0,:, :] ** 2 + Lv[1,:, :] ** 2).sum()*self.volumeElement

    def _compute_regularizer_3d(self, v, alpha, gamma):
        Lv = AdaptVal(Variable(torch.zeros(v.size()), requires_grad=False))
        for i in [0, 1, 2]:
            # None is refer to batch, which is added here for compatibility, the following [0] is used for this reason
            Lv[i,:, :, :] = v[i,:, :, :] * gamma - self.fdt.lap(v[None,i,:, :, :])[0] * alpha

        # now compute the norm
        return (Lv[0,:, :, :] ** 2 + Lv[1,:, :, :] ** 2 + Lv[2,:, :, :] ** 2).sum()*self.volumeElement


class RegularizerFactory(object):
    """
    Regularizer factory to instantiate a regularizer by name.
    """

    __metaclass__ = ABCMeta

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

    def create_regularizer(self, params):
        """
        Create the regularizer
        
        :param params: ParameterDict instance, expecting category 'regularizer', with variables 'type' and any settings the regularizer may require
          
        :return: returns the regularization energy
        """

        cparams = params[('regularizer',{},'Parameters for the regularizer')]
        regularizerType = cparams[('type',self.default_regularizer_type,
                                             'type of regularizer (only helmholtz at the moment)')]

        if regularizerType=='helmholtz':
            return HelmholtzRegularizer(self.spacing,cparams)
        else:
            raise ValueError( 'Regularizer: ' + regularizerType + ' not known')





