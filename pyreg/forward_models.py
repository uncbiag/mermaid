"""
Package defining various dynamic forward models as well as convenience methods to generate the
right hand sides (RHS) of the related partial differential equations.

Currently, the following forward models are implemented:
    #. An advection equation for images
    #. An advection equation for maps
    #. The EPDiff-equation parameterized using the vector-valued momentum for images
    #. The EPDiff-equation parameterized using the vector-valued momentum for maps
    #. The EPDiff-equation parameterized using the scalar-valued momentum for images
    #. The EPDiff-equation parameterized using the scalar-valued momentum for maps
    
The images are expected to be tensors of dimension: BxCxXxYxZ (or BxCxX in 1D and BxCxXxY in 2D),
where B is the batch-size, C the number of channels, and X, Y, and Z are the spatial coordinate indices.

Futhermore the following (RHSs) are provided
    #. Image advection
    #. Map advection
    #. Scalar conservation law
    #. EPDiff
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import finite_differences as fd
import smoother_factory as sf
import utils
from data_wrapper import MyTensor
import torch
from torch.autograd import Variable

class RHSLibrary(object):
    """
    Convenience class to quickly generate various right hand sides (RHSs) of popular partial differential 
    equations. In this way new forward models can be written with minimal code duplication.
    """

    def __init__(self, spacing, use_neumann_BC_for_map=False):
        """
        Constructor
        
        :param spacing: Spacing for the images. This will be an array with 1, 2, or 3 entries in 1D, 2D, and 3D respectively. 
        """
        self.spacing = spacing
        """spatial spacing"""
        self.fdt = fd.FD_torch( self.spacing )
        """torch finite differencing support"""
        self.fdt_le = fd.FD_torch( self.spacing, False)
        """torch finite differencing support w/ linear extrapolation"""
        self.dim = len(self.spacing)
        """spatial dimension"""
        self.use_neumann_BC_for_map = use_neumann_BC_for_map
        """If True uses zero Neumann boundary conditions also for evolutions of the map, if False uses linear extrapolation"""

    def rhs_advect_image_multiNC(self,I,v):
        '''
        Advects a batch of images which can be multi-channel. Expected image format here, is 
        BxCxXxYxZ, where B is the number of images (batch size), C, the number of channels
        per image and X, Y, Z are the spatial coordinates (X only in 1D; X,Y only in 2D)
        if the channel size is 1, then a fast version will be used, which batch can be calculate in one loop
        otherwise the traditional loop batch and loop channel way will be used
        
        :math:`-\\nabla I^Tv`

        
        :param I: Image batch
        :param v: Velocity fields (this will be one velocity field per image)
        :return: Returns the RHS of the advection equations involved
        '''
        sz = I.size()
        rhs_ret = Variable(MyTensor(sz).zero_(), requires_grad=False)
        if sz[1]==1:  #  when only has one channel use fast version
            self._rhs_advect_oneC_image(I,v,rhs_ret)
        else:
            for nrI in range(sz[0]):  # loop over all the images
                rhs_ret[nrI, ...] = self._rhs_advect_image_multiC(I[nrI, ...], v[nrI, ...])
        return rhs_ret

    def _rhs_advect_image_multiC(self,I,v):
        """
        Similar to *_rhs_advect_image_multiNC*, but only for one multi-channel image at at time.

        :param I: Multi-channel image: CxXxYxZ
        :param v: Same velocity field applied to each channel
        :return: Returns the RHS of the advection equation
        """
        sz = I.size()
        rhs_ret = Variable(MyTensor(sz).zero_(), requires_grad=False)
        for nrC in range(sz[0]): # loop over all the channels, just advect them all the same
            rhs_ret[nrC,...] = self._rhs_advect_image_singleC(I[None,nrC,...],v)  # additional None add here for compatiabilty
        return rhs_ret

    def _rhs_advect_image_singleC(self, I, v):
        """
        Similar to *_rhs_advect_image_multiC*, but only for one channel at a time.

        :param I: One-channel input image: 1xXxYxZ, here 1 is refer to batch, for compatibility
        :param v: velocity field
        :return: Returns the RHS of the advection equation for one channel
        ##############   additional None channel is for batch
        """
        if self.dim == 1:
            return -self.fdt.dXc(I) * v[None,0, :]
        elif self.dim == 2:
            return -self.fdt.dXc(I) * v[None,0, :, :] - self.fdt.dYc(I) * v[None,1, :, :]
        elif self.dim == 3:
            return -self.fdt.dXc(I) * v[None,0, :, :, :] - self.fdt.dYc(I) * v[None,1, :, :, :] - self.fdt.dZc(I) * v[None,2, :, :, :]
        else:
            raise ValueError('Only supported up to dimension 3')


    def _rhs_advect_oneC_image(self,I,v,rhs_ret):
        """
        a fast version for _rhs_advect_image_multiC,   batch will no longer to be calculated separately, one channel at a time.
        
        :param I: One-channel input image: batchxchxXxYxZ
        :param v: velocity field
        :return: Returns the RHS of the advection equation for one channel
        """
        if self.dim == 1:
            rhs_ret[:] = -self.fdt.dXc(I[:,0,...]) * v[:,0,...]
        elif self.dim == 2:
            rhs_ret[:] = -self.fdt.dXc(I[:,0,...]) * v[:,0,...] -self.fdt.dYc(I[:,0,...])*v[:,1,...]
        elif self.dim == 3:
            rhs_ret[:] = -self.fdt.dXc(I[:,0,...]) * v[:,0,...] -self.fdt.dYc(I[:,0,...])*v[:,1,...]-self.fdt.dZc(I[:,0,...])*v[:,2,...]
        else:
            raise ValueError('Only supported up to dimension 3')


    def rhs_scalar_conservation_multiNC(self, I, v):
        """
        Scalar conservation law for a batch of images which can be multi-channel. Expected image format here, is 
        BxCxXxYxZ, where B is the number of images (batch size), C, the number of channels
        per image and X, Y, Z are the spatial coordinates (X only in 1D; X,Y only in 2D)

        if the channel size is 1, then a fast version will be used, which batch can be calculate in one loop
        otherwise the traditional loop batch and loop channel way will be used

        :math:`-div(Iv)`

        :param I: Image batch
        :param v: Velocity fields (this will be one velocity field per image)
        :return: Returns the RHS of the scalar conservation law equations involved
        """

        sz = I.size()
        rhs_ret = Variable(MyTensor(sz).zero_(), requires_grad=False)
        # for nrI in range(sz[0]):  # loop over all the images
        #     rhs_ret[nrI, ...] = self._rhs_scalar_conservation_multiC(I[nrI, ...], v[nrI, ...])
        # return rhs_ret
        if sz[1]==1:  # if one channel case, a fast version will be used
            self._rhs_scalar_oneC_conservation(I, v, rhs_ret)
        else:
            for nrI in range(sz[0]):  # loop over all the images
                rhs_ret[nrI, ...] = self._rhs_scalar_conservation_multiC(I[nrI, ...], v[nrI, ...])
        return rhs_ret

    def _rhs_scalar_conservation_multiC(self,I,v):
        """
        Similar to *_rhs_scalar_conservation_multiNC*, but only for one multi-channel image at at time.

        :param I: Multi-channel image: CxXxYxZ
        :param v: Same velocity field applied to each channel
        :return: Returns the RHS of the scalar-conservation law equation
        """
        sz = I.size()
        rhs_ret = Variable(MyTensor(sz).zero_(), requires_grad=False)
        for nrC in range(sz[0]):  # loop over all the channels, just advect them all the same
            rhs_ret[nrC, ...] = self._rhs_scalar_conservation_singleC(I[None,nrC, ...], v)  # here None is added for compatiability
        return rhs_ret

    def _rhs_scalar_conservation_singleC(self,I,v):
        """
        Similar to *_rhs_scalar_conservation_multiC*, but only for one channel at a time.

        :param I: One-channel input image: 1xXxYxZ  here 1 refer to batch, for compatibility
        :param v: velocity field
        :return: Returns the RHS of the scalar-conservation law equation for one channel
        """
        if self.dim==1:
            return -self.fdt.dXc(I*v[None,0,:])
        elif self.dim==2:
            return -self.fdt.dXc(I*v[None,0,:,:]) -self.fdt.dYc(I*v[None,1,:,:])
        elif self.dim==3:
            return -self.fdt.dXc(I* v[None,0,:,:,:]) -self.fdt.dYc(I*v[None,1,:,:,:])-self.fdt.dZc(I*v[None,2,:,:,:])
        else:
            raise ValueError('Only supported up to dimension 3')


    def _rhs_scalar_oneC_conservation(self, I, v, rhs_ret):
        """
        a fast version for _rhs_scalar_conservation_multiC,   batch will no longer to be calculated sperately, one channel at a time.

        :param I: One-channel input image: batchxchXxYxZ
        :param v: velocity field
        :return: Returns the RHS of the scalar-conservation law equation for one channel
        """
        if self.dim==1:
            rhs_ret[:] = -self.fdt.dXc(I[:,0,...]*v[:,0,...])
        elif self.dim==2:
            rhs_ret[:] = -self.fdt.dXc(I[:,0,...]*v[:,0,...]) -self.fdt.dYc(I[:,0,...]*v[:,1,...])
        elif self.dim==3:
            rhs_ret[:] = -self.fdt.dXc(I[:,0,...]* v[:,0,...]) -self.fdt.dYc(I[:,0,...]*v[:,1,...])-self.fdt.dZc(I[:,0,...]*v[:,2,...])
        else:
            raise ValueError('Only supported up to dimension 3')


    def rhs_advect_map_multiN(self, phi, v):
        '''
        Advects a set of N maps (for N images). Expected format here, is 
        BxCxXxYxZ, where B is the number of images/maps (batch size), C, the number of channels
        per (here the spatial dimension for the map coordinate functions), 
        and X, Y, Z are the spatial coordinates (X only in 1D; X,Y only in 2D)

        new fast version is implemented, where batch is no longer calculated separately

        :math:`-D\\phi v`

        :param phi: map batch
        :param v: Velocity fields (this will be one velocity field per map)
        :return: Returns the RHS of the advection equations involved
        '''
        sz = phi.size()
        rhs_ret = Variable(MyTensor(sz).zero_(), requires_grad=False)
        # for nrI in range(sz[0]):  # loop over all the images
        #     rhs_ret[nrI, ...] = self._rhs_advect_map_singleN(phi[nrI, ...], v[nrI, ...])
        self._rhs_advect_map_singleN(phi, v,rhs_ret)
        return rhs_ret

    def _rhs_advect_map_singleN(self,phi,v,rhsphi):  ##########

        if self.use_neumann_BC_for_map:
            fdc = self.fdt # use zero Neumann boundary conditions
        else:
            fdc = self.fdt_le # do linear extrapolation

        if self.dim==1:
            rhsphi[:]= -fdc.dXc(phi[:,0,:]) * v[:,0,:]
        elif self.dim==2:
            #rhsphi = Variable(MyTensor( phi.size() ).zero_(), requires_grad=False)
            rhsphi[:,0,:, :] = -(v[:,0,:, :] * fdc.dXc(phi[:,0,:, :]) + v[:,1,:, :] * fdc.dYc(phi[:,0,:, :]))
            rhsphi[:,1,:, :] = -(v[:,0,:, :] * fdc.dXc(phi[:,1,:, :]) + v[:,1,:, :] * fdc.dYc(phi[:,1,:, :]))
            #return rhsphi
        elif self.dim==3:
            #rhsphi = Variable(MyTensor( phi.size() ).zero_(), requires_grad=False)
            rhsphi[:,0,:, :, :] = -(v[:,0,:, :, :] * fdc.dXc(phi[:,0,:, :, :]) +
                                   v[:,1,:, :, :] * fdc.dYc(phi[:,0,:, :, :]) +
                                   v[:,2,:, :, :] * fdc.dZc(phi[:,0,:, :, :]))

            rhsphi[:,1,:, :, :] = -(v[:,0,:, :, :] * fdc.dXc(phi[:,1,:, :, :]) +
                                   v[:,1,:, :, :] * fdc.dYc(phi[:,1,:, :, :]) +
                                   v[:,2,:, :, :] * fdc.dZc(phi[:,1,:, :, :]))

            rhsphi[:,2,:, :, :] = -(v[:,0,:, :, :] * fdc.dXc(phi[:,2,:, :, :]) +
                                   v[:,1,:, :, :] * fdc.dYc(phi[:,2,:, :, :]) +
                                   v[:,2,:, :, :] * fdc.dZc(phi[:,2,:, :, :]))
            #return rhsphi
        else:
            raise ValueError('Only supported up to dimension 3')




    def rhs_epdiff_multiN(self, m, v):
        '''
        Computes the right hand side of the EPDiff equation for of N momenta (for N images). 
        Expected format here, is BxCxXxYxZ, where B is the number of momenta (batch size), C, 
        the number of channels per (here the spatial dimension for the momenta), 
        and X, Y, Z are the spatial coordinates (X only in 1D; X,Y only in 2D)

        a new version, where batch is no longer calculated separately

        :math:`-(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`

        :param m: momenta batch
        :param v: Velocity fields (this will be one velocity field per momentum)
        :return: Returns the RHS of the EPDiff equations involved
        '''
        sz = m.size()
        rhs_ret = Variable(MyTensor(sz).zero_(), requires_grad=False)
        # for nrI in range(sz[0]):  # loop over all the images
        #     rhs_ret[nrI, ...] = self._rhs_epdiff_singleN(m[nrI, ...], v[nrI, ...])
        self._rhs_epdiff_singleN(m, v, rhs_ret)
        return rhs_ret

    def _rhs_epdiff_singleN(self, m, v,rhsm):
        if self.dim == 1:
            rhsm[:]= -self.fdt.dXc(m[:,:, 0] * v[:,0, :]) - self.fdt.dXc(v[:,0, :]) * m[:,0, :]
        elif self.dim == 2:
            #rhsm = Variable(MyTensor(m.size()).zero_(), requires_grad=False)
            # (m_1,...,m_d)^T_t = -(div(m_1v),...,div(m_dv))^T-(Dv)^Tm  (EPDiff equation)
            rhsm[:,0, :, :] = (-self.fdt.dXc(m[:,0, :, :] * v[:,0, :, :])
                             - self.fdt.dYc(m[:,0, :, :] * v[:,1, :, :])
                             - self.fdt.dXc(v[:,0, :, :]) * m[:,0, :, :]
                             - self.fdt.dXc(v[:,1, :, :]) * m[:,1, :, :])

            rhsm[:,1, :, :] = (-self.fdt.dXc(m[:,1, :, :] * v[:,0, :, :])
                             - self.fdt.dYc(m[:,1, :, :] * v[:,1, :, :])
                             - self.fdt.dYc(v[:,0, :, :]) * m[:,0, :, :]
                             - self.fdt.dYc(v[:,1, :, :]) * m[:,1, :, :])
            #return rhsm
        elif self.dim == 3:
            #rhsm = Variable(MyTensor(m.size()).zero_(), requires_grad=False)
            # (m_1,...,m_d)^T_t = -(div(m_1v),...,div(m_dv))^T-(Dv)^Tm  (EPDiff equation)
            rhsm[:,0, :, :, :] = (-self.fdt.dXc(m[:,0, :, :, :] * v[:,0, :, :, :])
                                - self.fdt.dYc(m[:,0, :, :, :] * v[:,1, :, :, :])
                                - self.fdt.dZc(m[:,0, :, :, :] * v[:,2, :, :, :])
                                - self.fdt.dXc(v[:,0, :, :, :]) * m[:,0, :, :, :]
                                - self.fdt.dXc(v[:,1, :, :, :]) * m[:,1, :, :, :]
                                - self.fdt.dXc(v[:,2, :, :, :]) * m[:,2, :, :, :])

            rhsm[:,1, :, :, :] = (-self.fdt.dXc(m[:,1, :, :, :] * v[:,0, :, :, :])
                                - self.fdt.dYc(m[:,1, :, :, :] * v[:,1, :, :, :])
                                - self.fdt.dZc(m[:,1, :, :, :] * v[:,2, :, :, :])
                                - self.fdt.dYc(v[:,0, :, :, :]) * m[:,0, :, :, :]
                                - self.fdt.dYc(v[:,1, :, :, :]) * m[:,1, :, :, :]
                                - self.fdt.dYc(v[:,2, :, :, :]) * m[:,2, :, :, :])

            rhsm[:,2, :, :, :] = (-self.fdt.dXc(m[:,2, :, :, :] * v[:,0, :, :, :])
                                - self.fdt.dYc(m[:,2, :, :, :] * v[:,1, :, :, :])
                                - self.fdt.dZc(m[:,2, :, :, :] * v[:,2, :, :, :])
                                - self.fdt.dZc(v[:,0, :, :, :]) * m[:,0, :, :, :]
                                - self.fdt.dZc(v[:,1, :, :, :]) * m[:,1, :, :, :]
                                - self.fdt.dZc(v[:,2, :, :, :]) * m[:,2, :, :, :])
            #return rhsm
        else:
            raise ValueError('Only supported up to dimension ')


    # def _rhs_epdiff_singleN(self,m,v):
    #     if self.dim==1:
    #         return -self.fdt.dXc(m[:,0] * v[0,:]) - self.fdt.dXc(v[0,:]) * m[0,:]
    #     elif self.dim==2:
    #         rhsm = Variable(MyTensor( m.size() ).zero_(), requires_grad=False)
    #          # (m_1,...,m_d)^T_t = -(div(m_1v),...,div(m_dv))^T-(Dv)^Tm  (EPDiff equation)
    #         rhsm[0,:, :] = (-self.fdt.dXc(m[0,:, :] * v[0,:, :])
    #                          - self.fdt.dYc(m[0,:, :] * v[1,:, :])
    #                          - self.fdt.dXc(v[0,:, :]) * m[0,:, :]
    #                          - self.fdt.dXc(v[1,:, :]) * m[1,:, :])
    #
    #         rhsm[1,:, :] = (-self.fdt.dXc(m[1,:, :] * v[0,:, :])
    #                          - self.fdt.dYc(m[1,:, :] * v[1,:, :])
    #                          - self.fdt.dYc(v[0,:, :]) * m[0,:, :]
    #                          - self.fdt.dYc(v[1,:, :]) * m[1,:, :])
    #         return rhsm
    #     elif self.dim==3:
    #         rhsm = Variable(MyTensor( m.size() ).zero_(), requires_grad=False)
    #         # (m_1,...,m_d)^T_t = -(div(m_1v),...,div(m_dv))^T-(Dv)^Tm  (EPDiff equation)
    #         rhsm[0,:, :, :] = (-self.fdt.dXc(m[0,:, :, :] * v[0,:, :, :])
    #                             - self.fdt.dYc(m[0,:, :, :] * v[1,:, :, :])
    #                             - self.fdt.dZc(m[0,:, :, :] * v[2,:, :, :])
    #                             - self.fdt.dXc(v[0,:, :, :]) * m[0,:, :, :]
    #                             - self.fdt.dXc(v[1,:, :, :]) * m[1,:, :, :]
    #                             - self.fdt.dXc(v[2,:, :, :]) * m[2,:, :, :])
    #
    #         rhsm[1,:, :, :] = (-self.fdt.dXc(m[1,:, :, :] * v[0,:, :, :])
    #                             - self.fdt.dYc(m[1,:, :, :] * v[1,:, :, :])
    #                             - self.fdt.dZc(m[1,:, :, :] * v[2,:, :, :])
    #                             - self.fdt.dYc(v[0,:, :, :]) * m[0,:, :, :]
    #                             - self.fdt.dYc(v[1,:, :, :]) * m[1,:, :, :]
    #                             - self.fdt.dYc(v[2,:, :, :]) * m[2,:, :, :])
    #
    #         rhsm[2,:, :, :] = (-self.fdt.dXc(m[2,:, :, :] * v[0,:, :, :])
    #                             - self.fdt.dYc(m[2,:, :, :] * v[1,:, :, :])
    #                             - self.fdt.dZc(m[2,:, :, :] * v[2,:, :, :])
    #                             - self.fdt.dZc(v[0,:, :, :]) * m[0,:, :, :]
    #                             - self.fdt.dZc(v[1,:, :, :]) * m[1,:, :, :]
    #                             - self.fdt.dZc(v[2,:, :, :]) * m[2,:, :, :])
    #         return rhsm
    #     else:
    #         raise ValueError('Only supported up to dimension 3')

class ForwardModel(object):
    """
    Abstract forward model class. Should never be instantiated.
    Derived classes require the definition of f(self,t,x,u,pars) and u(self,t,pars).
    These functions will be used for integration: x'(t) = f(t,x(t),u(t))
    """
    __metaclass__ = ABCMeta

    def __init__(self, sz, spacing, params=None):
        '''
        Constructor of abstract forward model class
        
        :param sz: size of images
        :param spacing: numpy array for spacing in x,y,z directions
        '''

        self.dim = spacing.size # spatial dimension of the problem
        """spatial dimension"""
        self.spacing = spacing
        """spatial spacing"""
        self.sz = sz
        """image size (BxCxXxYxZ)"""
        self.params = params
        """ParameterDict instance holding parameters"""
        self.rhs = RHSLibrary(self.spacing)
        """rhs library support"""

        if self.dim>3 or self.dim<1:
            raise ValueError('Forward models are currently only supported in dimensions 1 to 3')

        self.fdt = fd.FD_torch( self.spacing )
        """torch finite difference support"""

    @abstractmethod
    def f(self,t,x,u,pars):
        """
        Function to be integrated
        
        :param t: time
        :param x: state
        :param u: input
        :param pars: optional parameters
        :return: the function value, should return a list (to support easy concatenations of states)
        """
        pass

    def u(self,t,pars):
        """
        External input
        
        :param t: time
        :param pars: parameters
        :return: the external input
        """
        return []


class AdvectMap(ForwardModel):
    """
    Forward model to advect an n-D map using a transport equation: :math:`\\Phi_t + D\\Phi v = 0`.
    v is treated as an external argument and \Phi is the state
    """

    def __init__(self, sz, spacing, params=None):
        super(AdvectMap,self).__init__(sz,spacing,params)

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
        Function to be integrated, i.e., right hand side of transport equation: 
        
        :math:`-D\\phi v`
        
        :param t: time (ignored; not time-dependent) 
        :param x: state, here the map, \Phi, itself (assumes 3D-5D array; [nrI,0,:,:] x-coors; [nrI,1,:,:] y-coors; ...
        :param u: external input, will be the velocity field here
        :param pars: ignored (does not expect any additional inputs)
        :return: right hand side [phi]
        """
        return [self.rhs.rhs_advect_map_multiN(x[0],u)]

class AdvectImage(ForwardModel):
    """
    Forward model to advect an image using a transport equation: :math:`I_t + \\nabla I^Tv = 0`.
    v is treated as an external argument and I is the state
    """

    def __init__(self, sz, spacing, params=None):
        super(AdvectImage, self).__init__(sz, spacing,params)


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
        Function to be integrated, i.e., right hand side of transport equation: :math:`-\\nabla I^T v`
        
        :param t: time (ignored; not time-dependent) 
        :param x: state, here the image, I, itself (supports multiple images and channels)
        :param u: external input, will be the velocity field here
        :param pars: ignored (does not expect any additional inputs)
        :return: right hand side [I]
        """
        return [self.rhs.rhs_advect_image_multiNC(x[0],u)]


class EPDiffImage(ForwardModel):
    """
    Forward model for the EPdiff equation. State is the momentum, m, and the image I:
    :math:`(m_1,...,m_d)^T_t = -(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`
    
    :math:`v=Km`
    
    :math:`I_t+\\nabla I^Tv=0`
    """
    def __init__(self, sz, spacing, params=None):
        super(EPDiffImage, self).__init__(sz, spacing,params)
        self.smoother = sf.SmootherFactory(self.sz[2::],self.spacing).create_smoother(params)

    def f(self,t, x, u, pars):
        """
        Function to be integrated, i.e., right hand side of the EPDiff equation: 
        :math:`-(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`
        
        :math:`-\\nabla I^Tv`
        
        :param t: time (ignored; not time-dependent) 
        :param x: state, here the vector momentum, m, and the image, I
        :param u: ignored, no external input
        :param pars: ignored (does not expect any additional inputs)
        :return: right hand side [m,I]
        """
        # assume x[0] is m and x[1] is I for the state
        m = x[0]
        I = x[1]
        v = self.smoother.smooth_vector_field_multiN(m)
        # print('max(|v|) = ' + str( v.abs().max() ))
        return [self.rhs.rhs_epdiff_multiN(m,v), self.rhs.rhs_advect_image_multiNC(I,v)]


class EPDiffMap(ForwardModel):
    """
    Forward model for the EPDiff equation. State is the momentum, m, and the transform, :math:`\\phi` 
    (mapping the source image to the target image).
    :math:`(m_1,...,m_d)^T_t = -(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`
    
    :math:`v=Km`
    
    :math:`\\phi_t+D\\phi v=0`
    """

    def __init__(self, sz, spacing, params=None):
        super(EPDiffMap, self).__init__(sz,spacing,params)
        #self.smoother = sf.SmootherFactory(self.sz[2::],self.spacing).create_smoother(params)
        #######################################3
        self.smoother = params['sm_ins']
        self.use_net = True if self.params['smoother']['type'] == 'adaptiveNet' else False
        ##########################################

    def debugging(self,input,t):
        x = utils.checkNan(input)
        if np.sum(x):
            print("find nan at {} step".format(t))
            print("flag m: {}, ".format(x[0]))
            print("flag v: {},".format(x[1]))
            print("flag phi: {},".format(x[2]))
            print("flag new_m: {},".format(x[3]))
            print("flag new_phi: {},".format(x[4]))
            raise ValueError, "nan error"

    def f(self,t, x, u, pars):
        """
        Function to be integrated, i.e., right hand side of the EPDiff equation:
        :math:`-(div(m_1v),...,div(m_dv))^T-(Dv)^Tm'
        
        :math:`-D\\phi v`
        
        :param t: time (ignored; not time-dependent) 
        :param x: state, here the image, vector momentum, m, and the map, :math:`\\phi`
        :param u: ignored, no external input
        :param pars: ignored (does not expect any additional inputs)
        :return: right hand side [m,phi]
        """

        # assume x[0] is m and x[1] is phi for the state
        m = x[0]
        phi = x[1]
        if not self.use_net:
            v = self.smoother.smooth_vector_field_multiN(m)
        else:
            v = self.smoother.adaptive_smooth(m, phi, using_map=True)

        # print('max(|v|) = ' + str( v.abs().max() ))
        new_val= [self.rhs.rhs_epdiff_multiN(m,v),self.rhs.rhs_advect_map_multiN(phi,v)]
        #self.debugging([m, v, phi, new_val[0], new_val[1]], t)
        return new_val

class EPDiffScalarMomentum(ForwardModel):
    """
    Base class for scalar momentum EPDiff solutions. Defines a smoother that can be commonly used.
    """

    def __init__(self, sz, spacing, params):
        super(EPDiffScalarMomentum,self).__init__(sz,spacing,params)

        self.smoother = sf.SmootherFactory(self.sz[2::],self.spacing).create_smoother(params)


class EPDiffScalarMomentumImage(EPDiffScalarMomentum):
    """
    Forward model for the scalar momentum EPdiff equation. State is the scalar momentum, lam, and the image I
    :math:`(m_1,...,m_d)^T_t = -(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`

    :math:`v=Km`

    :math:'m=\\lambda\\nabla I`

    :math:`I_t+\\nabla I^Tv=0`

    :math:`\\lambda_t + div(\\lambda v)=0`
    """

    def __init__(self, sz, spacing, params=None):
        super(EPDiffScalarMomentumImage, self).__init__(sz, spacing, params)

    def f(self, t, x, u, pars):
        """
        Function to be integrated, i.e., right hand side of the EPDiff equation:

        :math:`-(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`

        :math:`-\\nabla I^Tv`

        :math: `-div(\\lambda v)`

        :param t: time (ignored; not time-dependent) 
        :param x: state, here the scalar momentum, lam, and the image, I, itself
        :param u: no external input
        :param pars: ignored (does not expect any additional inputs)
        :return: right hand side [lam,I]
        """
        # assume x[0] is \lambda and x[1] is I for the state
        lam = x[0]
        I = x[1]

        # now compute the momentum
        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(lam, I, self.sz, self.spacing)
        v = self.smoother.smooth_vector_field_multiN(m)

        # advection for I, scalar-conservation law for lam
        return [self.rhs.rhs_scalar_conservation_multiNC(lam, v), self.rhs.rhs_advect_image_multiNC(I, v)]


class EPDiffScalarMomentumImage(EPDiffScalarMomentum):
    """
    Forward model for the scalar momentum EPdiff equation. State is the scalar momentum, lam, and the image I
    :math:`(m_1,...,m_d)^T_t = -(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`
    
    :math:`v=Km`
    
    :math:'m=\\lambda\\nabla I`
    
    :math:`I_t+\\nabla I^Tv=0`
    
    :math:`\\lambda_t + div(\\lambda v)=0`
    """

    def __init__(self, sz, spacing, params=None):
        super(EPDiffScalarMomentumImage, self).__init__(sz, spacing,params)

    def f(self,t, x, u, pars):
        """
        Function to be integrated, i.e., right hand side of the EPDiff equation:
        
        :math:`-(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`
        
        :math:`-\\nabla I^Tv`
        
        :math: `-div(\\lambda v)`
        
        :param t: time (ignored; not time-dependent) 
        :param x: state, here the scalar momentum, lam, and the image, I, itself
        :param u: no external input
        :param pars: ignored (does not expect any additional inputs)
        :return: right hand side [lam,I]
        """
        # assume x[0] is \lambda and x[1] is I for the state
        lam = x[0]
        I = x[1]

        # now compute the momentum
        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(lam, I, self.sz, self.spacing)
        v = self.smoother.smooth_vector_field_multiN(m)

        # advection for I, scalar-conservation law for lam
        return [self.rhs.rhs_scalar_conservation_multiNC(lam,v), self.rhs.rhs_advect_image_multiNC(I,v)]


class EPDiffScalarMomentumMap(EPDiffScalarMomentum):
    """
    Forward model for the scalar momentum EPDiff equation. State is the scalar momentum, lam, the image, I, and the transform, phi.
    :math:`(m_1,...,m_d)^T_t = -(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`
    
    :math:`v=Km`
    
    :math:`m=\\lambda\\nabla I`
    
    :math:`I_t+\\nabla I^Tv=0`
    
    :math:`\\lambda_t + div(\\lambda v)=0`
    
    :math:`\\Phi_t+D\\Phi v=0`
    """

    def __init__(self, sz, spacing, params=None):
        super(EPDiffScalarMomentumMap, self).__init__(sz,spacing,params)


    def f(self,t, x, u, pars):
        """
        Function to be integrated, i.e., right hand side of the EPDiff equation:
        
        :math:`-(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`
        
        :math:`-\\nabla I^Tv`
        
        :math:`-div(\\lambda v)`
        
        :math:`-D\\Phi v`
        
        :param t: time (ignored; not time-dependent) 
        :param x: state, here the scalar momentum, lam, the image, I, and the transform, :math:`\\phi`
        :param u: ignored, no external input
        :param pars: ignored (does not expect any additional inputs)
        :return: right hand side [lam,I,phi]
        """

        # assume x[0] is lam and x[1] is I and x[2] is phi for the state
        lam = x[0]
        I = x[1]
        phi = x[2]

        # now compute the momentum
        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(lam, I, self.sz, self.spacing)
        v = self.smoother.smooth_vector_field_multiN(m)

        return [self.rhs.rhs_scalar_conservation_multiNC(lam,v),
                self.rhs.rhs_advect_image_multiNC(I,v),
                self.rhs.rhs_advect_map_multiN(phi,v)]
