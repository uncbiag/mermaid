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
from __future__ import print_function
from __future__ import absolute_import

from builtins import range
from builtins import object
from abc import ABCMeta, abstractmethod
import numpy as np
from . import finite_differences_multi_channel as fdm
from . import utils
from .data_wrapper import MyTensor
from future.utils import with_metaclass
import torch.nn as nn
import torch


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
        self.spacing_min = np.min(spacing)
        """ min of the spacing"""
        self.spacing_ratio = spacing/self.spacing_min
        self.fdt_ne = fdm.FD_torch_multi_channel(spacing,mode='neumann_zero')
        """torch finite differencing support neumann zero"""
        self.fdt_le = fdm.FD_torch_multi_channel( spacing, mode='linear')
        """torch finite differencing support linear extrapolation"""
        self.fdt_di = fdm.FD_torch_multi_channel(spacing, mode='dirichlet_zero')
        """torch finite differencing support dirichlet zero"""
        self.dim = len(self.spacing)
        """spatial dimension"""
        self.use_neumann_BC_for_map = use_neumann_BC_for_map
        """If True uses zero Neumann boundary conditions also for evolutions of the map, if False uses linear extrapolation"""

    def rhs_advect_image_multiNC(self,I,v):
        '''
        Advects a batch of images which can be multi-channel. Expected image format here, is 
        BxCxXxYxZ, where B is the number of images (batch size), C, the number of channels
        per image and X, Y, Z are the spatial coordinates (X only in 1D; X,Y only in 2D)

        
        :math:`-\\nabla I^Tv`

        
        :param I: Image batch BxCIxXxYxZ
        :param v: Velocity fields (this will be one velocity field per image) BxCxXxYxZ
        :return: Returns the RHS of the advection equations involved BxCxXxYxZ
        '''

        rhs_ret= self._rhs_advect_image_multiN(I, v )
        return rhs_ret


    def _rhs_advect_image_multiN(self,I,v):
        """
        :param I: One-channel input image: Bx1xXxYxZ
        :param v: velocity field BxCxXxYxZ
        :return: Returns the RHS of the advection equation for one channel BxXxYxZ
        """

        if self.dim == 1:
            rhs_ret = -self.fdt_ne.dXc(I) * v[:,0:1]
        elif self.dim == 2:
            rhs_ret = -self.fdt_ne.dXc(I) * v[:,0:1] -self.fdt_ne.dYc(I)*v[:,1:2]
        elif self.dim == 3:
            rhs_ret = -self.fdt_ne.dXc(I) * v[:,0:1] -self.fdt_ne.dYc(I)*v[:,1:2]-self.fdt_ne.dZc(I)*v[:,2:3]
        else:
            raise ValueError('Only supported up to dimension 3')
        return rhs_ret


    def rhs_scalar_conservation_multiNC(self, I, v):
        """
        Scalar conservation law for a batch of images which can be multi-channel. Expected image format here, is 
        BxCxXxYxZ, where B is the number of images (batch size), C, the number of channels
        per image and X, Y, Z are the spatial coordinates (X only in 1D; X,Y only in 2D)

        :math:`-div(Iv)`

        :param I: Image batch BxCIxXxYxZ
        :param v: Velocity fields (this will be one velocity field per image) BxCxXxYxZ
        :return: Returns the RHS of the scalar conservation law equations involved BxCxXxYxZ
        """

        rhs_ret=self._rhs_scalar_conservation_multiN(I, v)
        return rhs_ret



    def _rhs_scalar_conservation_multiN(self, I, v):
        """
        :param I: One-channel input image: Bx1xXxYxZ
        :param v: velocity field  BxCxXxYxZ
        :return: Returns the RHS of the scalar-conservation law equation for one channel BxXxYxZ
        """

        if self.dim==1:
            rhs_ret = -self.fdt_ne.dXc(I*v[:,0:1])
        elif self.dim==2:
            rhs_ret = -self.fdt_ne.dXc(I*v[:,0:1]) -self.fdt_ne.dYc(I*v[:,1:2])
        elif self.dim==3:
            rhs_ret = -self.fdt_ne.dXc(I* v[:,0:1]) -self.fdt_ne.dYc(I*v[:,1:2])-self.fdt_ne.dZc(I*v[:,2:3])
        else:
            raise ValueError('Only supported up to dimension 3')
        return rhs_ret


    def rhs_lagrangian_evolve_map_multiNC(self, phi, v):
        """
        Evolves a set of N maps (for N images). Expected format here, is
        BxCxXxYxZ, where B is the number of images/maps (batch size), C, the number of channels
        per (here the spatial dimension for the map coordinate functions),
        and X, Y, Z are the spatial coordinates (X only in 1D; X,Y only in 2D).
        This is used to evolve the map going from source to target image. Requires interpolation
        so should if at all possible not be used as part of an optimization.
        the idea of compute inverse map is due to the map is defined
        in the source space, referring to point move to where,(compared with the target space, refers to where it comes from)
        in this situation, we only need to capture the velocity at that place and accumulate along the time step
        since advecton function is moves the image (or phi based image) by v step, which means v is shared by different coordinate,
        so it is safe to compute in this way.

        :math:`v\circ\phi`

        :param phi: map batch BxCxXxYxZ
        :param v: Velocity fields (this will be one velocity field per map) BxCxXxYxZ
        :return: Returns the RHS of the evolution equations involved BxCxXxYxZ
        :param phi:
        :param v:
        :return:
        """

        rhs_ret = utils.compute_warped_image_multiNC(v, phi, spacing=self.spacing, spline_order=1,zero_boundary=False)
        return rhs_ret


    def rhs_advect_map_multiNC(self, phi, v):
        '''
        Advects a set of N maps (for N images). Expected format here, is 
        BxCxXxYxZ, where B is the number of images/maps (batch size), C, the number of channels
        per (here the spatial dimension for the map coordinate functions), 
        and X, Y, Z are the spatial coordinates (X only in 1D; X,Y only in 2D)


        :math:`-D\\phi v`

        :param phi: map batch BxCxXxYxZ
        :param v: Velocity fields (this will be one velocity field per map) BxCxXxYxZ
        :return: Returns the RHS of the advection equations involved BxCxXxYxZ
        '''

        sz = phi.size()
        rhs_ret = self._rhs_advect_map_call(phi, v)
        return rhs_ret

    def _rhs_advect_map_call(self,phi,v):
        """

         :param phi: map batch  BxCxXxYxZ
        :param v: Velocity fields (this will be one velocity field per map)  BxCxXxYxZ
        :return rhsphi: Returns the RHS of the advection equations involved  BxCxXxYxZ
        """

        fdc = self.fdt_le # use order boundary conditions (interpolation)

        if self.dim==1:
            dxc_phi = -fdc.dXc(phi)
            rhsphi = v[:, 0:1] * dxc_phi
        elif self.dim==2:
            dxc_phi = -fdc.dXc(phi)
            dyc_phi = -fdc.dYc(phi)
            rhsphi = v[:, 0:1] * dxc_phi + v[:, 1:2] * dyc_phi
        elif self.dim==3:
            dxc_phi = -fdc.dXc(phi)
            dyc_phi = -fdc.dYc(phi)
            dzc_phi = -fdc.dZc(phi)
            rhsphi = v[:,0:1]*dxc_phi + v[:,1:2]*dyc_phi + v[:,2:3]*dzc_phi
        else:
            raise ValueError('Only supported up to dimension 3')
        return rhsphi


    def rhs_epdiff_multiNC(self, m, v):
        '''
        Computes the right hand side of the EPDiff equation for of N momenta (for N images). 
        Expected format here, is BxCxXxYxZ, where B is the number of momenta (batch size), C, 
        the number of channels per (here the spatial dimension for the momenta), 
        and X, Y, Z are the spatial coordinates (X only in 1D; X,Y only in 2D)

        a new version, where batch is no longer calculated separately

        :math:`-(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`

        :param m: momenta batch BxCXxYxZ
        :param v: Velocity fields (this will be one velocity field per momentum) BxCXxYxZ
        :return: Returns the RHS of the EPDiff equations involved BxCXxYxZ
        '''

        sz = m.size()
        rhs_ret = MyTensor(sz).zero_()
        rhs_ret = self._rhs_epdiff_call(m, v, rhs_ret)
        return rhs_ret

    def _rhs_epdiff_call(self, m, v,rhsm):
        """
        :param m: momenta batch  BxCxXxYxZ
        :param v: Velocity fields (this will be one velocity field per momentum)  BxCxXxYxZ
        :return rhsm: Returns the RHS of the EPDiff equations involved  BxCxXxYxZ
        """

        # if self.use_neumann_BC_for_map:
        #     fdc = self.fdt_ne # use zero Neumann boundary conditions
        # else:
        #     fdc = self.fdt_le # do linear extrapolation

        fdc = self.fdt_ne
        #fdc = self.fdt_le
        if self.dim == 1:
            dxc_mv0 = -fdc.dXc(m*v[:,0:1])
            dxc_v = -fdc.dXc(v)
            dxc_v_multi_m = dxc_v * m
            rhsm[:]= dxc_mv0 + dxc_v_multi_m

        elif self.dim == 2:
            # (m_1,...,m_d)^T_t = -(div(m_1v),...,div(m_dv))^T-(Dv)^Tm  (EPDiff equation)
            dxc_mv0 = -fdc.dXc(m*v[:,0:1])
            dyc_mv1 = -fdc.dYc(m*v[:,1:2])
            dc_mv_sum = dxc_mv0 + dyc_mv1
            dxc_v = -fdc.dXc(v)
            dyc_v = -fdc.dYc(v)
            dxc_v_multi_m = dxc_v * m
            dyc_v_multi_m = dyc_v * m
            dxc_v_multi_m_sum = torch.sum(dxc_v_multi_m, 1)
            dyc_v_multi_m_sum = torch.sum(dyc_v_multi_m, 1)
            rhsm[:,0, :, :] = dc_mv_sum[:,0] + dxc_v_multi_m_sum

            rhsm[:,1, :, :] = dc_mv_sum[:,1] + dyc_v_multi_m_sum

        elif self.dim == 3:
            dxc_mv0 = -fdc.dXc(m*v[:,0:1])
            dyc_mv1 = -fdc.dYc(m*v[:,1:2])
            dzc_mv2 = -fdc.dZc(m*v[:,2:3])
            dc_mv_sum = dxc_mv0 + dyc_mv1 + dzc_mv2
            dxc_v = -fdc.dXc(v)
            dyc_v = -fdc.dYc(v)
            dzc_v = -fdc.dZc(v)
            dxc_v_multi_m = dxc_v*m
            dyc_v_multi_m = dyc_v*m
            dzc_v_multi_m = dzc_v*m
            dxc_v_multi_m_sum = torch.sum(dxc_v_multi_m,1)
            dyc_v_multi_m_sum = torch.sum(dyc_v_multi_m,1)
            dzc_v_multi_m_sum = torch.sum(dzc_v_multi_m,1)

            rhsm[:, 0] = dc_mv_sum[:,0] + dxc_v_multi_m_sum

            rhsm[:, 1] = dc_mv_sum[:,1] + dyc_v_multi_m_sum

            rhsm[:, 2] = dc_mv_sum[:,2] + dzc_v_multi_m_sum

        else:
            raise ValueError('Only supported up to dimension ')
        return rhsm



    def rhs_adapt_epdiff_wkw_multiNC(self, m, v,w, sm_wm,smoother):
        '''
        Computes the right hand side of the EPDiff equation for of N momenta (for N images).
        Expected format here, is BxCxXxYxZ, where B is the number of momenta (batch size), C,
        the number of channels per (here the spatial dimension for the momenta),
        and X, Y, Z are the spatial coordinates (X only in 1D; X,Y only in 2D)

        a new version, where batch is no longer calculated separately

        :math:`-(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`

        :param m: momenta batch BxCXxYxZ
        :param v: Velocity fields (this will be one velocity field per momentum) BxCXxYxZ
        :return: Returns the RHS of the EPDiff equations involved BxCXxYxZ
        '''

        sz = m.size()
        rhs_ret = MyTensor(sz).zero_()
        rhs_ret = self._rhs_adapt_epdiff_wkw_call(m, v,w,sm_wm,smoother, rhs_ret)
        return rhs_ret

    def _rhs_adapt_epdiff_wkw_call(self, m, v,w,sm_wm, smoother, rhsm):
        """
        :param m: momenta batch  BxCxXxYxZ
        :param sm_wm: smoothed(wm)  batch x K x dim x X x Y x ...
        :param w: smoothed(wm)  batch x K x X x Y x ...
        :param v: Velocity fields (this will be one velocity field per momentum)  BxCxXxYxZ
        :return rhsm: Returns the RHS of the EPDiff equations involved  BxCxXxYxZ
        """
        # if self.use_neumann_BC_for_map:
        #     fdc = self.fdt_ne # use zero Neumann boundary conditions
        # else:
        #     fdc = self.fdt_le # do linear extrapolation

        fdc = self.fdt_ne
        rhs = self._rhs_epdiff_call(m,v,rhsm)
        ret_var = torch.empty_like(rhs)
        # ret_var, rhs should batch x dim x X x Yx ..
        dim = m.shape[1]
        sz = [m.shape[0]]+[1]+list(m.shape[1:]) # batchx1xdimx X x Y
        m = m.view(*sz)
        m_sm_wm = m* sm_wm
        m_sm_wm = m_sm_wm.sum(dim=2)
        sm_m_sm_wm = smoother.smooth(m_sm_wm)  # batchx K x X xY...
        dxc_w = fdc.dXc(w)
        dc_w_list = [dxc_w]
        if dim == 2 or dim == 3:
            dyc_w = fdc.dYc(w)
            dc_w_list.append(dyc_w)
        if dim == 3:
            dzc_w = fdc.dZc(w)  # batch x K x X xY ...
            dc_w_list.append(dzc_w)
        for i in range(dim):
            ret_var[:, i] = rhs[:, i] + (sm_m_sm_wm* dc_w_list[i]).sum(1)

        return ret_var



class ForwardModel(with_metaclass(ABCMeta, object)):
    """
    Abstract forward model class. Should never be instantiated.
    Derived classes require the definition of f(self,t,x,u,pars) and u(self,t,pars).
    These functions will be used for integration: x'(t) = f(t,x(t),u(t))
    """

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

        self.debug_mode_on =False

    @abstractmethod
    def f(self,t,x,u,pars,variables_from_optimizer=None):
        """
        Function to be integrated
        
        :param t: time
        :param x: state
        :param u: input
        :param pars: optional parameters
        :param variables_from_optimizer: variables that can be passed from the optimizer
        :return: the function value, should return a list (to support easy concatenations of states)
        """

        pass

    def u(self,t,pars,variables_from_optimizer=None):
        """
        External input
        
        :param t: time
        :param pars: parameters
        :param variables_from_optimizer: variables that can be passed from the optimizer
        :return: the external input
        """

        return []


class AdvectMap(ForwardModel):
    """
    Forward model to advect an n-D map using a transport equation: :math:`\\Phi_t + D\\Phi v = 0`.
    v is treated as an external argument and \Phi is the state
    """

    def __init__(self, sz, spacing, params=None,compute_inverse_map=False):
        super(AdvectMap,self).__init__(sz,spacing,params)
        self.compute_inverse_map = compute_inverse_map
        """If True then computes the inverse map on the fly for a map-based solution"""

    def u(self,t, pars, variables_from_optimizer=None):
        """
        External input, to hold the velocity field
        
        :param t: time (ignored; not time-dependent) 
        :param pars: assumes an n-D velocity field is passed as the only input argument
        :param variables_from_optimizer: variables that can be passed from the optimizer
        :return: Simply returns this velocity field
        """

        return pars['v']

    def f(self,t, x, u, pars=None, variables_from_optimizer=None):
        """
        Function to be integrated, i.e., right hand side of transport equation: 
        
        :math:`-D\\phi v`
        
        :param t: time (ignored; not time-dependent) 
        :param x: state, here the map, \Phi, itself (assumes 3D-5D array; [nrI,0,:,:] x-coors; [nrI,1,:,:] y-coors; ...
        :param u: external input, will be the velocity field here
        :param pars: ignored (does not expect any additional inputs)
        :param variables_from_optimizer: variables that can be passed from the optimizer
        :return: right hand side [phi]
        """

        if self.compute_inverse_map:
            return [self.rhs.rhs_advect_map_multiNC(x[0], u),self.rhs.rhs_lagrangian_evolve_map_multiNC(x[1], u)]
        else:
            return [self.rhs.rhs_advect_map_multiNC(x[0],u)]

class AdvectImage(ForwardModel):
    """
    Forward model to advect an image using a transport equation: :math:`I_t + \\nabla I^Tv = 0`.
    v is treated as an external argument and I is the state
    """

    def __init__(self, sz, spacing, params=None):
        super(AdvectImage, self).__init__(sz, spacing,params)


    def u(self,t, pars, variables_from_optimizer=None):
        """
        External input, to hold the velocity field
        
        :param t: time (ignored; not time-dependent) 
        :param pars: assumes an n-D velocity field is passed as the only input argument
        :param variables_from_optimizer: variables that can be passed from the optimizer
        :return: Simply returns this velocity field
        """

        return pars['v']

    def f(self,t, x, u, pars=None, variables_from_optimizer=None):
        """
        Function to be integrated, i.e., right hand side of transport equation: :math:`-\\nabla I^T v`
        
        :param t: time (ignored; not time-dependent) 
        :param x: state, here the image, I, itself (supports multiple images and channels)
        :param u: external input, will be the velocity field here
        :param pars: ignored (does not expect any additional inputs)
        :param variables_from_optimizer: variables that can be passed from the optimizer
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
    def __init__(self, sz, spacing, smoother, params=None):
        super(EPDiffImage, self).__init__(sz, spacing,params)
        self.smoother = smoother

    def f(self,t, x, u, pars=None, variables_from_optimizer=None):
        """
        Function to be integrated, i.e., right hand side of the EPDiff equation: 
        :math:`-(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`
        
        :math:`-\\nabla I^Tv`
        
        :param t: time (ignored; not time-dependent) 
        :param x: state, here the vector momentum, m, and the image, I
        :param u: ignored, no external input
        :param pars: ignored (does not expect any additional inputs)
        :param variables_from_optimizer: variables that can be passed from the optimizer
        :return: right hand side [m,I]
        """

        # assume x[0] is m and x[1] is I for the state
        m = x[0]
        I = x[1]
        v = self.smoother.smooth(m,None,utils.combine_dict(pars,{'I': I}),variables_from_optimizer)
        # print('max(|v|) = ' + str( v.abs().max() ))
        return [self.rhs.rhs_epdiff_multiNC(m,v), self.rhs.rhs_advect_image_multiNC(I,v)]


class EPDiffMap(ForwardModel):
    """
    Forward model for the EPDiff equation. State is the momentum, m, and the transform, :math:`\\phi` 
    (mapping the source image to the target image).

    :math:`(m_1,...,m_d)^T_t = -(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`
    
    :math:`v=Km`
    
    :math:`\\phi_t+D\\phi v=0`
    """

    def __init__(self, sz, spacing, smoother, params=None,compute_inverse_map=False):
        super(EPDiffMap, self).__init__(sz,spacing,params)
        self.compute_inverse_map = compute_inverse_map
        """If True then computes the inverse map on the fly for a map-based solution"""

        self.smoother = smoother
        self.use_net = True if self.params['smoother']['type'] == 'adaptiveNet' else False

    def debugging(self,input,t):
        x = utils.checkNan(input)
        if np.sum(x):
            print("find nan at {} step".format(t))
            print("flag m: {}, ".format(x[0]))
            print("flag v: {},".format(x[1]))
            print("flag phi: {},".format(x[2]))
            print("flag new_m: {},".format(x[3]))
            print("flag new_phi: {},".format(x[4]))
            raise ValueError("nan error")

    def f(self,t, x, u, pars=None, variables_from_optimizer=None):
        """
        Function to be integrated, i.e., right hand side of the EPDiff equation:
        :math:`-(div(m_1v),...,div(m_dv))^T-(Dv)^Tm'
        
        :math:`-D\\phi v`
        
        :param t: time (ignored; not time-dependent) 
        :param x: state, here the image, vector momentum, m, and the map, :math:`\\phi`
        :param u: ignored, no external input
        :param pars: ignored (does not expect any additional inputs)
        :param variables_from_optimizer: variables that can be passed from the optimizer
        :return: right hand side [m,phi]
        """

        # assume x[0] is m and x[1] is phi for the state
        m = x[0]
        m = m.clamp(max=1., min=-1.)
        phi = x[1]

        if self.compute_inverse_map:
            phi_inv = x[2]

        if not self.use_net:
            v = self.smoother.smooth(m,None,utils.combine_dict(pars,{'phi':phi}),variables_from_optimizer)
        else:
            v = self.smoother.adaptive_smooth(m, phi, using_map=True)

        # print('max(|v|) = ' + str( v.abs().max() ))

        if self.compute_inverse_map:
            ret_val= [self.rhs.rhs_epdiff_multiNC(m,v),
                      self.rhs.rhs_advect_map_multiNC(phi,v),
                      self.rhs.rhs_lagrangian_evolve_map_multiNC(phi_inv,v)]
        else:
            new_m = self.rhs.rhs_epdiff_multiNC(m,v)
            new_phi = self.rhs.rhs_advect_map_multiNC(phi,v)
            ret_val= [new_m, new_phi]
        return ret_val



class EPDiffAdaptMap(ForwardModel):
    """
    Forward model for the EPDiff equation. State is the momentum, m, and the transform, :math:`\\phi`
    (mapping the source image to the target image).

    :math:`(m_1,...,m_d)^T_t = -(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`

    :math:`v=Km`

    :math:`\\phi_t+D\\phi v=0`
    """

    def __init__(self, sz, spacing, smoother, params=None, compute_inverse_map=False, update_sm_by_advect= True, update_sm_with_interpolation=True,compute_on_initial_map=True):
        super(EPDiffAdaptMap, self).__init__(sz, spacing, params)
        from . import module_parameters as pars
        from . import smoother_factory as sf
        self.compute_inverse_map = compute_inverse_map
        """If True then computes the inverse map on the fly for a map-based solution"""

        self.smoother = smoother
        self.update_sm_by_advect = update_sm_by_advect
        self.use_the_first_step_penalty = True
        self.update_sm_with_interpolation = update_sm_with_interpolation
        self.compute_on_initial_map=compute_on_initial_map
        self.update_sm_weight=None
        self.velocity_mask = None
        self.debug_mode_on = False
        s_m_params = pars.ParameterDict()
        s_m_params['smoother']['type'] = 'gaussian'
        s_m_params['smoother']['gaussian_std'] =self.params['smoother']['deep_smoother']['deep_network_local_weight_smoothing']
        self.embedded_smoother  = sf.SmootherFactory(sz[2:], spacing).create_smoother(
            s_m_params)

        """ if only take the first step penalty as the total penalty, otherwise accumluate the penalty"""
    def debug_nan(self, input, t,name=''):
        x = utils.checkNan([input])
        if np.sum(x):
            # print(input[0])
            print("find nan at {} step, {} with number {}".format(t,name,x[0]))

            raise ValueError("nan error")
    def init_zero_sm_weight(self,sm_weight):
        self.update_sm_weight = torch.zeros_like(sm_weight).detach()


    def init_velocity_mask(self,velocity_mask):
        self.velocity_mask = velocity_mask


    def debug_distrib(self,var,name):
        var = var.detach().cpu().numpy()
        density,_= np.histogram(var,[-100,-10,-1,0,1,10,100],density=True)
        print("{} distri:{}".format(name,density))


    def f(self, t, x, u, pars=None, variables_from_optimizer=None):
        """
        Function to be integrated, i.e., right hand side of the EPDiff equation:
        :math:`-(div(m_1v),...,div(m_dv))^T-(Dv)^Tm'

        :math:`-D\\phi v`

        :param t: time (ignored; not time-dependent)
        :param x: state, here the image, vector momentum, m, and the map, :math:`\\phi`
        :param u: ignored, no external input
        :param pars: ignored (does not expect any additional inputs)
        :param variables_from_optimizer: variables that can be passed from the optimizer
        :return: right hand side [m,phi]
        """

        # assume x[0] is m and x[1] is phi for the state
        m = x[0]
        m=m.clamp(max=1., min=-1.)
        phi = x[1]
        return_val_name = []
        sm_weight = None
        if self.update_sm_by_advect:
            if not self.update_sm_with_interpolation:
                sm_weight_pre = x[2]
                sm_weight = self.embedded_smoother.smooth(sm_weight_pre)

                v, extra_ret = self.smoother.smooth(m, None, {'w':sm_weight},multi_output=True)
                if self.velocity_mask is not None:
                    v = v* self.velocity_mask
                new_phi = self.rhs.rhs_advect_map_multiNC(phi, v)
                new_sm_weight_pre =  self.rhs.rhs_advect_map_multiNC(sm_weight_pre, v)
                new_m = self.rhs.rhs_adapt_epdiff_wkw_multiNC(m, v, new_sm_weight_pre, extra_ret,
                                                                  self.embedded_smoother)

                ret_val = [new_m, new_phi,new_sm_weight_pre]
                return_val_name  =['new_m','new_phi','new_sm_weight']
            else:
                if self.compute_on_initial_map:
                    sm_weight = x[2]
                    sm_phi = x[3]
                    new_sm_weight = utils.compute_warped_image_multiNC(sm_weight, sm_phi, self.spacing, 1,
                                                                    zero_boundary=False)
                    pre_weight = sm_weight
                    new_sm_weight = self.embedded_smoother.smooth(new_sm_weight)
                    #print('t{},m min, mean,max {} {} {}'.format(t,m.min().item(),m.mean().item(),m.max().item()))
                    v,extra_ret = self.smoother.smooth(m,None,{'w': new_sm_weight},multi_output=True)
                    if self.velocity_mask is not None:
                        v = v * self.velocity_mask

                    new_m = self.rhs.rhs_adapt_epdiff_wkw_multiNC(m,v,pre_weight,extra_ret,self.embedded_smoother)
                    new_phi = self.rhs.rhs_advect_map_multiNC(phi, v)
                    new_sm_phi = self.rhs.rhs_advect_map_multiNC(sm_phi, v)
                    new_sm_weight = self.update_sm_weight.detach()
                    ret_val = [new_m, new_phi,new_sm_weight,new_sm_phi]
                    return_val_name = ['new_m', 'new_phi', 'new_sm_weight','new_sm_phi']
                else: #todo  just attention here is what we currently used
                    sm_weight = x[2]
                    new_sm_weight = utils.compute_warped_image_multiNC(sm_weight, phi, self.spacing, 1,
                                                                       zero_boundary=False)

                    pre_weight = sm_weight
                    new_sm_weight = self.embedded_smoother.smooth(new_sm_weight)

                    v, extra_ret = self.smoother.smooth(m, None,{'w':new_sm_weight}, multi_output=True)

                    if self.velocity_mask is not None:
                        v = v * self.velocity_mask

                    new_m = self.rhs.rhs_adapt_epdiff_wkw_multiNC(m,v,pre_weight,extra_ret,self.embedded_smoother)
                    new_phi = self.rhs.rhs_advect_map_multiNC(phi, v)
                    new_sm_weight = self.update_sm_weight.detach()
                    ret_val = [new_m, new_phi, new_sm_weight]
                    return_val_name = ['new_m', 'new_phi', 'new_sm_weight']

        else:
            if not t==0:
                if self.use_the_first_step_penalty:
                    self.smoother.disable_penalty_computation()
                else:
                    self.smoother.enable_accumulated_penalty()

            I = utils.compute_warped_image_multiNC(pars['I0'], phi, self.spacing, 1,zero_boundary=True)
            pars['I'] = I.detach()  # TODO  check whether I should be detached here
            v = self.smoother.smooth(m, None, pars, variables_from_optimizer)
            if self.velocity_mask is not None:
                v = v * self.velocity_mask
            new_m = self.rhs.rhs_epdiff_multiNC(m, v)
            new_phi = self.rhs.rhs_advect_map_multiNC(phi, v)
            ret_val = [new_m, new_phi]
            return_val_name =['new_m','new_phi']


        if self.debug_mode_on:
            toshows = [m, v,phi]+ret_val if sm_weight is None else  [m, v,phi]+ret_val +[sm_weight]
            name = ['m', 'v','phi']+return_val_name if sm_weight is None else ['m', 'v','phi']+return_val_name +['sm_weight']
            for i, toshow in enumerate(toshows):
                print('t{},{} min, mean,max {} {} {}'.format(t, name[i], toshow.min().item(), toshow.mean().item(),
                                                             toshow.max().item()))
                self.debug_distrib(toshow, name[i])
                self.debug_nan(toshow,t,name[i])
        return ret_val



        # print('max(|v|) = ' + str( v.abs().max() ))



class EPDiffScalarMomentum(ForwardModel):
    """
    Base class for scalar momentum EPDiff solutions. Defines a smoother that can be commonly used.
    """

    def __init__(self, sz, spacing, smoother, params):
        super(EPDiffScalarMomentum,self).__init__(sz,spacing,params)

        self.smoother = smoother


class EPDiffScalarMomentumImage(EPDiffScalarMomentum):
    """
    Forward model for the scalar momentum EPdiff equation. State is the scalar momentum, lam, and the image I
    :math:`(m_1,...,m_d)^T_t = -(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`

    :math:`v=Km`

    :math:'m=\\lambda\\nabla I`

    :math:`I_t+\\nabla I^Tv=0`

    :math:`\\lambda_t + div(\\lambda v)=0`
    """

    def __init__(self, sz, spacing, smoother, params=None):
        super(EPDiffScalarMomentumImage, self).__init__(sz, spacing, smoother, params)

    def f(self, t, x, u, pars=None, variables_from_optimizer=None):
        """
        Function to be integrated, i.e., right hand side of the EPDiff equation:

        :math:`-(div(m_1v),...,div(m_dv))^T-(Dv)^Tm`

        :math:`-\\nabla I^Tv`

        :math: `-div(\\lambda v)`

        :param t: time (ignored; not time-dependent) 
        :param x: state, here the scalar momentum, lam, and the image, I, itself
        :param u: no external input
        :param pars: ignored (does not expect any additional inputs)
        :param variables_from_optimizer: variables that can be passed from the optimizer
        :return: right hand side [lam,I]
        """

        # assume x[0] is \lambda and x[1] is I for the state
        lam = x[0]
        I = x[1]

        # now compute the momentum
        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(lam, I, self.sz, self.spacing)
        v = self.smoother.smooth(m,None,utils.combine_dict(pars,{'I':I}),variables_from_optimizer)

        # advection for I, scalar-conservation law for lam
        return [self.rhs.rhs_scalar_conservation_multiNC(lam, v), self.rhs.rhs_advect_image_multiNC(I, v)]



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

    def __init__(self, sz, spacing, smoother, params=None, compute_inverse_map=False):
        super(EPDiffScalarMomentumMap, self).__init__(sz,spacing, smoother, params)
        self.compute_inverse_map = compute_inverse_map
        """If True then computes the inverse map on the fly for a map-based solution"""

    def f(self,t, x, u, pars=None, variables_from_optimizer=None):
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
        :param variables_from_optimizer: variables that can be passed from the optimizer
        :return: right hand side [lam,I,phi]
        """

        # assume x[0] is lam and x[1] is I and x[2] is phi for the state
        lam = x[0]
        I = x[1]
        phi = x[2]

        if self.compute_inverse_map:
            phi_inv = x[3]

        # now compute the momentum
        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(lam, I, self.sz, self.spacing)
        # todo: replace this by phi again
        #v = self.smoother.smooth(m,None,[phi,True],variables_from_optimizer)
        v = self.smoother.smooth(m,None,utils.combine_dict(pars,{'I':I}),variables_from_optimizer)

        if self.compute_inverse_map:
            ret_val = [self.rhs.rhs_scalar_conservation_multiNC(lam,v),
                self.rhs.rhs_advect_image_multiNC(I,v),
                self.rhs.rhs_advect_map_multiNC(phi,v),
                self.rhs.rhs_lagrangian_evolve_map_multiNC(phi_inv,v)]
        else:
            ret_val = [self.rhs.rhs_scalar_conservation_multiNC(lam,v),
                self.rhs.rhs_advect_image_multiNC(I,v),
                self.rhs.rhs_advect_map_multiNC(phi,v)]

        return ret_val
