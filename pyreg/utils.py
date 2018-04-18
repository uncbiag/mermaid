"""
Various utility functions.

.. todo::
    Reorganize this package in a more meaningful way.
"""
from __future__ import print_function
from __future__ import absolute_import
# TODO

from builtins import str
from builtins import range
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from .libraries.modules.stn_nd import STN_ND_BCXYZ
from .data_wrapper import AdaptVal
from .data_wrapper import MyTensor
import numpy as np
from . import finite_differences as fd
import torch.nn as nn
import torch.nn.init as init

import pyreg.module_parameters as pars

from pyreg.spline_interpolation import SplineInterpolation_ND_BCXYZ

import os

try:
    from .libraries.functions.nn_interpolation import get_nn_interpolation
except ImportError:
    print('WARNING: nn_interpolation could not be imported (only supported in CUDA at the moment), some functionality may not be available.')


def create_symlink_with_correct_ext(sf,tf):
    abs_s = os.path.abspath(sf)
    ext_s = os.path.splitext(abs_s)[1]

    abs_t = os.path.abspath(tf)
    root_t,ext_t = os.path.splitext(abs_t)

    abs_t_with_right_ext = root_t + ext_s

    if os.path.isfile(abs_t_with_right_ext):
        if os.path.samefile(abs_s,abs_t_with_right_ext):
            # nothing to do here, these are already the same file
            return
        else:
            os.remove(abs_t_with_right_ext)

    # now we can do the symlink
    os.symlink(abs_s,abs_t_with_right_ext)

    
def combine_dict(d1,d2):
    """
    Creates a dictionary which has entries from both of them

    :param d1: dictionary 1
    :param d2: dictionary 2
    :return: resulting dictionary
    """
    d = d1.copy()
    d.update(d2)
    return d

def get_parameter_list_from_parameter_dict(pd):
    """
    Takes a dictionary which contains key value pairs for model parameters and converts it into a list of parameters that can
    be used as an input to an optimizer.

    :param pd: parameter dictionary
    :return: list of parameters
    """
    pl = []
    for key in pd:
        pl.append(pd[key])
    return pl

def get_parameter_list_and_par_to_name_dict_from_parameter_dict(pd):
    """
    Same as get_parameter_list_from_parameter_dict; but also returns a dictionary which keeps track of the keys based on memory id

    :param pd: parameter dictionary
    :return: tuple of (parameter_list,name_dictionary)
    """

    par_to_name_dict = dict()
    pl = []
    for key in pd:
        pl.append(pd[key])
        par_to_name_dict[pd[key]] = key
    return pl,par_to_name_dict

def remove_infs_from_variable(v):
    # 32 - bit floating point: torch.FloatTensor, torch.cuda.FloatTensor
    # 64 - bit floating point: torch.DoubleTensor, torch.cuda.DoubleTensor
    # 16 - bit floating point: torch.HalfTensor, torch.cuda.HalfTensor

    # todo: maybe find a cleaner way of handling this
    # this is to make sure that subsequent sums work (hence will be smaller than it could be,
    # but values of this size should not occur in practice anyway
    sz = v.size()
    reduction_factor = np.prod(np.array(sz))

    if type(v.data)==torch.FloatTensor or type(v.data)==torch.cuda.FloatTensor:
        return torch.clamp(v,
                           min=(np.asscalar(np.finfo('float32').min))/reduction_factor,
                           max=(np.asscalar(np.finfo('float32').max))/reduction_factor)
    elif type(v.data)==torch.DoubleTensor or type(v.data)==torch.cuda.DoubleTensor:
        return torch.clamp(v,
                           min=(np.asscalar(np.finfo('float64').min))/reduction_factor,
                           max=(np.asscalar(np.finfo('float64').max))/reduction_factor)
    elif type(v.data)==torch.HalfTensor or type(v.data)==torch.cuda.HalfTensor:
        return torch.clamp(v,
                           min=(np.asscalar(np.finfo('float16').min))/reduction_factor,
                           max=(np.asscalar(np.finfo('float16').max))/reduction_factor)
    else:
        raise ValueError('Unknown data type: ' + str( type(v.data)))


def lift_to_dimension(A,dim):
    """
    Creates a view of A of dimension dim (by adding dummy dimensions if necessary).
    Assumes a numpy array as input

    :param A: numpy array
    :param dim: desired dimension of view
    :return: returns view of A of appropriate dimension
    """

    current_dim = len(A.shape)
    if current_dim>dim:
        raise ValueError('Can only add dimensions, but not remove them')

    if current_dim==dim:
        return A
    else:
        return A.reshape([1]*(dim-current_dim)+list(A.shape))

def get_dim_of_affine_transform(Ab):
    """
    Returns the number of dimensions corresponding to an affine transformation  of the form y=Ax+b 
    stored in a column vector. For A =[a1,a2,a3] the parameter vector is simply [a1;a2;a3;b], i.e.,
    all columns stacked on top of each other
    :param Ab: parameter vector
    :return: dimensionality of transform (1,2,or 3)
    """
    nr = len(Ab)
    if nr==2:
        return 1
    elif nr==6:
        return 2
    elif nr==12:
        return 3
    else:
        raise ValueError('Only supports dimensions 1, 2, and 3.')

def set_affine_transform_to_identity(Ab):
    """
    Sets the affine transformation as given by the column vector Ab to the identity transform.
    :param Ab: Affine parameter vector (will be overwritten with the identity transform)
    :return: n/a
    """
    dim = get_dim_of_affine_transform(Ab)

    if dim==1:
        Ab.zero_()
        Ab[0]=1.
    elif dim==2:
        Ab.zero_()
        Ab[0]=1.
        Ab[3]=1.
    elif dim==3:
        Ab.zero_()
        Ab[0]=1.
        Ab[4]=1.
        Ab[8]=1.
    else:
        raise ValueError('Only supports dimensions 1, 2, and 3.')

def set_affine_transform_to_identity_multiN(Ab):
    """
    Set the affine transforms to the identity (in the case of arbitrary batch size)
    :param Ab: Parameter vectors Bxpars (batch size x parameter vector); will be overwritten with identity transforms
    :return: n/a
    """
    sz = Ab.size()
    nrOfImages = sz[0]
    for nrI in range(nrOfImages):
        set_affine_transform_to_identity(Ab[nrI,:])


def apply_affine_transform_to_map(Ab,phi):
    """
    Applies an affine transform to a map
    :param Ab: affine transform parameter column vector
    :param phi: map; format nrCxXxYxZ (nrC corresponds to dimension)
    :return: returns transformed map
    """
    sz = phi.size()
    nrOfChannels = sz[0]

    dim = len(sz) - 1
    if dim not in [1,2,3]:
        raise ValueError('Only supports dimensions 1, 2, and 3.')

    phiR = Variable(MyTensor(sz).zero_(), requires_grad=False).type_as(phi)

    if dim == 1:
        phiR = phi * Ab[0] + Ab[1]
    elif dim == 2:
        phiR[0, ...] = Ab[0] * phi[0, ...] + Ab[2] * phi[1, ...] + Ab[4]  # a_11x+a_21y+b1
        phiR[1, ...] = Ab[1] * phi[0, ...] + Ab[3] * phi[1, ...] + Ab[5]  # a_12x+a_22y+b2
    elif dim == 3:
        phiR[0, ...] = Ab[0] * phi[0, ...] + Ab[3] * phi[1, ...] + Ab[6] * phi[2, ...] + Ab[9]
        phiR[1, ...] = Ab[1] * phi[0, ...] + Ab[4] * phi[1, ...] + Ab[7] * phi[2, ...] + Ab[10]
        phiR[2, ...] = Ab[2] * phi[0, ...] + Ab[5] * phi[1, ...] + Ab[8] * phi[2, ...] + Ab[11]
    else:
        raise ValueError('Only supports dimensions 1, 2, and 3.')

    return phiR

def apply_affine_transform_to_map_multiNC(Ab,phi):
    """
    Applies an affine transform to maps (for arbitrary batch size)
    :param Ab: affine transform parameter column vectors (batchSize x parameter vector)
    :param phi: maps; format batchxnrCxXxYxZ (nrC corresponds to dimension)
    :return: returns transformed maps
    """
    sz = phi.size()
    dim = get_dim_of_affine_transform(Ab[0,:])
    nrOfImages = Ab.size()[0]
    if nrOfImages != sz[0]:
        raise ValueError('Incompatible number of affine transforms')
    if dim != len(sz)-2:
        raise ValueError('Incompatible number of affine transforms')

    phiR = Variable(MyTensor(sz).zero_(), requires_grad=False).type_as(phi)
    for nrI in range(nrOfImages):
        phiR[nrI,...] = apply_affine_transform_to_map(Ab[nrI,:],phi[nrI,...])

    return phiR

def compute_normalized_gaussian(X, mu, sig):
    """
    Computes a normalized Gaussian
    
    :param X: map with coordinates at which to evaluate 
    :param mu: array indicating the mean
    :param sig: array indicating the standard deviations for the different dimensions
    :return: normalized Gaussian
    """
    dim = len(mu)
    if dim==1:
        g = np.exp(-np.power(X[0,:]-mu[0],2.)/(2*np.power(sig[0],2.)))
        g = g/g.sum()
        return g
    elif dim==2:
        g = np.exp(-np.power(X[0,:,:]-mu[0],2.)/(2*np.power(sig[0],2.))
                   - np.power(X[1,:, :] - mu[1], 2.) / (2 * np.power(sig[1], 2.)))
        g = g/g.sum()
        return g
    elif dim==3:
        g = np.exp(-np.power(X[0,:, :, :] - mu[0], 2.) / (2 * np.power(sig[0], 2.))
                   -np.power(X[1,:, :, :] - mu[1], 2.) / (2 * np.power(sig[1], 2.))
                   -np.power(X[2,:, :, :] - mu[2], 2.) / (2 * np.power(sig[2], 2.)))
        g = g / g.sum()
        return g
    else:
        raise ValueError('Can only compute Gaussians in dimensions 1-3')


def _compute_warped_image_multiNC_1d(I0, phi, spacing, spline_order):

    if spline_order not in [1,2,3,4,5,6,7,8,9]:
        raise ValueError('Currently only orders 1 to 9 are supported')

    if spline_order==1:
        stn = STN_ND_BCXYZ(spacing)
    else:
        stn = SplineInterpolation_ND_BCXYZ(spacing, spline_order)

    I1_warped = stn(I0, phi)

    return I1_warped

def _compute_warped_image_multiNC_2d(I0, phi, spacing, spline_order):

    if spline_order not in [1,2,3,4,5,6,7,8,9]:
        raise ValueError('Currently only orders 1 to 9 are supported')

    if spline_order==1:
        stn = STN_ND_BCXYZ(spacing)
    else:
        stn = SplineInterpolation_ND_BCXYZ(spacing, spline_order)

    I1_warped = stn(I0, phi)

    return I1_warped

def _compute_warped_image_multiNC_3d(I0, phi, spacing, spline_order):

    if spline_order not in [1,2,3,4,5,6,7,8,9]:
        raise ValueError('Currently only orders 1 to 9 are supported')

    if spline_order==1:
        stn = STN_ND_BCXYZ(spacing)
    else:
        stn = SplineInterpolation_ND_BCXYZ(spacing,spline_order)

    I1_warped = stn(I0, phi)

    return I1_warped


def compute_warped_image(I0,phi,spacing,spline_order):
    """
    Warps image.

    :param I0: image to warp, image size XxYxZ
    :param phi: map for the warping, size dimxXxYxZ
    :param spacing: image spacing [dx,dy,dz]
    :return: returns the warped image of size XxYxZ
    """

    # implements this by creating a different view (effectively adding dimensions)
    Iw = compute_warped_image_multiNC(I0.view(torch.Size([1, 1]+ list(I0.size()))),
                                        phi.view(torch.Size([1]+ list(phi.size()))),spacing,spline_order)
    return Iw.view(I0.size())

def compute_warped_image_multiNC(I0, phi, spacing, spline_order):
    """
    Warps image.
    
    :param I0: image to warp, image size BxCxXxYxZ
    :param phi: map for the warping, size BxdimxXxYxZ
    :param spacing: image spacing [dx,dy,dz]
    :return: returns the warped image of size BxCxXxYxZ
    """

    dim = I0.dim()-2
    if dim == 1:
        return _compute_warped_image_multiNC_1d(I0, phi, spacing, spline_order)
    elif dim == 2:
        return _compute_warped_image_multiNC_2d(I0, phi, spacing, spline_order)
    elif dim == 3:
        return _compute_warped_image_multiNC_3d(I0, phi, spacing, spline_order)
    else:
        raise ValueError('Images can only be warped in dimensions 1 to 3')


def compute_vector_momentum_from_scalar_momentum_multiNC(lam, I, sz, spacing):
    """
    Computes the vector momentum from the scalar momentum: :math:`m=\\lambda\\nabla I`
    
    :param lam: scalar momentum, BxCxXxYxZ
    :param I: image, BxCxXxYxZ
    :param sz: size of image
    :param spacing: spacing of image
    :return: returns the vector momentum
    """
    nrOfI = sz[0] # number of images
    m = create_ND_vector_field_variable_multiN(sz[2::], nrOfI)  # attention that the second dimension here is image dim, not nrOfC
    nrOfC = sz[1]
    for c in range(nrOfC): # loop over all the channels and add the results
        m = m + compute_vector_momentum_from_scalar_momentum_multiN(lam[:,c, ...], I[:,c, ...], nrOfI, sz[2::], spacing)
    return m


def compute_vector_momentum_from_scalar_momentum_multiN(lam, I, nrOfI, sz, spacing):
    """
    Computes the vector momentum from the scalar momentum: :math:`m=\\lambda\\nabla I`

    :param lam: scalar momentum, batchxXxYxZ
    :param I: image, batchXxYxZ
    :param sz: size of image
    :param spacing: spacing of image
    :return: returns the vector momentum
    """
    fdt = fd.FD_torch(spacing)
    dim = len(sz)
    m = create_ND_vector_field_variable_multiN(sz, nrOfI)
    if dim==1:
        m[:,0,:] = fdt.dXc(I)*lam
    elif dim==2:
        m[:,0,:,:] = fdt.dXc(I)*lam
        m[:,1,:,:] = fdt.dYc(I)*lam
    elif dim==3:
        m[:,0,:,:,:] = fdt.dXc(I)*lam
        m[:,1,:,:,:] = fdt.dYc(I)*lam
        m[:,2,:,:,:] = fdt.dZc(I)*lam
    else:
        raise ValueError('Can only convert scalar to vector momentum in dimensions 1-3')
    return m

def create_ND_vector_field_variable_multiN(sz, nrOfI=1):
    """
    Create vector field torch Variable of given size
    
    :param sz: just the spatial sizes (e.g., [5] in 1D, [5,10] in 2D, [5,10,10] in 3D)
    :param nrOfI: number of images
    :return: returns vector field of size nrOfIxdimxXxYxZ
    """
    dim = len(sz)
    csz = np.array(sz) # just to make sure it is a numpy array
    csz = np.array([nrOfI,dim]+list(csz))
    return Variable(MyTensor(*(csz.tolist())).zero_(), requires_grad=False)

def create_ND_vector_field_variable(sz):
    """
    Create vector field torch Variable of given size
    
    :param sz: just the spatial sizes (e.g., [5] in 1D, [5,10] in 2D, [5,10,10] in 3D)
    :return: returns vector field of size dimxXxYxZ
    """
    dim = len(sz)
    csz = np.array(sz) # just to make sure it is a numpy array
    csz = np.array([dim]+list(csz))
    return Variable(MyTensor(*(csz.tolist())).zero_(), requires_grad=False)

def create_vector_parameter(nr_of_elements):
    """
    Creates a vector parameters with a specified number of elements
    :param nr_of_elements: number of vector elements
    :return: returns the parameter vector
    """
    return Parameter(MyTensor(nr_of_elements).zero_())

def create_ND_vector_field_parameter_multiN(sz, nrOfI=1):
    """
    Create vector field torch Parameter of given size

    :param sz: just the spatial sizes (e.g., [5] in 1D, [5,10] in 2D, [5,10,10] in 3D)
    :param nrOfI: number of images
    :return: returns vector field of size nrOfIxdimxXxYxZ
    """
    dim = len(sz)
    csz = np.array(sz) # just to make sure it is a numpy array
    csz = np.array([nrOfI,dim]+list(csz))
    return Parameter(MyTensor(*(csz.tolist())).zero_())

def create_ND_scalar_field_parameter_multiNC(sz, nrOfI=1, nrOfC=1):
    """
    Create vector field torch Parameter of given size

    :param sz: just the spatial sizes (e.g., [5] in 1D, [5,10] in 2D, [5,10,10] in 3D)
    :param nrOfI: number of images
    :param nrOfC: number of channels
    :return: returns vector field of size nrOfIxnrOfCxXxYxZ
    """

    csz = np.array(sz) # just to make sure it is a numpy array
    csz = np.array([nrOfI,nrOfC]+list(csz))
    return Parameter(MyTensor(*(csz.tolist())).zero_())

def centered_identity_map_multiN(sz, spacing, dtype='float32'):
    """
    Create a centered identity map (shifted so it is centered around 0)

    :param sz: size of an image in BxCxXxYxZ format
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map
    """
    dim = len(sz) - 2
    nrOfI = sz[0]

    if dim == 1:
        id = np.zeros([nrOfI, 1, sz[2]], dtype=dtype)
    elif dim == 2:
        id = np.zeros([nrOfI, 2, sz[2], sz[3]], dtype=dtype)
    elif dim == 3:
        id = np.zeros([nrOfI, 3, sz[2], sz[3], sz[4]], dtype=dtype)
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')

    for n in range(nrOfI):
        id[n, ...] = centered_identity_map(sz[2::], spacing,dtype=dtype)

    return id


def identity_map_multiN(sz,spacing,dtype='float32'):
    """
    Create an identity map
    
    :param sz: size of an image in BxCxXxYxZ format
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map
    """
    dim = len(sz)-2
    nrOfI = sz[0]

    if dim == 1:
        id = np.zeros([nrOfI,1,sz[2]],dtype=dtype)
    elif dim == 2:
        id = np.zeros([nrOfI,2,sz[2],sz[3]],dtype=dtype)
    elif dim == 3:
        id = np.zeros([nrOfI,3,sz[2],sz[3],sz[4]],dtype=dtype)
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')

    for n in range(nrOfI):
        id[n,...] = identity_map(sz[2::],spacing,dtype=dtype)

    return id


def centered_identity_map(sz, spacing, dtype='float32'):
    """
    Returns a centered identity map (with 0 in the middle) if the sz is odd
    Otherwise shifts everything by 0.5*spacing

    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0:sz[0]]
    elif dim == 2:
        id = np.mgrid[0:sz[0], 0:sz[1]]
    elif dim == 3:
        id = np.mgrid[0:sz[0], 0:sz[1], 0:sz[2]]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')

    # now get it into range [0,(sz-1)*spacing]^d
    id = np.array(id.astype(dtype))
    if dim == 1:
        id = id.reshape(1, sz[0])  # add a dummy first index

    for d in range(dim):
        id[d] *= spacing[d]
        if sz[d]%2==0:
            #even
            id[d] -= spacing[d]*(sz[d]//2)
        else:
            #odd
            id[d] -= spacing[d]*((sz[d]+1)//2)

    # and now store it in a dim+1 array
    if dim == 1:
        idnp = np.zeros([1, sz[0]], dtype=dtype)
        idnp[0, :] = id[0]
    elif dim == 2:
        idnp = np.zeros([2, sz[0], sz[1]], dtype=dtype)
        idnp[0, :, :] = id[0]
        idnp[1, :, :] = id[1]
    elif dim == 3:
        idnp = np.zeros([3, sz[0], sz[1], sz[2]], dtype=dtype)
        idnp[0, :, :, :] = id[0]
        idnp[1, :, :, :] = id[1]
        idnp[2, :, :, :] = id[2]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the centered identity map')

    return idnp


def identity_map(sz,spacing,dtype='float32'):
    """
    Returns an identity map.
    
    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim==1:
        id = np.mgrid[0:sz[0]]
    elif dim==2:
        id = np.mgrid[0:sz[0],0:sz[1]]
    elif dim==3:
        id = np.mgrid[0:sz[0],0:sz[1],0:sz[2]]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')

    # now get it into range [0,(sz-1)*spacing]^d
    id = np.array( id.astype(dtype) )
    if dim==1:
        id = id.reshape(1,sz[0]) # add a dummy first index

    for d in range(dim):
        id[d]*=spacing[d]

        #id[d]*=2./(sz[d]-1)
        #id[d]-=1.

    # and now store it in a dim+1 array
    if dim==1:
        idnp = np.zeros([1, sz[0]], dtype=dtype)
        idnp[0,:] = id[0]
    elif dim==2:
        idnp = np.zeros([2, sz[0], sz[1]], dtype=dtype)
        idnp[0,:, :] = id[0]
        idnp[1,:, :] = id[1]
    elif dim==3:
        idnp = np.zeros([3,sz[0], sz[1], sz[2]], dtype=dtype)
        idnp[0,:, :, :] = id[0]
        idnp[1,:, :, :] = id[1]
        idnp[2,:, :, :] = id[2]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')

    return idnp



def get_warped_label_map(label_map, phi, spacing, sched='nn'):
    if sched == 'nn':
        warped_label_map = get_nn_interpolation(label_map, phi, spacing)
        # check if here should be add assert
        assert abs(torch.sum(warped_label_map.data -warped_label_map.data.round()))< 0.1, "nn interpolation is not precise"
    else:
        raise ValueError(" the label warpping method is not implemented")
    return warped_label_map


def t2np( v ):
    """
    Takes a torch array and returns it as a numpy array on the cpu
    
    :param v: torch array
    :return: numpy array
    """

    if type( v ) == torch.autograd.variable.Variable:
        return (v.data).cpu().numpy()
    else:
        return v.cpu().numpy()

def checkNan(x):
    """"
    input should be list of Variable
    """
    return [len(np.argwhere(np.isnan(elem.cpu().data.numpy()))) for elem in x]


##########################################  Adaptive Net ###################################################3
def space_normal(tensors, std=0.1):
    """
    space normalize for the net kernel
    :param tensor:
    :param mean:
    :param std:
    :return:
    """
    if isinstance(tensors, Variable):
        space_normal(tensors.data, std=std)
        return tensors
    for n in range(tensors.size()[0]):
        for c in range(tensors.size()[1]):
            dim = tensors[n][c].dim()
            sz = tensors[n][c].size()
            mus = np.zeros(dim)
            stds = std * np.ones(dim)
            print('WARNING: What should the spacing be here? Needed for new identity map code')
            raise ValueError('Double check the spacing here before running this code')
            spacing = np.ones(dim)
            centered_id = centered_identity_map(sz,spacing)
            g = compute_normalized_gaussian(centered_id, mus, stds)
            tensors[n,c] = torch.from_numpy(g)


def weights_init_uniform(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.038, 0.042)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        space_normal(m.weight.data)
    elif classname.find('Linear') != -1:
        space_normal(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_rd_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'rd_normal':
        net.apply(weights_init_rd_normal)
    elif init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'uniform':
        net.apply(weights_init_uniform)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def organize_data(moving, target, sched='depth_concat'):
    if sched == 'depth_concat':
        input = torch.cat([moving, target], dim=1)
    elif sched == 'width_concat':
        input = torch.cat((moving, target), dim=3)
    elif sched =='list_concat':
        input = torch.cat((moving.unsqueeze(0),target.unsqueeze(0)),dim=0)
    elif sched == 'difference':
        input = moving-target
    return input


def bh(m,gi,go):
    print("Grad Input")
    print((torch.sum(gi[0].data), torch.sum(gi[1].data)))
    print("Grad Output")
    print(torch.sum(go[0].data))
    return gi[0],gi[1], gi[2]



class ConvBnRel(nn.Module):
    # conv + bn (optional) + relu
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, active_unit='relu', same_padding=False,
                 bn=False, reverse=False, bias=False):
        super(ConvBnRel, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        if not reverse:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding,bias=bias)
        #y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
        #When affine=False the output of BatchNorm is equivalent to considering gamma=1 and beta=0 as constants.
        self.bn = nn.BatchNorm2d(out_channels, eps=0.0001, momentum=0, affine=True) if bn else None
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x

class FcRel(nn.Module):
    # fc+ relu(option)
    def __init__(self, in_features, out_features, active_unit='relu'):
        super(FcRel, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.fc(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x


class AdpSmoother(nn.Module):
    """
    a simple conv implementation, generate displacement field
    """
    def __init__(self, inputs, dim, net_sched=None):
        # settings should include [using_bias, using bn, using elu]
        # inputs should be a dictionary could contain ['s'],['t']
        super(AdpSmoother, self).__init__()
        self.dim = dim
        self.net_sched = 'm_only'
        self.s = inputs['s'].detach()
        self.t = inputs['t'].detach()
        self.mask = Parameter(torch.cat([torch.ones(inputs['s'].size())]*dim, 1), requires_grad = True)
        self.get_net_sched()
        #self.net.register_backward_hook(bh)

    def get_net_sched(self, debugging=True, using_bn=True, active_unit='relu', using_sigmoid=False , kernel_size=5):
        # return the self.net and self.net_input
        padding_size = (kernel_size-1)//2
        if self.net_sched == 'm_only':
            if debugging:
                self.net = nn.Conv2d(2, 2, kernel_size, 1, padding=padding_size, bias=False,groups=2)
            else:
                net = \
                    [ConvBnRel(self.dim, 20, 5, active_unit=active_unit, same_padding=True, bn=using_bn),
                     ConvBnRel(20,self.dim, 5, active_unit=active_unit, same_padding=True, bn=using_bn)]
                if using_sigmoid:
                    net += [nn.Sigmoid()]
                self.net = nn.Sequential(*net)

        elif self.net_sched =='m_f_s':
            if debugging:
                self.net = nn.Conv2d(self.dim+1, self.dim, kernel_size, 1, padding=padding_size, bias=False)
            else:
                net = \
                    [ConvBnRel(self.dim +1, 20, 5, active_unit=active_unit, same_padding=True, bn=using_bn),
                       ConvBnRel(20, self.dim, 5, active_unit=active_unit, same_padding=True, bn=using_bn)]
                if using_sigmoid:
                    net += [nn.Sigmoid()]
                self.net = nn.Sequential(*net)

        elif self.net_sched == 'm_d_s':
            if debugging:
                self.net = nn.Conv2d(self.dim+1, self.dim, kernel_size, 1, padding=padding_size, bias=False)
            else:
                net = \
                    [ConvBnRel(self.dim + 1, 20, 5, active_unit=active_unit, same_padding=True, bn=using_bn),
                       ConvBnRel(20, self.dim, 5, active_unit=active_unit, same_padding=True, bn=using_bn)]
                if using_sigmoid:
                    net += [nn.Sigmoid()]
                self.net = nn.Sequential(*net)

        elif self.net_sched == 'm_f_s_t':
            if debugging:
                self.net = nn.Conv2d(self.dim+2, self.dim, kernel_size, 1, padding=padding_size, bias=False)
            else:
                net = \
                    [ConvBnRel(self.dim + 2, 20, 5, active_unit=active_unit, same_padding=True, bn=using_bn),
                       ConvBnRel(20, self.dim, 5, active_unit=active_unit, same_padding=True, bn=using_bn)]
                if using_sigmoid:
                    net += [nn.Sigmoid()]
                self.net = nn.Sequential(*net)
        elif self.net_sched == 'm_d_s_f_t':
            if debugging:
                self.net = nn.Conv2d(self.dim + 2, self.dim, kernel_size, 1, padding=padding_size, bias=False)
            else:
                net = \
                    [ConvBnRel(self.dim + 2, 20, 5, active_unit=active_unit, same_padding=True, bn=using_bn),
                     ConvBnRel(20, self.dim, 5, active_unit=active_unit, same_padding=True, bn=using_bn)]
                if using_sigmoid:
                    net += [nn.Sigmoid()]
                self.net = nn.Sequential(*net)


    def prepare_data(self, m, new_s):
        input=None
        if self.net_sched == 'm_only':
            input = m
        elif self.net_sched == 'm_f_s':
            input = organize_data(m,self.s,sched='depth_concat')
        elif self.net_sched == 'm_d_s':
            input = organize_data(m, new_s, sched='depth_concat')
        elif self.net_sched == 'm_f_s_t':
            input = organize_data(m, self.s, sched='depth_concat')
            input = organize_data(input, self.t, sched='depth_concat')
        elif self.net_sched == 'm_f_s_t':
            input = organize_data(m, self.s, sched='depth_concat')
            input = organize_data(input, self.t, sched='depth_concat')
        elif self.net_sched == 'm_d_s_f_t':
            input = organize_data(m, new_s, sched='depth_concat')
            input = organize_data(input, self.t, sched='depth_concat')

        return input





    def forward(self, m,new_s=None):
        m = m * self.mask
        input = self.prepare_data(m,new_s)
        x= input
        x = self.net(x)
        return x





