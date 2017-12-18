"""
Various utility functions.

.. todo::
    Reorganize this package in a more meaningful way.
"""
# TODO

import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from libraries.modules.stn_nd import STN_ND_BCXYZ
from data_wrapper import AdaptVal
from data_wrapper import MyTensor
import numpy as np
import finite_differences as fd

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

'''
def computeWarpedImage_1d( I0, phi):
    stn = STN_ND(1)
    sz = I0.size()
    I0_stn = I0.view(torch.Size([1, sz[0], 1]))
    phi_stn = Variable( torch.zeros([1,sz[0],1]), requires_grad=False )
    phi_stn[0,:,0] = phi
    I1_warped = stn(I0_stn, phi_stn)
    return I1_warped[0,:,0]

def computeWarpedImage_2d(I0, phi):
    stn = STN_ND(2)
    sz = I0.size()
    I0_stn = I0.view(torch.Size([1, sz[0], sz[1], 1]))
    phi_stn = Variable( torch.zeros([1,sz[0],sz[1],2]), requires_grad=False )
    phi_stn[0,:,:,0]=phi[0,:,:]
    phi_stn[0,:,:,1]=phi[1,:,:]
    I1_warped = stn(I0_stn, phi_stn)
    return I1_warped[0, :, :, 0]

def computeWarpedImage_3d(I0, phi):
    stn = STN_ND(3)
    sz = I0.size()
    I0_stn = I0.view(torch.Size([1, sz[0], sz[1], sz[2], 1]))
    phi_stn = Variable( torch.zeros([1,sz[0],sz[1],sz[2],3]), requires_grad=False )
    phi_stn[0,:,:,:,0] = phi[0,:,:,:]
    phi_stn[0,:,:,:,1] = phi[1,:,:,:]
    phi_stn[0,:,:,:,2] = phi[2,:,:,:]
    I1_warped = stn(I0_stn, phi_stn)
    return I1_warped[0, :, :, :, 0]
'''

def _compute_warped_image_multiNC_1d(I0, phi):

    stn = STN_ND_BCXYZ(1)
    I1_warped = stn(I0, phi)
    return I1_warped

def _compute_warped_image_multiNC_2d(I0, phi):
    stn = STN_ND_BCXYZ(2)
    I1_warped = stn(I0, phi)
    return I1_warped

def _compute_warped_image_multiNC_3d(I0, phi):
    stn = STN_ND_BCXYZ(3)
    I1_warped = stn(I0, phi)
    return I1_warped


def compute_warped_image_multiNC(I0, phi):
    """
    Warps image.
    
    :param I0: image to warp, image size BxCxXxYxZ
    :param phi: map for the warping, size BxdimxXxYxZ 
    :return: returns the warped image of size BxCxXxYxZ
    """
    dim = I0.dim()-2
    if dim == 1:
        return _compute_warped_image_multiNC_1d(I0, phi)
    elif dim == 2:
        return _compute_warped_image_multiNC_2d(I0, phi)
    elif dim == 3:
        return _compute_warped_image_multiNC_3d(I0, phi)
    else:
        raise ValueError('Images can only be warped in dimensions 1 to 3')

'''
def computeWarpedImage(I0, phi):
    dim = I0.dim()
    if dim == 1:
        return computeWarpedImage_1d(I0, phi)
    elif dim == 2:
        return computeWarpedImage_2d(I0, phi)
    elif dim == 3:
        return computeWarpedImage_3d(I0, phi)
    else:
        raise ValueError('Images can only be warped in dimensions 1 to 3')
'''

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
        m = m + compute_vector_momentum_from_scalar_momentum_singleC(lam[:,c, ...], I[:,c, ...], nrOfI, sz[2::], spacing)
    return m

def compute_vector_momentum_from_scalar_momentum_multiC(lam, I, sz, spacing):
    """
    Computes the vector momentum from the scalar momentum: :math:`m=\\lambda\\nabla I`

    :param lam: scalar momentum, CxXxYxZ
    :param I: image, CxXxYxZ
    :param sz: size of image
    :param spacing: spacing of image
    :return: returns the vector momentum
    """
    nrOfC = sz[0]
    m = create_ND_vector_field_variable(sz[1::])
    for c in range(nrOfC): # loop over all the channels and add the results
        m = m + compute_vector_momentum_from_scalar_momentum_singleC(lam[c, ...], I[c, ...], sz[1::], spacing)
    return m

def compute_vector_momentum_from_scalar_momentum_singleC(lam, I, nrOfI, sz, spacing):
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
    return AdaptVal(Variable(torch.zeros(csz.tolist()), requires_grad=False))

def create_ND_vector_field_variable(sz):
    """
    Create vector field torch Variable of given size
    
    :param sz: just the spatial sizes (e.g., [5] in 1D, [5,10] in 2D, [5,10,10] in 3D)
    :return: returns vector field of size dimxXxYxZ
    """
    dim = len(sz)
    csz = np.array(sz) # just to make sure it is a numpy array
    csz = np.array([dim]+list(csz))
    return AdaptVal(Variable(torch.zeros(csz.tolist()), requires_grad=False))

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
    return Parameter(AdaptVal(torch.zeros(csz.tolist())))

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
    return Parameter(AdaptVal(torch.zeros(csz.tolist())))

def identity_map_multiN(sz):
    """
    Create an identity map
    
    :param sz: size of an image in BxCxXxYxZ format
    :return: returns the identity map
    """
    dim = len(sz)-2
    nrOfI = sz[0]

    if dim == 1:
        id = np.zeros([nrOfI,1,sz[2]],dtype='float32')
    elif dim == 2:
        id = np.zeros([nrOfI,2,sz[2],sz[3]],dtype='float32')
    elif dim == 3:
        id = np.zeros([nrOfI,3,sz[2],sz[3],sz[4]],dtype='float32')
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')

    for n in range(nrOfI):
        id[n,...] = identity_map(sz[2::])

    return id

def identity_map(sz):
    """
    Returns an identity map.
    
    :param sz: just the spatial dimensions, i.e., XxYxZ
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

    # now get it into range [-1,1]^d
    id = np.array( id.astype('float32') )
    if dim==1:
        id = id.reshape(1,sz[0]) # add a dummy first index

    for d in range(dim):
        id[d]*=2./(sz[d]-1)
        id[d]-=1

    # and now store it in a dim+1 array
    if dim==1:
        idnp = np.zeros([1, sz[0]], dtype='float32')
        idnp[0,:] = id[0]
    elif dim==2:
        idnp = np.zeros([2, sz[0], sz[1]], dtype='float32')
        idnp[0,:, :] = id[0]
        idnp[1,:, :] = id[1]
    elif dim==3:
        idnp = np.zeros([3,sz[0], sz[1], sz[2]], dtype='float32')
        idnp[0,:, :, :] = id[0]
        idnp[1,:, :, :] = id[1]
        idnp[2,:, :, :] = id[2]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')

    return idnp

def t2np( v ):
    """
    Takes a torch array and returns it as a numpy array on the cpu
    
    :param v: torch array
    :return: numpy array
    """
    return (v.data).cpu().numpy()

