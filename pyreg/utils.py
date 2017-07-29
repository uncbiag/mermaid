"""
Various utility functions
"""

import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from modules.stn_nd import STN_ND

import numpy as np
import finite_differences as fd
import utils

# TODO: maybe reorganize them in a more meaningful way

def computeNormalizedGaussian(X,mu,sig):
    dim = len(mu)
    if dim==1:
        g = np.exp(-np.power(X-mu[0],2.)/(2*np.power(sig[0],2.)))
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

def computeVectorMomentumFromScalarMomentum( lam, I, sz, spacing ):
    fdt = fd.FD_torch(spacing)
    dim = len(sz)
    m = utils.createNDVectorFieldVariable( sz )
    if dim==1:
        m = fdt.dXc(I)*lam
    elif dim==2:
        m[0,:,:] = fdt.dXc(I)*lam
        m[1,:,:] = fdt.dYc(I)*lam
    elif dim==3:
        m[0,:,:,:] = fdt.dXc(I)*lam
        m[1,:,:,:] = fdt.dYc(I)*lam
        m[2,:,:,:] = fdt.dZc(I)*lam
    else:
        raise ValueError('Can only convert scalar to vector momentum in dimensions 1-3')
    return m

def createNDVectorFieldVariable( sz ):

    dim = len(sz)
    csz = np.array(sz) # just to make sure it is a numpy array

    if dim==1:
        return Variable(torch.zeros(csz.tolist()), requires_grad=False )
    elif dim>1:
        csz = np.array([dim]+list(csz))
        return Variable(torch.zeros(csz.tolist()), requires_grad=False)
    else:
        raise ValueError('Cannot create a ' + str( dim ) + ' dimensional vector field')

def createNDVectorFieldParameter( sz ):

    dim = len(sz)
    csz = np.array(sz) # just to make sure it is a numpy array

    if dim==1:
        return Parameter(torch.zeros(csz.tolist()))
    elif dim>1:
        csz = np.array([dim]+list(csz))
        return Parameter(torch.zeros(csz.tolist()))
    else:
        raise ValueError('Cannot create a ' + str( dim ) + ' dimensional vector field')

def createNDScalarFieldParameter( sz ):

    dim = len(sz)
    csz = np.array(sz) # just to make sure it is a numpy array

    return Parameter(torch.zeros(csz.tolist()))


def identityMap(sz):
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
        id*=2./(sz[0]-1)
        id-=1
    else:
        for d in range(dim):
            id[d]*=2./(sz[d]-1)
            id[d]-=1

    # and now store it in a dim+1 array
    if dim==1:
        idnp = id
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

def getpar( params, key, defaultval ):
    if params is not None:
        if params.has_key( key ):
            return params[ key ]

    print( 'Using default value: ' + str( defaultval) + ' for key = ' + key )
    return defaultval

def t2np( v ):
    return (v.data).cpu().numpy()

