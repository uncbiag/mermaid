"""
Various utility functions
"""

import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from libraries.modules.stn_nd import STN_ND_BCXYZ

import numpy as np
import finite_differences as fd

# TODO: maybe reorganize them in a more meaningful way

def transformImageToNCImageFormat(I):
    '''
    Takes an input image and returns it in the format which is typical for torch.
    I.e., two dimensions are added (the first one for number of images and the second for the 
    number of channels). As were are dealing with individual single-channel intensity images here, these
    dimensions will be 1x1
    :param I: input image of size, sz
    :return: input image, reshaped to size [1,1] + sz
    '''
    return I.reshape( [1,1] + list(I.shape) )

def computeNormalizedGaussian(X,mu,sig):
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

def computeWarpedImage_multiNC_1d( I0, phi):
    stn = STN_ND_BCXYZ(1)
    I1_warped = stn(I0, phi)
    return I1_warped

def computeWarpedImage_multiNC_2d(I0, phi):
    stn = STN_ND_BCXYZ(2)
    I1_warped = stn(I0, phi)
    return I1_warped

def computeWarpedImage_multiNC_3d(I0, phi):
    stn = STN_ND_BCXYZ(3)
    I1_warped = stn(I0, phi)
    return I1_warped


def computeWarpedImage_multiNC(I0, phi):
    dim = I0.dim()-2
    if dim == 1:
        return computeWarpedImage_multiNC_1d(I0, phi)
    elif dim == 2:
        return computeWarpedImage_multiNC_2d(I0, phi)
    elif dim == 3:
        return computeWarpedImage_multiNC_3d(I0, phi)
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

def computeVectorMomentumFromScalarMomentum_multiNC( lam, I, sz, spacing ):
    nrOfI = sz[0] # number of images
    m = createNDVectorFieldVariable_multiN(sz[2::],nrOfI)
    for nrI in range(nrOfI):  # loop over all the images
        m[nrI, ...] = computeVectorMomentumFromScalarMomentum_multiC( lam[nrI, ...], I[nrI,...], sz[1::], spacing )
    return m

def computeVectorMomentumFromScalarMomentum_multiC(lam, I, sz, spacing):
    nrOfC = sz[0]
    m = createNDVectorFieldVariable( sz[1::] )
    for c in range(nrOfC): # loop over all the channels and add the results
        m = m + computeVectorMomentumFromScalarMomentum_singleC( lam[c,...], I[c,...], sz[1::], spacing )
    return m

def computeVectorMomentumFromScalarMomentum_singleC( lam, I, sz, spacing ):
    fdt = fd.FD_torch(spacing)
    dim = len(sz)
    m = createNDVectorFieldVariable( sz )
    if dim==1:
        m[0,:] = fdt.dXc(I)*lam
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

def createNDVectorFieldVariable_multiN( sz, nrOfI=1 ):
    dim = len(sz)
    csz = np.array(sz) # just to make sure it is a numpy array
    csz = np.array([nrOfI,dim]+list(csz))
    return Variable(torch.zeros(csz.tolist()), requires_grad=False)

def createNDVectorFieldVariable( sz ):
    dim = len(sz)
    csz = np.array(sz) # just to make sure it is a numpy array
    csz = np.array([dim]+list(csz))
    return Variable(torch.zeros(csz.tolist()), requires_grad=False)

def createNDVectorFieldParameter_multiN( sz, nrOfI=1 ):

    dim = len(sz)
    csz = np.array(sz) # just to make sure it is a numpy array
    csz = np.array([nrOfI,dim]+list(csz))
    return Parameter(torch.zeros(csz.tolist()))

def createNDScalarFieldParameter_multiNC( sz, nrOfI=1, nrOfC=1 ):

    dim = len(sz)
    csz = np.array(sz) # just to make sure it is a numpy array
    csz = np.array([nrOfI,nrOfC]+list(csz))
    return Parameter(torch.zeros(csz.tolist()))

def identityMap_multiN(sz):
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
        id[n,...] = identityMap(sz[2::])

    return id

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
    return (v.data).cpu().numpy()

