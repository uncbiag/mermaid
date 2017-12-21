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
from data_wrapper import MyTensor,AdaptVal
import numpy as np
import finite_differences as fd
import torch.nn as nn
import torch.nn.init as init
from libraries.functions.nn_interpolation import get_nn_interpolation
import pandas as pd


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



def get_warped_label_map(label_map, phi, sched='nn'):
    if sched == 'nn':
        warped_label_map = get_nn_interpolation(label_map, phi)
        # check if here should be add assert
        assert abs(torch.sum(warped_label_map.data -warped_label_map.data.round()))< 0.1, "nn interpolation is not precise"
    else:
        raise ValueError, " the label warpping method is not implemented"
    return warped_label_map






def t2np( v ):
    """
    Takes a torch array and returns it as a numpy array on the cpu
    
    :param v: torch array
    :return: numpy array
    """
    return (v.data).cpu().numpy()

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
            id =identity_map(sz)
            g = compute_normalized_gaussian(id, mus, stds)
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
    print(torch.sum(gi[0].data), torch.sum(gi[1].data))
    print("Grad Output")
    print(torch.sum(go[0].data))
    return gi[0],gi[1], gi[2]



class ConvBnRel(nn.Module):
    # conv + bn (optional) + relu
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, active_unit='relu', same_padding=False,
                 bn=False, reverse=False, bias=False):
        super(ConvBnRel, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
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





