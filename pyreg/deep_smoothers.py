from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from builtins import object
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .data_wrapper import USE_CUDA, MyTensor, AdaptVal

import math
import pyreg.finite_differences as fd
import pyreg.module_parameters as pars
import pyreg.fileio as fio
import pyreg.custom_pytorch_extensions as ce

import os
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable

batch_norm_momentum_val = 0.1

# def _custom_colorbar(mappable):
#     ax = mappable.axes
#     fig = ax.figure
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     return fig.colorbar(mappable, cax=cax)
#
# def _plot_edgemap_2d(image,gradient_norm,edge_map):
#     plt.clf()
#
#     fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
#     c1 = ax1.imshow(image)
#     _custom_colorbar(c1)
#     ax1.set_title('Original image')
#
#     c2 = ax2.imshow(gradient_norm)
#     _custom_colorbar(c2)
#     ax2.set_title('Gradient norm')
#
#     c3 = ax3.imshow(edge_map)
#     _custom_colorbar(c3)
#     ax3.set_title('edge map')
#
#     plt.tight_layout()
#     #plt.tight_layout(h_pad=1)

def _plot_edgemap_2d(image,gradient_norm,edge_map,gamma):
    plt.clf()

    plt.subplot(221)
    plt.imshow(image)
    plt.colorbar()
    plt.title('Original image')

    plt.subplot(222)
    plt.imshow(edge_map)
    plt.colorbar()
    plt.title('edge map')

    plt.subplot(223)
    plt.imshow(gradient_norm)
    plt.colorbar()
    plt.title('Gradient norm')

    plt.subplot(224)
    plt.imshow(gamma*gradient_norm)
    plt.colorbar()
    plt.title('gamma * Gradient norm')

    plt.tight_layout()
    #plt.tight_layout(h_pad=1)

def _plot_edgemap_3d(image,gradient_norm,edge_map,gamma):
    plt.clf()

    sz = image.shape
    szh = np.array(sz) // 2

    plt.subplot(3,4,1)
    plt.imshow(image[szh[0],...])
    plt.colorbar()
    plt.title('O-image')

    plt.subplot(3,4,2)
    plt.imshow(edge_map[szh[0],...])
    plt.colorbar()
    plt.title('edge')

    plt.subplot(3,4,3)
    plt.imshow(gradient_norm[szh[0],...])
    plt.colorbar()
    plt.title('Grad-norm')

    plt.subplot(3,4,4)
    plt.imshow(gamma*gradient_norm[szh[0],...])
    plt.colorbar()
    plt.title('gamma * grad-norm')

    plt.subplot(3,4,5)
    plt.imshow(image[:,szh[1], ...])
    plt.colorbar()

    plt.subplot(3,4,6)
    plt.imshow(edge_map[:,szh[1], ...])
    plt.colorbar()

    plt.subplot(3,4,7)
    plt.imshow(gradient_norm[:,szh[1], ...])
    plt.colorbar()
    plt.title('Grad-norm')

    plt.subplot(3,4,8)
    plt.imshow(gamma * gradient_norm[:,szh[1], ...])
    plt.colorbar()

    plt.subplot(3,4,9)
    plt.imshow(image[:,:,szh[2]])
    plt.colorbar()

    plt.subplot(3,4,10)
    plt.imshow(edge_map[:,:,szh[2]])
    plt.colorbar()

    plt.subplot(3,4,11)
    plt.imshow(gradient_norm[:,:,szh[2]])
    plt.colorbar()
    plt.title('Grad-norm')

    plt.subplot(3,4,12)
    plt.imshow(gamma * gradient_norm[:,:,szh[2]])
    plt.colorbar()
    plt.title('gamma * grad-norm')

    plt.tight_layout()
    #plt.tight_layout(h_pad=1)


def _edgemap_plot_and_write_to_pdf(image,gradient_norm,edge_map,gamma,pdf_filename):
    dim = len(edge_map.size())

    if dim==1:
        print('INFO: PDF output not yet implemented for 1D image; implement it in deep_smoothers.py')
    elif dim==2:
        _plot_edgemap_2d(image=image,gradient_norm=gradient_norm,edge_map=edge_map,gamma=gamma)
        plt.savefig(pdf_filename,bbox_inches='tight',pad_inches=0)
        _plot_edgemap_2d(image=image,gradient_norm=gradient_norm,edge_map=edge_map,gamma=gamma)
        plt.show()
    elif dim==3:
        _plot_edgemap_3d(image=image,gradient_norm=gradient_norm,edge_map=edge_map,gamma=gamma)
        plt.savefig(pdf_filename,bbox_inches='tight',pad_inches=0)
        _plot_edgemap_3d(image=image,gradient_norm=gradient_norm,edge_map=edge_map,gamma=gamma)
        plt.show()
    else:
        raise ValueError('Unknown dimension; dimension must be 1, 2, or 3')

def half_sigmoid(x,alpha=1):
    r = 2.0/(1+torch.exp(-x*alpha))-1.0
    return r

def _compute_localized_edge_penalty(I,spacing,gamma):
    # needs to be batch B x X x Y x Z format
    fdt = fd.FD_torch(spacing=spacing)
    gnI = float(np.min(spacing)) * fdt.grad_norm_sqr(I) ** 0.5

    # compute edge penalty
    localized_edge_penalty = 1.0 / (1.0 + gamma * gnI)  # this is what we weight the OMT values with

    return localized_edge_penalty,gnI

def compute_localized_edge_penalty(I,spacing,params=None):

    if params==None:
        # will not be tracked, but this is to keep track of the parameters
        params = pars.ParameterDict()

    gamma = params[('edge_penalty_gamma',10.0,'Constant for edge penalty: 1.0/(1.0+gamma*||\\nabla I||*min(spacing)')]
    write_edge_penalty_to_file = params[('edge_penalty_write_to_file',False,'If set to True the edge penalty is written into a file so it can be debugged')]
    edge_penalty_filename = params[('edge_penalty_filename','DEBUG_edge_penalty.nrrd','Edge penalty image')]
    terminate_after_writing_edge_penalty = params[('edge_penalty_terminate_after_writing',False,'Terminates the program after the edge file has been written; otherwise file may be constantly overwritten')]

    # compute edge penalty
    localized_edge_penalty,gnI = _compute_localized_edge_penalty(I,spacing=spacing,gamma=gamma) # this is what we weight the OMT values with

    if write_edge_penalty_to_file:

        current_image_filename = str(edge_penalty_filename)
        current_pdf_filename = (os.path.splitext(current_image_filename)[0])+'.pdf'
        current_pt_filename = (os.path.splitext(current_image_filename)[0])+'.pt'

        debug_dict = dict()
        debug_dict['image'] = I

        torch.save(debug_dict,current_pt_filename)
        fio.ImageIO().write(current_image_filename,localized_edge_penalty[0,...].data.cpu())
        _edgemap_plot_and_write_to_pdf(image=I[0,...].data.cpu(),gradient_norm=gnI[0,...].data.cpu(),edge_map=localized_edge_penalty[0,...].data.cpu(),gamma=gamma,pdf_filename=current_pdf_filename)

        if terminate_after_writing_edge_penalty:
            print('Terminating, because terminate_after_writing_edge_penalty was set to True')
            exit(code=0)

    return localized_edge_penalty

def _compute_weighted_total_variation_1d(d_in,w, spacing, bc_val, pnorm=2):

    d = torch.zeros_like(d_in)
    d[:] = d_in

    # now force the boundary condition
    d[:, 0] = bc_val
    d[:, -1] = bc_val

    fdt = fd.FD_torch(spacing=spacing)
    # need to use torch.abs here to make sure the proper subgradient is computed at zero
    batch_size = d.size()[0]
    volumeElement = spacing.prod()
    t0 = torch.abs(fdt.dXc(d))

    tm = t0*w

    return (tm).sum()*volumeElement/batch_size

def _compute_weighted_total_variation_2d(d_in,w, spacing, bc_val, pnorm=2):

    d = torch.zeros_like(d_in)
    d[:] = d_in

    # now force the boundary condition
    d[:, 0, :] = bc_val
    d[:, -1, :] = bc_val
    d[:, :, 0] = bc_val
    d[:, :, -1] = bc_val

    fdt = fd.FD_torch(spacing=spacing)
    # need to use torch.norm here to make sure the proper subgradient is computed at zero
    batch_size = d.size()[0]
    volumeElement = spacing.prod()
    t0 = torch.norm(torch.stack((fdt.dXc(d),fdt.dYc(d))),pnorm,0)

    tm = t0*w

    return (tm).sum()*volumeElement/batch_size

def _compute_weighted_total_variation_3d(d_in,w, spacing, bc_val, pnorm=2):

    d = torch.zeros_like(d_in)
    d[:] = d_in

    # now force the boundary condition
    d[:, 0, :, :] = bc_val
    d[:, -1, :, :] = bc_val
    d[:, :, 0, :] = bc_val
    d[:, :, -1, :] = bc_val
    d[:, :, :, 0] = bc_val
    d[:, :, :, -1] = bc_val

    fdt = fd.FD_torch(spacing=spacing)
    # need to use torch.norm here to make sure the proper subgradient is computed at zero
    batch_size = d.size()[0]
    volumeElement = spacing.prod()

    t0 = torch.norm(torch.stack((fdt.dXc(d),
                                 fdt.dYc(d),
                                 fdt.dZc(d))), pnorm, 0)

    tm = t0*w

    return (tm).sum()*volumeElement/batch_size

def compute_weighted_total_variation(d, w, spacing,bc_val,pnorm=2):
    # just do the standard component-wise Euclidean norm of the gradient, but muliplied locally by a weight
    # format needs to be B x X x Y x Z

    dim = len(d.size())-1

    if dim == 1:
        return _compute_weighted_total_variation_1d(d,w, spacing,bc_val,pnorm)
    elif dim == 2:
        return _compute_weighted_total_variation_2d(d,w, spacing,bc_val,pnorm)
    elif dim == 3:
        return _compute_weighted_total_variation_3d(d,w, spacing,bc_val,pnorm)
    else:
        raise ValueError('Total variation computation is currently only supported in dimensions 1 to 3')

def _compute_local_norm_of_gradient_1d(d,spacing,pnorm=2):

    fdt = fd.FD_torch(spacing=spacing)
    # need to use torch.abs here to make sure the proper subgradient is computed at zero
    t0 = torch.abs(fdt.dXc(d))

    return t0

def _compute_local_norm_of_gradient_2d(d,spacing,pnorm=2):

    fdt = fd.FD_torch(spacing=spacing)
    # need to use torch.norm here to make sure the proper subgradient is computed at zero
    #t0 = torch.norm(torch.stack((fdt.dXc(d),fdt.dYc(d))),pnorm,0)

    dX = fdt.dXc(d)
    dY = fdt.dYc(d)

    t0 = torch.norm(torch.stack((dX, dY)), pnorm, 0)

    #print('DEBUG:min(dX)={}, max(dX)={}, min(dY)={}, max(dY)={}, min(t0)={}, max(t0)={}'.format(dX.min().data.cpu().numpy(),
    #                                                                                      dX.max().data.cpu().numpy(),
    #                                                                                      dY.min().data.cpu().numpy(),
    #                                                                                      dY.max().data.cpu().numpy(),
    #                                                                                      t0.min().data.cpu().numpy(),
    #                                                                                      t0.max().data.cpu().numpy()))

    return t0

def _compute_local_norm_of_gradient_3d(d,spacing, pnorm=2):

    fdt = fd.FD_torch(spacing=spacing)
    # need to use torch.norm here to make sure the proper subgradient is computed at zero

    t0 = torch.norm(torch.stack((fdt.dXc(d),
                                 fdt.dYc(d),
                                 fdt.dZc(d))), pnorm, 0)

    return t0

def _compute_local_norm_of_gradient(d, spacing, pnorm=2):
    # just do the standard component-wise Euclidean norm of the gradient, but muliplied locally by a weight
    # format needs to be B x X x Y x Z

    dim = len(d.size())-1

    if dim == 1:
        return _compute_local_norm_of_gradient_1d(d,spacing,pnorm)
    elif dim == 2:
        return _compute_local_norm_of_gradient_2d(d,spacing,pnorm)
    elif dim == 3:
        return _compute_local_norm_of_gradient_3d(d,spacing,pnorm)
    else:
        raise ValueError('Local norm of gradient computation is currently only supported in dimensions 1 to 3')

def _compute_total_variation(d, spacing, pnorm=2):
    # just do the standard component-wise Euclidean norm of the gradient, but muliplied locally by a weight
    # format needs to be B x X x Y x Z

    batch_size = d.size()[0]
    volumeElement = spacing.prod()

    tv = _compute_local_norm_of_gradient(d,spacing,pnorm)
    return (tv).sum()*volumeElement/batch_size


def _compute_localized_omt_weight_1d(weights,g,I,spacing,pnorm):

    r = torch.zeros_like(g)

    # set the boundary values to 1
    r[:, 0] = 1
    r[:, -1] = 1

    # compute the sum of the total variations of the weights
    sum_tv = torch.zeros_like(g)
    nr_of_weights = weights.size()[1]

    for n in range(nr_of_weights):
        sum_tv += _compute_local_norm_of_gradient(weights[:, n, ...], spacing, pnorm)
    sum_tv /= nr_of_weights

    #sum_tv = torch.max(sum_tv,_compute_local_norm_of_gradient(I[:,0,...], spacing, pnorm))
    sum_tv += _compute_local_norm_of_gradient(I[:, 0, ...], spacing, pnorm)

    #r_tv = half_sigmoid(sum_tv) * g
    r_tv = sum_tv*g

    r_tv[:, 0] = 0
    r_tv[:, -1] = 0

    r = r + r_tv

    S0 = 2

    return r,S0


def _compute_localized_omt_weight_2d(weights,g,I,spacing,pnorm):

    r = torch.zeros_like(g)

    ## set the boundary values to 1
    #r[:,0,:] = 1
    #r[:,-1,:] = 1
    #r[:,:,0] = 1
    #r[:,:,-1] = 1

    sz = r.size()
    S0 = 2 * sz[2] + 2 * sz[1]

    # compute the sum of the total variations of the weights
    sum_tv = torch.zeros_like(g)
    nr_of_weights = weights.size()[1]

    for n in range(nr_of_weights):
        sum_tv += _compute_local_norm_of_gradient(weights[:,n,...],spacing,pnorm)
    sum_tv /= nr_of_weights

    #sum_tv = torch.max(sum_tv,_compute_local_norm_of_gradient(I[:,0,...], spacing, pnorm))
    tvl_I = _compute_local_norm_of_gradient(I[:, 0, ...], spacing, pnorm)
    sum_tv += tvl_I

    #r_tv = half_sigmoid(sum_tv)*g
    r_tv = sum_tv * g

    batch_size = I.size()[0]
    r_tv *= batch_size/(tvl_I*g).sum()

    #r_tv[:,0,:] = 0
    #r_tv[:,-1,:] = 0
    #r_tv[:,:,0] = 0
    #r_tv[:,:,-1] = 0

    #r = r + r_tv
    r = r_tv

    return r,S0

def _compute_localized_omt_weight_3d(weights,g,I,spacing,pnorm):

    r = torch.zeros_like(g)

    # set the boundary values to 1
    r[:, 0, :, :] = 1
    r[:, -1, :, :] = 1
    r[:, :, 0, :] = 1
    r[:, :, -1, :] = 1
    r[:, :, :, 0] = 1
    r[:, :, :, -1] = 1

    sz = r.size()
    S0 = 2*sz[2]*sz[3] + 2*sz[1]*sz[3] + 2*sz[1]*sz[2]

    # compute the sum of the total variations of the weights
    sum_tv = torch.zeros_like(g)
    nr_of_weights = weights.size()[1]

    for n in range(nr_of_weights):
        sum_tv += _compute_local_norm_of_gradient(weights[:, n, ...], spacing, pnorm)
    sum_tv /= nr_of_weights

    #sum_tv = torch.max(sum_tv,_compute_local_norm_of_gradient(I[:,0,...], spacing, pnorm))
    sum_tv += _compute_local_norm_of_gradient(I[:, 0, ...], spacing, pnorm)

    #r_tv = half_sigmoid(sum_tv) * g
    r_tv = sum_tv * g

    r_tv[:, 0, :, :] = 0
    r_tv[:, -1, :, :] = 0
    r_tv[:, :, 0, :] = 0
    r_tv[:, :, -1, :] = 0
    r_tv[:, :, :, 0] = 0
    r_tv[:, :, :, -1] = 0

    r = r + r_tv

    return r,S0

def compute_localized_omt_weight(weights, I, spacing,pnorm=2):
    # just do the standard component-wise Euclidean norm of the gradient, but muliplied locally by a weight
    # format needs to be B x X x Y x Z

    if I.size()[1]!=1:
        raise ValueError('Only scalar images are currently supported')

    g = compute_localized_edge_penalty(I[:,0,...],spacing)

    dim = len(g.size())-1

    if dim == 1:
        return _compute_localized_omt_weight_1d(weights,g, I, spacing,pnorm)
    elif dim == 2:
        return _compute_localized_omt_weight_2d(weights,g, I, spacing,pnorm)
    elif dim == 3:
        return _compute_localized_omt_weight_3d(weights,g, I, spacing,pnorm)
    else:
        raise ValueError('Total variation computation is currently only supported in dimensions 1 to 3')



def compute_localized_omt_penalty(weights, I, multi_gaussian_stds,spacing,volume_element,desired_power=2.0,use_log_transform=False):
    # weights: B x weights x X x Y

    if weights.size()[1] != len(multi_gaussian_stds):
        raise ValueError('Number of weights need to be the same as number of Gaussians. Format recently changed for weights to B x weights x X x Y')

    penalty = Variable(MyTensor(1).zero_(), requires_grad=False)

    # first compute the gradient of the image as the penalty is only evaluated at gradients of the image
    # and where the gradients of the total variation of the weights are

    nr_of_image_channels = I.size()[1]
    if nr_of_image_channels!=1:
        raise ValueError('localized omt is currently only supported for single channel images')

    batch_size = I.size()[0]

    max_std = max(multi_gaussian_stds)
    min_std = min(multi_gaussian_stds)

    nr_of_multi_gaussians = len(multi_gaussian_stds)
    if multi_gaussian_stds[nr_of_multi_gaussians-1]!=max_std:
        raise ValueError('Assuming that the last standard deviation is the largest')

    gamma,S0 = compute_localized_omt_weight(weights, I, spacing)

    if desired_power == 2:
        for i, s in enumerate(multi_gaussian_stds):

            weighted_tv_penalty = (gamma*weights[:,i,...]).sum()

            if use_log_transform:
                penalty += weighted_tv_penalty * ((np.log(max_std / s)) ** desired_power)
            else:
                penalty += weighted_tv_penalty * ((s - max_std) ** desired_power)

        if use_log_transform:
            penalty /= (np.log(max_std / min_std)) ** desired_power
        else:
            penalty /= (max_std - min_std) ** desired_power

        #penalty/=S0

    else:
        for i, s in enumerate(multi_gaussian_stds):

            weighted_tv_penalty = (gamma*weights[:,i,...]).sum()

            if use_log_transform:
                penalty += weighted_tv_penalty * (abs(np.log(max_std / s)) ** desired_power)
            else:
                penalty += weighted_tv_penalty * (abs(s - max_std) ** desired_power)

        if use_log_transform:
            penalty /= abs(np.log(max_std / min_std)) ** desired_power
        else:
            penalty /= abs(max_std - min_std) ** desired_power

        #penalty/=S0

    # todo: check why division by batch size appears not to be necessary (probably because of division by S0)
    penalty /= batch_size
    #penalty *= volume_element

    return penalty


def compute_omt_penalty(weights, multi_gaussian_stds,volume_element,desired_power=2.0,use_log_transform=False):

    # weights: B x weights x X x Y

    if weights.size()[1] != len(multi_gaussian_stds):
        raise ValueError('Number of weights need to be the same as number of Gaussians. Format recently changed for weights to B x weights x X x Y')

    penalty = Variable(MyTensor(1).zero_(), requires_grad=False)
    batch_size = weights.size()[0]

    max_std = max(multi_gaussian_stds)
    min_std = min(multi_gaussian_stds)

    if desired_power==2:
        for i, s in enumerate(multi_gaussian_stds):
            if use_log_transform:
                penalty += ((weights[:, i, ...]).sum()) * ((np.log(max_std/s)) ** desired_power)
            else:
                penalty += ((weights[:, i, ...]).sum()) * ((s - max_std) ** desired_power)

        if use_log_transform:
            penalty /= (np.log(max_std/min_std))** desired_power
        else:
            penalty /= (max_std - min_std)** desired_power
    else:
        for i,s in enumerate(multi_gaussian_stds):
            if use_log_transform:
                penalty += ((weights[:,i,...]).sum())*(abs(np.log(max_std/s))**desired_power)
            else:
                penalty += ((weights[:,i,...]).sum())*(abs(s-max_std)**desired_power)

        if use_log_transform:
            penalty /= abs(np.log(max_std/min_std))**desired_power
        else:
            penalty /= abs(max_std-min_std)**desired_power

    penalty /= batch_size
    penalty *= volume_element

    return penalty

def DimConv(dim):
    if dim==1:
        return nn.Conv1d
    elif dim==2:
        return nn.Conv2d
    elif dim==3:
        return nn.Conv3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')

def DimBatchNorm(dim):
    if dim==1:
        return nn.BatchNorm1d
    elif dim==2:
        return nn.BatchNorm2d
    elif dim==3:
        return nn.BatchNorm3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')

def DimConvTranspose(dim):
    if dim==1:
        return nn.ConvTranspose1d
    elif dim==2:
        return nn.ConvTranspose2d
    elif dim==3:
        return nn.ConvTranspose3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')

def weighted_softmax(input, dim=None, weights=None ):
    r"""Applies a softmax function.

    Weighted_softmax is defined as:

    :math:`weighted_softmax(x) = \frac{w_i exp(x_i)}{\sum_j w_j exp(x_j)}`

    It is applied to all slices along dim, and will rescale them so that the elements
    lie in the range `(0, 1)` and sum to 1.

    See :class:`~torch.nn.WeightedSoftmax` for more details.

    Arguments:
        input (Variable): input
        dim (int): A dimension along which weighted_softmax will be computed.

    """
    if dim is None:
        raise ValueError('dimension needs to be defined!')

    sz = input.size()
    if weights is None: # just make them all one; this is the default softmax
        weights = [1.]*sz[dim]

    nr_of_weights = len(weights)
    assert( sz[dim]==nr_of_weights )

    ret = torch.zeros_like(input)

    if dim==0:
        norm = torch.zeros_like(input[0,...])
        for c in range(sz[0]):
            norm += weights[c]*torch.exp(input[c,...])
        for c in range(sz[0]):
            ret[c,...] = weights[c]*torch.exp(input[c,...])/norm
    elif dim==1:
        norm = torch.zeros_like(input[:,0, ...])
        for c in range(sz[1]):
            norm += weights[c] * torch.exp(input[:,c, ...])
        for c in range(sz[1]):
            ret[:,c, ...] = weights[c] * torch.exp(input[:,c, ...]) / norm
    elif dim==2:
        norm = torch.zeros_like(input[:,:,0, ...])
        for c in range(sz[2]):
            norm += weights[c] * torch.exp(input[:,:,c, ...])
        for c in range(sz[2]):
            ret[:,:,c, ...] = weights[c] * torch.exp(input[:,:,c, ...]) / norm
    elif dim==3:
        norm = torch.zeros_like(input[:,:,:,0, ...])
        for c in range(sz[3]):
            norm += weights[c] * torch.exp(input[:,:,:,c, ...])
        for c in range(sz[3]):
            ret[:,:,:,c, ...] = weights[c] * torch.exp(input[:,:,:,c, ...]) / norm
    elif dim==4:
        norm = torch.zeros_like(input[:,:,:,:,0, ...])
        for c in range(sz[4]):
            norm += weights[c] * torch.exp(input[:,:,:,:,c, ...])
        for c in range(sz[4]):
            ret[:,:,:,:,c, ...] = weights[c] * torch.exp(input[:,:,:,:,c, ...]) / norm
    else:
        raise ValueError('weighted_softmax is only supported for dimensions 0, 1, 2, 3, and 4.')

    return ret


class WeightedSoftmax(nn.Module):
    r"""Applies the WeightedSoftmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    WeightedSoftmax is defined as
    :math:`f_i(x) = \frac{w_i\exp(x_i)}{\sum_j w_j\exp(x_j)}`

    It is assumed that w_i>0 and that the weights sum up to one.
    The effect of this weighting is that for a zero input (x=0) the output for f_i(x) will be w_i.
    I.e., we can obtain a default output which is not 1/n.

    Shape:
        - Input: any shape
        - Output: same as input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Arguments:
        dim (int): A dimension along which WeightedSoftmax will be computed (so every slice
            along dim will sum to 1).

    Examples::

        >>> m = nn.WeightedSoftmax()
        >>> input = autograd.Variable(torch.randn(2, 3))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, dim=None, weights=None):
        super(WeightedSoftmax, self).__init__()
        self.dim = dim
        self.weights = weights

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None
        if not hasattr(self, 'weights'):
            self.weights = None

    def forward(self, input):

        return weighted_softmax(input, self.dim, self.weights, _stacklevel=5)

    def __repr__(self):

        return self.__class__.__name__ + '()'

def weighted_sqrt_softmax(input, dim=None, weights=None ):
    r"""Applies a weighted square-root softmax function.

    Weighted_sqrt_softmax is defined as:

    :math:`weighted_sqrt_softmax(x) = \frac{\sqrt{w_i} exp(x_i)}{\sqrt{\sum_j w_j (exp(x_j))^2}}`

    It is applied to all slices along dim, and will rescale them so that the elements
    lie in the range `(0, 1)` and sum to 1.

    See :class:`~torch.nn.WeightedSoftmax` for more details.

    Arguments:
        input (Variable): input
        dim (int): A dimension along which weighted_softmax will be computed.

    """
    if dim is None:
        raise ValueError('dimension needs to be defined!')

    sz = input.size()
    if weights is None: # just make them all one; this is the default softmax
        weights = [1.]*sz[dim]

    nr_of_weights = len(weights)
    assert( sz[dim]==nr_of_weights )

    ret = torch.zeros_like(input)

    if dim==0:
        norm_sqr = torch.zeros_like(input[0,...])
        for c in range(sz[0]):
            norm_sqr += weights[c]*(torch.exp(input[c,...]))**2
        norm = torch.sqrt(norm_sqr)
        for c in range(sz[0]):
            ret[c,...] = torch.sqrt(weights[c])*torch.exp(input[c,...])/norm
    elif dim==1:
        norm_sqr = torch.zeros_like(input[:,0, ...])
        for c in range(sz[1]):
            norm_sqr += weights[c] * (torch.exp(input[:,c, ...]))**2
        norm = torch.sqrt(norm_sqr)
        for c in range(sz[1]):
            ret[:,c, ...] = torch.sqrt(weights[c]) * torch.exp(input[:,c, ...]) / norm
    elif dim==2:
        norm_sqr = torch.zeros_like(input[:,:,0, ...])
        for c in range(sz[2]):
            norm_sqr += weights[c] * (torch.exp(input[:,:,c, ...]))**2
        norm = torch.sqrt(norm_sqr)
        for c in range(sz[2]):
            ret[:,:,c, ...] = torch.sqrt(weights[c]) * torch.exp(input[:,:,c, ...]) / norm
    elif dim==3:
        norm_sqr = torch.zeros_like(input[:,:,:,0, ...])
        for c in range(sz[3]):
            norm_sqr += weights[c] * (torch.exp(input[:,:,:,c, ...]))**2
        norm = torch.sqrt(norm_sqr)
        for c in range(sz[3]):
            ret[:,:,:,c, ...] = torch.sqrt(weights[c]) * torch.exp(input[:,:,:,c, ...]) / norm
    elif dim==4:
        norm_sqr = torch.zeros_like(input[:,:,:,:,0, ...])
        for c in range(sz[4]):
            norm_sqr += weights[c] * (torch.exp(input[:,:,:,:,c, ...]))**2
        norm = torch.sqrt(norm_sqr)
        for c in range(sz[4]):
            ret[:,:,:,:,c, ...] = torch.sqrt(weights[c]) * torch.exp(input[:,:,:,:,c, ...]) / norm
    else:
        raise ValueError('weighted_softmax is only supported for dimensions 0, 1, 2, 3, and 4.')

    return ret

class WeightedSqrtSoftmax(nn.Module):
    r"""Applies the WeightedSqrtSoftmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and their squares sum to 1

    WeightedSoftmax is defined as
    :math:`f_i(x) = \frac{\sqrt{w_i}\exp(x_i)}{\sqrt{\sum_j w_j\exp(x_j)^2}}`

    It is assumed that w_i>=0 and that the weights sum up to one.
    The effect of this weighting is that for a zero input (x=0) the output for f_i(x) will be \sqrt{w_i}.
    I.e., we can obtain a default output which is not 1/n and if we sqaure the outputs we are back
    to the original weights for zero (input). This is useful behavior to implement, for example, local
    kernel weightings while avoiding square roots of weights that may be close to zero (and hence potential
    numerical issues with the gradient). The assumption is, of course, here that the weights are fixed and are not being
    optimized over, otherwise there would still be numerical issues. TODO: check that this is indeed working as planned.

    Shape:
        - Input: any shape
        - Output: same as input

    Returns:
        a Tensor of the same dimension and shape as the input with
        positive values such that their squares sum up to one.

    Arguments:
        dim (int): A dimension along which WeightedSqrtSoftmax will be computed (so every slice
            along dim will sum to 1).

    Examples::

        >>> m = nn.WeightedSqrtSoftmax()
        >>> input = autograd.Variable(torch.randn(2, 3))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, dim=None, weights=None):
        super(WeightedSqrtSoftmax, self).__init__()
        self.dim = dim
        self.weights = weights

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None
        if not hasattr(self, 'weights'):
            self.weights = None

    def forward(self, input):

        return weighted_sqrt_softmax(input, self.dim, self.weights, _stacklevel=5)

    def __repr__(self):

        return self.__class__.__name__ + '()'



class DeepSmoothingModel(nn.Module):
    """
    Base class for mini neural network which takes as an input a set of smoothed velocity field as
    well as input images and predicts weights for a multi-Gaussian smoothing from this
    Enforces the same weighting for all the dimensions of the vector field to be smoothed

    """

    def __init__(self, nr_of_gaussians, gaussian_stds, dim, spacing, nr_of_image_channels=1, omt_power=1.0,params=None):
        super(DeepSmoothingModel, self).__init__()

        self.nr_of_image_channels = nr_of_image_channels

        if nr_of_image_channels!=1:
            raise ValueError('Currently only implemented for images with 1 channel')

        self.dim = dim
        self.omt_power = omt_power
        self.pnorm = 2

        self.spacing = spacing
        self.fdt = fd.FD_torch(self.spacing)
        self.volumeElement = self.spacing.prod()

        # check that the largest standard deviation is the largest one
        if max(gaussian_stds) > gaussian_stds[-1]:
            raise ValueError('The last standard deviation needs to be the largest')

        self.omt_weight_penalty = params[('omt_weight_penalty', 25.0, 'Penalty for the optimal mass transport')]
        self.omt_use_log_transformed_std = params[('omt_use_log_transformed_std', False,
                                                        'If set to true the standard deviations are log transformed for the computation of OMT')]
        """if set to true the standard deviations are log transformed for the OMT computation"""

        self.omt_power = params[('omt_power', 1.0, 'Power for the optimal mass transport (i.e., to which power distances are penalized')]
        """optimal mass transport power"""

        self.gaussianWeight_min = params[('gaussian_weight_min', 0.001, 'minimal allowed weight for the Gaussians')]
        """minimal allowed weight during optimization"""

        cparams = params[('deep_smoother',{})]
        self.params = cparams

        self.weighting_type = self.params[('weighting_type','w_K','Type of weighting: w_K|w_K_w|sqrt_w_K_sqrt_w')]
        admissible_weighting_types = ['w_K','w_K_w','sqrt_w_K_sqrt_w']
        if self.weighting_type not in admissible_weighting_types:
            raise ValueError('Unknown weighting_type: needs to be  w_K|w_K_w|sqrt_w_K_sqrt_w')

        self.diffusion_weight_penalty = self.params[('diffusion_weight_penalty', 0.0, 'Penalized the squared gradient of the weights')]
        self.total_variation_weight_penalty = self.params[('total_variation_weight_penalty', 0.1, 'Penalize the total variation of the weights if desired')]

        self.standardize_input_images = self.params[('standardize_input_images',True,'if true, subtracts the value specified by standardize_subtract_from_input_images followed by division by standardize_divide_input_images from all input images to the network')]
        """if true then we subtract standardize_subtract_from_input_images from all network input images"""

        self.standardize_subtract_from_input_images = self.params[('standardize_subtract_from_input_images',0.5,'Subtracts this value from all images input into a network')]
        """Subtracts this value from all images input into a network"""

        self.standardize_divide_input_images = self.params[('standardize_divide_input_images',1.0,'Value to divide the input images by *AFTER* subtraction')]
        """Value to divide the input images by *AFTER* subtraction"""

        self.standardize_input_momentum = self.params[('standardize_input_momentum', True, 'if true, subtracts the value specified by standardize_subtract_from_input_momentum followed by division by standardize_divide_input_momentum from the input momentum to the network')]
        """if true then we subtract standardize_subtract_from_input_momentum from the network input momentum"""

        self.standardize_subtract_from_input_momentum = self.params[('standardize_subtract_from_input_momentum', 0.0, 'Subtracts this value from the input momentum into a network')]
        """Subtracts this value from the momentum input to a network"""

        self.standardize_divide_input_momentum = self.params[('standardize_divide_input_momentum', 1.0, 'Value to divide the input momentum by *AFTER* subtraction')]
        """Value to divide the input momentum by *AFTER* subtraction"""

        self.standardize_display_standardization = self.params[('standardize_display_standardization',True,'Outputs statistical values before and after standardization')]
        """Outputs statistical values before and after standardization"""

        self.nr_of_gaussians = nr_of_gaussians
        self.gaussian_stds = gaussian_stds

        self.computed_weights = None
        """stores the computed weights if desired"""

        self.computed_pre_weights = None
        """stores the computed pre weights if desired"""

        self.current_penalty = None
        """to stores the current penalty (for example OMT) after running through the model"""

        self.deep_network_local_weight_smoothing = self.params[('deep_network_local_weight_smoothing', 0.02, 'How much to smooth the local weights (implemented by smoothing the resulting velocity field) to assure sufficient regularity')]
        """Smoothing of the local weight fields to assure sufficient regularity of the resulting velocity"""

        self.deep_network_weight_smoother = None
        """The smoother that does the smoothing of the weights; needs to be initialized in the forward model"""


    def _display_stats_before_after(self, Ib, Ia, iname):

        Ib_min = Ib.min().data.cpu().numpy()[0]
        Ib_max = Ib.max().data.cpu().numpy()[0]
        Ib_mean = Ib.mean().data.cpu().numpy()[0]
        Ib_std = Ib.std().data.cpu().numpy()[0]

        Ia_min = Ia.min().data.cpu().numpy()[0]
        Ia_max = Ia.max().data.cpu().numpy()[0]
        Ia_mean = Ia.mean().data.cpu().numpy()[0]
        Ia_std = Ia.std().data.cpu().numpy()[0]

        print('{}: before: [{:.2f},{:.2f},{:.2f}]({:.2f}); after: [{:.2f},{:.2f},{:.2f}]({:.2f})'.format(iname, Ib_min,Ib_mean,Ib_max,Ib_std,Ia_min,Ia_mean,Ia_max,Ia_std))

    def _standardize_input_if_necessary(self, I, momentum, I0, I1):

        sI = None
        sM = None
        sI0 = None
        sI1 = None

        # first standardize the input images

        if self.standardize_input_images:

            if not np.isclose(self.standardize_divide_input_images,1.0):
                sI = (I - self.standardize_subtract_from_input_images)/self.standardize_divide_input_images
                if self.use_source_image_as_input:
                    sI0 =  (I0 - self.standardize_subtract_from_input_images)/self.standardize_divide_input_images
                if self.use_target_image_as_input:
                    sI1 = (I1 - self.standardize_subtract_from_input_images)/self.standardize_divide_input_images
            else:
                sI = I - self.standardize_subtract_from_input_images
                if self.use_source_image_as_input:
                    sI0 = I0 - self.standardize_subtract_from_input_images
                if self.use_target_image_as_input:
                    sI1 = I1 - self.standardize_subtract_from_input_images

            if self.standardize_display_standardization:
                self._display_stats_before_after(I,sI,'I')
                if self.use_source_image_as_input:
                    self._display_stats_before_after(I0, sI0, 'I0')
                if self.use_target_image_as_input:
                    self._display_stats_before_after(I1, sI1, 'I1')

        else: # if we do not standardize the input images, just do the I/O mapping

            sI = I
            if self.use_source_image_as_input:
                sI0 = I0
            if self.use_target_image_as_input:
                sI1 = I1

        # now standardize the input momentum

        if self.use_momentum_as_input:
            if self.standardize_input_momentum:
                if not np.isclose(self.standardize_divide_input_momentum, 1.0):
                    sM = (momentum - self.standardize_subtract_from_input_momentum)/self.standardize_divide_input_momentum
                else:
                    sM = momentum - self.standardize_subtract_from_input_momentum

                if self.standardize_display_standardization:
                    self._display_stats_before_after(momentum,sM,'m')

            else:
                sM = momentum

        return sI, sM, sI0, sI1


    def get_omt_weight_penalty(self):
        return self.omt_weight_penalty

    def get_omt_power(self):
        return self.omt_power

    def _initialize_weights(self):

        print('WARNING: weight initialization DISABLED; using pyTorch default network initialization, which is probably good enough (famous last words).')

        # self.weight_initalization_constant = self.params[('weight_initialization_constant', 0.01, 'Weights are initialized via normal_(0, math.sqrt(weight_initialization_constant / n))')]
        # print('WARNING: weight initialization ENABLED')
        #
        # for m in self.modules():
        #     if isinstance(m, DimConv(self.dim)):
        #         n = m.out_channels
        #         for d in range(self.dim):
        #             n *= m.kernel_size[d]
        #         m.weight.data.normal_(0, math.sqrt(self.weight_initalization_constant / n))
        #     elif isinstance(m, DimBatchNorm(self.dim)):
        #         pass
        #     elif isinstance(m, nn.Linear):
        #         pass

    def get_number_of_image_channels_from_state_dict(self, state_dict, dim):
        """legacy; to support velocity fields as input channels"""
        return self.nr_of_image_channels

    def get_number_of_input_channels(self, nr_of_image_channels, dim):
        """
        legacy; to support velocity fields as input channels
        currently only returns the number of image channels, but if something else would be used as
        the network input, would need to return the total number of inputs
        """
        return self.nr_of_image_channels

    def get_computed_weights(self):
        return self.computed_weights

    def get_computed_pre_weights(self):
        return self.computed_pre_weights

    def compute_diffusion(self, d):
        # just do the standard component-wise Euclidean squared norm of the gradient

        if self.dim == 1:
            return self._compute_diffusion_1d(d)
        elif self.dim == 2:
            return self._compute_diffusion_2d(d)
        elif self.dim == 3:
            return self._compute_diffusion_3d(d)
        else:
            raise ValueError('Diffusion computation is currently only supported in dimensions 1 to 3')

    def _compute_diffusion_1d(self, d):

        # need to use torch.abs here to make sure the proper subgradient is computed at zero
        batch_size = d.size()[0]
        t0 = (self.fdt.dXc(d))**2

        return (t0).sum() * self.volumeElement / batch_size

    def _compute_diffusion_2d(self, d):

        # need to use torch.norm here to make sure the proper subgradient is computed at zero
        batch_size = d.size()[0]
        t0 = self.fdt.dXc(d)**2+self.fdt.dYc(d)**2

        return t0.sum() * self.volumeElement / batch_size

    def _compute_diffusion_3d(self, d):

        # need to use torch.norm here to make sure the proper subgradient is computed at zero
        batch_size = d.size()[0]
        t0 = self.fdt.dXc(d)**2 + self.fdt.dYc(d)**2 + self.fdt.dZc(d)**2

        return t0.sum() * self.volumeElement / batch_size


    def compute_total_variation(self, d):
        # just do the standard component-wise Euclidean norm of the gradient

        if self.dim == 1:
            return self._compute_total_variation_1d(d)
        elif self.dim == 2:
            return self._compute_total_variation_2d(d)
        elif self.dim == 3:
            return self._compute_total_variation_3d(d)
        else:
            raise ValueError('Total variation computation is currently only supported in dimensions 1 to 3')

    def _compute_total_variation_1d(self, d):

        # need to use torch.abs here to make sure the proper subgradient is computed at zero
        batch_size = d.size()[0]
        t0 = torch.abs(self.fdt.dXc(d))

        return (t0).sum()*self.volumeElement/batch_size

    def _compute_total_variation_2d(self, d):

        # need to use torch.norm here to make sure the proper subgradient is computed at zero
        batch_size = d.size()[0]
        t0 = torch.norm(torch.stack((self.fdt.dXc(d),self.fdt.dYc(d))),self.pnorm,0)

        return t0.sum()*self.volumeElement/batch_size

    def _compute_total_variation_3d(self, d):

        # need to use torch.norm here to make sure the proper subgradient is computed at zero
        batch_size = d.size()[0]
        t0 = torch.norm(torch.stack((self.fdt.dXc(d),
                                     self.fdt.dYc(d),
                                     self.fdt.dZc(d))), self.pnorm, 0)

        return t0.sum()*self.volumeElement/batch_size

    def spatially_average(self, x):
        """
        does spatial averaging of a 2D image with potentially multiple batches: format B x X x Y
        :param x:
        :return:
        """

        # first set the boundary to zero (first dimension is batch and this only works for 2D for now)

        y = torch.zeros_like(x)

        # now do local averaging in the interior using the four neighborhood
        y[:, 1:-1, 1:-1] = 0.5 * x[:, 1:-1, 1:-1] + 0.125 * (
                    x[:, 0:-2, 1:-1] + x[:, 2:, 1:-1] + x[:, 1:-1, 0:-2] + x[:, 1:-1, 2:])

        # do the corners
        y[:, 0, 0] = 8. / 6. * (0.5 * x[:, 0, 0] + 0.125 * (x[:, 1, 0] + x[:, 0, 1]))
        y[:, 0, -1] = 8. / 6. * (0.5 * x[:, 0, -1] + 0.125 * (x[:, 1, -1] + x[:, 0, -2]))
        y[:, -1, 0] = 8. / 6. * (0.5 * x[:, -1, 0] + 0.125 * (x[:, -2, 0] + x[:, -1, 1]))
        y[:, -1, -1] = 8. / 6. * (0.5 * x[:, -1, -1] + 0.125 * (x[:, -2, -1] + x[:, -1, -2]))

        # and lastly the edges
        y[:, 1:-1, 0] = 8. / 7. * (0.5 * x[:, 1:-1, 0] + 0.125 * (x[:, 1:-1, 1] + x[:, 0:-2, 0] + x[:, 2:, 0]))
        y[:, 0, 1:-1] = 8. / 7. * (0.5 * x[:, 0, 1:-1] + 0.125 * (x[:, 1, 1:-1] + x[:, 0, 0:-2] + x[:, 0, 2:]))
        y[:, 1:-1, -1] = 8. / 7. * (0.5 * x[:, 1:-1, -1] + 0.125 * (x[:, 1:-1, -2] + x[:, 0:-2, -1] + x[:, 2:, -1]))
        y[:, -1, 1:-1] = 8. / 7. * (0.5 * x[:, -1, 1:-1] + 0.125 * (x[:, -2, 1:-1] + x[:, -1, 0:-2] + x[:, -1, 2:]))

        return y

    def get_current_penalty(self):
        """
        returns the current penalty for the weights (OMT penalty here)
        :return:
        """
        return self.current_penalty


class encoder_block_2d(nn.Module):
    def __init__(self, input_feature, output_feature, use_dropout, use_batch_normalization):
        super(encoder_block_2d, self).__init__()

        if use_batch_normalization:
            conv_bias = False
        else:
            conv_bias = True

        self.conv_input = DimConv(self.dim)(in_channels=input_feature, out_channels=output_feature,
                                    kernel_size=3, stride=1, padding=1, dilation=1,bias=conv_bias)
        self.conv_inblock1 = DimConv(self.dim)(in_channels=output_feature, out_channels=output_feature,
                                       kernel_size=3, stride=1, padding=1, dilation=1,bias=conv_bias)
        self.conv_inblock2 = DimConv(self.dim)(in_channels=output_feature, out_channels=output_feature,
                                       kernel_size=3, stride=1, padding=1, dilation=1,bias=conv_bias)
        self.conv_pooling = DimConv(self.dim)(in_channels=output_feature, out_channels=output_feature,
                                      kernel_size=2, stride=2, padding=0, dilation=1,bias=conv_bias)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()

        if use_batch_normalization:
            self.bn_1 = DimBatchNorm(self.dim)(output_feature,momentum=batch_norm_momentum_val)
            self.bn_2 = DimBatchNorm(self.dim)(output_feature,momentum=batch_norm_momentum_val)
            self.bn_3 = DimBatchNorm(self.dim)(output_feature,momentum=batch_norm_momentum_val)
            self.bn_4 = DimBatchNorm(self.dim)(output_feature,momentum=batch_norm_momentum_val)

        self.use_dropout = use_dropout
        self.use_batch_normalization = use_batch_normalization
        self.dropout = nn.Dropout(0.2)

    def apply_dropout(self, input):
        if self.use_dropout:
            return self.dropout(input)
        else:
            return input

    def forward_with_batch_normalization(self,x):
        output = self.conv_input(x)
        output = self.apply_dropout(self.prelu1(self.bn_1(output)))
        output = self.apply_dropout(self.prelu2(self.bn_2(self.conv_inblock1(output))))
        output = self.apply_dropout(self.prelu3(self.bn_3(self.conv_inblock2(output))))
        return self.prelu4(self.bn_4(self.conv_pooling(output)))

    def forward_without_batch_normalization(self,x):
        output = self.conv_input(x)
        output = self.apply_dropout(self.prelu1(output))
        output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)))
        output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)))
        return self.prelu4(self.conv_pooling(output))

    def forward(self, x):
        if self.use_batch_normalization:
            return self.forward_with_batch_normalization(x)
        else:
            return self.forward_without_batch_normalization(x)

class decoder_block_2d(nn.Module):
    def __init__(self, input_feature, output_feature, pooling_filter,use_dropout, use_batch_normalization,last_block=False):
        super(decoder_block_2d, self).__init__()
        # todo: check padding here, not sure if it is the right thing to do

        if use_batch_normalization:
            conv_bias = False
        else:
            conv_bias = True

        self.conv_unpooling = DimConvTranspose(self.dim)(in_channels=input_feature, out_channels=input_feature,
                                                 kernel_size=pooling_filter, stride=2, padding=0,output_padding=0,bias=conv_bias)
        self.conv_inblock1 = DimConv(self.dim)(in_channels=input_feature, out_channels=input_feature,
                                       kernel_size=3, stride=1, padding=1, dilation=1,bias=conv_bias)
        self.conv_inblock2 = DimConv(self.dim)(in_channels=input_feature, out_channels=input_feature,
                                       kernel_size=3, stride=1, padding=1, dilation=1,bias=conv_bias)
        if last_block:
            self.conv_output = DimConv(self.dim)(in_channels=input_feature, out_channels=output_feature,
                                         kernel_size=3, stride=1, padding=1, dilation=1)
        else:
            self.conv_output = DimConv(self.dim)(in_channels=input_feature, out_channels=output_feature,
                                         kernel_size=3, stride=1, padding=1, dilation=1,bias=conv_bias)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()

        if use_batch_normalization:
            self.bn_1 = DimBatchNorm(self.dim)(input_feature,momentum=batch_norm_momentum_val)
            self.bn_2 = DimBatchNorm(self.dim)(input_feature,momentum=batch_norm_momentum_val)
            self.bn_3 = DimBatchNorm(self.dim)(input_feature,momentum=batch_norm_momentum_val)
            if not last_block:
                self.bn_4 = DimBatchNorm(self.dim)(output_feature,momentum=batch_norm_momentum_val)

        self.use_dropout = use_dropout
        self.use_batch_normalization = use_batch_normalization
        self.last_block = last_block
        self.dropout = nn.Dropout(0.2)
        self.output_feature = output_feature

    def apply_dropout(self, input):
        if self.use_dropout:
            return self.dropout(input)
        else:
            return input

    def forward_with_batch_normalization(self,x):
        output = self.prelu1(self.bn_1(self.conv_unpooling(x)))
        output = self.apply_dropout(self.prelu2(self.bn_2(self.conv_inblock1(output))))
        output = self.apply_dropout(self.prelu3(self.bn_3(self.conv_inblock2(output))))
        if self.last_block:  # generates final output
            return self.conv_output(output);
        else:  # generates intermediate results
            return self.apply_dropout(self.prelu4(self.bn_4(self.conv_output(output))))

    def forward_without_batch_normalization(self,x):
        output = self.prelu1(self.conv_unpooling(x))
        output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)))
        output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)))
        if self.last_block:  # generates final output
            return self.conv_output(output);
        else:  # generates intermediate results
            return self.apply_dropout(self.prelu4(self.conv_output(output)))

    def forward(self, x):
        if self.use_batch_normalization:
            return self.forward_with_batch_normalization(x)
        else:
            return self.forward_without_batch_normalization(x)

class EncoderDecoderSmoothingModel(DeepSmoothingModel):
    """
    Similar to the model used in Quicksilver
    """
    def __init__(self, nr_of_gaussians, gaussian_stds, dim, spacing, nr_of_image_channels=1, omt_power=1.0, params=None ):
        super(EncoderDecoderSmoothingModel, self).__init__(nr_of_gaussians=nr_of_gaussians,\
                                                                     gaussian_stds=gaussian_stds,\
                                                                     dim=dim,\
                                                                     spacing=spacing,\
                                                                     nr_of_image_channels=nr_of_image_channels,\
                                                                     omt_power=omt_power,\
                                                                     params=params)

        # is 64 in quicksilver paper
        feature_num = self.params[('number_of_features_in_first_layer', 32, 'number of features in the first encoder layer (64 in quicksilver)')]
        use_dropout= self.params[('use_dropout',False,'use dropout for the layers')]
        self.use_separate_decoders_per_gaussian = self.params[('use_separate_decoders_per_gaussian',True,'if set to true separte decoder branches are used for each Gaussian')]
        self.use_momentum_as_input = self.params[('use_momentum_as_input',True,'If true, uses the image and the momentum as input')]
        use_batch_normalization = self.params[('use_batch_normalization',True,'If true, uses batch normalization between layers')]
        self.use_one_encoder_decoder_block = self.params[('use_one_encoder_decoder_block',True,'If False, using two each as in the quicksilver paper')]
        self.estimate_around_global_weights = self.params[('estimate_around_global_weights', True, 'If true, a weighted softmax is used so the default output (for input zero) are the global weights')]

        self.use_source_image_as_input = self.params[('use_source_image_as_input',True,'If true, uses the source image as additional input')]
        self.use_target_image_as_input = self.params[('use_target_image_as_input', True, 'If true, uses the target image as additional input')]

        if self.use_momentum_as_input or self.use_target_image_as_input or self.use_source_image_as_input:
            self.encoder_1 = encoder_block_2d(self.get_number_of_input_channels(nr_of_image_channels,dim), feature_num,
                                              use_dropout, use_batch_normalization)
        else:
            self.encoder_1 = encoder_block_2d(1, feature_num, use_dropout, use_batch_normalization)

        if not self.use_one_encoder_decoder_block:
            self.encoder_2 = encoder_block_2d(feature_num, feature_num * 2, use_dropout, use_batch_normalization)

        # todo: maybe have one decoder for each Gaussian here.
        # todo: the current version seems to produce strange gridded results

        if self.use_separate_decoders_per_gaussian:
            if not self.use_one_encoder_decoder_block:
                self.decoder_1 = nn.ModuleList()
            self.decoder_2 = nn.ModuleList()
            # create two decoder blocks for each Gaussian
            for g in range(nr_of_gaussians):
                if not self.use_one_encoder_decoder_block:
                    self.decoder_1.append( decoder_block_2d(feature_num * 2, feature_num, 2, use_dropout, use_batch_normalization) )
                self.decoder_2.append( decoder_block_2d(feature_num, 1, 2, use_dropout, use_batch_normalization, last_block=True) )
        else:
            if not self.use_one_encoder_decoder_block:
                self.decoder_1 = decoder_block_2d(feature_num * 2, feature_num, 2, use_dropout, use_batch_normalization)
            self.decoder_2 = decoder_block_2d(feature_num, nr_of_gaussians, 2, use_dropout, use_batch_normalization, last_block=True)  # 3?

        self._initialize_weights()

    def get_number_of_input_channels(self, nr_of_image_channels, dim):
        """
        legacy; to support velocity fields as input channels
        currently only returns the number of image channels, but if something else would be used as
        the network input, would need to return the total number of inputs
        """
        add_channels = 0
        if self.use_momentum_as_input:
            add_channels+=dim
        if self.use_source_image_as_input:
            add_channels+=1
        if self.use_target_image_as_input:
            add_channels+=1

        return self.nr_of_image_channels + add_channels

    def forward(self, I, additional_inputs, global_multi_gaussian_weights, gaussian_fourier_filter_generator,
                encourage_spatial_weight_consistency=True, retain_weights=False):

        # format of multi_smooth_v is multi_v x batch x channels x X x Y
        # (channels here are the vector field components)
        # I is the image, m is the momentum. multi_smooth_v is the momentum smoothed with the various kernels

        """
        First make sure that the multi_smooth_v has the correct dimension.
        I.e., the correct spatial dimension and one output for each Gaussian (multi_v)
        """

        # get the size of the input momentum batch x channels x X x Y
        momentum = additional_inputs['m']
        sz_m = momentum.size()
        # get the size of the multi-velocity field; multi_v x batch x channels x X x Y
        sz_mv = [self.nr_of_gaussians] + list(sz_m)

        dim_mv = sz_mv[2]  # format is
        # currently only implemented in 2D
        assert (dim_mv == 2)
        assert (sz_mv[0] == self.nr_of_gaussians)

        # create the output tensor: will be of dimension: batch x channels x X x Y
        ret = AdaptVal(Variable(MyTensor(*sz_mv[1:]), requires_grad=False))

        # now determine the size for the weights
        # Since the smoothing will be the same for all spatial directions (for a velocity field),
        # this basically amounts to cutting out the channels; i.e., multi_v x batch x X x Y
        sz_weight = list(sz_mv)
        sz_weight = [sz_weight[1]] + [sz_weight[0]] + sz_weight[3:]

        # if the weights should be stored (for debugging), create the tensor to store them here
        if retain_weights:
            if self.computed_weights is None:
                print('DEBUG: retaining smoother weights - turn off to minimize memory consumption')
                # create storage; batch x size v x X x Y
                self.computed_weights = MyTensor(*sz_weight)

            if self.deep_network_local_weight_smoothing > 0:
                if self.computed_pre_weights is None:
                    print('DEBUG: retaining smoother pre weights - turn off to minimize memory consumption')
                    self.computed_pre_weights = MyTensor(*sz_weight)

        # here is the actual network; maybe abstract this out later

        sI,sM,sI0,sI1 = self._standardize_input_if_necessary(I,momentum,additional_inputs['I0'],additional_inputs['I1'])

        # network input
        x = sI
        if self.use_momentum_as_input:
            x = torch.cat([x, sM], dim=1)
        if self.use_source_image_as_input:
            x = torch.cat([x, sI0], dim=1)
        if self.use_target_image_as_input:
            x = torch.cat([x, sI1], dim=1)

        if self.use_one_encoder_decoder_block:
            encoder_output = self.encoder_1(x)
        else:
            encoder_output = self.encoder_2(self.encoder_1(x))

        if self.use_one_encoder_decoder_block:
            if self.use_separate_decoders_per_gaussian:
                # here we have separate decoder outputs for the different Gaussians
                # should give it more flexibility
                decoder_output_individual = []
                for g in range(self.nr_of_gaussians):
                    decoder_output_individual.append(self.decoder_2[g]((encoder_output)))
                decoder_output = torch.cat(decoder_output_individual, dim=1);
            else:
                decoder_output = self.decoder_2(encoder_output)
        else:
            if self.use_separate_decoders_per_gaussian:
                # here we have separate decoder outputs for the different Gaussians
                # should give it more flexibility
                decoder_output_individual = []
                for g in range(self.nr_of_gaussians):
                    decoder_output_individual.append( self.decoder_2[g](self.decoder_1[g](encoder_output)) )
                decoder_output = torch.cat(decoder_output_individual, dim=1);
            else:
                decoder_output = self.decoder_2(self.decoder_1(encoder_output))

        # now we are ready for the softmax
        if self.estimate_around_global_weights:
            weights = weighted_softmax(decoder_output, dim=1, weights=global_multi_gaussian_weights)
        else:
            weights = F.softmax(decoder_output, dim=1)

        # compute the total variation penalty
        total_variation_penalty = Variable(MyTensor(1).zero_(), requires_grad=False)
        if self.total_variation_weight_penalty > 0:
            # first compute the edge map
            g_I = compute_localized_edge_penalty(I[:, 0, ...], self.spacing, self.params)
            batch_size = I.size()[0]
            for g in range(self.nr_of_gaussians):
                # total_variation_penalty += self.compute_total_variation(weights[:,g,...])
                c_local_norm_grad = _compute_local_norm_of_gradient(weights[:, g, ...], self.spacing, self.pnorm)
                total_variation_penalty += (g_I * c_local_norm_grad).sum() * self.volumeElement / batch_size

        diffusion_penalty = Variable(MyTensor(1).zero_(), requires_grad=False)
        if self.diffusion_weight_penalty > 0:
            for g in range(self.nr_of_gaussians):
                diffusion_penalty += self.compute_diffusion(weights[:, g, ...])

        # enforce minimum weight for numerical reasons
        weights = _project_weights_to_min(weights, self.gaussianWeight_min)

        # instantiate the extra smoother if weight is larger than 0 and it has not been initialized yet
        if self.deep_network_local_weight_smoothing > 0 and self.deep_network_weight_smoother is None:
            import pyreg.smoother_factory as sf
            s_m_params = pars.ParameterDict()
            s_m_params['smoother']['type'] = 'gaussian'
            s_m_params['smoother']['gaussian_std'] = self.deep_network_local_weight_smoothing
            self.deep_network_weight_smoother = sf.SmootherFactory(ret.size()[2::], self.spacing).create_smoother(
                s_m_params)

        if self.deep_network_local_weight_smoothing > 0:  # and retain_weights:
            # now we smooth all the weights
            if retain_weights:
                # todo: change visualization to work with this new format:
                # B x weights x X x Y instead of weights x B x X x Y
                self.computed_pre_weights[:] = weights.data

            weights = self.deep_network_weight_smoother.smooth(weights)
            # make sure they are all still positive (#todo: may not be necessary, since we set a minumum weight above now)
            weights = torch.clamp(weights, 0.0, 1.0)

        # ends here

        # multiply the velocity fields by the weights and sum over them
        # this is then the multi-Gaussian output

        if self.weighting_type=='sqrt_w_K_sqrt_w':
            sqrt_weights = torch.sqrt(weights)
            sqrt_weighted_multi_smooth_v = compute_weighted_multi_smooth_v( momentum=momentum, weights=sqrt_weights, gaussian_stds=self.gaussian_stds,
                                                                   gaussian_fourier_filter_generator=gaussian_fourier_filter_generator )
        elif self.weighting_type=='w_K_w':
            # now create the weighted multi-smooth-v
            weighted_multi_smooth_v = compute_weighted_multi_smooth_v( momentum=momentum, weights=weights, gaussian_stds=self.gaussian_stds,
                                                                       gaussian_fourier_filter_generator=gaussian_fourier_filter_generator )
        elif self.weighting_type=='w_K':
            multi_smooth_v = ce.fourier_set_of_gaussian_convolutions(momentum,
                                                                     gaussian_fourier_filter_generator=gaussian_fourier_filter_generator,
                                                                     sigma=Variable(torch.from_numpy(self.gaussian_stds),requires_grad=False),
                                                                     compute_std_gradients=False)
        else:
            raise ValueError('Unknown weighting_type: {}'.format(self.weighting_type))

        # now we apply this weight across all the channels; weight output is B x weights x X x Y
        for n in range(self.dim):
            # reverse the order so that for a given channel we have batch x multi_velocity x X x Y
            # i.e., the multi-velocity field output is treated as a channel
            # reminder: # format of multi_smooth_v is multi_v x batch x channels x X x Y
            # (channels here are the vector field components); i.e. as many as there are dimensions
            # each one of those should be smoothed the same

            # let's smooth this on the fly, as the smoothing will be of form
            # w_i*K_i*(w_i m)

            if self.weighting_type == 'sqrt_w_K_sqrt_w':
                # roc should be: batch x multi_v x X x Y
                roc = torch.transpose(sqrt_weighted_multi_smooth_v[:, :, n, ...], 0, 1)
                yc = torch.sum(roc * sqrt_weights, dim=1)
            elif self.weighting_type == 'w_K_w':
                # roc should be: batch x multi_v x X x Y
                roc = torch.transpose(weighted_multi_smooth_v[:, :, n, ...], 0, 1)
                yc = torch.sum(roc * weights, dim=1)
            elif self.weighting_type == 'w_K':
                # roc should be: batch x multi_v x X x Y
                roc = torch.transpose(multi_smooth_v[:, :, n, ...], 0, 1)
                yc = torch.sum(roc * weights, dim=1)
            else:
                raise ValueError('Unknown weighting_type: {}'.format(self.weighting_type))

            ret[:, n, ...] = yc  # ret is: batch x channels x X x Y

        #if self.deep_network_local_weight_smoothing>0 and not retain_weights:
        #    # this is the actual optimization, just smooth the resulting velocity field which is more efficicient
        # (no longer possible, as we now have the weights twice)
        #    # than smoothing the individual weights
        #    # if weights are retained then instead the weights are smoothed, see above (as this is what we were after)
        #    ret = self.deep_network_weight_smoother.smooth(ret)

        current_omt_penalty = self.omt_weight_penalty * compute_omt_penalty(weights, self.gaussian_stds,
                                                                        self.volumeElement, self.omt_power, self.omt_use_log_transformed_std)
        self.current_penalty = current_omt_penalty
        if self.total_variation_weight_penalty > 0:
            current_tv_penalty = self.total_variation_weight_penalty * total_variation_penalty
            print('TV_penalty = ' + str(current_tv_penalty.data.cpu().numpy()) + \
                  'OMT_penalty = ' + str(current_omt_penalty.data.cpu().numpy()))
            self.current_penalty += current_tv_penalty

        if retain_weights:
            # todo: change visualization to work with this new format:
            # B x weights x X x Y instead of weights x B x X x Y
            self.computed_weights[:] = weights.data

        return ret

def compute_weighted_multi_smooth_v(momentum, weights, gaussian_stds, gaussian_fourier_filter_generator):
    # computes the weighted smoothed velocity field i.e., K_i*( w_i m ) for all i in one data structure
    # dimension will be multi_v x batch x X x Y x ...

    sz_m = momentum.size()
    sz_mv = [len(gaussian_stds)] + list(sz_m)
    dim = sz_m[1]
    weighted_multi_smooth_v = AdaptVal(Variable(MyTensor(*sz_mv), requires_grad=False))
    # and fill it with weighted smoothed velocity fields
    for i,g in enumerate(gaussian_stds):
        weighted_momentum_i = torch.zeros_like(momentum)
        for c in range(dim):
            weighted_momentum_i[:,c,...] = weights[:,i,...]*momentum[:,c,...]
        current_g = Variable(torch.from_numpy(np.array([g])),requires_grad=False)
        current_weighted_smoothed_v_i = ce.fourier_set_of_gaussian_convolutions(weighted_momentum_i, gaussian_fourier_filter_generator,current_g, compute_std_gradients=False)
        weighted_multi_smooth_v[i,...] = current_weighted_smoothed_v_i[0,...]

    return weighted_multi_smooth_v


def _project_weights_to_min(weights,min_val):
    clamped_weights = torch.clamp(weights,min_val,1.0)
    clamped_weight_sum = torch.sum(clamped_weights, dim=1)
    sz = clamped_weights.size()
    nr_of_weights = sz[1]
    projected_weights = torch.zeros_like(clamped_weights)
    for n in range(nr_of_weights):
        projected_weights[:,n,...] = clamped_weights[:,n,...]/clamped_weight_sum

    #plt.clf()
    #plt.subplot(1,2,1)
    #plt.imshow(weights[0,nr_of_weights-1,...].data.cpu().numpy())
    #plt.colorbar()
    #plt.subplot(1, 2, 2)
    #plt.imshow(projected_weights[0, nr_of_weights - 1, ...].data.cpu().numpy())
    #plt.colorbar()
    #plt.show()


    return clamped_weights

class SimpleConsistentWeightedSmoothingModel(DeepSmoothingModel):
    """
    Mini neural network which takes as an input a set of smoothed velocity field as
    well as input images and predicts weights for a multi-Gaussian smoothing from this
    Enforces the same weighting for all the dimensions of the vector field to be smoothed

    """
    def __init__(self, nr_of_gaussians, gaussian_stds, dim, spacing, nr_of_image_channels=1, omt_power=1.0, params=None ):
        super(SimpleConsistentWeightedSmoothingModel, self).__init__(nr_of_gaussians=nr_of_gaussians,\
                                                                     gaussian_stds=gaussian_stds,\
                                                                     dim=dim,\
                                                                     spacing=spacing,\
                                                                     nr_of_image_channels=nr_of_image_channels,\
                                                                     omt_power=omt_power,
                                                                     params=params)

        self.kernel_sizes = self.params[('kernel_sizes',[7,7],'size of the convolution kernels')]
        self.use_momentum_as_input = self.params[('use_momentum_as_input',True,'If true, uses the momentum as an additional input')]
        self.use_source_image_as_input = self.params[('use_source_image_as_input',True,'If true, uses the target image as additional input')]
        self.use_target_image_as_input = self.params[('use_target_image_as_input',True,'If true, uses the target image as additional input')]
        self.estimate_around_global_weights = self.params[('estimate_around_global_weights', True, 'If true, a weighted softmax is used so the default output (for input zero) are the global weights')]
        self.use_batch_normalization = self.params[('use_batch_normalization', True, 'If true, uses batch normalization between layers')]

        # check that all the kernel-size are odd
        for ks in self.kernel_sizes:
            if ks%2== 0:
                raise ValueError('Kernel sizes need to be odd')

        # the last layers feature number is not specified as it will simply be the number of Gaussians
        self.nr_of_features_per_layer = self.params[('number_of_features_per_layer',[20],'Number of features for the convolution later; last one is set to number of Gaussians')]
        # add the number of Gaussians to the last layer
        self.nr_of_features_per_layer = self.nr_of_features_per_layer + [nr_of_gaussians]

        self.nr_of_layers = len(self.kernel_sizes)
        assert( self.nr_of_layers== len(self.nr_of_features_per_layer) )

        self.use_relu = self.params[('use_relu',True,'if set to True uses Relu, otherwise sigmoid')]

        self.conv_layers = None
        self.batch_normalizations = None

        # needs to be initialized here, otherwise the optimizer won't see the modules from ModuleList
        # todo: figure out how to do ModuleList initialization not in __init__
        # todo: this would allow removing dim and nr_of_image_channels from interface
        # todo: because it could be compute on the fly when forward is executed
        self._init(self.nr_of_image_channels,dim=self.dim)


    def get_number_of_input_channels(self, nr_of_image_channels, dim):
        """
        legacy; to support velocity fields as input channels
        currently only returns the number of image channels, but if something else would be used as
        the network input, would need to return the total number of inputs
        """
        add_channels = 0
        if self.use_momentum_as_input:
            add_channels+=dim
        if self.use_target_image_as_input:
            add_channels+=1
        if self.use_source_image_as_input:
            add_channels+=1

        return self.nr_of_image_channels + add_channels

    def _init(self,nr_of_image_channels,dim):
        """
        Initalizes all the conv layers
        :param nr_of_image_channels:
        :param dim:
        :return:
        """

        assert(self.nr_of_layers>0)

        nr_of_input_channels = self.get_number_of_input_channels(nr_of_image_channels,dim)

        convs = [None]*self.nr_of_layers

        # first layer
        convs[0] = DimConv(self.dim)(nr_of_input_channels,
                                  self.nr_of_features_per_layer[0],
                                  self.kernel_sizes[0], padding=(self.kernel_sizes[0]-1)//2)

        # all the intermediate layers and the last one
        for l in range(self.nr_of_layers-1):
            convs[l+1] = DimConv(self.dim)(self.nr_of_features_per_layer[l],
                                        self.nr_of_features_per_layer[l+1],
                                        self.kernel_sizes[l+1],
                                        padding=(self.kernel_sizes[l+1]-1)//2)

        self.conv_layers = nn.ModuleList()
        for c in convs:
            self.conv_layers.append(c)

        if self.use_batch_normalization:

            batch_normalizations = [None]*(self.nr_of_layers-1) # not on the last layer
            for b in range(self.nr_of_layers-1):
                batch_normalizations[b] = DimBatchNorm(self.dim)(self.nr_of_features_per_layer[b],momentum=batch_norm_momentum_val)

            self.batch_normalizations = nn.ModuleList()
            for b in batch_normalizations:
                self.batch_normalizations.append(b)

        self._initialize_weights()

    def forward(self, I, additional_inputs, global_multi_gaussian_weights, gaussian_fourier_filter_generator,
                encourage_spatial_weight_consistency=True, retain_weights=False):

        # format of multi_smooth_v is multi_v x batch x channels x X x Y
        # (channels here are the vector field components)
        # I is the image, m is the momentum. multi_smooth_v is the momentum smoothed with the various kernels

        """
        First make sure that the multi_smooth_v has the correct dimension.
        I.e., the correct spatial dimension and one output for each Gaussian (multi_v)
        """

        # get the size of the input momentum batch x channels x X x Y
        momentum = additional_inputs['m']
        sz_m = momentum.size()
        # get the size of the multi-velocity field; multi_v x batch x channels x X x Y
        sz_mv = [self.nr_of_gaussians] + list(sz_m)

        # create the output tensor: will be of dimension: batch x channels x X x Y
        ret = AdaptVal(Variable(MyTensor(*sz_m), requires_grad=False))

        # now determine the size for the weights
        # Since the smoothing will be the same for all spatial directions (for a velocity field),
        # this basically amounts to cutting out the channels; i.e., multi_v x batch x X x Y
        sz_weight = list(sz_mv)
        sz_weight = [sz_weight[1]] + [sz_weight[0]] + sz_weight[3:]

        # if the weights should be stored (for debugging), create the tensor to store them here
        if retain_weights:
            if self.computed_weights is None:
                print('DEBUG: retaining smoother weights - turn off to minimize memory consumption')
                # create storage; batch x size v x X x Y
                self.computed_weights = MyTensor(*sz_weight)

            if self.deep_network_local_weight_smoothing > 0:
                if self.computed_pre_weights is None:
                    print('DEBUG: retaining smoother pre weights - turn off to minimize memory consumption')
                    self.computed_pre_weights = MyTensor(*sz_weight)

        sI, sM, sI0, sI1 = self._standardize_input_if_necessary(I, momentum, additional_inputs['I0'],additional_inputs['I1'])

        # network input
        x = sI
        if self.use_momentum_as_input:
            x = torch.cat([x, sM], dim=1)
        if self.use_source_image_as_input:
            x = torch.cat([x, sI0], dim=1)
        if self.use_target_image_as_input:
            x = torch.cat([x, sI1], dim=1)

        # now let's apply all the convolution layers, until the last
        # (because the last one is not relu-ed

        if self.batch_normalizations:
            for l in range(len(self.conv_layers) - 1):
                if self.use_relu:
                    x = F.relu(self.batch_normalizations[l](self.conv_layers[l](x)))
                else:
                    x = F.sigmoid(self.batch_normalizations[l](self.conv_layers[l](x)))
        else:
            for l in range(len(self.conv_layers) - 1):
                if self.use_relu:
                    x = F.relu(self.conv_layers[l](x))
                else:
                    x = F.sigmoid(self.conv_layers[l](x))

        # and now apply the last one without an activation for now
        # because we want to have the ability to smooth *before* the softmax
        # this is similar to smoothing in the logit domain for an active mean field approach
        x = self.conv_layers[-1](x)

        # now do the smoothing if desired
        y = torch.zeros_like(x)

        if encourage_spatial_weight_consistency:
            # now we do local averaging for the weights and force the boundaries to zero
            for g in range(self.nr_of_gaussians):
                y[:, g, ...] = self.spatially_average(x[:, g, ...])
        else:
            y = x

        # now we are ready for the weighted softmax (will be like softmax if no weights are specified)

        if self.estimate_around_global_weights:
            weights = weighted_softmax(y, dim=1, weights=global_multi_gaussian_weights)
            #plt.clf()
            #for sf in range(4):
            #    plt.subplot(2,2,sf+1)
            #    plt.imshow(weights[0,sf,...].data.cpu().numpy())
            #    plt.colorbar()
            #plt.show()
        else:
            weights = F.softmax(y, dim=1)

        # compute the total variation penalty; compute this on the pre (non-smoothed) weights
        total_variation_penalty = Variable(MyTensor(1).zero_(), requires_grad=False)
        if self.total_variation_weight_penalty > 0:
            # first compute the edge map
            g_I = compute_localized_edge_penalty(I[:, 0, ...], self.spacing, self.params)
            batch_size = I.size()[0]
            for g in range(self.nr_of_gaussians):
                # total_variation_penalty += self.compute_total_variation(weights[:,g,...])
                c_local_norm_grad = _compute_local_norm_of_gradient(weights[:, g, ...], self.spacing, self.pnorm)
                total_variation_penalty += (g_I * c_local_norm_grad).sum() * self.volumeElement / batch_size

        diffusion_penalty = Variable(MyTensor(1).zero_(), requires_grad=False)
        if self.diffusion_weight_penalty > 0:
            for g in range(self.nr_of_gaussians):
                diffusion_penalty += self.compute_diffusion(weights[:, g, ...])

        # enforce minimum weight for numerical reasons
        weights = _project_weights_to_min(weights, self.gaussianWeight_min)

        # instantiate the extra smoother if weight is larger than 0 and it has not been initialized yet
        if self.deep_network_local_weight_smoothing > 0 and self.deep_network_weight_smoother is None:
            import pyreg.smoother_factory as sf
            s_m_params = pars.ParameterDict()
            s_m_params['smoother']['type'] = 'gaussian'
            s_m_params['smoother']['gaussian_std'] = self.deep_network_local_weight_smoothing
            self.deep_network_weight_smoother = sf.SmootherFactory(ret.size()[2::], self.spacing).create_smoother(
                s_m_params)

        if self.deep_network_local_weight_smoothing>0:
            # now we smooth all the weights
            if retain_weights:
                # todo: change visualization to work with this new format:
                # B x weights x X x Y instead of weights x B x X x Y
                self.computed_pre_weights[:] = weights.data

            weights = self.deep_network_weight_smoother.smooth(weights)
            # make sure they are all still positive (#todo: may not be necessary, since we set a minumum weight above now; but risky as we take the square root below)
            weights = torch.clamp(weights,0.0,1.0)

        # multiply the velocity fields by the weights and sum over them
        # this is then the multi-Gaussian output

        if self.weighting_type=='sqrt_w_K_sqrt_w':
            sqrt_weights = torch.sqrt(weights)
            sqrt_weighted_multi_smooth_v = compute_weighted_multi_smooth_v( momentum=momentum, weights=sqrt_weights, gaussian_stds=self.gaussian_stds,
                                                                   gaussian_fourier_filter_generator=gaussian_fourier_filter_generator )
        elif self.weighting_type=='w_K_w':
            # now create the weighted multi-smooth-v
            weighted_multi_smooth_v = compute_weighted_multi_smooth_v( momentum=momentum, weights=weights, gaussian_stds=self.gaussian_stds,
                                                                       gaussian_fourier_filter_generator=gaussian_fourier_filter_generator )
        elif self.weighting_type=='w_K':
            multi_smooth_v = ce.fourier_set_of_gaussian_convolutions(momentum,
                                                                     gaussian_fourier_filter_generator=gaussian_fourier_filter_generator,
                                                                     sigma=Variable(torch.from_numpy(self.gaussian_stds),requires_grad=False),
                                                                     compute_std_gradients=False)
        else:
            raise ValueError('Unknown weighting_type: {}'.format(self.weighting_type))

        # now we apply this weight across all the channels; weight output is B x weights x X x Y
        for n in range(self.dim):
            # reverse the order so that for a given channel we have batch x multi_velocity x X x Y
            # i.e., the multi-velocity field output is treated as a channel
            # reminder: # format of multi_smooth_v is multi_v x batch x channels x X x Y
            # (channels here are the vector field components); i.e. as many as there are dimensions
            # each one of those should be smoothed the same

            # let's smooth this on the fly, as the smoothing will be of form
            # w_i*K_i*(w_i m)

            if self.weighting_type=='sqrt_w_K_sqrt_w':
                # roc should be: batch x multi_v x X x Y
                roc = torch.transpose(sqrt_weighted_multi_smooth_v[:, :, n, ...], 0, 1)
                yc = torch.sum(roc * sqrt_weights, dim=1)
            elif self.weighting_type=='w_K_w':
                # roc should be: batch x multi_v x X x Y
                roc = torch.transpose(weighted_multi_smooth_v[:, :, n, ...], 0, 1)
                yc = torch.sum(roc*weights,dim=1)
            elif self.weighting_type=='w_K':
                # roc should be: batch x multi_v x X x Y
                roc = torch.transpose(multi_smooth_v[:, :, n, ...], 0, 1)
                yc = torch.sum(roc * weights, dim=1)
            else:
                raise ValueError('Unknown weighting_type: {}'.format(self.weighting_type))

            ret[:, n, ...] = yc # ret is: batch x channels x X x Y

        #if self.deep_network_local_weight_smoothing>0 and not retain_weights:
        #    # this is the actual optimization, just smooth the resulting velocity field which is more efficicient
        # no longer possible, as the weights are in there twice now
        #    # than smoothing the individual weights
        #    # if weights are retained then instead the weights are smoothed, see above (as this is what we were after)
        #    ret = self.deep_network_weight_smoother.smooth(ret)

        current_diffusion_penalty = self.diffusion_weight_penalty * diffusion_penalty

        current_omt_penalty = self.omt_weight_penalty*compute_omt_penalty(weights,self.gaussian_stds,self.volumeElement,self.omt_power,self.omt_use_log_transformed_std)
        current_tv_penalty = self.total_variation_weight_penalty * total_variation_penalty
        self.current_penalty = current_omt_penalty + current_tv_penalty + current_diffusion_penalty

        print('TV_penalty = ' + str(current_tv_penalty.data.cpu().numpy()) + \
              '; OMT_penalty = ' + str(current_omt_penalty.data.cpu().numpy()) + \
              '; diffusion_penalty = ' + str(current_diffusion_penalty.data.cpu().numpy()))


        if retain_weights:
            # todo: change visualization to work with this new format:
            # B x weights x X x Y instead of weights x B x X x Y
            self.computed_weights[:] = weights.data

        return ret


class DeepSmootherFactory(object):
    """
    Factory to quickly create different types of deep smoothers.
    """

    def __init__(self, nr_of_gaussians, gaussian_stds, dim, spacing, nr_of_image_channels=1 ):
        self.nr_of_gaussians = nr_of_gaussians
        """number of Gaussians as input"""
        self.gaussian_stds = gaussian_stds
        """stds of the Gaussians"""
        self.dim = dim
        """dimension of input image"""
        self.nr_of_image_channels = nr_of_image_channels
        """number of channels the image has (currently only one is supported)"""
        self.spacing = spacing
        """Spacing of the image"""

        if self.nr_of_image_channels!=1:
            raise ValueError('Currently only one image channel supported')


    def create_deep_smoother(self, params ):
        """
        Create the desired deep smoother
        :param params: ParamterDict() object to hold paramters which should be passed on
        :return: returns the deep smoother
        """

        cparams = params[('deep_smoother',{})]
        smootherType = cparams[('type', 'simple_consistent','type of deep smoother (simple_consistent|encoder_decoder)')]
        if smootherType=='simple_consistent':
            return SimpleConsistentWeightedSmoothingModel(nr_of_gaussians=self.nr_of_gaussians,
                                                          gaussian_stds=self.gaussian_stds,
                                                          dim=self.dim,
                                                          spacing=self.spacing,
                                                          nr_of_image_channels=self.nr_of_image_channels,
                                                          params=params)
        elif smootherType=='encoder_decoder':
            return EncoderDecoderSmoothingModel(nr_of_gaussians=self.nr_of_gaussians,
                                                  gaussian_stds=self.gaussian_stds,
                                                  dim=self.dim,
                                                  spacing=self.spacing,
                                                  nr_of_image_channels=self.nr_of_image_channels,
                                                  params=params)
        else:
            raise ValueError('Deep smoother: ' + smootherType + ' not known')
