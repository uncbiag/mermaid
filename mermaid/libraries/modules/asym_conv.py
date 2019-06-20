from __future__ import absolute_import

from torch.nn.modules import Module
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import sys
import os
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('/playpen/zyshen/reg_clean/mermaid'))
import mermaid.utils as utils

class AsymConv(Module):
    """
    the implementation of location-dependent convolution method
    the input should include a BxCxXxYxZ image, a 1x1xHxWxDxK3 kernel, where each location has a corresponding filter
    we would extend into guassian based convolution method
    the input should include a BxCxXxYxZ image, a 1x1xXxYxZ kernel, where each location refers to the deviation of a gaussian filter
    the implementaiton is based on img2col convolution
    """
    def __init__(self, kernel_size):
        super(AsymConv,self).__init__()
        self.k_sz =kernel_size

    def forward(self,X, W):
        assert self.k_sz%2==1,  'the kernel size must be odd'
        hk_sz =self.k_sz//2
        X= F.pad(X, (hk_sz,hk_sz,hk_sz,hk_sz,hk_sz,hk_sz), "replicate")
        print(X.shape)
        X_col = X.unfold(2, self.k_sz, 1).unfold(3, self.k_sz, 1).unfold(4, self.k_sz, 1)# BxCxXxYxZxKxKxK
        print(X_col.shape)
        dim_B, dim_C, dim_X, dim_Y, dim_Z,K,_,_ = X_col.shape
        X_col = X_col.contiguous().view(dim_B, dim_C, dim_X, dim_Y, dim_Z,-1) # BxCxXxYxZxK3
        print(X_col.shape, W.shape)
        res = X_col*W
        res = res.sum(5)
        print(res.shape)
        return res





def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = (H + 2 * padding - field_height) / stride + 1
  out_width = (W + 2 * padding - field_width) / stride + 1

  i0 = np.repeat(np.arange(field_height), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
  """ An implementation of col2im based on fancy indexing and np.add.at """
  N, C, H, W = x_shape
  H_padded, W_padded = H + 2 * padding, W + 2 * padding
  x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
  k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
  cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)
  np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
  if padding == 0:
    return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]

pass





class AsymConvFunc(Function):
    """
    the implementation of location-dependent convolution method
    the input should include a BxCxXxYxZ image, a 1x1xHxWxDxK3 kernel, where each location has a corresponding filter
    we would extend into guassian based convolution method
    the input should include a BxCxXxYxZ image, a 1x1xXxYxZ kernel, where each location refers to the deviation of a gaussian filter
    the implementaiton is based on img2col convolution
    """
    def __init__(self, kernel_size,spacing):
        super(AsymConvFunc,self).__init__()
        self.k_sz =kernel_size
        self.spacing = spacing
        self.dim = 3
        self.__init_center_id_coord()

    def __init_center_id_coord(self):

        self.centered_id = utils.centered_identity_map([self.k_sz]*3, self.spacing)  # should be 3xKxKxK
        self.centered_id_2sum  = (self.centered_id **2).sum(0)  # should be KxKxK
        self.centered_id_2sum = torch.from_numpy(self.centered_id_2sum).cuda()


    def forward(self,input,sig):
        """
        :param X: the X should be BxCxXxYxZ
        :param sig:  the sigma should be 1x1xXxYxZ
        :return: the output should be BxCxXxYxZ
        """
        frac_sig2 =1./ (2 *(sig**2))  # here should be 1x1xXxYxZ
        kernel_sz = list(frac_sig2.shape)+[1,1,1]
        frac_sig2 = frac_sig2.view(kernel_sz) # should be 1x1xXxYxZx1x1x1
        W = torch.exp(-self.centered_id_2sum*frac_sig2)  #  KxKxK * 1x1xXxYxZx1x1x1  should be 1x1xXxYxZxKxKxK
        W = W/(W.sum(7).sum(6).sum(5).view(kernel_sz))  # should be 1x1xXxYxZxKxKxK
        del frac_sig2
        hk_sz = self.k_sz // 2
        batch = input.shape[0]
        channel = input.shape[1]
        convoled_input = torch.zeros_like(input)
        for bh in range(batch):
            for ch in range(channel):
                X = input[bh:bh+1,ch:ch+1,...]
                X = F.pad(X, (hk_sz, hk_sz, hk_sz, hk_sz, hk_sz, hk_sz), "replicate")
                X_col = X.unfold(2, self.k_sz, 1).unfold(3, self.k_sz, 1).unfold(4, self.k_sz, 1)  # BxCxXxYxZxKxKxK
                dim_B, dim_C, dim_X, dim_Y, dim_Z, K, _, _ = X_col.shape
                res = X_col * W # BxCxXxYxZxKxKxK
                res = res.sum(7).sum(6).sum(5)
                convoled_input[bh:bh+1,ch:ch+1,...] = res
                del X
        return convoled_input


    def compute_normalized_gaussian(self,X, mu, sig):
        """
        Computes a normalized Gaussian

        :param X: map with coordinates at which to evaluate
        :param mu: array indicating the mean
        :param sig: array indicating the standard deviations for the different dimensions
        :return: normalized Gaussian
        """
        g = np.exp(-np.power(X[0, :, :, :] - mu[0], 2.) / (2 * np.power(sig[0], 2.))
                   - np.power(X[1, :, :, :] - mu[1], 2.) / (2 * np.power(sig[1], 2.))
                   - np.power(X[2, :, :, :] - mu[2], 2.) / (2 * np.power(sig[2], 2.)))
        g = g / g.sum()
        return g


    def backward(self,grad_output):
        pass






def compute_normalized_gaussian(X, mu, sig):
    """
    Computes a normalized Gaussian

    :param X: map with coordinates at which to evaluate
    :param mu: array indicating the mean
    :param sig: array indicating the standard deviations for the different dimensions
    :return: normalized Gaussian
    """
    g = np.exp(-np.power(X[0, :, :, :] - mu[0], 2.) / (2 * np.power(sig[0], 2.))
               - np.power(X[1, :, :, :] - mu[1], 2.) / (2 * np.power(sig[1], 2.))
               - np.power(X[2, :, :, :] - mu[2], 2.) / (2 * np.power(sig[2], 2.)))
    print(g.sum())
    g = 1./(np.sqrt((2*np.pi)**3 *((sig[0]**2)**3)))*g
    print(g.sum())
    return g



#
# def conv_forward(X, W, b, stride=1, padding=1):
#     cache = W, b, stride, padding
#     n_filters, d_filter, h_filter, w_filter = W.shape
#     n_x, d_x, h_x, w_x = X.shape
#     h_out = (h_x - h_filter + 2 * padding) / stride + 1
#     w_out = (w_x - w_filter + 2 * padding) / stride + 1
#
#     if not h_out.is_integer() or not w_out.is_integer():
#         raise Exception('Invalid output dimension!')
#
#     h_out, w_out = int(h_out), int(w_out)
#
#     X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
#     W_col = W.reshape(n_filters, -1)
#
#     out = W_col @ X_col + b
#     out = out.reshape(n_filters, h_out, w_out, n_x)
#     out = out.transpose(3, 0, 1, 2)
#
#     cache = (X, W, b, stride, padding, X_col)
#
#     return out, cache
#
#
# def conv_backward(dout, cache):
#     X, W, b, stride, padding, X_col = cache
#     n_filter, d_filter, h_filter, w_filter = W.shape
#
#     db = np.sum(dout, axis=(0, 2, 3))
#     db = db.reshape(n_filter, -1)
#
#     dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
#     dW = dout_reshaped @ X_col.T
#     dW = dW.reshape(W.shape)
#
#     W_reshape = W.reshape(n_filter, -1)
#     dX_col = W_reshape.T @ dout_reshaped
#     dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)
#
#     return dX, dW, db



import torch
kernel_sz = 7
torch.set_grad_enabled(True)
# X = torch.nn.Parameter(torch.rand(1,3,40,96,96)).cuda()
# W = torch.rand(1,1,40,96,96).cuda()
# aconv_f = AsymConvFunc (kernel_sz,[1./40,1./96,1./96])
# g= aconv_f(X,W)
spacing =[1./7]*3 # [1./40,1./96,1./96]
centered_id = utils.centered_identity_map([kernel_sz] * 3, spacing)
compute_normalized_gaussian(centered_id,np.zeros(3),np.ones(3)*5)
# torch.set_grad_enabled(True)
# X = torch.nn.Parameter(torch.rand(1,3,40,96,96)).cuda()
# W = torch.rand(1,1,40,96,96,kernel_sz*kernel_sz*kernel_sz).cuda()
# print(W.shape)
# aconv =AsymConv(kernel_sz)
#
#
# res = aconv(X,W)
# print(res.sum(), res.requires_grad)

