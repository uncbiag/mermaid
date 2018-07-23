from __future__ import absolute_import
from pyreg.data_wrapper import USE_CUDA, MyTensor, AdaptVal
from cffi import FFI

import sys

import torch
from pyreg.libraries._ext import my_lib_nn
from . import map_scale_utils

ffi = FFI()

if USE_CUDA:
    if sys.version_info >= (3, 0):
        from pyreg.libraries._ext import nn_interpolation
    else:
        from pyreg.libraries._ext import nn_interpolation

def nn_interpolation_fn_sel(input1, input2, output, ndim, device_c, use_cuda=USE_CUDA):
    if use_cuda:
        if ndim == 1:
            nn_interpolation.nearestNeighBCW_updateOutput_cuda_1D(input1, input2, output, device_c)
        elif ndim == 2:
            nn_interpolation.nearestNeighBCWH_updateOutput_cuda_2D(input1, input2, output, device_c)
        elif ndim == 3:
            nn_interpolation.nearestNeighBCWHD_updateOutput_cuda_3D(input1, input2, output, device_c)
    else:
        my_lib_nn.nearestNeighBCXYZ_updateOutput_ND(input1, input2, output, ndim)


def get_nn_interpolation(input1, input2, spacing):
    device_c = ffi.new("int *")

    map_scaled = map_scale_utils.scale_map(input2,spacing)
    ndim = len(spacing)

    if ndim == 1:
        output = MyTensor(input1.size()[0], input1.size()[1], input2.size()[2]).zero_()
    elif ndim == 2:
        output = MyTensor(input1.size()[0], input1.size()[1], input2.size()[2], input2.size()[3]).zero_()
    elif ndim == 3:
        #print(type(input1),type(input2))
        output = MyTensor(input1.size()[0], input1.size()[1], input2.size()[2], input2.size()[3], input2.size()[4]).zero_()
    else:
        raise ValueError('Can only process dimensions 1-3')
    # print('decice %d' % torch.cuda.current_device())
    if USE_CUDA:
        device_c[0] = torch.cuda.current_device()
    else:
        device_c[0] = -1
    nn_interpolation_fn_sel(input1.data, map_scaled.data, output, ndim, device_c)
    return AdaptVal(output)
