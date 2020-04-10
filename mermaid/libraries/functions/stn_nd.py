"""
Spatial transform functions in 1D, 2D, and 3D.

.. todo::
    Add CUDA implementation. Could be based of the existing 2D CUDA implementation.
"""
from __future__ import absolute_import

import sys

import torch
from torch.autograd import Function
from torch.nn import Module
from cffi import FFI
try:
    from mermaid.data_wrapper import USE_CUDA, STNTensor, STNVal
except:
    from mermaid.data_wrapper import USE_CUDA, STNTensor, STNVal


###########TODO temporal comment for torch1 compatability

# if sys.version_info >= (3, 0):
#     if USE_CUDA:
#         from mermaid.libraries._ext import my_lib_1D, my_lib_2D, my_lib_3D
#     from mermaid.libraries._ext import my_lib_nd
# else:
#     if USE_CUDA:
#         from mermaid.libraries._ext import my_lib_1D, my_lib_2D, my_lib_3D
#     from mermaid.libraries._ext import my_lib_nd
###########################################################3
from . import map_scale_utils

ffi = FFI()



class STNFunction_ND_BCXYZ(Module):
    """
   Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
   """

    def __init__(self, spacing, zero_boundary = False,using_bilinear=True,using_01_input=True):
        """
        Constructor

        :param ndim: (int) spatial transformation of the transform
        """
        super(STNFunction_ND_BCXYZ, self).__init__()
        self.spacing = spacing
        self.ndim = len(spacing)
        #zero_boundary = False
        self.zero_boundary = 'zeros' if zero_boundary else 'border'
        self.mode = 'bilinear' if using_bilinear else 'nearest'
        self.using_01_input=using_01_input

    def forward_stn(self, input1, input2, ndim):
        if ndim==1:
            # use 2D interpolation to mimick 1D interpolation
            # now test this for 1D
            phi_rs = input2.reshape(list(input2.size()) + [1])
            input1_rs = input1.reshape(list(input1.size()) + [1])

            phi_rs_size = list(phi_rs.size())
            phi_rs_size[1] = 2

            phi_rs_ordered = torch.zeros(phi_rs_size,dtype=phi_rs.dtype,device=phi_rs.device)
            # keep dimension 1 at zero
            phi_rs_ordered[:, 1, ...] = phi_rs[:, 0, ...]

            output_rs = torch.nn.functional.grid_sample(input1_rs, phi_rs_ordered.permute([0, 2, 3, 1]), mode=self.mode, padding_mode=self.zero_boundary,align_corners=True)
            output = output_rs[:, :, :, 0]

        if ndim==2:
            # todo double check, it seems no transpose is need for 2d, already in height width design
            input2_ordered = torch.zeros_like(input2)
            input2_ordered[:,0,...] = input2[:,1,...]
            input2_ordered[:,1,...] = input2[:,0,...]
            output = torch.nn.functional.grid_sample(input1, input2_ordered.permute([0, 2, 3, 1]), mode=self.mode,
                                          padding_mode=self.zero_boundary,align_corners=True)
        if ndim==3:
            input2_ordered = torch.zeros_like(input2)
            input2_ordered[:, 0, ...] = input2[:, 2, ...]
            input2_ordered[:, 1, ...] = input2[:, 1, ...]
            input2_ordered[:, 2, ...] = input2[:, 0, ...]
            output = torch.nn.functional.grid_sample(input1, input2_ordered.permute([0, 2, 3, 4, 1]), mode=self.mode, padding_mode=self.zero_boundary,align_corners=True)
        return output

    def forward(self, input1, input2):
        """
        Perform the actual spatial transform

        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """

        assert(len(self.spacing)+2==len(input2.size()))
        if self.using_01_input:
            output = self.forward_stn(input1, map_scale_utils.scale_map(input2,self.spacing), self.ndim)
        else:
            output = self.forward_stn(input1, input2, self.ndim)
        # print(STNVal(output, ini=-1).sum())
        return output















class STNFunction_ND_BCXYZ_Compile(Function):
    """
   Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
     TODO, the boundary issue is still there and would be triggered at 1, so it would cause the boundary a little bit shrink,
     this can be solved by adding more strick judgement when boundary is 1, it would inflence a lot at low-resolution case, and
     will influence the high resolution case by upsampling the map
     currently we put it aside
     """

    def __init__(self, spacing,zero_boundary=True):
        """
        Constructor

        :param ndim: (int) spatial transformation of the transform
        """
        super(STNFunction_ND_BCXYZ_Compile, self).__init__()
        self.spacing = spacing
        self.ndim = len(spacing)
        #zero_boundary = False
        self.zero_boundary = zero_boundary

    def forward_stn(self, input1, input2, output, ndim, device_c, use_cuda=USE_CUDA,zero_boundary=True):
        if use_cuda:
            if ndim == 1:
                my_lib_1D.BilinearSamplerBCW_updateOutput_cuda_1D(input1, input2, output, device_c, int(zero_boundary))
            elif ndim == 2:
                my_lib_2D.BilinearSamplerBCWH_updateOutput_cuda_2D(input1, input2, output, device_c, int(zero_boundary))
            elif ndim == 3:
                my_lib_3D.BilinearSamplerBCWHD_updateOutput_cuda_3D(input1, input2, output, device_c, int(zero_boundary))
        else:
            my_lib_nd.BilinearSamplerBCXYZ_updateOutput_ND(input1, input2, output, ndim, int(zero_boundary))

    def backward_stn(self, input1, input2, grad_input1, grad_input2, grad_output, ndim, device_c, use_cuda=USE_CUDA,zero_boundary=True):
        if use_cuda:
            if ndim == 1:
                my_lib_1D.BilinearSamplerBCW_updateGradInput_cuda_1D(input1, input2, grad_input1, grad_input2,
                                                                     grad_output, device_c, int(zero_boundary))
            elif ndim == 2:
                my_lib_2D.BilinearSamplerBCWH_updateGradInput_cuda_2D(input1, input2, grad_input1, grad_input2,
                                                                      grad_output, device_c, int(zero_boundary))
            elif ndim == 3:
                my_lib_3D.BilinearSamplerBCWHD_updateGradInput_cuda_3D(input1, input2, grad_input1, grad_input2,
                                                                       grad_output, device_c, int(zero_boundary))
        else:
            my_lib_nd.BilinearSamplerBCXYZ_updateGradInput_ND(input1, input2, grad_input1, grad_input2, grad_output,
                                                              ndim, int(zero_boundary))

    def forward(self, input1, input2):
        """
        Perform the actual spatial transform

        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """

        assert(len(self.spacing)+2==len(input2.size()))

        self.input1 = STNVal(input1, ini=1)
        self.input2 = STNVal(input2, ini=1)
        self.device_c = ffi.new("int *")
        if self.ndim == 1:
            output = STNTensor(input1.size()[0], input1.size()[1], input2.size()[2]).zero_()
        elif self.ndim == 2:
            output = STNTensor(input1.size()[0], input1.size()[1], input2.size()[2], input2.size()[3]).zero_()
        elif self.ndim == 3:
            output = STNTensor(input1.size()[0], input1.size()[1], input2.size()[2], input2.size()[3],
                               input2.size()[4]).zero_()
        else:
            raise ValueError('Can only process dimensions 1-3')
        # print('decice %d' % torch.cuda.current_device())
        if USE_CUDA:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1
        self.device_c[0] = self.device

        # the spatial transformer code expects maps in the range of [-1,1]^d
        # So first rescale the map (i.e., input2) and then account for this rescaling in the gradient

        self.forward_stn(input1, map_scale_utils.scale_map(input2,self.spacing), output, self.ndim, self.device_c, zero_boundary= self.zero_boundary)
        # print(STNVal(output, ini=-1).sum())
        return STNVal(output, ini=-1)

    def backward(self, grad_output):
        """
        Computes the gradient

        :param grad_output: grad output from previous "layer"
        :return: gradient
        """
        grad_input1 = STNTensor(self.input1.size()).zero_()
        grad_input2 = STNTensor(self.input2.size()).zero_()
        grad_output = STNVal(grad_output, ini=1)
        # print grad_output.view(1, -1).sum()
        # print('backward decice %d' % self.device)

        # also needs to scale the input map first
        self.backward_stn(self.input1, map_scale_utils.scale_map(self.input2,self.spacing), grad_input1, grad_input2, grad_output, self.ndim, self.device_c, zero_boundary=  self.zero_boundary)
        # print( STNVal(grad_input1, ini=-1).sum(), STNVal(grad_input2, ini=-1).sum())

        map_scale_utils.scale_map_grad(grad_input2,self.spacing)

        return STNVal(grad_input1, ini=-1), STNVal(grad_input2, ini=-1)


#################################################################################################################

