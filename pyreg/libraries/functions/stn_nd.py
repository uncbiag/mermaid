"""
Spatial transform functions in 1D, 2D, and 3D.

.. todo::
    Add CUDA implementation. Could be based of the existing 2D CUDA implementation.
"""

import torch
from torch.autograd import Function
from cffi import FFI
from pyreg.data_wrapper import USE_CUDA, STNTensor, STNVal
if USE_CUDA:
    from pyreg.libraries._ext import my_lib_1D, my_lib_2D, my_lib_3D
from pyreg.libraries._ext import my_lib_nd
ffi = FFI()


class STNFunction_ND_BCXYZ(Function):
    """
   Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
   """

    def __init__(self, ndim):
        """
        Constructor

        :param ndim: (int) spatial transformation of the transform
        """
        super(STNFunction_ND_BCXYZ, self).__init__()
        self.ndim = ndim

    def forward_stn(self, input1, input2, output, ndim, device_c, use_cuda=USE_CUDA):
        if use_cuda:
            if ndim == 1:
                my_lib_1D.BilinearSamplerBCW_updateOutput_cuda_1D(input1, input2, output, device_c)
            elif ndim == 2:
                my_lib_2D.BilinearSamplerBCWH_updateOutput_cuda_2D(input1, input2, output, device_c)
            elif ndim == 3:
                my_lib_3D.BilinearSamplerBCWHD_updateOutput_cuda_3D(input1, input2, output, device_c)
        else:
            my_lib_nd.BilinearSamplerBCXYZ_updateOutput_ND(input1, input2, output, ndim)

    def backward_stn(self, input1, input2, grad_input1, grad_input2, grad_output, ndim, device_c, use_cuda=USE_CUDA):
        if use_cuda:
            if ndim == 1:
                my_lib_1D.BilinearSamplerBCW_updateGradInput_cuda_1D(input1, input2, grad_input1, grad_input2,
                                                                     grad_output, device_c)
            elif ndim == 2:
                my_lib_2D.BilinearSamplerBCWH_updateGradInput_cuda_2D(input1, input2, grad_input1, grad_input2,
                                                                      grad_output, device_c)
            elif ndim == 3:
                my_lib_3D.BilinearSamplerBCWHD_updateGradInput_cuda_3D(input1, input2, grad_input1, grad_input2,
                                                                       grad_output, device_c)
        else:
            my_lib_nd.BilinearSamplerBCXYZ_updateGradInput_ND(input1, input2, grad_input1, grad_input2, grad_output,
                                                              ndim)

    def forward(self, input1, input2):
        """
        Perform the actual spatial transform

        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """
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
        self.forward_stn(input1, input2, output, self.ndim, self.device_c)
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
        self.backward_stn(self.input1, self.input2, grad_input1, grad_input2, grad_output, self.ndim, self.device_c)
        # print( STNVal(grad_input1, ini=-1).sum(), STNVal(grad_input2, ini=-1).sum())
        return STNVal(grad_input1, ini=-1), STNVal(grad_input2, ini=-1)


###################################################################################################################

class STNFunction_ND(Function):
    """
    Spatial transform function for 1D, 2D, and 3D. In BXYZC format (NOT the format used in the current toolbox).
    """

    def __init__(self, ndim):
        """
        Constructor

        :param ndim: (int) spatial transformation of the transform
        """
        super(STNFunction_ND, self).__init__()
        self.ndim = ndim
        """spatial dimension"""

    def forward(self, input1, input2):
        """
        Perform the actual spatial transform

        :param input1: image in BXYZC format
        :param input2: spatial transform in BXYZdim format
        :return: spatially transformed image in BXYZC format
        """
        self.input1 = input1
        self.input2 = input2
        self.device_c = ffi.new("int *")
        if self.ndim == 1:
            output = torch.zeros(input1.size()[0], input2.size()[1], input1.size()[2])
        elif self.ndim == 2:
            output = torch.zeros(input1.size()[0], input2.size()[1], input2.size()[2], input1.size()[3])
        elif self.ndim == 3:
            output = torch.zeros(input1.size()[0], input2.size()[1], input2.size()[2], input2.size()[3],
                                 input1.size()[4])
        else:
            raise ValueError('Can only process dimensions 1-3')
        # print('decice %d' % torch.cuda.current_device())
        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1
        self.device_c[0] = self.device
        if not input1.is_cuda:
            my_lib_nd.BilinearSamplerBXYZC_updateOutput_ND(input1, input2, output, self.ndim)
        else:
            output = output.cuda(self.device)
            my_lib_nd.BilinearSamplerBXYZC_updateOutput_ND_cuda(input1, input2, output, self.device_c)
        return output

    def backward(self, grad_output):
        """
        Computes the gradient

        :param grad_output: grad output from previous "layer"
        :return: gradient
        """
        grad_input1 = STNTensor(self.input1.size()).zero_()
        grad_input2 = STNTensor(self.input2.size()).zero_()
        # print('backward decice %d' % self.device)
        if not USE_CUDA:
            my_lib_nd.BilinearSamplerBXYZC_updateGradInput_ND(self.input1, self.input2, grad_input1, grad_input2,
                                                              grad_output, self.ndim)
        else:
            my_lib_nd.BilinearSamplerBXYZC_updateGradInput_ND_cuda(self.input1, self.input2, grad_input1, grad_input2,
                                                                   grad_output, self.device_c)
        return grad_input1, grad_input2


# Old code starts here

class STNFunction(Function):
    """
    Legacy 2D implementation. Ignore.
    """

    def forward(self, input1, input2):
        self.input1 = input1
        self.input2 = input2
        self.device_c = ffi.new("int *")
        output = torch.zeros(input1.size()[0], input2.size()[1], input2.size()[2], input1.size()[3])
        # print('decice %d' % torch.cuda.current_device())
        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1
        self.device_c[0] = self.device
        if not input1.is_cuda:
            my_lib_nd.BilinearSamplerBHWD_updateOutput(input1, input2, output)
        else:
            output = output.cuda(self.device)
            my_lib_nd.BilinearSamplerBHWD_updateOutput_cuda(input1, input2, output, self.device_c)
        return output

    def backward(self, grad_output):
        if USE_CUDA:
            grad_input1 = STNTensor(self.input1.size()).zero_()
            grad_input2 = STNTensor(self.input2.size()).zero_()
        # print('backward decice %d' % self.device)
        if not USE_CUDA:
            my_lib_nd.BilinearSamplerBHWD_updateGradInput(self.input1, self.input2, grad_input1, grad_input2,
                                                          grad_output)
        else:
            my_lib_nd.BilinearSamplerBHWD_updateGradInput_cuda(self.input1, self.input2, grad_input1, grad_input2,
                                                               grad_output, self.device_c)
        return grad_input1, grad_input2
