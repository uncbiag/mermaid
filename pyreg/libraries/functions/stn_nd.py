# functions/add.py
import torch
from torch.autograd import Function
from _ext import my_lib_nd
from cffi import FFI
ffi = FFI()

class STNFunction_ND(Function):
    def __init__(self, ndim):
        super(STNFunction_ND,self).__init__()
        self.ndim = ndim
        
    def forward(self, input1, input2):
        self.input1 = input1
        self.input2 = input2
        self.device_c = ffi.new("int *")
        if self.ndim==1:
            output = torch.zeros(input1.size()[0], input2.size()[1], input1.size()[2])
        elif self.ndim==2:
            output = torch.zeros(input1.size()[0], input2.size()[1], input2.size()[2], input1.size()[3])
        elif self.ndim==3:
            output = torch.zeros(input1.size()[0], input2.size()[1], input2.size()[2], input2.size()[3], input1.size()[4])
        else:
            raise ValueError('Can only process dimensions 1-3')
        #print('decice %d' % torch.cuda.current_device())
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
        grad_input1 = torch.zeros(self.input1.size())
        grad_input2 = torch.zeros(self.input2.size())
        #print('backward decice %d' % self.device)
        if not grad_output.is_cuda:
            my_lib_nd.BilinearSamplerBXYZC_updateGradInput_ND(self.input1, self.input2, grad_input1, grad_input2, grad_output, self.ndim)
        else:
            grad_input1 = grad_input1.cuda(self.device)
            grad_input2 = grad_input2.cuda(self.device)
            my_lib_nd.BilinearSamplerBXYZC_updateGradInput_ND_cuda(self.input1, self.input2, grad_input1, grad_input2, grad_output, self.device_c)
        return grad_input1, grad_input2


class STNFunction_ND_BCXYZ(Function):
    def __init__(self, ndim):
        super(STNFunction_ND_BCXYZ,self).__init__()
        self.ndim = ndim
        
    def forward(self, input1, input2): # input one is image and 2 is the map
        self.input1 = input1
        self.input2 = input2
        self.device_c = ffi.new("int *")
        if self.ndim==1:
            output = torch.zeros(input1.size()[0], input1.size()[1], input2.size()[2])
        elif self.ndim==2:
            output = torch.zeros(input1.size()[0], input1.size()[1], input2.size()[2], input2.size()[3])
        elif self.ndim==3:
            output = torch.zeros(input1.size()[0], input1.size()[1], input2.size()[2], input2.size()[3], input2.size()[4])
        else:
            raise ValueError('Can only process dimensions 1-3')
        #print('decice %d' % torch.cuda.current_device())
        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1
        self.device_c[0] = self.device
        if not input1.is_cuda:
            my_lib_nd.BilinearSamplerBCXYZ_updateOutput_ND(input1, input2, output, self.ndim)
        else:
            output = output.cuda(self.device)
            my_lib_nd.BilinearSamplerBCXYZ_updateOutput_ND_cuda(input1, input2, output, self.device_c)
        return output

    def backward(self, grad_output):
        grad_input1 = torch.zeros(self.input1.size())
        grad_input2 = torch.zeros(self.input2.size())
        #print('backward decice %d' % self.device)
        if not grad_output.is_cuda:
            my_lib_nd.BilinearSamplerBCXYZ_updateGradInput_ND(self.input1, self.input2, grad_input1, grad_input2, grad_output, self.ndim)
        else:
            grad_input1 = grad_input1.cuda(self.device)
            grad_input2 = grad_input2.cuda(self.device)
            my_lib_nd.BilinearSamplerBCXYZ_updateGradInput_ND_cuda(self.input1, self.input2, grad_input1, grad_input2, grad_output, self.device_c)
        return grad_input1, grad_input2

# Old code starts here

class STNFunction(Function):
    def forward(self, input1, input2):
        self.input1 = input1
        self.input2 = input2
        self.device_c = ffi.new("int *")
        output = torch.zeros(input1.size()[0], input2.size()[1], input2.size()[2], input1.size()[3])
        #print('decice %d' % torch.cuda.current_device())
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
        grad_input1 = torch.zeros(self.input1.size())
        grad_input2 = torch.zeros(self.input2.size())
        #print('backward decice %d' % self.device)
        if not grad_output.is_cuda:
            my_lib_nd.BilinearSamplerBHWD_updateGradInput(self.input1, self.input2, grad_input1, grad_input2, grad_output)
        else:
            grad_input1 = grad_input1.cuda(self.device)
            grad_input2 = grad_input2.cuda(self.device)
            my_lib_nd.BilinearSamplerBHWD_updateGradInput_cuda(self.input1, self.input2, grad_input1, grad_input2, grad_output, self.device_c)
        return grad_input1, grad_input2
