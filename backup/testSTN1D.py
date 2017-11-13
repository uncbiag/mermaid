from __future__ import print_function

import torch
from cffi import FFI
import time
ffi = FFI()

from pyreg.libraries._ext import my_lib_1D
from pyreg.libraries._ext import my_lib_nd

device = torch.cuda.current_device()
device_c = ffi.new("int *")
device_c[0] = device
nframes = 1
width = 1000 #233333
ratio =1
grid_width= int(width/ratio)
channels = 5
device = torch.cuda.current_device()

inputImage = torch.randn(nframes, channels, width)
inputGrids = (torch.rand(nframes, 1,  grid_width)*2 -1)
output = torch.rand(nframes, channels, grid_width)



inputImage_cuda = inputImage.cuda(device)
inputGrids_cuda = inputGrids.cuda(device)
output_cuda = output.cuda(device)

start = time.time()
my_lib_nd.BilinearSamplerBCX_updateOutput_1D(inputImage, inputGrids, output)
print('sampling cpu time taking:', time.time() - start)

start = time.time()
my_lib_1D.BilinearSamplerBCW_updateOutput_cuda_1D(inputImage_cuda, inputGrids_cuda, output_cuda, device_c)
print('sampling gpu time taking:', time.time() - start)
out1 = (output-(output_cuda).cpu()).view(-1,1).sum()

print('difference of sampling cpu sum of output:{}'.format(out1))
grad_output = torch.Tensor(output.cpu())
grad_input = torch.zeros(inputImage.size())
grad_grids = torch.zeros(inputGrids.size())
grad_output_cuda = grad_output.cuda(device)
grad_input_cuda = grad_input.cuda(device)
grad_grids_cuda = grad_grids.cuda(device)

start = time.time()
my_lib_nd.BilinearSamplerBCX_updateGradInput_1D(inputImage, inputGrids,grad_input , grad_grids, grad_output)

print('cpu backward time taking:', time.time() - start)

#grad_output_cuda=grad_output.cuda(device)
start = time.time()
my_lib_1D.BilinearSamplerBCW_updateGradInput_cuda_1D(inputImage_cuda, inputGrids_cuda, grad_input_cuda, grad_grids_cuda, grad_output_cuda,
                                                  device_c)
print('gpu backward time taking:', time.time() - start)

out_input_grad = (grad_input-grad_input_cuda.cpu()).view(-1,1).sum()
out_grids_grad = (grad_grids-grad_grids_cuda.cpu()).view(-1,1).sum()
out_output_grad= (grad_output-grad_output_cuda.cpu()).view(-1,1).sum()
print('difference of sampling cpu sum of \n input_grad:{}, \n grid_grad: {},\n out_output_grad:{}'.format(out_input_grad,out_grids_grad,out_output_grad))
