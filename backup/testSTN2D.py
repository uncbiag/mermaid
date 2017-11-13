from __future__ import print_function

import torch
from cffi import FFI
import time
ffi = FFI()

from pyreg.libraries._ext import my_lib_2D
from pyreg.libraries._ext import my_lib_nd

device = torch.cuda.current_device()
device_c = ffi.new("int *")
device_c[0] = device
nframes = 3
height = 120
width = 126
ratio =2
grid_height = int(height/ratio)
grid_width= int(width/ratio)
channels = 3
device = torch.cuda.current_device()

inputImage = torch.randn(nframes, channels, width, height)
inputGrids = (torch.rand(nframes, 2,  grid_width,grid_height)*2 -1)
output = torch.rand(nframes, channels, grid_width,grid_height)



inputImage_cuda = inputImage.cuda(device)
inputGrids_cuda = inputGrids.cuda(device)
output_cuda = output.cuda(device)

start = time.time()
my_lib_nd.BilinearSamplerBCXY_updateOutput_2D(inputImage, inputGrids, output)
print('sampling cpu time taking:', time.time() - start)

start = time.time()
my_lib_2D.BilinearSamplerBCWH_updateOutput_cuda_2D(inputImage_cuda, inputGrids_cuda, output_cuda, device_c)
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
my_lib_nd.BilinearSamplerBCXY_updateGradInput_2D(inputImage, inputGrids,grad_input , grad_grids, grad_output)

print('cpu backward time taking:', time.time() - start)

#grad_output_cuda=grad_output.cuda(device)
start = time.time()
my_lib_2D.BilinearSamplerBCWH_updateGradInput_cuda_2D(inputImage_cuda, inputGrids_cuda, grad_input_cuda, grad_grids_cuda, grad_output_cuda,
                                                  device_c)
print('gpu backward time taking:', time.time() - start)

out_input_grad = (grad_input-grad_input_cuda.cpu()).view(-1,1).sum()
out_grids_grad = (grad_grids-grad_grids_cuda.cpu()).view(-1,1).sum()
out_output_grad= (grad_output-grad_output_cuda.cpu()).view(-1,1).sum()
print('difference of sampling cpu sum of \n input_grad:{}, \n grid_grad: {},\n out_output_grad:{}'.format(out_input_grad,out_grids_grad,out_output_grad))
