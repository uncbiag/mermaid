from __future__ import print_function
# start with the setup

import os
import sys
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../mermaid'))
sys.path.insert(0,os.path.abspath('../mermaid/libraries'))

import torch
from cffi import FFI
import time
ffi = FFI()

import numpy as np
import numpy.testing as npt
import torch

from mermaid.libraries.functions.stn_nd import STNFunction_ND_BCXYZ

import unittest
import imp

try:
    imp.find_module('HtmlTestRunner')
    foundHTMLTestRunner = True
    import HtmlTestRunner
except ImportError:
    foundHTMLTestRunner = False

# done with all the setup

# testing code starts here

torch.backends.cudnn.deterministic = True

torch.manual_seed(999)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(999)


class Test_stn_1d(unittest.TestCase):

    def setUp(self):
        device = torch.cuda.current_device()
        device_c = ffi.new("int *")
        device_c[0] = device
        nframes = 3
        width = 10000  # 233333
        ratio = 2
        grid_width = int(width / ratio)
        channels = 5
        device = torch.cuda.current_device()
        device_c = ffi.new("int *")
        device_c[0] = device
        self.device = device
        self.device_c = device_c
        self.inputImage = torch.randn(nframes, channels, width)
        self.inputGrids = (torch.rand(nframes, 1, grid_width))
        self.output = torch.rand(nframes, channels, grid_width)
        self.inputImage_cuda = self.inputImage.cuda(device)
        self.inputGrids_cuda = self.inputGrids.cuda(device)
        self.output_cuda = self.output.cuda(device)

        spacing = np.array([1./(grid_width-1.)])
        self.stn = STNFunction_ND_BCXYZ(spacing)

    def tearDown(self):
        pass

    def test_forward(self):
        self.stn.forward_stn(self.inputImage, self.inputGrids, self.output,1, self.device_c,use_cuda=False)
        self.stn.forward_stn(self.inputImage_cuda, self.inputGrids_cuda, self.output_cuda, 1, self.device_c, use_cuda=True)
        npt.assert_almost_equal(self.output.numpy(), self.output_cuda.cpu().numpy(), decimal=4)

    def test_backward(self):
        grad_output = torch.randn(self.output.size())
        grad_input = torch.zeros(self.inputImage.size())
        grad_grids = torch.zeros(self.inputGrids.size())
        grad_output_cuda = grad_output.cuda(self.device)
        grad_input_cuda = grad_input.cuda(self.device)
        grad_grids_cuda = grad_grids.cuda(self.device)
        self.stn.backward_stn(self.inputImage, self.inputGrids, grad_input, grad_grids, grad_output,1, self.device_c,use_cuda=False)
        self.stn.backward_stn(self.inputImage_cuda, self.inputGrids_cuda, grad_input_cuda, grad_grids_cuda, grad_output_cuda,1, self.device_c, use_cuda=True)
        npt.assert_almost_equal(grad_input.numpy(), grad_input_cuda.cpu().numpy(), decimal=4)
        # easy to fail when the width large, it may due to two reason, one is the numerical unstable of derivative of floor, second is  atomicadd in gpu
        # in the code the dgrid is factor by image length, if the image length is too long, it will lead large dgrid
        npt.assert_almost_equal(grad_grids.numpy(), grad_grids_cuda.cpu().numpy(), decimal=1)
        npt.assert_almost_equal(grad_output.numpy(),grad_output_cuda.cpu().numpy(), decimal=4)


class Test_stn_2d(unittest.TestCase):

    def setUp(self):
        device = torch.cuda.current_device()
        device_c = ffi.new("int *")
        device_c[0] = device
        nframes = 3
        height = 120
        width = 100
        ratio = 1
        grid_height = int(height / ratio)
        grid_width = int(width / ratio)
        channels = 3
        device = torch.cuda.current_device()
        self.device = device
        self.device_c = device_c
        self.inputImage = torch.randn(nframes, channels, width, height)
        self.inputGrids = (torch.rand(nframes, 2, grid_width, grid_height))*2-1
        self.output = torch.zeros(nframes, channels, grid_width, grid_height)
        self.inputImage_cuda = self.inputImage.cuda(device)
        self.inputGrids_cuda = self.inputGrids.cuda(device)
        self.output_cuda = self.output.cuda(device)
        try:
            spacing = np.array([1./(grid_width-1.),1./(grid_height-1.)])
        except:
            spacing = [1,1,1]
        self.stn = STNFunction_ND_BCXYZ(spacing)

    def tearDown(self):
        pass

    def test_forward(self):
        self.stn.forward_stn(self.inputImage, self.inputGrids, self.output,2, self.device_c,use_cuda=False)
        self.stn.forward_stn(self.inputImage_cuda, self.inputGrids_cuda, self.output_cuda, 2, self.device_c, use_cuda=True)
        npt.assert_almost_equal(self.output.numpy(), self.output_cuda.cpu().numpy(), decimal=4)

    def test_backward(self):
        grad_output = torch.randn(self.output.size())
        grad_input = torch.zeros(self.inputImage.size())
        grad_grids = torch.zeros(self.inputGrids.size())
        grad_output_cuda = grad_output.cuda(self.device)
        grad_input_cuda = grad_input.cuda(self.device)
        grad_grids_cuda = grad_grids.cuda(self.device)
        self.stn.backward_stn(self.inputImage, self.inputGrids, grad_input, grad_grids, grad_output,2, self.device_c,use_cuda=False)
        self.stn.backward_stn(self.inputImage_cuda, self.inputGrids_cuda, grad_input_cuda, grad_grids_cuda, grad_output_cuda,2, self.device_c, use_cuda=True)
        npt.assert_almost_equal(grad_input.numpy(), grad_input_cuda.cpu().numpy(), decimal=4)
        # easy to fail when the width large, it may due to the atomicadd in gpu
        npt.assert_almost_equal(grad_grids.numpy(), grad_grids_cuda.cpu().numpy(), decimal=3)
        npt.assert_almost_equal(grad_output.numpy(),grad_output_cuda.cpu().numpy(), decimal=4)



class Test_stn_3d(unittest.TestCase):

    def setUp(self):
        device = torch.cuda.current_device()
        device_c = ffi.new("int *")
        device_c[0] = device
        nframes = 3
        depth = 10
        height = 137
        width = 100
        ratio = 1
        grid_depth = int(depth / ratio)
        grid_height = int(height / ratio)
        grid_width = int(width / ratio)
        channels = 3
        device = torch.cuda.current_device()
        self.device = device
        self.device_c = device_c
        self.inputImage = torch.randn(nframes, channels, width, height, depth)
        self.inputGrids = (torch.rand(nframes, 3, grid_width, grid_height, grid_depth))
        self.output = torch.rand(nframes, channels, grid_width, grid_height, grid_depth)
        self.inputImage_cuda = self.inputImage.cuda(device)
        self.inputGrids_cuda = self.inputGrids.cuda(device)
        self.output_cuda = self.output.cuda(device)
        try:
            spacing = np.array([1./(grid_width-1.),1./(grid_height-1.),1./(grid_depth-1.)])
        except:
            spacing  = np.array([1,1,1])
        self.stn = STNFunction_ND_BCXYZ(spacing)

    def tearDown(self):
        pass

    def test_forward(self):
        self.stn.forward_stn(self.inputImage, self.inputGrids, self.output,3, self.device_c,use_cuda=False)
        self.stn.forward_stn(self.inputImage_cuda, self.inputGrids_cuda, self.output_cuda, 3, self.device_c, use_cuda=True)
        npt.assert_almost_equal(self.output.numpy(), self.output_cuda.cpu().numpy(), decimal=4)

    def test_backward(self):
        grad_output = torch.randn(self.output.size())
        grad_input = torch.zeros(self.inputImage.size())
        grad_grids = torch.zeros(self.inputGrids.size())
        grad_output_cuda = grad_output.cuda(self.device)
        grad_input_cuda = grad_input.cuda(self.device)
        grad_grids_cuda = grad_grids.cuda(self.device)
        self.stn.backward_stn(self.inputImage, self.inputGrids, grad_input, grad_grids, grad_output,3, self.device_c,use_cuda=False)
        self.stn.backward_stn(self.inputImage_cuda, self.inputGrids_cuda, grad_input_cuda, grad_grids_cuda, grad_output_cuda,3, self.device_c, use_cuda=True)
        npt.assert_almost_equal(grad_input.numpy(), grad_input_cuda.cpu().numpy(), decimal=4)
        # easy to fail when the width large, it may due to the atomicadd in gpu
        npt.assert_almost_equal(grad_grids.numpy(), grad_grids_cuda.cpu().numpy(), decimal=3)
        npt.assert_almost_equal(grad_output.numpy(),grad_output_cuda.cpu().numpy(), decimal=4)


if __name__ == '__main__':
    raise("this test is for cuda version stn, since pytorch provide official support, it is no longer maintained")
    if foundHTMLTestRunner:
        if torch.cuda.device_count()==0:
            print('No CUDA devices found. Ignoring test.')
        else:
            unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='test_output'))
    else:
        if torch.cuda.device_count()==0:
            print('No CUDA devices found. Ignoring test.')
        else:
            unittest.main()
