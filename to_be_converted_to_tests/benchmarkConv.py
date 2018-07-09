from __future__ import print_function
from builtins import range
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np

import custom_pytorch_extensions as ce


sz = 50

dim=-1

if dim==-1: # test nd version via fft
    f = np.random.rand(sz,sz,sz)
    fconvFilter = ce.FourierConvolution(f,[sz,sz,sz])

    #F = np.fft.fftn(f)
    A = torch.from_numpy(np.random.rand(sz,sz,sz))
    A.requires_grad = True

    start = time.time()
    for iter in range(100):
        # print( 'Iteration: ' + str(iter))
        #np.fft.fftn(np.fft.fftn(A)*F)
        fconvFilter(A)
    print(('time:', time.time() - start))


elif dim==2:
    filters = torch.randn(1, 1, 32, 32)
    inputs = torch.randn(1, 1, 170, 170)

    start = time.time()
    for iter in range(1000):
        # print( 'Iteration: ' + str(iter))
        F.conv2d(inputs, filters)
    print(('time:', time.time() - start))

elif dim==3:
    filters = torch.randn(1, 1, 10, 10, 10)
    inputs = torch.randn(1, 1, sz, sz, sz)

    start = time.time()
    for iter in range(100):
        #print( 'Iteration: ' + str(iter))
        F.conv3d(inputs, filters)
    print(('time:', time.time() - start))