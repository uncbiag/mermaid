from __future__ import print_function

import torch
import pytorch_fft.fft as fft
import time


A_real = torch.randn(1,50,50).cuda()
A_imag = torch.zeros(1,50,50).cuda()

A2_real = torch.zeros(1,2,50,50).cuda()
A2_real[0,0] = A_real[0]
A2_real[0,1] = A_real[0]

A_imag = torch.zeros(1,2,50,50).cuda()

ar, ai = fft.rfft2(A_real)
a2r, a2i = fft.rfft2(A2_real)




start = time.time()

ar, ai = fft.fft2(A_real, A_imag)

print("the complete fft cost {} time".format(time.time()-start))


start = time.time()

ar, ai = fft.rfft2(A_real)

print("the real fft cost {} time".format(time.time()-start))