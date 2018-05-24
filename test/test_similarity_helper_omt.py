from __future__ import print_function
import os
import sys

from pyreg.similarity_helper_omt import OTSimilarityGradient,OTSimilarityHelper

sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../pyreg'))
sys.path.insert(0,os.path.abspath('../pyreg/libraries'))

import pylab as pl
import torch
import numpy as np
from torch.autograd import Variable









if __name__ == '__main__':
    n = 50
    m = 60
    dx = 1. / n
    dy = 1. / m
    spacing = torch.Tensor(2)
    spacing[0] = dx
    spacing[1] = dy
    ot = OTSimilarityGradient([dx, dy], [n,m], sinkhorn_iterations=500, std_dev=0.07)
    temp = torch.zeros(n, m)
    temp[10:30, 10:30] = 1
    I0 = Variable(temp, requires_grad=True)
    temp2 = torch.zeros(n, m)
    temp2[9:31, 9:31] = 1
    I1 = Variable(temp2, requires_grad=True)
    a, convergence = ot.compute_similarity(I0, I1)
    #pl.plot(range(len(convergence)), convergence)
    #pl.show()
    print(a)

    pl.figure()
    xxx, yyy = np.meshgrid(np.linspace(0, n, n), np.linspace(0, m, m))
    xx = xxx.transpose()
    yy = yyy.transpose()
    spacingbis = 2
    phi = Variable(torch.zeros(2,n,m),requires_grad = True)
    out = OTSimilarityHelper.apply(phi,I0, I1, Variable(torch.zeros(I0.size())), Variable(torch.zeros(I1.size())),
                              Variable(spacing))
    out.backward()
    gradientTorch = -phi.grad.data.numpy()
    pl.imshow((I1.data.numpy() - I0.data.numpy()).transpose(), origin="lower")
    pl.quiver(xx[0:n:spacingbis, 0:m:spacingbis], yy[0:n:spacingbis, 0:m:spacingbis],gradientTorch[0, 0:n:spacingbis, 0:m:spacingbis], gradientTorch[1, 0:n:spacingbis, 0:m:spacingbis],color="red")
    pl.show()
