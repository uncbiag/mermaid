from __future__ import print_function
import os
import sys

from mermaid.similarity_helper_omt import OTSimilarityGradient,OTSimilarityHelper

sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../mermaid'))
sys.path.insert(0,os.path.abspath('../mermaid/libraries'))

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
    I0 = temp
    I0.requires_grad = True
    temp2 = torch.zeros(n, m)
    temp2[9:31, 9:31] = 1
    I1 = temp2
    I1.requires_grad = True
    a, convergence = ot.compute_similarity(I0, I1)
    #pl.plot(range(len(convergence)), convergence)
    #pl.show()
    print(a)

    pl.figure()
    xxx, yyy = np.meshgrid(np.linspace(0, n, n), np.linspace(0, m, m))
    xx = xxx.transpose()
    yy = yyy.transpose()
    spacingbis = 2
    phi = torch.zeros(2,n,m)
    phi.requires_grad = True
    out = OTSimilarityHelper.apply(phi,I0, I1, torch.zeros(I0.size()), torch.zeros(I1.size(),
                              spacing))
    out.backward()
    gradientTorch = -phi.grad.detach().cpu().numpy()
    pl.imshow((I1.detach().cpu().numpy() - I0.detach().cpu().numpy()).transpose(), origin="lower")
    pl.quiver(xx[0:n:spacingbis, 0:m:spacingbis], yy[0:n:spacingbis, 0:m:spacingbis],gradientTorch[0, 0:n:spacingbis, 0:m:spacingbis], gradientTorch[1, 0:n:spacingbis, 0:m:spacingbis],color="red")
    pl.show()
