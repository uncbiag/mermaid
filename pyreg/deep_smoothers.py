import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from data_wrapper import USE_CUDA, MyTensor, AdaptVal

class WeightedSmoothingModel(nn.Module):
    """
    Mini neural network which takes as an input a set of smoothed velocity field as
    well as input images and predicts weights for a multi-Gaussian smoothing from this

    """
    def __init__(self, nr_of_gaussians):
        super(WeightedSmoothingModel, self).__init__()
        self.nr_of_gaussians = nr_of_gaussians
        self.is_initialized = False
        self.conv1 = None
        self.conv2 = None

    def _init(self,nr_of_image_channels):
        self.is_initialized = True
        self.conv1 = nn.Conv2d(self.nr_of_gaussians + nr_of_image_channels, 20, 5, padding=2)
        self.conv2 = nn.Conv2d(20, self.nr_of_gaussians, 5, padding=2)

    def forward(self, multi_smooth_v, I):
        if not self.is_initialized:
            nr_of_image_channels = I.size()[1]
            self._init(nr_of_image_channels)
        sz = multi_smooth_v.size()
        assert(sz[0]==self.nr_of_gaussians)
        nr_c = sz[2]

        new_sz = sz[1:]
        ret = AdaptVal(Variable(torch.FloatTensor(*new_sz),requires_grad=False))

        # loop over all channels
        for n in range(nr_c):
            # reverse the order so that for a given channel we have batchxmulti_velocityxXxYxZ
            # i.e., the multi-velocity field output is treated as a channel
            ro = torch.transpose(multi_smooth_v[:,:,n,...],0,1)
            # concatenate the data that should help identify how to smooth (this is an image)
            x = torch.cat( (ro,I), 1 )
            x = F.relu(self.conv1(x))
            # make the output non-negative
            x = F.relu(self.conv2(x))
            # now project it onto the unit ball
            x = x/torch.sum(x,dim=1,keepdim=True)
            # multiply the velocity fields by the weights and sum over them
            # this is then the multi-Gaussian output
            y = torch.sum(ro*x,dim=1)
            ret[:,n,...] = y

        return ret