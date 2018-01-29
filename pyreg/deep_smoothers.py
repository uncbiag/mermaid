import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from data_wrapper import USE_CUDA, MyTensor, AdaptVal

class ConsistentWeightedSmoothingModel(nn.Module):
    """
    Mini neural network which takes as an input a set of smoothed velocity field as
    well as input images and predicts weights for a multi-Gaussian smoothing from this
    Enforces the same weighting for all the dimensions of the vector field to be smoothed

    """
    def __init__(self, nr_of_gaussians,min_weight=0.0001):
        super(ConsistentWeightedSmoothingModel, self).__init__()
        self.nr_of_gaussians = nr_of_gaussians
        self.min_weight = 0.0001
        self.is_initialized = False
        self.conv1 = None
        self.conv2 = None
        self.computed_weights = None
        self.dim = None

    def get_computed_weights(self):
        return self.computed_weights

    def _init(self,nr_of_image_channels,dim):
        self.is_initialized = True
        self.conv1 = nn.Conv2d(self.nr_of_gaussians*dim + nr_of_image_channels, 20, 5, padding=2)
        self.conv2 = nn.Conv2d(20, self.nr_of_gaussians, 5, padding=2)

    def spatially_average_and_set_boundary_to_zero(self,x):

        # first set the boundary to zero (first dimension is batch and this only works for 2D for now)

        y = torch.zeros_like(x)
        y[:] = x

        y[:,0,:] = 0
        y[:,-1,:] = 0
        y[:,:,0] = 0
        y[:,:,-1] =0

        # now do local averaging in the interior using the four neighborhood
        y[:,1:-2,1:-2] = 0.25*(y[:,0:-3,1:-2] + y[:,2:-1,1:-2] + y[:,1:-2,0:-3] + y[:,1:-2,2:-1])

        return y

    def forward(self, multi_smooth_v, I, default_multi_gaussian_weights,
                encourage_spatial_weight_consistency=True,retain_weights=False):

        self.dim = multi_smooth_v.size()[2] # format is multi_v x batch x channels x X x Y x Z (channels here are the vector field components)

        # currently only implemented in 2D
        assert(self.dim==2)

        if not self.is_initialized:
            nr_of_image_channels = I.size()[1]
            self._init(nr_of_image_channels,self.dim)
        sz = multi_smooth_v.size()

        assert(sz[0]==self.nr_of_gaussians)
        nr_c = sz[2]

        new_sz = sz[1:] # simply the size of the resulting smoothed vector fields (collapsed along the multi-direction)
        ret = AdaptVal(Variable(torch.FloatTensor(*new_sz),requires_grad=False))

        sz_weight = list(sz)
        sz_weight = sz_weight[0:2] + sz_weight[3:]  # cut out the channels, since the smoothing will be the same for all spatial directions

        if retain_weights and self.computed_weights is None:
            print('DEBUG: retaining smoother weights - turn off to minimize memory consumption')
            # create storage; size vxbatchxxXxYxZ
            self.computed_weights = torch.FloatTensor(*sz_weight)

        # loop over all channels

        # todo: not clear what the input to the smoother should be. Maybe it should be the
        # todo: vector field in all spatial dimensions and not only component-by-component

        # first we stack up the response over all the channels to compute
        # consistent weights over the dimensions

        # we take multi_velocity x batch x channel x X x Y x Z format
        # and convert it to batch x [multi_velocity*channel] x X x Y x Z format

        # 1st transpose
        rot = torch.transpose(multi_smooth_v,0,1)
        # now create a new view that essentially stacks the channels
        sz_stacked = [sz[1]] + [sz[0]*sz[2]] + list(sz[3:])
        ro = rot.contiguous().view(*sz_stacked)
        # now we concatenate the image input
        # This image data should help identify how to smooth (this is an image)
        x = torch.cat((ro, I), 1)
        x = F.relu(self.conv1(x))
        # make the output non-negative
        #x = F.relu(self.conv2(x))

        x = self.conv2(x) # they do not need to be positive (hence ReLU removed); only the absolute weights need to be

        # need to be adapted for multiple D
        if encourage_spatial_weight_consistency:
            # now we do local averaging for the weights and force the boundaries to zero
            y = torch.zeros_like(x)
            for g in range(self.nr_of_gaussians):
                y[:,g,...] = self.spatially_average_and_set_boundary_to_zero(x[:,g,...])
                x[:, g, ...] = y[:, g, ...] + default_multi_gaussian_weights[g]
        else: # no spatial consistency
            # loop over all the responses
            # we want to model deviations from the default values instead of the values themselves
            for g in range(self.nr_of_gaussians):
                x[:, g, ...] = x[:, g, ...] + default_multi_gaussian_weights[g]

        #todo: maybe run through a sigmoid here

        # safeguard against values that are too small; this also safeguards against having all zero weights
        x = torch.clamp(x, min=self.min_weight)
        # now project it onto the unit ball
        x = x / torch.sum(x, dim=1, keepdim=True)
        # multiply the velocity fields by the weights and sum over them
        # this is then the multi-Gaussian output

        # now we apply this weight across all the channels; weight output is B x weights x X x Y x Z
        for n in range(nr_c):
            # reverse the order so that for a given channel we have batchxmulti_velocityxXxYxZ
            # i.e., the multi-velocity field output is treated as a channel
            roc = torch.transpose(multi_smooth_v[:, :, n, ...], 0, 1)
            yc = torch.sum(roc*x,dim=1)
            ret[:, n, ...] = yc

        if retain_weights:
            self.computed_weights[:] = torch.transpose(x.data, 0, 1)  # flip it back

        return ret

class WeightedSmoothingModel(nn.Module):
    """
    Mini neural network which takes as an input a set of smoothed velocity field as
    well as input images and predicts weights for a multi-Gaussian smoothing from this

    """
    def __init__(self, nr_of_gaussians,default_multi_gaussian_weights,min_weight=0.0001):
        super(WeightedSmoothingModel, self).__init__()
        self.nr_of_gaussians = nr_of_gaussians
        self.default_multi_gaussian_weights = default_multi_gaussian_weights
        self.min_weight = 0.0001
        self.is_initialized = False
        self.conv1 = None
        self.conv2 = None
        self.computed_weights = None

    def get_computed_weights(self):
        return self.computed_weights

    def _init(self,nr_of_image_channels,dim=None):
        self.is_initialized = True
        kernel_size = 7
        self.conv1 = nn.Conv2d(self.nr_of_gaussians + nr_of_image_channels, 20, kernel_size, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(20, self.nr_of_gaussians, kernel_size, padding=(kernel_size-1)//2)

    def forward(self, multi_smooth_v, I, retain_weights=False):

        if not self.is_initialized:
            nr_of_image_channels = I.size()[1]
            self._init(nr_of_image_channels)
        sz = multi_smooth_v.size()
        assert(sz[0]==self.nr_of_gaussians)
        nr_c = sz[2]

        new_sz = sz[1:]
        ret = AdaptVal(Variable(torch.FloatTensor(*new_sz),requires_grad=False))

        if retain_weights and self.computed_weights is None:
            print('DEBUG: retaining smoother weights - turn off to minimize memory consumption')
            # create storage; same dimension as the multi velocity field input vxbatchxchannelxXxYxZ
            self.computed_weights = torch.FloatTensor(*sz)

        # loop over all channels

        # todo: not clear what the input to the smoother should be. Maybe it should be the
        # todo: vector field in all spatial dimensions and not only component-by-component

        for n in range(nr_c):
            # reverse the order so that for a given channel we have batchxmulti_velocityxXxYxZ
            # i.e., the multi-velocity field output is treated as a channel
            ro = torch.transpose(multi_smooth_v[:,:,n,...],0,1)
            # concatenate the data that should help identify how to smooth (this is an image)
            x = torch.cat( (ro,I), 1 )
            x = F.relu(self.conv1(x))
            # make the output non-negative

            #x = F.relu(self.conv2(x))
            x = self.conv2(x)

            # loop over all the responses
            # we want to model deviations from the default values instead of the values themselves
            for g in range(self.nr_of_gaussians):
                x[:,g,...] = x[:,g,...] + self.default_multi_gaussian_weights[g]

            # safeguard against values that are too small; this also safeguards against having all zero weights
            x = torch.clamp(x,min=self.min_weight)
            # now project it onto the unit ball
            x = x/torch.sum(x,dim=1,keepdim=True)
            # multiply the velocity fields by the weights and sum over them
            # this is then the multi-Gaussian output
            y = torch.sum(ro*x,dim=1)

            ret[:,n,...] = y

            if retain_weights:
                self.computed_weights[:,:,n,...] =  torch.transpose(x.data,0,1) # flip it back

        return ret

class old_WeightedSmoothingModel(nn.Module):
    """
    Mini neural network which takes as an input a set of smoothed velocity field as
    well as input images and predicts weights for a multi-Gaussian smoothing from this

    """
    def __init__(self, nr_of_gaussians,default_multi_gaussian_weights=None):
        super(old_WeightedSmoothingModel, self).__init__()
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