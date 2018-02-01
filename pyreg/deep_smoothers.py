import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from data_wrapper import USE_CUDA, MyTensor, AdaptVal

import math

def compute_omt_penalty(weights, multi_gaussian_stds):

    # weights: weights x B x X x Y x Z

    penalty = Variable(MyTensor(1).zero_(), requires_grad=False)
    batch_size = weights.size()[1]
    max_std = max(multi_gaussian_stds)
    for i,s in enumerate(multi_gaussian_stds):
        penalty += ((weights[i,...]).sum())*((s-max_std)**2)

    penalty /= batch_size

    return penalty

class ConsistentWeightedSmoothingModel(nn.Module):
    """
    Mini neural network which takes as an input a set of smoothed velocity field as
    well as input images and predicts weights for a multi-Gaussian smoothing from this
    Enforces the same weighting for all the dimensions of the vector field to be smoothed

    """
    def __init__(self, nr_of_gaussians, gaussian_stds, dim, nr_of_image_channels=1, params=None ):
        super(ConsistentWeightedSmoothingModel, self).__init__()

        self.nr_of_image_channels = nr_of_image_channels
        self.dim = dim

        # check that the largest standard deviation is the largest one
        if max(gaussian_stds)>gaussian_stds[-1]:
            raise ValueError('The last standard deviation needs to be the largest')

        cparams = params[('deep_smoother', {})]
        self.params = cparams

        self.use_velocity_fields_as_network_input = self.params[('use_velocity_fields_as_network_input', True, 'Also uses the velocity fields as inputs of the network')]
        self.kernel_sizes = self.params[('kernel_sizes',[7,7],'size of the convolution kernels')]
        self.one_directional_weight_change = self.params[('one_directional_weight_change',
                                                          True,
                                                          'If True modifies the network so that the largest standard deviation can only be reduced and the others increased')]

        # check that all the kernel-size are odd
        for ks in self.kernel_sizes:
            if ks%2== 0:
                raise ValueError('Kernel sizes need to be odd')

        self.nr_of_features_per_layer = self.params[('number_of_features_per_layer',[20],'Number of feartures for the convolution later; last one is set to number of Gaussians')]

        # add the number of Gaussians to the last layer
        self.nr_of_features_per_layer = self.nr_of_features_per_layer

        self.nr_of_layers = len(self.kernel_sizes)
        assert( self.nr_of_layers== len(self.nr_of_features_per_layer)+1 )

        self.nr_of_gaussians = nr_of_gaussians
        self.gaussian_stds = gaussian_stds
        self.min_weight = 0.0001
        self.is_initialized = False

        self.conv_layers = None

        # needs to be initialized here, otherwise the optimizer won't see the modules from ModuleList
        # todo: figure out how to do ModuleList initialization not in __init__
        # todo: this would allow removing dim and nr_of_image_channels from interface
        # todo: because it could be compute on the fly when forward is executed
        self._init(self.nr_of_image_channels,dim=self.dim)

        self.computed_weights = None
        self.dim = None

        self.current_penalty = None


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(0.25 / n))
            elif isinstance(m, nn.BatchNorm2d):
                pass
            elif isinstance(m, nn.Linear):
                pass


    def get_number_of_image_channels_from_state_dict(self,state_dict,dim):
        nr_of_input_channels = state_dict['weighted_smoothing_net.conv_layers.0.weight'].size()[1]

        if self.use_velocity_fields_as_network_input:
            nr_of_image_channels = nr_of_input_channels-self.nr_of_gaussians * dim
        else:
            nr_of_image_channels = nr_of_input_channels

        return nr_of_image_channels


    def get_number_of_input_channels(self,nr_of_image_channels,dim):
        nr_of_input_channels = nr_of_image_channels
        if self.use_velocity_fields_as_network_input:
            nr_of_input_channels += self.nr_of_gaussians * dim

        return nr_of_input_channels

    def get_computed_weights(self):
        return self.computed_weights

    def _init(self,nr_of_image_channels,dim):
        self.is_initialized = True

        assert(self.nr_of_layers>0)

        nr_of_input_channels = self.get_number_of_input_channels(nr_of_image_channels,dim)

        convs = [None]*self.nr_of_layers

        # first layer
        convs[0] = nn.Conv2d(nr_of_input_channels,
                                  self.nr_of_features_per_layer[0],
                                  self.kernel_sizes[0], padding=(self.kernel_sizes[0]-1)//2)

        # all the intermediate layers
        for l in range(self.nr_of_layers-2):
            convs[l+1] = nn.Conv2d(self.nr_of_features_per_layer[l],
                                        self.nr_of_features_per_layer[l+1],
                                        self.kernel_size[l+1],
                                        padding=(self.kernel_sizes[l+1]-1)//2)

        # and now the last layer
        convs[-1] = nn.Conv2d(self.nr_of_features_per_layer[-1],
                                   self.nr_of_gaussians,
                                   self.kernel_sizes[-1],
                                   padding=(self.kernel_sizes[-1]-1)//2)

        self.conv_layers = nn.ModuleList()
        for c in convs:
            self.conv_layers.append(c)

        self._initialize_weights()

    def spatially_average(self,x):

        # first set the boundary to zero (first dimension is batch and this only works for 2D for now)

        y = torch.zeros_like(x)

        # now do local averaging in the interior using the four neighborhood
        y[:,1:-1,1:-1] = 0.5*x[:,1:-1,1:-1]+ 0.125*(x[:,0:-2,1:-1] + x[:,2:,1:-1] + x[:,1:-1,0:-2] + x[:,1:-1,2:])

        # do the corners
        y[:,0,0] = 8./6.*( 0.5*x[:,0,0] + 0.125*(x[:,1,0]+x[:,0,1]))
        y[:,0,-1] = 8./6.*(0.5*x[:,0,-1] + 0.125*(x[:,1,-1]+x[:,0,-2]))
        y[:,-1,0] = 8./6.*(0.5*x[:,-1,0] + 0.125*(x[:,-2,0]+x[:,-1,1]))
        y[:,-1,-1] = 8./6.*(0.5*x[:,-1,-1] + 0.125*(x[:,-2,-1]+x[:,-1,-2]))

        # and lastly the edges
        y[:,1:-1,0] = 8./7.*(0.5*x[:,1:-1,0] + 0.125*(x[:,1:-1,1]+x[:,0:-2,0]+x[:,2:,0]))
        y[:,0,1:-1] = 8./7.*(0.5*x[:,0,1:-1] + 0.125*(x[:,1,1:-1]+x[:,0,0:-2]+x[:,0,2:]))
        y[:,1:-1,-1] = 8./7.*(0.5*x[:,1:-1,-1] + 0.125*(x[:,1:-1,-2]+x[:,0:-2,-1]+x[:,2:,-1]))
        y[:,-1,1:-1] = 8./7.*(0.5*x[:,-1,1:-1] + 0.125*(x[:,-2,1:-1]+x[:,-1,0:-2]+x[:,-1,2:]))

        return y

    def spatially_average_and_set_boundary_to_zero(self,x):

        # first set the boundary to zero (first dimension is batch and this only works for 2D for now)

        y = torch.zeros_like(x)
        y[:] = x

        y[:,0,:] = 0
        y[:,-1,:] = 0
        y[:,:,0] = 0
        y[:,:,-1] =0

        # todo: do a less agressive averaging as inspatially_average
        # now do local averaging in the interior using the four neighborhood
        y[:,1:-1,1:-1] = 0.5*x[:,1:-1,1:-1]+ 0.125*(x[:,0:-2,1:-1] + x[:,2:,1:-1] + x[:,1:-1,0:-2] + x[:,1:-1,2:])

        return y

    def get_current_penalty(self):
        return self.current_penalty

    def forward(self, multi_smooth_v, I, global_multi_gaussian_weights,
                encourage_spatial_weight_consistency=True, retain_weights=False):

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

        # we need to deal with different types of input here
        # it can either be only the image, or the image and the velocity field

        if self.use_velocity_fields_as_network_input:
            # 1st transpose
            rot = torch.transpose(multi_smooth_v,0,1)
            # now create a new view that essentially stacks the channels
            sz_stacked = [sz[1]] + [sz[0]*sz[2]] + list(sz[3:])
            ro = rot.contiguous().view(*sz_stacked)

            # now we concatenate the image input
            # This image data should help identify how to smooth (this is an image)
            x = torch.cat((ro, I), 1)
        else: # only uses the image
            x = I

        # now let's apply all the convolution layers, until the last (because the last one is not relu-ed
        for l in range(len(self.conv_layers)-1):
            #x = F.relu(self.conv_layers[l](x))
            x = F.sigmoid(self.conv_layers[l](x))

        # and now apply the last one without a relu (because we want to support positive and negative weigths)
        x = self.conv_layers[-1](x) # they do not need to be positive (hence ReLU removed); only the absolute weights need to be

        if self.one_directional_weight_change: # force them to be positive; we only want to change weights in one direction
            x = F.sigmoid(x-6.0) # shifted sigmoid so that zero output roughly amounts to zero and only very positive output amounts to 1

        # need to be adapted for multiple D

        # now add the default weights and encourage spatial weight-consistency if desired

        y = torch.zeros_like(x)

        if encourage_spatial_weight_consistency:
            # now we do local averaging for the weights and force the boundaries to zero
            for g in range(self.nr_of_gaussians):
                y[:,g,...] = self.spatially_average(x[:,g,...])
        else:
            y = x

        z = torch.zeros_like(x)

        # loop over all the responses
        # we want to model deviations from the default values instead of the values themselves
        if self.one_directional_weight_change:
            for g in range(self.nr_of_gaussians-1):
                z[:, g, ...] = y[:, g, ...] + global_multi_gaussian_weights[g]
            # subtract for the last one (which will be the largest)
            g = self.nr_of_gaussians-1
            z[:, g, ...] =  -y[:, g, ...] + global_multi_gaussian_weights[g]
        else:
            for g in range(self.nr_of_gaussians):
                z[:, g, ...] = y[:, g, ...] + global_multi_gaussian_weights[g]

        #todo: maybe run through a sigmoid here

        # safeguard against values that are too small; this also safeguards against having all zero weights
        x = torch.clamp(z, min=self.min_weight)
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

        x = torch.transpose(x, 0, 1) # flip it back

        self.current_penalty = compute_omt_penalty(x,self.gaussian_stds)

        #import matplotlib.pyplot as plt
        #plt.clf()
        #plt.imshow(x[0,0,...].data.numpy())
        #plt.colorbar()
        #plt.title(str(self.current_penalty.data.numpy()))
        #plt.show()



        if retain_weights:
            self.computed_weights[:] = x.data

        return ret


class old_ConsistentWeightedSmoothingModel(nn.Module):
    """
    Mini neural network which takes as an input a set of smoothed velocity field as
    well as input images and predicts weights for a multi-Gaussian smoothing from this
    Enforces the same weighting for all the dimensions of the vector field to be smoothed
    """
    def __init__(self, nr_of_gaussians,multi_gaussian_stds, dim, nr_of_image_channels=1, params=None):
        super(old_ConsistentWeightedSmoothingModel, self).__init__()
        self.nr_of_image_channels = nr_of_image_channels
        self.dim = dim
        self.nr_of_gaussians = nr_of_gaussians
        self.min_weight = 0.0001
        self.is_initialized = False
        self.conv1 = None
        self.conv2 = None
        self.computed_weights = None
        self.dim = None
        self.gaussian_stds = multi_gaussian_stds
        self.params = params
        self.current_penalty = None

        self._int(self.nr_of_image_channels,self.dim)

    def get_current_penalty(self):
        return self.current_penalty

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

        x = torch.transpose(x.data, 0, 1)

        self.current_penalty = compute_omt_penalty(x,self.gaussian_stds)


        if retain_weights:
            #self.computed_weights[:] = torch.transpose(x.data, 0, 1)  # flip it back
            self.computed_weights[:] = x

        return ret

class WeightedSmoothingModel(nn.Module):
    """
    Mini neural network which takes as an input a set of smoothed velocity field as
    well as input images and predicts weights for a multi-Gaussian smoothing from this

    """
    def __init__(self, nr_of_gaussians,default_multi_gaussian_weights,dim,nr_of_image_channels=1, params=None):
        super(WeightedSmoothingModel, self).__init__()

        self.params = params
        self.nr_of_image_channels = nr_of_image_channels
        self.dim = dim
        self.nr_of_gaussians = nr_of_gaussians
        self.default_multi_gaussian_weights = default_multi_gaussian_weights
        self.min_weight = 0.0001
        self.is_initialized = False
        self.conv1 = None
        self.conv2 = None
        self.computed_weights = None

        self._init(self.nr_of_image_channels,self.dim)

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

