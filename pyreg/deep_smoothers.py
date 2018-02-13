import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from data_wrapper import USE_CUDA, MyTensor, AdaptVal

import math

def compute_omt_penalty(weights, multi_gaussian_stds,desired_power=2.0):

    # weights: B x weights x X x Y

    if weights.size()[1] != len(multi_gaussian_stds):
        raise ValueError('Number of weights need to be the same as number of Gaussians. Format recently changed for weights to B x weights x X x Y')

    penalty = Variable(MyTensor(1).zero_(), requires_grad=False)
    batch_size = weights.size()[0]
    max_std = max(multi_gaussian_stds)
    for i,s in enumerate(multi_gaussian_stds):
        penalty += ((weights[:,i,...]).sum())*((s-max_std)**desired_power)

    penalty /= batch_size

    return penalty


class DeepSmoothingModel(nn.Module):
    """
    Base class for mini neural network which takes as an input a set of smoothed velocity field as
    well as input images and predicts weights for a multi-Gaussian smoothing from this
    Enforces the same weighting for all the dimensions of the vector field to be smoothed

    """

    def __init__(self, nr_of_gaussians, gaussian_stds, dim, nr_of_image_channels=1, params=None):
        super(DeepSmoothingModel, self).__init__()

        self.nr_of_image_channels = nr_of_image_channels
        self.dim = dim

        # check that the largest standard deviation is the largest one
        if max(gaussian_stds) > gaussian_stds[-1]:
            raise ValueError('The last standard deviation needs to be the largest')

        self.params = params

        self.nr_of_gaussians = nr_of_gaussians
        self.gaussian_stds = gaussian_stds
        self.min_weight = 0.0001

        self.computed_weights = None
        """stores the computed weights if desired"""
        self.current_penalty = None
        """to stores the current penalty (for example OMT) after running through the model"""

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(0.25 / n))
            elif isinstance(m, nn.BatchNorm2d):
                pass
            elif isinstance(m, nn.Linear):
                pass

    def get_number_of_image_channels_from_state_dict(self, state_dict, dim):
        """legacy; to support velocity fields as input channels"""
        return self.nr_of_image_channels

    def get_number_of_input_channels(self, nr_of_image_channels, dim):
        """
        legacy; to support velocity fields as input channels
        currently only returns the number of image channels, but if something else would be used as
        the network input, would need to return the total number of inputs
        """
        return self.nr_of_image_channels

    def get_computed_weights(self):
        return self.computed_weights

    def spatially_average(self, x):
        """
        does spatial averaging of a 2D image with potentially multiple batches: format B x X x Y
        :param x:
        :return:
        """

        # first set the boundary to zero (first dimension is batch and this only works for 2D for now)

        y = torch.zeros_like(x)

        # now do local averaging in the interior using the four neighborhood
        y[:, 1:-1, 1:-1] = 0.5 * x[:, 1:-1, 1:-1] + 0.125 * (
                    x[:, 0:-2, 1:-1] + x[:, 2:, 1:-1] + x[:, 1:-1, 0:-2] + x[:, 1:-1, 2:])

        # do the corners
        y[:, 0, 0] = 8. / 6. * (0.5 * x[:, 0, 0] + 0.125 * (x[:, 1, 0] + x[:, 0, 1]))
        y[:, 0, -1] = 8. / 6. * (0.5 * x[:, 0, -1] + 0.125 * (x[:, 1, -1] + x[:, 0, -2]))
        y[:, -1, 0] = 8. / 6. * (0.5 * x[:, -1, 0] + 0.125 * (x[:, -2, 0] + x[:, -1, 1]))
        y[:, -1, -1] = 8. / 6. * (0.5 * x[:, -1, -1] + 0.125 * (x[:, -2, -1] + x[:, -1, -2]))

        # and lastly the edges
        y[:, 1:-1, 0] = 8. / 7. * (0.5 * x[:, 1:-1, 0] + 0.125 * (x[:, 1:-1, 1] + x[:, 0:-2, 0] + x[:, 2:, 0]))
        y[:, 0, 1:-1] = 8. / 7. * (0.5 * x[:, 0, 1:-1] + 0.125 * (x[:, 1, 1:-1] + x[:, 0, 0:-2] + x[:, 0, 2:]))
        y[:, 1:-1, -1] = 8. / 7. * (0.5 * x[:, 1:-1, -1] + 0.125 * (x[:, 1:-1, -2] + x[:, 0:-2, -1] + x[:, 2:, -1]))
        y[:, -1, 1:-1] = 8. / 7. * (0.5 * x[:, -1, 1:-1] + 0.125 * (x[:, -2, 1:-1] + x[:, -1, 0:-2] + x[:, -1, 2:]))

        return y

    def get_current_penalty(self):
        """
        returns the current penalty for the weights (OMT penalty here)
        :return:
        """
        return self.current_penalty

#nn.Conv2d(nr_of_input_channels,self.nr_of_features_per_layer[0],self.kernel_sizes[0], padding=(self.kernel_sizes[0]-1)//2)
#torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

class encoder_block_2d(nn.Module):
    def __init__(self, input_feature, output_feature, use_dropout):
        super(encoder_block_2d, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=input_feature, out_channels=output_feature,
                                    kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_inblock1 = nn.Conv2d(in_channels=output_feature, out_channels=output_feature,
                                       kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_inblock2 = nn.Conv2d(in_channels=output_feature, out_channels=output_feature,
                                       kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_pooling = nn.Conv2d(in_channels=output_feature, out_channels=output_feature,
                                      kernel_size=2, stride=2, padding=0, dilation=1)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2)
    def apply_dropout(self, input):
        if self.use_dropout:
            return self.dropout(input)
        else:
            return input
    def forward(self, x):
        output = self.conv_input(x)
        output = self.apply_dropout(self.prelu1(output))
        output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)))
        output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)))
        return self.prelu4(self.conv_pooling(output))


class decoder_block_2d(nn.Module):
    def __init__(self, input_feature, output_feature, pooling_filter,use_dropout,last_block=False):
        super(decoder_block_2d, self).__init__()
        # todo: check padding here, not sure if it is the right thing to do
        self.conv_unpooling = nn.ConvTranspose2d(in_channels=input_feature, out_channels=input_feature,
                                                 kernel_size=pooling_filter, stride=2, padding=0,output_padding=0)
        self.conv_inblock1 = nn.Conv2d(in_channels=input_feature, out_channels=input_feature,
                                       kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_inblock2 = nn.Conv2d(in_channels=input_feature, out_channels=input_feature,
                                       kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv_output = nn.Conv2d(in_channels=input_feature, out_channels=output_feature,
                                     kernel_size=3, stride=1, padding=1, dilation=1)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        self.use_dropout = use_dropout
        self.last_block = last_block
        self.dropout = nn.Dropout(0.2)
        self.output_feature = output_feature

    def apply_dropout(self, input):
        if self.use_dropout:
            return self.dropout(input)
        else:
            return input
    def forward(self, x):
        output = self.prelu1(self.conv_unpooling(x))
        output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)))
        output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)))
        if self.last_block: # generates final output
            return self.conv_output(output);
        else: # generates intermediate results
            return self.apply_dropout(self.prelu4(self.conv_output(output)))


class EncoderDecoderSmoothingModel(DeepSmoothingModel):
    """
    Similar to the model used in Quicksilver
    """
    def __init__(self, nr_of_gaussians, gaussian_stds, dim, nr_of_image_channels=1, params=None ):
        super(EncoderDecoderSmoothingModel, self).__init__(nr_of_gaussians=nr_of_gaussians,\
                                                                     gaussian_stds=gaussian_stds,\
                                                                     dim=dim,\
                                                                     nr_of_image_channels=nr_of_image_channels,\
                                                                     params=params)

        # is 64 in quicksilver paper
        feature_num = self.params[('number_of_features_in_first_layer', 32, 'number of features in the first encoder layer (64 in quicksilver)')]
        use_dropout= self.params[('use_dropout',False,'use dropout for the layers')]
        self.use_separate_decoders_per_gaussian = self.params[('use_separate_decoders_per_gaussian',True,'if set to true separte decoder branches are used for each Gaussian')]
        self.use_momentum_as_input = self.params[('use_momentum_as_input',True,'If true, uses the image and the momentum as input')]

        if self.use_momentum_as_input:
            self.encoder_1 = encoder_block_2d(self.get_number_of_input_channels(nr_of_image_channels,dim), feature_num, use_dropout)
        else:
            self.encoder_1 = encoder_block_2d(1, feature_num, use_dropout)

        self.encoder_2 = encoder_block_2d(feature_num, feature_num * 2, use_dropout)

        # todo: maybe have one decoder for each Gaussian here.
        # todo: the current version seems to produce strange gridded results

        if self.use_separate_decoders_per_gaussian:
            self.decoder_1 = nn.ModuleList()
            self.decoder_2 = nn.ModuleList()
            # create two decoder blocks for each Gaussian
            for g in range(nr_of_gaussians):
                self.decoder_1.append( decoder_block_2d(feature_num * 2, feature_num, 2, use_dropout) )
                self.decoder_2.append( decoder_block_2d(feature_num, 1, 2, use_dropout, last_block=True) )
        else:
            self.decoder_1 = decoder_block_2d(feature_num * 2, feature_num, 2, use_dropout)
            self.decoder_2 = decoder_block_2d(feature_num, nr_of_gaussians, 2, use_dropout, last_block=True)  # 3?


    def get_number_of_input_channels(self, nr_of_image_channels, dim):
        """
        legacy; to support velocity fields as input channels
        currently only returns the number of image channels, but if something else would be used as
        the network input, would need to return the total number of inputs
        """
        if self.use_momentum_as_input:
            return self.nr_of_image_channels + dim
        else:
            return self.nr_of_image_channels

    def forward(self, multi_smooth_v, I, m, global_multi_gaussian_weights,
                encourage_spatial_weight_consistency=True, retain_weights=False):

        # format of multi_smooth_v is multi_v x batch x channels x X x Y
        # (channels here are the vector field components)
        # I is the image, m is the momentum. multi_smooth_v is the momentum smoothed with the various kernels

        """
        First make sure that the multi_smooth_v has the correct dimension.
        I.e., the correct spatial dimension and one output for each Gaussian (multi_v)
        """
        sz_mv = multi_smooth_v.size()
        dim_mv = sz_mv[2]  # format is
        # currently only implemented in 2D
        assert (dim_mv == 2)
        assert (sz_mv[0] == self.nr_of_gaussians)

        # create the output tensor: will be of dimension: batch x channels x X x Y
        ret = AdaptVal(Variable(torch.FloatTensor(*sz_mv[1:]), requires_grad=False))

        # now determine the size for the weights
        # Since the smoothing will be the same for all spatial directions (for a velocity field),
        # this basically amounts to cutting out the channels; i.e., multi_v x batch x X x Y
        sz_weight = list(sz_mv)
        sz_weight = [sz_weight[1]] + [sz_weight[0]] + sz_weight[3:]

        # if the weights should be stored (for debugging), create the tensor to store them here
        if retain_weights and self.computed_weights is None:
            print('DEBUG: retaining smoother weights - turn off to minimize memory consumption')
            # create storage; batch x size v x X x Y
            self.computed_weights = torch.FloatTensor(*sz_weight)

        # here is the actual network; maybe abstract this out later
        if self.use_momentum_as_input:
            x = torch.cat([I,m],dim=1)
        else:
            # the input to the network is simply the image
            x = I

        encoder_output = self.encoder_2(self.encoder_1(x))

        if self.use_separate_decoders_per_gaussian:
            # here we have seperate decoder outputs for the different Gaussians
            # should give it more flexibility
            decoder_output_individual = []
            for g in range(self.nr_of_gaussians):
                decoder_output_individual.append( self.decoder_2[g](self.decoder_1[g](encoder_output)) )
            decoder_output = torch.cat(decoder_output_individual, dim=1);
        else:
            decoder_output = self.decoder_2(self.decoder_1(encoder_output))

        # now we are ready for the softmax
        weights = F.softmax(decoder_output, dim=1)

        # ends here

        # multiply the velocity fields by the weights and sum over them
        # this is then the multi-Gaussian output

        # now we apply this weight across all the channels; weight output is B x weights x X x Y
        for n in range(self.dim):
            # reverse the order so that for a given channel we have batch x multi_velocity x X x Y
            # i.e., the multi-velocity field output is treated as a channel
            # reminder: # format of multi_smooth_v is multi_v x batch x channels x X x Y
            # (channels here are the vector field components); i.e. as many as there are dimensions
            # each one of those should be smoothed the same

            # roc should be: batch x multi_v x X x Y
            roc = torch.transpose(multi_smooth_v[:, :, n, ...], 0, 1)
            yc = torch.sum(roc * weights, dim=1)
            ret[:, n, ...] = yc  # ret is: batch x channels x X x Y

        self.current_penalty = compute_omt_penalty(weights, self.gaussian_stds)

        if retain_weights:
            # todo: change visualization to work with this new format:
            # B x weights x X x Y instead of weights x B x X x Y
            self.computed_weights[:] = weights.data

        return ret

class SimpleConsistentWeightedSmoothingModel(DeepSmoothingModel):
    """
    Mini neural network which takes as an input a set of smoothed velocity field as
    well as input images and predicts weights for a multi-Gaussian smoothing from this
    Enforces the same weighting for all the dimensions of the vector field to be smoothed

    """
    def __init__(self, nr_of_gaussians, gaussian_stds, dim, nr_of_image_channels=1, params=None ):
        super(SimpleConsistentWeightedSmoothingModel, self).__init__(nr_of_gaussians=nr_of_gaussians,\
                                                                     gaussian_stds=gaussian_stds,\
                                                                     dim=dim,\
                                                                     nr_of_image_channels=nr_of_image_channels,\
                                                                     params=params)

        self.kernel_sizes = self.params[('kernel_sizes',[7,7],'size of the convolution kernels')]
        self.use_momentum_as_input = self.params[('use_momentum_as_input',True,'If true, uses the image and the momentum as input')]

        # check that all the kernel-size are odd
        for ks in self.kernel_sizes:
            if ks%2== 0:
                raise ValueError('Kernel sizes need to be odd')

        # the last layers feature number is not specified as it will simply be the number of Gaussians
        self.nr_of_features_per_layer = self.params[('number_of_features_per_layer',[20],'Number of features for the convolution later; last one is set to number of Gaussians')]
        # add the number of Gaussians to the last layer
        self.nr_of_features_per_layer = self.nr_of_features_per_layer + [nr_of_gaussians]

        self.nr_of_layers = len(self.kernel_sizes)
        assert( self.nr_of_layers== len(self.nr_of_features_per_layer) )

        self.use_relu = self.params[('use_relu',True,'if set to True uses Relu, otherwise sigmoid')]

        self.conv_layers = None

        # needs to be initialized here, otherwise the optimizer won't see the modules from ModuleList
        # todo: figure out how to do ModuleList initialization not in __init__
        # todo: this would allow removing dim and nr_of_image_channels from interface
        # todo: because it could be compute on the fly when forward is executed
        self._init(self.nr_of_image_channels,dim=self.dim)


    def get_number_of_input_channels(self, nr_of_image_channels, dim):
        """
        legacy; to support velocity fields as input channels
        currently only returns the number of image channels, but if something else would be used as
        the network input, would need to return the total number of inputs
        """
        if self.use_momentum_as_input:
            return self.nr_of_image_channels + dim
        else:
            return self.nr_of_image_channels

    def _init(self,nr_of_image_channels,dim):
        """
        Initalizes all the conv layers
        :param nr_of_image_channels:
        :param dim:
        :return:
        """

        assert(self.nr_of_layers>0)
        assert(dim==2)

        nr_of_input_channels = self.get_number_of_input_channels(nr_of_image_channels,dim)

        convs = [None]*self.nr_of_layers

        # first layer
        convs[0] = nn.Conv2d(nr_of_input_channels,
                                  self.nr_of_features_per_layer[0],
                                  self.kernel_sizes[0], padding=(self.kernel_sizes[0]-1)//2)

        # all the intermediate layers and the last one
        for l in range(self.nr_of_layers-1):
            convs[l+1] = nn.Conv2d(self.nr_of_features_per_layer[l],
                                        self.nr_of_features_per_layer[l+1],
                                        self.kernel_sizes[l+1],
                                        padding=(self.kernel_sizes[l+1]-1)//2)

        self.conv_layers = nn.ModuleList()
        for c in convs:
            self.conv_layers.append(c)

        self._initialize_weights()

    def forward(self, multi_smooth_v, I, m, global_multi_gaussian_weights,
                encourage_spatial_weight_consistency=True, retain_weights=False):

        # format of multi_smooth_v is multi_v x batch x channels x X x Y
        # (channels here are the vector field components)
        # I is the image, m is the momentum. multi_smooth_v is the momentum smoothed with the various kernels

        """
        First make sure that the multi_smooth_v has the correct dimension.
        I.e., the correct spatial dimension and one output for each Gaussian (multi_v)
        """
        sz_mv = multi_smooth_v.size()
        dim_mv = sz_mv[2] # format is
        # currently only implemented in 2D
        assert(dim_mv==2)
        assert(sz_mv[0]==self.nr_of_gaussians)

        # create the output tensor: will be of dimension: batch x channels x X x Y
        ret = AdaptVal(Variable(torch.FloatTensor(*sz_mv[1:]), requires_grad=False))

        # now determine the size for the weights
        # Since the smoothing will be the same for all spatial directions (for a velocity field),
        # this basically amounts to cutting out the channels; i.e., multi_v x batch x X x Y
        sz_weight = list(sz_mv)
        sz_weight = [sz_weight[1]] + [sz_weight[0]] + sz_weight[3:]

        # if the weights should be stored (for debugging), create the tensor to store them here
        if retain_weights and self.computed_weights is None:
            print('DEBUG: retaining smoother weights - turn off to minimize memory consumption')
            # create storage; batch x size v x X x Y
            self.computed_weights = torch.FloatTensor(*sz_weight)


        if self.use_momentum_as_input:
            x = torch.cat([I,m],dim=1)
        else:
            # the input to the network is simply the image
            x = I

        # now let's apply all the convolution layers, until the last
        # (because the last one is not relu-ed
        for l in range(len(self.conv_layers) - 1):
            if self.use_relu:
                x = F.relu(self.conv_layers[l](x))
            else:
                x = F.sigmoid(self.conv_layers[l](x))

        # and now apply the last one without an activation for now
        # because we want to have the ability to smooth *before* the softmax
        # this is similar to smoothing in the logit domain for an active mean field approach
        x = self.conv_layers[-1](x)

        # now do the smoothing if desired
        y = torch.zeros_like(x)
        if encourage_spatial_weight_consistency:
            # now we do local averaging for the weights and force the boundaries to zero
            for g in range(self.nr_of_gaussians):
                y[:, g, ...] = self.spatially_average(x[:, g, ...])
        else:
            y = x

        # now we are ready for the softmax
        weights = F.softmax(y, dim=1)

        # multiply the velocity fields by the weights and sum over them
        # this is then the multi-Gaussian output

        # now we apply this weight across all the channels; weight output is B x weights x X x Y
        for n in range(self.dim):
            # reverse the order so that for a given channel we have batch x multi_velocity x X x Y
            # i.e., the multi-velocity field output is treated as a channel
            # reminder: # format of multi_smooth_v is multi_v x batch x channels x X x Y
            # (channels here are the vector field components); i.e. as many as there are dimensions
            # each one of those should be smoothed the same

            # roc should be: batch x multi_v x X x Y
            roc = torch.transpose(multi_smooth_v[:, :, n, ...], 0, 1)
            yc = torch.sum(roc*weights,dim=1)
            ret[:, n, ...] = yc # ret is: batch x channels x X x Y

        self.current_penalty = compute_omt_penalty(weights,self.gaussian_stds)

        if retain_weights:
            # todo: change visualization to work with this new format:
            # B x weights x X x Y instead of weights x B x X x Y
            self.computed_weights[:] = weights.data

        return ret


class DeepSmootherFactory(object):
    """
    Factory to quickly create different types of deep smoothers.
    """

    def __init__(self, nr_of_gaussians, gaussian_stds, dim, nr_of_image_channels=1 ):
        self.nr_of_gaussians = nr_of_gaussians
        """number of Gaussians as input"""
        self.gaussian_stds = gaussian_stds
        """stds of the Gaussians"""
        self.dim = dim
        """dimension of input image"""
        self.nr_of_image_channels = nr_of_image_channels
        """number of channels the image has (currently only one is supported)"""

        if self.nr_of_image_channels!=1:
            raise ValueError('Currently only one image channel supported')


    def create_deep_smoother(self, params ):
        """
        Create the desired deep smoother
        :param params: ParamterDict() object to hold paramters which should be passed on
        :return: returns the deep smoother
        """

        cparams = params[('deep_smoother',{})]
        smootherType = cparams[('type', 'simple_consistent','type of deep smoother (simple_consistent|encoder_decoder)')]
        if smootherType=='simple_consistent':
            return SimpleConsistentWeightedSmoothingModel(nr_of_gaussians=self.nr_of_gaussians,
                                                          gaussian_stds=self.gaussian_stds,
                                                          dim=self.dim,
                                                          nr_of_image_channels=self.nr_of_image_channels,
                                                          params=cparams)
        elif smootherType=='encoder_decoder':
            return EncoderDecoderSmoothingModel(nr_of_gaussians=self.nr_of_gaussians,
                                                  gaussian_stds=self.gaussian_stds,
                                                  dim=self.dim,
                                                  nr_of_image_channels=self.nr_of_image_channels,
                                                  params=cparams)
        else:
            raise ValueError('Deep smoother: ' + smootherType + ' not known')