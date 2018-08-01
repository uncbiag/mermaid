from __future__ import print_function
from __future__ import absolute_import
from builtins import object

from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass

import torch
import torch.nn as nn

import pyreg.finite_differences as fd
import pyreg.module_parameters as pars
import pyreg.noisy_convolution as nc

import numpy as np

from pyreg.data_wrapper import MyTensor, AdaptVal, USE_CUDA

device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

import math

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

def DimNoisyConv(dim):
    if dim == 1:
        return nc.NoisyConv1d
    elif dim == 2:
        return nc.NoisyConv2d
    elif dim == 3:
        return nc.NoisyConv3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')


def DimConv(dim):
    if dim==1:
        return nn.Conv1d
    elif dim==2:
        return nn.Conv2d
    elif dim==3:
        return nn.Conv3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')

def DimBatchNorm(dim):
    if dim==1:
        return nn.BatchNorm1d
    elif dim==2:
        return nn.BatchNorm2d
    elif dim==3:
        return nn.BatchNorm3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')

def DimInstanceNorm(dim):
    if dim == 1:
        return nn.InstanceNorm1d
    elif dim == 2:
        return nn.InstanceNorm2d
    elif dim == 3:
        return nn.InstanceNorm3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')

def DimNormalization(dim,normalization_type,nr_channels,im_sz):
    normalization_types = ['batch', 'instance', 'layer', 'group']
    if normalization_type is not None:
        if normalization_type.lower() not in normalization_types:
            raise ValueError("normalization type either needs to be None or in ['layer'|'batch'|'instance']")
    else:
        return None

    if normalization_type.lower()=='batch':
        return DimBatchNorm(dim)(nr_channels, eps=0.0001, momentum=0.1, affine=True)
    elif normalization_type.lower()=='layer':
        int_im_sz = [int(elem) for elem in im_sz]
        layer_sz = [int(nr_channels)] + int_im_sz
        return nn.LayerNorm(layer_sz)
    elif normalization_type.lower()=='instance':
        return DimInstanceNorm(dim)(nr_channels, eps=0.0001, momentum=0.1, affine=True)
    elif normalization_type.lower()=='group':
        channels_per_group = 10
        nr_groups = max(1,nr_channels//channels_per_group)
        return nn.GroupNorm(num_groups=nr_groups,num_channels=nr_channels)
    else:
        raise ValueError('Unknown normalization type: {}'.format(normalization_type))


def DimConvTranspose(dim):
    if dim==1:
        return nn.ConvTranspose1d
    elif dim==2:
        return nn.ConvTranspose2d
    elif dim==3:
        return nn.ConvTranspose3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')


def DimNoisyConvTranspose(dim):
    if dim==1:
        return nc.NoisyConvTranspose1d
    elif dim==2:
        return nc.NoisyConvTranspose2d
    elif dim==3:
        return nc.NoisyConvTranspose3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')

def DimMaxPool(dim):
    if dim==1:
        return nn.MaxPool1d
    elif dim==2:
        return nn.MaxPool2d
    elif dim==3:
        return nn.MaxPool3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')


class conv_norm_in_rel(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size, im_sz, stride=1, active_unit='relu', same_padding=False,
                 normalization_type='layer', reverse=False, group = 1,dilation = 1,
                 use_noisy_convolution=False,
                 noisy_convolution_std=0.5,
                 noisy_convolution_optimize_over_std=False):

        super(conv_norm_in_rel, self).__init__()
        self.use_noisy_convolution = use_noisy_convolution

        if normalization_type is None:
            conv_bias = True
        else:
            conv_bias = False

        padding = int((kernel_size - 1) / 2) if same_padding else 0
        if not reverse:
            if self.use_noisy_convolution:
                self.conv = DimNoisyConv(dim)(in_channels, out_channels, kernel_size, stride, padding=padding, groups=group,dilation=dilation,bias=conv_bias,
                                              scalar_sigmas=True, optimize_sigmas=noisy_convolution_optimize_over_std, std_init=noisy_convolution_std)
            else:
                self.conv = DimConv(dim)(in_channels, out_channels, kernel_size, stride, padding=padding, groups=group,dilation=dilation,bias=conv_bias)
        else:
            if self.use_noisy_convolution:
                self.conv = DimNoisyConvTranspose(dim)(in_channels, out_channels, kernel_size, stride, padding=padding,groups=group,dilation=dilation,bias=conv_bias,
                                                       scalar_sigmas=True,
                                                       optimize_sigmas=noisy_convolution_optimize_over_std,
                                                       std_init=noisy_convolution_std)
            else:
                self.conv = DimConvTranspose(dim)(in_channels, out_channels, kernel_size, stride, padding=padding,groups=group,dilation=dilation,bias=conv_bias)

        self.normalization = DimNormalization(dim,normalization_type,out_channels,im_sz) if normalization_type else None

        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        elif active_unit =='leaky_relu':
            self.active_unit = nn.LeakyReLU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x, iter=0):
        if self.use_noisy_convolution:
            x = self.conv(x,iter=iter)
        else:
            x = self.conv(x)
        if self.normalization is not None:
            x = self.normalization(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x


###########################################

# Quicksilver style 2d encoder
class encoder_block_2d(nn.Module):
    def __init__(self, input_feature, output_feature, im_sz, use_dropout, normalization_type, dim):
        super(encoder_block_2d, self).__init__()

        self.dim = dim

        if normalization_type is None:
            conv_bias = True
        else:
            conv_bias = False

        self.conv_input = DimConv(self.dim)(in_channels=input_feature, out_channels=output_feature,
                                    kernel_size=3, stride=1, padding=1, dilation=1,bias=conv_bias)
        self.conv_inblock1 = DimConv(self.dim)(in_channels=output_feature, out_channels=output_feature,
                                       kernel_size=3, stride=1, padding=1, dilation=1,bias=conv_bias)
        self.conv_inblock2 = DimConv(self.dim)(in_channels=output_feature, out_channels=output_feature,
                                       kernel_size=3, stride=1, padding=1, dilation=1,bias=conv_bias)
        self.conv_pooling = DimConv(self.dim)(in_channels=output_feature, out_channels=output_feature,
                                      kernel_size=2, stride=2, padding=0, dilation=1,bias=conv_bias)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()

        if normalization_type:
            self.norm_1 = DimNormalization(self.dim,normalization_type=normalization_type,nr_channels=output_feature,im_sz=im_sz)
            self.norm_2 = DimNormalization(self.dim,normalization_type=normalization_type,nr_channels=output_feature,im_sz=im_sz)
            self.norm_3 = DimNormalization(self.dim,normalization_type=normalization_type,nr_channels=output_feature,im_sz=im_sz)
            self.norm_4 = DimNormalization(self.dim,normalization_type=normalization_type,nr_channels=output_feature,im_sz=im_sz)

        self.use_dropout = use_dropout
        self.normalization_type = normalization_type
        self.dropout = nn.Dropout(0.2)

    def apply_dropout(self, input):
        if self.use_dropout:
            return self.dropout(input)
        else:
            return input

    def forward_with_normalization(self,x):
        output = self.conv_input(x)
        output = self.apply_dropout(self.prelu1(self.norm_1(output)))
        output = self.apply_dropout(self.prelu2(self.norm_2(self.conv_inblock1(output))))
        output = self.apply_dropout(self.prelu3(self.norm_3(self.conv_inblock2(output))))
        return self.prelu4(self.norm_4(self.conv_pooling(output)))

    def forward_without_normalization(self,x):
        output = self.conv_input(x)
        output = self.apply_dropout(self.prelu1(output))
        output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)))
        output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)))
        return self.prelu4(self.conv_pooling(output))

    def forward(self, x):
        if self.normalization_type:
            return self.forward_with_normalization(x)
        else:
            return self.forward_without_normalization(x)

# quicksilver style 2d decoder
class decoder_block_2d(nn.Module):
    def __init__(self, input_feature, output_feature, im_sz, pooling_filter,use_dropout, normalization_type, dim, last_block=False):
        super(decoder_block_2d, self).__init__()
        # todo: check padding here, not sure if it is the right thing to do

        self.dim = dim

        if normalization_type is None:
            conv_bias = True
        else:
            conv_bias = False

        self.conv_unpooling = DimConvTranspose(self.dim)(in_channels=input_feature, out_channels=input_feature,
                                                 kernel_size=pooling_filter, stride=2, padding=0,output_padding=0,bias=conv_bias)
        self.conv_inblock1 = DimConv(self.dim)(in_channels=input_feature, out_channels=input_feature,
                                       kernel_size=3, stride=1, padding=1, dilation=1,bias=conv_bias)
        self.conv_inblock2 = DimConv(self.dim)(in_channels=input_feature, out_channels=input_feature,
                                       kernel_size=3, stride=1, padding=1, dilation=1,bias=conv_bias)
        if last_block:
            self.conv_output = DimConv(self.dim)(in_channels=input_feature, out_channels=output_feature,
                                         kernel_size=3, stride=1, padding=1, dilation=1)
        else:
            self.conv_output = DimConv(self.dim)(in_channels=input_feature, out_channels=output_feature,
                                         kernel_size=3, stride=1, padding=1, dilation=1,bias=conv_bias)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()

        if normalization_type:
            self.norm_1 = DimNormalization(self.dim,normalization_type=normalization_type,nr_channels=input_feature,im_sz=im_sz)
            self.norm_2 = DimNormalization(self.dim,normalization_type=normalization_type,nr_channels=input_feature,im_sz=im_sz)
            self.norm_3 = DimNormalization(self.dim,normalization_type=normalization_type,nr_channels=input_feature,im_sz=im_sz)
            if not last_block:
                self.norm_4 = DimNormalization(self.dim,normalization_type=normalization_type,nr_channels=input_feature,im_sz=im_sz)

        self.use_dropout = use_dropout
        self.normalization_type = normalization_type
        self.last_block = last_block
        self.dropout = nn.Dropout(0.2)
        self.output_feature = output_feature

    def apply_dropout(self, input):
        if self.use_dropout:
            return self.dropout(input)
        else:
            return input

    def forward_with_normalization(self,x):
        output = self.prelu1(self.norm_1(self.conv_unpooling(x)))
        output = self.apply_dropout(self.prelu2(self.norm_2(self.conv_inblock1(output))))
        output = self.apply_dropout(self.prelu3(self.norm_3(self.conv_inblock2(output))))
        if self.last_block:  # generates final output
            return self.conv_output(output);
        else:  # generates intermediate results
            return self.apply_dropout(self.prelu4(self.norm_4(self.conv_output(output))))

    def forward_without_normalization(self,x):
        output = self.prelu1(self.conv_unpooling(x))
        output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)))
        output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)))
        if self.last_block:  # generates final output
            return self.conv_output(output);
        else:  # generates intermediate results
            return self.apply_dropout(self.prelu4(self.conv_output(output)))

    def forward(self, x):
        if self.normalization_type:
            return self.forward_with_normalization(x)
        else:
            return self.forward_without_normalization(x)


# general definition of the different networks
class DeepNetwork(with_metaclass(ABCMeta,nn.Module)):

    def __init__(self, dim, n_in_channel, n_out_channel, im_sz, params):

        super(DeepNetwork,self).__init__()

        self.nr_of_gaussians = n_out_channel
        self.nr_of_image_channels = n_in_channel
        self.im_sz = im_sz
        self.dim = dim
        self.params = params

        self.normalization_type = self.params[('normalization_type', 'layer',
                                               "Normalization type between layers: ['batch'|'layer'|'instance'|'group'|'none']")]
        if self.normalization_type.lower() == 'none':
            self.normalization_type = None

        self.use_noisy_convolution = self.params[
            ('use_noisy_convolution', True, 'when true then the convolution layers will be replaced by noisy convolution layer')]

        self.noisy_convolution_std = self.params[('noisy_convolution_std', 0.25, 'Standard deviation for the noise')]
        self.noisy_convolution_optimize_over_std = self.params[
            ('noisy_convolution_optimize_over_std', False, 'If set to True, noise standard deviations are optimized')]

    def initialize_network_weights(self):
        for m in self.modules():
            if isinstance(m, DimConv(self.dim)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, DimConvTranspose(self.dim)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, DimNoisyConv(self.dim)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, DimNoisyConvTranspose(self.dim)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, DimBatchNorm(self.dim)):
                pass
            elif isinstance(m, nn.Linear):
                pass

# actual definitions of the UNets

class Unet(DeepNetwork):
    """
    unet include 4 down path (1/16)  and 4 up path (16)
    """
    def __init__(self, dim, n_in_channel, n_out_channel, im_sz, params):

        super(Unet, self).__init__(dim, n_in_channel, n_out_channel, im_sz, params)

        im_sz_down_1 = [elem//2 for elem in im_sz]
        im_sz_down_2 = [elem//2 for elem in im_sz_down_1]
        im_sz_down_3 = [elem//2 for elem in im_sz_down_2]
        im_sz_down_4 = [elem//2 for elem in im_sz_down_3]

        # each dimension of the input should be 16x
        self.down_path_1 = conv_norm_in_rel(dim, n_in_channel, 16, kernel_size=3, im_sz=im_sz, stride=1, active_unit='relu', same_padding=True, normalization_type=self.normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std
                                            )
        self.down_path_2_1 = conv_norm_in_rel(dim, 16, 32, kernel_size=3, im_sz=im_sz_down_1, stride=2, active_unit='relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std
                                              )
        self.down_path_2_2 = conv_norm_in_rel(dim, 32, 32, kernel_size=3, im_sz=im_sz_down_1, stride=1, active_unit='relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std
                                              )
        self.down_path_4_1 = conv_norm_in_rel(dim, 32, 32, kernel_size=3, im_sz=im_sz_down_2, stride=2, active_unit='relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std
                                              )
        self.down_path_4_2 = conv_norm_in_rel(dim, 32, 32, kernel_size=3, im_sz=im_sz_down_2, stride=1, active_unit='relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std
                                              )
        self.down_path_8_1 = conv_norm_in_rel(dim, 32, 64, kernel_size=3, im_sz=im_sz_down_3, stride=2, active_unit='relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std
                                              )
        self.down_path_8_2 = conv_norm_in_rel(dim, 64, 64, kernel_size=3, im_sz=im_sz_down_3, stride=1, active_unit='relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std
                                              )
        self.down_path_16 = conv_norm_in_rel(dim, 64, 64, kernel_size=3, im_sz=im_sz_down_4, stride=2, active_unit='relu', same_padding=True, normalization_type=self.normalization_type,
                                             use_noisy_convolution=self.use_noisy_convolution,
                                             noisy_convolution_std=self.noisy_convolution_std,
                                             noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std
                                             )


        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.up_path_8_1 = conv_norm_in_rel(dim, 64, 64, kernel_size=2, im_sz=im_sz_down_3, stride=2, active_unit='leaky_relu', same_padding=False, normalization_type=self.normalization_type, reverse=True,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std
                                            )
        self.up_path_8_2 = conv_norm_in_rel(dim, 128, 64, kernel_size=3, im_sz=im_sz_down_3, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std
                                            )
        self.up_path_4_1 = conv_norm_in_rel(dim, 64, 64, kernel_size=2, im_sz=im_sz_down_2, stride=2, active_unit='leaky_relu', same_padding=False, normalization_type=self.normalization_type, reverse=True,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std
                                            )
        self.up_path_4_2 = conv_norm_in_rel(dim, 96, 32, kernel_size=3, im_sz=im_sz_down_2, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std
                                            )
        self.up_path_2_1 = conv_norm_in_rel(dim, 32, 32, kernel_size=2, im_sz=im_sz_down_1, stride=2, active_unit='leaky_relu', same_padding=False, normalization_type=self.normalization_type,reverse=True,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std
                                            )
        self.up_path_2_2 = conv_norm_in_rel(dim, 64, 8, kernel_size=3, im_sz=im_sz_down_1, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std
                                            )
        self.up_path_1_1 = conv_norm_in_rel(dim, 8, 8, kernel_size=2, im_sz=im_sz, stride=2, active_unit='None', same_padding=False, normalization_type=self.normalization_type, reverse=True,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std
                                            )

        # we do not want to normalize the last layer as it will create the output
        self.up_path_1_2 = conv_norm_in_rel(dim, 24, n_out_channel, kernel_size=3, im_sz=im_sz, stride=1, active_unit='None', same_padding=True, normalization_type=None,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std)

    def forward(self, x, iter=0):
        d1 = self.down_path_1(x, iter=iter)
        d2_1 = self.down_path_2_1(d1, iter=iter)
        d2_2 = self.down_path_2_2(d2_1, iter=iter)
        d4_1 = self.down_path_4_1(d2_2, iter=iter)
        d4_2 = self.down_path_4_2(d4_1, iter=iter)
        d8_1 = self.down_path_8_1(d4_2, iter=iter)
        d8_2 = self.down_path_8_2(d8_1, iter=iter)
        d16 = self.down_path_16(d8_2, iter=iter)


        u8_1 = self.up_path_8_1(d16, iter=iter)
        u8_2 = self.up_path_8_2(torch.cat((d8_2,u8_1),1), iter=iter)
        u4_1 = self.up_path_4_1(u8_2, iter=iter)
        u4_2 = self.up_path_4_2(torch.cat((d4_2,u4_1),1), iter=iter)
        u2_1 = self.up_path_2_1(u4_2, iter=iter)
        u2_2 = self.up_path_2_2(torch.cat((d2_2, u2_1), 1), iter=iter)
        u1_1 = self.up_path_1_1(u2_2, iter=iter)
        output = self.up_path_1_2(torch.cat((d1, u1_1), 1), iter=iter)

        return output

class Encoder_decoder(DeepNetwork):

    def __init__(self, dim, n_in_channel, n_out_channel, im_sz, params):

        super(Encoder_decoder, self).__init__(dim, n_in_channel, n_out_channel, im_sz, params)

        # is 64 in quicksilver paper
        feature_num = self.params[('number_of_features_in_first_layer', 32, 'number of features in the first encoder layer (64 in quicksilver)')]
        use_dropout = self.params[('use_dropout', False, 'use dropout for the layers')]
        self.use_separate_decoders_per_gaussian = self.params[('use_separate_decoders_per_gaussian', True, 'if set to true separte decoder branches are used for each Gaussian')]

        self.use_one_encoder_decoder_block = self.params[('use_one_encoder_decoder_block', True, 'If False, using two each as in the quicksilver paper')]

        # im_sz, use_dropout, normalization_type, dim

        if self.use_momentum_as_input or self.use_target_image_as_input or self.use_source_image_as_input:
            self.encoder_1 = encoder_block_2d(self.get_number_of_input_channels(self.nr_of_image_channels, dim),
                                              feature_num, im_sz=im_sz, use_dropout=use_dropout,
                                              normalization_type=self.normalization_type, dim=dim)
        else:
            self.encoder_1 = encoder_block_2d(1, feature_num, im_sz=im_sz, use_dropout=use_dropout,
                                              normalization_type=self.normalization_type, dim=dim)

        if not self.use_one_encoder_decoder_block:
            self.encoder_2 = encoder_block_2d(feature_num, feature_num * 2, im_sz=im_sz, use_dropout=use_dropout,
                                              normalization_type=self.normalization_type, dim=dim)

        # todo: maybe have one decoder for each Gaussian here.
        # todo: the current version seems to produce strange gridded results

        # input_feature, output_feature, im_sz, pooling_filter,use_dropout, normalization_type, dim, last_block=False):

        if self.use_separate_decoders_per_gaussian:
            if not self.use_one_encoder_decoder_block:
                self.decoder_1 = nn.ModuleList()
            self.decoder_2 = nn.ModuleList()
            # create two decoder blocks for each Gaussian
            for g in range(self.nr_of_gaussians):
                if not self.use_one_encoder_decoder_block:
                    self.decoder_1.append(
                        decoder_block_2d(feature_num * 2, feature_num, im_sz=im_sz, pooling_filter=2,
                                         use_dropout=use_dropout, normalization_type=self.normalization_type, dim=dim))
                self.decoder_2.append(decoder_block_2d(feature_num, 1, im_sz=im_sz, pooling_filter=2,
                                                       use_dropout=use_dropout, normalization_type=self.normalization_type, dim=dim, last_block=True))
        else:
            if not self.use_one_encoder_decoder_block:
                self.decoder_1 = decoder_block_2d(feature_num * 2, feature_num, im_sz=im_sz, pooling_filter=2,
                                                  use_dropout=use_dropout, normalization_type=self.normalization_type,dim=dim)
            self.decoder_2 = decoder_block_2d(feature_num, self.nr_of_gaussians, im_sz=im_sz, pooling_filter=2,
                                              use_dropout=use_dropout, normalization_type=self.normalization_type,dim=dim, last_block=True) # 3?

    def forward(self, x):

        if self.use_one_encoder_decoder_block:
            encoder_output = self.encoder_1(x)
        else:
            encoder_output = self.encoder_2(self.encoder_1(x))

        if self.use_one_encoder_decoder_block:
            if self.use_separate_decoders_per_gaussian:
                # here we have separate decoder outputs for the different Gaussians
                # should give it more flexibility
                decoder_output_individual = []
                for g in range(self.nr_of_gaussians):
                    decoder_output_individual.append(self.decoder_2[g]((encoder_output)))
                decoder_output = torch.cat(decoder_output_individual, dim=1);
            else:
                decoder_output = self.decoder_2(encoder_output)
        else:
            if self.use_separate_decoders_per_gaussian:
                # here we have separate decoder outputs for the different Gaussians
                # should give it more flexibility
                decoder_output_individual = []
                for g in range(self.nr_of_gaussians):
                    decoder_output_individual.append(self.decoder_2[g](self.decoder_1[g](encoder_output)))
                decoder_output = torch.cat(decoder_output_individual, dim=1);
            else:
                decoder_output = self.decoder_2(self.decoder_1(encoder_output))

        return decoder_output

class Simple_consistent(DeepNetwork):

    def __init__(self, dim, n_in_channel, n_out_channel, im_sz, params):

        super(Simple_consistent, self).__init__(dim, n_in_channel, n_out_channel, im_sz, params)

        self.kernel_sizes = self.params[('kernel_sizes', [7, 7], 'size of the convolution kernels')]

        # check that all the kernel-size are odd
        for ks in self.kernel_sizes:
            if ks % 2 == 0:
                raise ValueError('Kernel sizes need to be odd')

        # the last layers feature number is not specified as it will simply be the number of Gaussians
        self.nr_of_features_per_layer = self.params[('number_of_features_per_layer', [20],
                                                     'Number of features for the convolution later; last one is set to number of Gaussians')]
        # add the number of Gaussians to the last layer
        self.nr_of_features_per_layer = self.nr_of_features_per_layer + [self.nr_of_gaussians]

        self.nr_of_layers = len(self.kernel_sizes)
        assert (self.nr_of_layers == len(self.nr_of_features_per_layer))

        self.active_unit_to_use = self.params[('active_unit_to_use', 'leaky_relu', "what type of active unit to use ['relu'|'sigmoid'|'elu'|'leaky_relu']")]

        if self.active_unit_to_use.lower() == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif self.active_unit_to_use.lower() == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        elif self.active_unit_to_use.lower() == 'leaky_relu':
            self.active_unit = nn.LeakyReLU(inplace=True)
        elif self.active_unit_to_use.lower() == 'sigmoid':
            self.active_unit = nn.Sigmoid
        else:
            raise ValueError('Active unit needs to be specified: unkown active unit: {}'.format(self.active_unit_to_use))

        self.conv_layers = None
        self.normalizations = None

        # needs to be initialized here, otherwise the optimizer won't see the modules from ModuleList
        # todo: figure out how to do ModuleList initialization not in __init__
        # todo: this would allow removing dim and nr_of_image_channels from interface
        # todo: because it could be compute on the fly when forward is executed
        self._init(self.nr_of_image_channels, dim=self.dim, im_sz=self.im_sz)

    def _init(self, nr_of_image_channels, dim, im_sz):
        """
        Initalizes all the conv layers
        :param nr_of_image_channels:
        :param dim:
        :return:
        """

        assert (self.nr_of_layers > 0)

        nr_of_input_channels = nr_of_image_channels
        convs = [None] * self.nr_of_layers

        if self.normalization_type is None:
            conv_bias = True
        else:
            conv_bias = False

        if self.use_noisy_convolution:
            DimConvType = DimNoisyConv
        else:
            DimConvType = DimConv

        # first layer
        convs[0] = DimConvType(self.dim)(nr_of_input_channels,
                                     self.nr_of_features_per_layer[0],
                                     self.kernel_sizes[0], padding=(self.kernel_sizes[0] - 1) // 2, bias=conv_bias)

        # all the intermediate layers and the last one
        for l in range(self.nr_of_layers - 1):
            convs[l + 1] = DimConvType(self.dim)(self.nr_of_features_per_layer[l],
                                             self.nr_of_features_per_layer[l + 1],
                                             self.kernel_sizes[l + 1],
                                             padding=(self.kernel_sizes[l + 1] - 1) // 2, bias=conv_bias)

        self.conv_layers = nn.ModuleList()
        for c in convs:
            self.conv_layers.append(c)

        if self.normalization_type is not None:

            normalizations = [None] * (self.nr_of_layers - 1)  # not on the last layer
            for b in range(self.nr_of_layers - 1):
                normalizations[b] = DimNormalization(self.dim,normalization_type=self.normalization_type,nr_channels=self.nr_of_features_per_layer[b],im_sz=im_sz)

            self.normalizations = nn.ModuleList()
            for b in normalizations:
                self.normalizations.append(b)

    def forward(self, x, iter=0):

        # now let's apply all the convolution layers, until the last
        # (because the last one is not relu-ed

        if self.normalization_type is not None:
            for l in range(len(self.conv_layers) - 1):
                if self.use_noisy_convolution:
                    x = self.active_unit(self.normalizations[l](self.conv_layers[l](x,iter=iter)))
                else:
                    x = self.active_unit(self.normalizations[l](self.conv_layers[l](x)))
        else:
            for l in range(len(self.conv_layers) - 1):
                if self.use_noisy_convolution:
                    x = self.active_unit(self.conv_layers[l](x,iter=iter))
                else:
                    x = self.active_unit(self.conv_layers[l](x))


        # and now apply the last one without an activation for now
        # because we want to have the ability to smooth *before* the softmax
        # this is similar to smoothing in the logit domain for an active mean field approach
        if self.use_noisy_convolution:
            x = self.conv_layers[-1](x,iter=iter)
        else:
            x = self.conv_layers[-1](x)

        return x


class Unet_no_skip(DeepNetwork):
    """
    unet include 4 down path (1/16)  and 4 up path (16)
    """
    def __init__(self, dim, n_in_channel, n_out_channel, im_sz, params):
        # each dimension of the input should be 16x
        super(Unet_no_skip,self).__init__(dim, n_in_channel, n_out_channel, im_sz, params)

        im_sz_down_1 = [elem // 2 for elem in im_sz]
        im_sz_down_2 = [elem // 2 for elem in im_sz_down_1]
        im_sz_down_3 = [elem // 2 for elem in im_sz_down_2]
        im_sz_down_4 = [elem // 2 for elem in im_sz_down_3]

        self.down_path_1 = conv_norm_in_rel(dim, n_in_channel, 16, kernel_size=3, im_sz=im_sz, stride=1, active_unit='relu', same_padding=True, normalization_type=self.normalization_type)
        self.down_path_2_1 = conv_norm_in_rel(dim, 16, 32, kernel_size=3, im_sz=im_sz_down_1, stride=2, active_unit='relu', same_padding=True, normalization_type=self.normalization_type)
        self.down_path_2_2 = conv_norm_in_rel(dim, 32, 32, kernel_size=3, im_sz=im_sz_down_1, stride=1, active_unit='relu', same_padding=True, normalization_type=self.normalization_type)
        self.down_path_4_1 = conv_norm_in_rel(dim, 32, 32, kernel_size=3, im_sz=im_sz_down_2, stride=2, active_unit='relu', same_padding=True, normalization_type=self.normalization_type)
        self.down_path_4_2 = conv_norm_in_rel(dim, 32, 32, kernel_size=3, im_sz=im_sz_down_2, stride=1, active_unit='relu', same_padding=True, normalization_type=self.normalization_type)
        self.down_path_8_1 = conv_norm_in_rel(dim, 32, 64, kernel_size=3, im_sz=im_sz_down_3, stride=2, active_unit='relu', same_padding=True, normalization_type=self.normalization_type)
        self.down_path_8_2 = conv_norm_in_rel(dim, 64, 64, kernel_size=3, im_sz=im_sz_down_3, stride=1, active_unit='relu', same_padding=True, normalization_type=self.normalization_type)
        self.down_path_16 = conv_norm_in_rel(dim, 64, 64, kernel_size=3, im_sz=im_sz_down_4, stride=2, active_unit='relu', same_padding=True, normalization_type=self.normalization_type)


        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.up_path_8_1 = conv_norm_in_rel(dim, 64, 64, kernel_size=2, im_sz=im_sz_down_3, stride=2, active_unit='leaky_relu', same_padding=False, normalization_type=self.normalization_type,reverse=True)
        self.up_path_8_2 = conv_norm_in_rel(dim, 64, 64, kernel_size=3, im_sz=im_sz_down_3, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type)
        self.up_path_4_1 = conv_norm_in_rel(dim, 64, 64, kernel_size=2, im_sz=im_sz_down_2, stride=2, active_unit='leaky_relu', same_padding=False, normalization_type=self.normalization_type,reverse=True)
        self.up_path_4_2 = conv_norm_in_rel(dim, 64, 32, kernel_size=3, im_sz=im_sz_down_2, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type)
        self.up_path_2_1 = conv_norm_in_rel(dim, 32, 32, kernel_size=2, im_sz=im_sz_down_1, stride=2, active_unit='leaky_relu', same_padding=False, normalization_type=self.normalization_type)
        self.up_path_1_1 = conv_norm_in_rel(dim, 8, 8, kernel_size=2, im_sz=im_sz, stride=2, active_unit='None', same_padding=False, normalization_type=self.normalization_type, reverse=True)

        self.up_path_1_2 = conv_norm_in_rel(dim, 8, n_out_channel, kernel_size=3, im_sz=im_sz, stride=1, active_unit='None', same_padding=True,
                                            normalization_type=None,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std)

    def forward(self, x):
        d1 = self.down_path_1(x)
        d2_1 = self.down_path_2_1(d1)
        d2_2 = self.down_path_2_2(d2_1)
        d4_1 = self.down_path_4_1(d2_2)
        d4_2 = self.down_path_4_2(d4_1)
        d8_1 = self.down_path_8_1(d4_2)
        d8_2 = self.down_path_8_2(d8_1)
        d16 = self.down_path_16(d8_2)


        u8_1 = self.up_path_8_1(d16)
        u8_2 = self.up_path_8_2(u8_1)
        u4_1 = self.up_path_4_1(u8_2)
        u4_2 = self.up_path_4_2(u4_1)
        u2_1 = self.up_path_2_1(u4_2)
        u2_2 = self.up_path_2_2(u2_1)
        u1_1 = self.up_path_1_1(u2_2)
        output = self.up_path_1_2(u1_1)

        return output

# custom loss function

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, spacing):
        volumeElement = spacing.prod()
        batch_size = x.size()[0]
        b = x*torch.log(x)
        #F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        #b = -1.0 * b.sum()*volumeElement/batch_size
        b = -1.0*b.sum()*volumeElement/batch_size
        return b

class GlobalHLoss(nn.Module):
    def __init__(self):
        super(GlobalHLoss, self).__init__()

    def forward(self, x, spacing):

        nr_of_labels = x.size()[1]
        P = MyTensor(nr_of_labels).zero_()
        sz = list(x.size())
        nr_of_elements = [sz[0]] + sz[2:]
        current_norm = float(np.array(nr_of_elements).prod().astype('float32')) # number of pixels
        for n in range(nr_of_labels):
            P[n] = (x[:,n,...].sum())/current_norm

        # now compute the entropy over this
        b = MyTensor(1).zero_()
        for n in range(nr_of_labels):
            b = b-P[n]*torch.log(P[n])

        return b

class TotalVariationLoss(nn.Module):
    """
    Loss function for image clustering (this is here a relaxation of normalized cuts)
    """

    def __init__(self, dim, params):
        """

        :param params: ParameterDict() object to hold and keep track of general parameters
        """
        super(TotalVariationLoss, self).__init__()
        self.params = params
        """ParameterDict() parameters"""
        self.dim = dim
        # dimension

    # todo: merge this with the code in deep_smoothers.py
    def compute_local_weighted_tv_norm(self, I, weights, spacing, nr_of_gaussians, use_color_tv, pnorm=2):

        import pyreg.deep_smoothers as deep_smoothers

        volumeElement = spacing.prod()
        if use_color_tv:
            individual_sum_of_total_variation_penalty = MyTensor(nr_of_gaussians).zero_()
        else:
            sum_of_total_variation_penalty = MyTensor(1).zero_()
        # first compute the edge map
        g_I = deep_smoothers.compute_localized_edge_penalty(I[:, 0, ...], spacing, self.params)
        batch_size = I.size()[0]

        # now computed weighted TV norm channel-by-channel, square it and then take the square root (this is like in color TV)
        for g in range(nr_of_gaussians):
            c_local_norm_grad = deep_smoothers._compute_local_norm_of_gradient(weights[:, g, ...], spacing, pnorm)

            to_sum = g_I * c_local_norm_grad * volumeElement / batch_size
            current_tv = (to_sum).sum()
            if use_color_tv:
                individual_sum_of_total_variation_penalty[g] = current_tv
            else:
                sum_of_total_variation_penalty += current_tv

        if use_color_tv:
            total_variation_penalty = torch.norm(individual_sum_of_total_variation_penalty,p=2)
        else:
            total_variation_penalty = sum_of_total_variation_penalty

        return total_variation_penalty


    def forward(self, input_images, spacing, label_probabilities, use_color_tv=False):
        # first compute the weighting functions

        nr_of_gaussians = label_probabilities.size()[1]
        current_penalty = self.compute_local_weighted_tv_norm(input_images, label_probabilities, spacing, nr_of_gaussians, use_color_tv)
        return current_penalty


class ClusteringLoss(nn.Module):
    """
    Loss function for image clustering (this is here a relaxation of normalized cuts)
    """

    def __init__(self, dim, params):
        """

        :param params: ParameterDict() object to hold and keep track of general parameters
        """
        super(ClusteringLoss, self).__init__()
        self.params = params
        """ParameterDict() parameters"""
        self.dim = dim
        # dimension

    def _compute_cut_cost_for_label_k_1d(self,w_edge,p):
        raise ValueError('Not yet implemented')

    def _compute_cut_cost_for_label_k_3d(self, w_edge, p):
        raise ValueError('Not yet implemented')

    def _compute_cut_cost_for_label_k_2d(self,w_edge,p):
        # todo: maybe the current cost is not ideal and we should try the one from the paper
        # todo: but at least it would be consistent with our total variation implementation
        batch_size = p.size()[0]
        cut_cost = AdaptVal(torch.zeros(batch_size))

        # needs to be batch B x X x Y x Z format (as input)
        fdt = fd.FD_torch(spacing=np.array([1.0]*self.dim))
        p_xp = fdt.dXf(p)
        p_yp = fdt.dYf(p)

        for b in range(batch_size):

            nom = (p[b,...]*(p[b,...] + w_edge[b,...]*(p_xp[b,...]+p_yp[b,...]))).sum()
            denom = (p[b,...]*(1.0+2*w_edge[b,...])).sum()

            cut_cost[b] = nom/denom

        return cut_cost

    def _compute_cut_cost_for_label_k(self, w_edge, p):
        if self.dim==1:
            return self._compute_cut_cost_for_label_k_1d(w_edge, p)
        elif self.dim==2:
            return self._compute_cut_cost_for_label_k_2d(w_edge, p)
        elif self.dim==3:
            return self._compute_cut_cost_for_label_k_3d(w_edge, p)
        else:
            raise ValueError('Only defined for dimensions {1,2,3}')

    def forward(self, input_images, spacing, label_probabilities):
        import pyreg.deep_smoothers as deep_smoothers

        # first compute the weighting functions
        localized_edge_penalty = deep_smoothers.compute_localized_edge_penalty(input_images[:,0,...],spacing,self.params)

        batch_size = label_probabilities.size()[0]
        nr_of_clusters = label_probabilities.size()[1]
        current_penalties = AdaptVal(torch.ones(batch_size)*nr_of_clusters)

        for k in range(nr_of_clusters):
            current_penalties -= self._compute_cut_cost_for_label_k(w_edge=localized_edge_penalty,p=label_probabilities[:,k,...])

        current_penalty = current_penalties.sum()
        return current_penalty
