from __future__ import print_function
from __future__ import absolute_import
from builtins import object

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyreg.finite_differences as fd
import pyreg.module_parameters as pars

import numpy as np

from pyreg.data_wrapper import MyTensor, AdaptVal, USE_CUDA

device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA) else "cpu")


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

def DimConvTranspose(dim):
    if dim==1:
        return nn.ConvTranspose1d
    elif dim==2:
        return nn.ConvTranspose2d
    elif dim==3:
        return nn.ConvTranspose3d
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


class conv_bn_in_rel(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride=1, active_unit='relu', same_padding=False,
                 bn=False, inn=False, reverse=False,group = 1,dilation = 1):
        super(conv_bn_in_rel, self).__init__()

        if bn and inn:
            raise ValueError('Simultaneous batch and instance normalization is not supported')

        padding = int((kernel_size - 1) / 2) if same_padding else 0
        if not reverse:
            self.conv = DimConv(dim)(in_channels, out_channels, kernel_size, stride, padding=padding, groups=group,dilation=dilation)
        else:
            self.conv = DimConvTranspose(dim)(in_channels, out_channels, kernel_size, stride, padding=padding,groups=group,dilation=dilation)

        self.bn = DimBatchNorm(dim)(out_channels, eps=0.0001, momentum=0.1, affine=True) if bn else None
        self.inn = DimInstanceNorm(dim)(out_channels, eps=0.0001, momentum=0.1, affine=True) if inn else None
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        elif active_unit =='leaky_relu':
            self.active_unit = nn.LeakyReLU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.inn is not None:
            x = self.inn(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x


###########################################

# Quicksilver style 2d encoder
class encoder_block_2d(nn.Module):
    def __init__(self, input_feature, output_feature, use_dropout, use_batch_normalization, use_instance_normalization, dim):
        super(encoder_block_2d, self).__init__()

        if use_batch_normalization and use_instance_normalization:
            raise ValueError('Batch normalization and instance normalization cannot be specified at the same time')

        self.dim = dim

        if use_batch_normalization or use_instance_normalization:
            conv_bias = False
        else:
            conv_bias = True

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

        if use_batch_normalization:
            self.bn_1 = DimBatchNorm(self.dim)(output_feature,momentum=batch_norm_momentum_val)
            self.bn_2 = DimBatchNorm(self.dim)(output_feature,momentum=batch_norm_momentum_val)
            self.bn_3 = DimBatchNorm(self.dim)(output_feature,momentum=batch_norm_momentum_val)
            self.bn_4 = DimBatchNorm(self.dim)(output_feature,momentum=batch_norm_momentum_val)

        if use_instance_normalization:
            self.in_1 = DimInstanceNorm(self.dim)(output_feature, momentum=batch_norm_momentum_val)
            self.in_2 = DimInstanceNorm(self.dim)(output_feature, momentum=batch_norm_momentum_val)
            self.in_3 = DimInstanceNorm(self.dim)(output_feature, momentum=batch_norm_momentum_val)
            self.in_4 = DimInstanceNorm(self.dim)(output_feature, momentum=batch_norm_momentum_val)

        self.use_dropout = use_dropout
        self.use_batch_normalization = use_batch_normalization
        self.use_instance_normalization = use_instance_normalization
        self.dropout = nn.Dropout(0.2)

    def apply_dropout(self, input):
        if self.use_dropout:
            return self.dropout(input)
        else:
            return input

    def forward_with_batch_normalization(self,x):
        output = self.conv_input(x)
        output = self.apply_dropout(self.prelu1(self.bn_1(output)))
        output = self.apply_dropout(self.prelu2(self.bn_2(self.conv_inblock1(output))))
        output = self.apply_dropout(self.prelu3(self.bn_3(self.conv_inblock2(output))))
        return self.prelu4(self.bn_4(self.conv_pooling(output)))

    def forward_with_instance_normalization(self,x):
        output = self.conv_input(x)
        output = self.apply_dropout(self.prelu1(self.in_1(output)))
        output = self.apply_dropout(self.prelu2(self.in_2(self.conv_inblock1(output))))
        output = self.apply_dropout(self.prelu3(self.in_3(self.conv_inblock2(output))))
        return self.prelu4(self.in_4(self.conv_pooling(output)))

    def forward_without_batch_normalization(self,x):
        output = self.conv_input(x)
        output = self.apply_dropout(self.prelu1(output))
        output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)))
        output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)))
        return self.prelu4(self.conv_pooling(output))

    def forward(self, x):
        if self.use_batch_normalization:
            return self.forward_with_batch_normalization(x)
        elif self.use_instance_normalization:
            return self.forward_with_instance_normalization(x)
        else:
            return self.forward_without_batch_normalization(x)

# quicksilver style 2d decoder
class decoder_block_2d(nn.Module):
    def __init__(self, input_feature, output_feature, pooling_filter,use_dropout, use_batch_normalization, use_instance_normalization, dim, last_block=False):
        super(decoder_block_2d, self).__init__()
        # todo: check padding here, not sure if it is the right thing to do

        if use_batch_normalization and use_instance_normalization:
            raise ValueError('Instance and batch normalization cannot be used together')

        self.dim = dim

        if use_batch_normalization or use_instance_normalization:
            conv_bias = False
        else:
            conv_bias = True

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

        if use_batch_normalization:
            self.bn_1 = DimBatchNorm(self.dim)(input_feature,momentum=batch_norm_momentum_val)
            self.bn_2 = DimBatchNorm(self.dim)(input_feature,momentum=batch_norm_momentum_val)
            self.bn_3 = DimBatchNorm(self.dim)(input_feature,momentum=batch_norm_momentum_val)
            if not last_block:
                self.bn_4 = DimBatchNorm(self.dim)(output_feature,momentum=batch_norm_momentum_val)

        if use_instance_normalization:
            self.in_1 = DimInstanceNorm(self.dim)(input_feature,momentum=batch_norm_momentum_val)
            self.in_2 = DimInstanceNorm(self.dim)(input_feature,momentum=batch_norm_momentum_val)
            self.in_3 = DimInstanceNorm(self.dim)(input_feature,momentum=batch_norm_momentum_val)
            if not last_block:
                self.in_4 = DimInstanceNorm(self.dim)(output_feature,momentum=batch_norm_momentum_val)

        self.use_dropout = use_dropout
        self.use_batch_normalization = use_batch_normalization
        self.use_instance_normalization = use_instance_normalization
        self.last_block = last_block
        self.dropout = nn.Dropout(0.2)
        self.output_feature = output_feature

    def apply_dropout(self, input):
        if self.use_dropout:
            return self.dropout(input)
        else:
            return input

    def forward_with_batch_normalization(self,x):
        output = self.prelu1(self.bn_1(self.conv_unpooling(x)))
        output = self.apply_dropout(self.prelu2(self.bn_2(self.conv_inblock1(output))))
        output = self.apply_dropout(self.prelu3(self.bn_3(self.conv_inblock2(output))))
        if self.last_block:  # generates final output
            return self.conv_output(output);
        else:  # generates intermediate results
            return self.apply_dropout(self.prelu4(self.bn_4(self.conv_output(output))))

    def forward_with_instance_normalization(self,x):
        output = self.prelu1(self.in_1(self.conv_unpooling(x)))
        output = self.apply_dropout(self.prelu2(self.in_2(self.conv_inblock1(output))))
        output = self.apply_dropout(self.prelu3(self.in_3(self.conv_inblock2(output))))
        if self.last_block:  # generates final output
            return self.conv_output(output);
        else:  # generates intermediate results
            return self.apply_dropout(self.prelu4(self.in_4(self.conv_output(output))))

    def forward_without_batch_normalization(self,x):
        output = self.prelu1(self.conv_unpooling(x))
        output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)))
        output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)))
        if self.last_block:  # generates final output
            return self.conv_output(output);
        else:  # generates intermediate results
            return self.apply_dropout(self.prelu4(self.conv_output(output)))

    def forward(self, x):
        if self.use_batch_normalization:
            return self.forward_with_batch_normalization(x)
        elif self.use_instance_normalization:
            return self.forward_with_instance_normalization(x)
        else:
            return self.forward_without_batch_normalization(x)


# actual definitions of the UNets

class Unet(nn.Module):
    """
    unet include 4 down path (1/16)  and 4 up path (16)
    """
    def __init__(self, dim, n_in_channel, n_out_channel, use_batch_normalization=False, use_instance_normalization=True):
        # each dimension of the input should be 16x
        super(Unet,self).__init__()
        self.down_path_1 = conv_bn_in_rel(dim, n_in_channel, 16, 3, stride=1, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.down_path_2_1 = conv_bn_in_rel(dim, 16, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.down_path_2_2 = conv_bn_in_rel(dim, 32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.down_path_4_1 = conv_bn_in_rel(dim, 32, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.down_path_4_2 = conv_bn_in_rel(dim, 32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.down_path_8_1 = conv_bn_in_rel(dim, 32, 64, 3, stride=2, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.down_path_8_2 = conv_bn_in_rel(dim, 64, 64, 3, stride=1, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.down_path_16 = conv_bn_in_rel(dim, 64, 64, 3, stride=2, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)


        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.up_path_8_1 = conv_bn_in_rel(dim, 64, 64, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=use_batch_normalization, inn=use_instance_normalization, reverse=True)
        self.up_path_8_2 = conv_bn_in_rel(dim, 128, 64, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.up_path_4_1 = conv_bn_in_rel(dim, 64, 64, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=use_batch_normalization, inn=use_instance_normalization,reverse=True)
        self.up_path_4_2 = conv_bn_in_rel(dim, 96, 32, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.up_path_2_1 = conv_bn_in_rel(dim, 32, 32, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=use_batch_normalization, inn=use_instance_normalization,reverse=True)
        self.up_path_2_2 = conv_bn_in_rel(dim, 64, 8, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.up_path_1_1 = conv_bn_in_rel(dim, 8, 8, 2, stride=2, active_unit='None', same_padding=False, bn=use_batch_normalization, inn=use_instance_normalization, reverse=True)
        self.up_path_1_2 = conv_bn_in_rel(dim, 24, n_out_channel, 3, stride=1, active_unit='None', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)

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
        u8_2 = self.up_path_8_2(torch.cat((d8_2,u8_1),1))
        u4_1 = self.up_path_4_1(u8_2)
        u4_2 = self.up_path_4_2(torch.cat((d4_2,u4_1),1))
        u2_1 = self.up_path_2_1(u4_2)
        u2_2 = self.up_path_2_2(torch.cat((d2_2, u2_1), 1))
        u1_1 = self.up_path_1_1(u2_2)
        output = self.up_path_1_2(torch.cat((d1, u1_1), 1))

        return output

batch_norm_momentum_val = 0.1

class Simple_consistent(nn.Module):

    def __init__(self, dim, n_in_channel, n_out_channel, use_batch_normalization=False, use_instance_normalization=True):

        super(Simple_consistent, self).__init__()

        self.nr_of_gaussians = n_out_channel
        self.nr_of_image_channels = n_in_channel
        self.dim = dim

        self.estimate_around_global_weights = False

        self.params = pars.ParameterDict()
        self.kernel_sizes = self.params[('kernel_sizes', [7, 7], 'size of the convolution kernels')]
        self.use_batch_normalization = self.params[('use_batch_normalization', use_batch_normalization, 'If true, uses batch normalization between layers')]
        self.use_instance_normalization = self.params[('use_instance_normalization', use_instance_normalization, 'If true, uses instance normalization between layers')]


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

        self.use_relu = self.params[('use_relu', False, 'if set to True uses Relu, otherwise sigmoid')]

        self.conv_layers = None
        self.batch_normalizations = None
        self.instance_normalizations = None

        # needs to be initialized here, otherwise the optimizer won't see the modules from ModuleList
        # todo: figure out how to do ModuleList initialization not in __init__
        # todo: this would allow removing dim and nr_of_image_channels from interface
        # todo: because it could be compute on the fly when forward is executed
        self._init(self.nr_of_image_channels, dim=self.dim)

    def _init(self, nr_of_image_channels, dim):
        """
        Initalizes all the conv layers
        :param nr_of_image_channels:
        :param dim:
        :return:
        """

        assert (self.nr_of_layers > 0)

        nr_of_input_channels = nr_of_image_channels

        convs = [None] * self.nr_of_layers

        # first layer
        convs[0] = DimConv(self.dim)(nr_of_input_channels,
                                     self.nr_of_features_per_layer[0],
                                     self.kernel_sizes[0], padding=(self.kernel_sizes[0] - 1) // 2)

        # all the intermediate layers and the last one
        for l in range(self.nr_of_layers - 1):
            convs[l + 1] = DimConv(self.dim)(self.nr_of_features_per_layer[l],
                                             self.nr_of_features_per_layer[l + 1],
                                             self.kernel_sizes[l + 1],
                                             padding=(self.kernel_sizes[l + 1] - 1) // 2)

        self.conv_layers = nn.ModuleList()
        for c in convs:
            self.conv_layers.append(c)

        if self.use_batch_normalization:

            batch_normalizations = [None] * (self.nr_of_layers - 1)  # not on the last layer
            for b in range(self.nr_of_layers - 1):
                batch_normalizations[b] = DimBatchNorm(self.dim)(self.nr_of_features_per_layer[b],
                                                                 momentum=batch_norm_momentum_val)

            self.batch_normalizations = nn.ModuleList()
            for b in batch_normalizations:
                self.batch_normalizations.append(b)

        if self.use_instance_normalization:

            instance_normalizations = [None] * (self.nr_of_layers - 1)  # not on the last layer
            for b in range(self.nr_of_layers - 1):
                instance_normalizations[b] = DimInstanceNorm(self.dim)(self.nr_of_features_per_layer[b],
                                                                 momentum=batch_norm_momentum_val)

            self.instance_normalization = nn.ModuleList()
            for b in instance_normalizations:
                self.instance_normalizations.append(b)

        #self._initialize_weights()

    def forward(self, x):

        # now let's apply all the convolution layers, until the last
        # (because the last one is not relu-ed

        if self.use_batch_normalization:
            for l in range(len(self.conv_layers) - 1):
                if self.use_relu:
                    x = F.relu(self.batch_normalizations[l](self.conv_layers[l](x)))
                else:
                    x = F.sigmoid(self.batch_normalizations[l](self.conv_layers[l](x)))
        elif self.use_instance_normalization:
            for l in range(len(self.conv_layers) - 1):
                if self.use_relu:
                    x = F.relu(self.instance_normalizations[l](self.conv_layers[l](x)))
                else:
                    x = F.sigmoid(self.instance_normalizations[l](self.conv_layers[l](x)))
        else:
            for l in range(len(self.conv_layers) - 1):
                if self.use_relu:
                    x = F.relu(self.conv_layers[l](x))
                else:
                    x = F.sigmoid(self.conv_layers[l](x))

        # and now apply the last one without an activation for now
        # because we want to have the ability to smooth *before* the softmax
        # this is similar to smoothing in the logit domain for an active mean field approach
        x = self.conv_layers[-1](x)

        # now we are ready for the weighted softmax (will be like softmax if no weights are specified)

        if self.estimate_around_global_weights:
            import pyreg.deep_smoothers as deep_smoothers
            weights = deep_smoothers.weighted_softmax(x, dim=1, weights=self.global_multi_gaussian_weights)
        else:
            weights = F.softmax(x, dim=1)

        return weights


class Unet_no_skip(nn.Module):
    """
    unet include 4 down path (1/16)  and 4 up path (16)
    """
    def __init__(self, dim, n_in_channel, n_out_channel, use_batch_normalization=False, use_instance_normalization=True):
        # each dimension of the input should be 16x
        super(Unet_no_skip,self).__init__()
        self.down_path_1 = conv_bn_in_rel(dim, n_in_channel, 16, 3, stride=1, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.down_path_2_1 = conv_bn_in_rel(dim, 16, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.down_path_2_2 = conv_bn_in_rel(dim, 32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.down_path_4_1 = conv_bn_in_rel(dim, 32, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.down_path_4_2 = conv_bn_in_rel(dim, 32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.down_path_8_1 = conv_bn_in_rel(dim, 32, 64, 3, stride=2, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.down_path_8_2 = conv_bn_in_rel(dim, 64, 64, 3, stride=1, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.down_path_16 = conv_bn_in_rel(dim, 64, 64, 3, stride=2, active_unit='relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)


        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.up_path_8_1 = conv_bn_in_rel(dim, 64, 64, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=use_batch_normalization, inn=use_instance_normalization,reverse=True)
        self.up_path_8_2 = conv_bn_in_rel(dim, 64, 64, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.up_path_4_1 = conv_bn_in_rel(dim, 64, 64, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=use_batch_normalization, inn=use_instance_normalization,reverse=True)
        self.up_path_4_2 = conv_bn_in_rel(dim, 64, 32, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.up_path_2_1 = conv_bn_in_rel(dim, 32, 32, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=use_batch_normalization, inn=use_instance_normalization,reverse=True)
        self.up_path_2_2 = conv_bn_in_rel(dim, 32, 8, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)
        self.up_path_1_1 = conv_bn_in_rel(dim, 8, 8, 2, stride=2, active_unit='None', same_padding=False, bn=use_batch_normalization, inn=use_instance_normalization, reverse=True)
        self.up_path_1_2 = conv_bn_in_rel(dim, 8, n_out_channel, 3, stride=1, active_unit='None', same_padding=True, bn=use_batch_normalization, inn=use_instance_normalization)

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
