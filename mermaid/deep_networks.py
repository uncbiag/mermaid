from __future__ import print_function
from __future__ import absolute_import
from builtins import object

from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass

import torch
import torch.nn as nn

from . import finite_differences as fd
from . import module_parameters as pars
from . import noisy_convolution as nc

import numpy as np

from  .data_wrapper import MyTensor, AdaptVal, USE_CUDA

device = torch.device("cuda:0" if (USE_CUDA and torch.cuda.is_available()) else "cpu")

import math

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from . import utils as utils

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

    #todo: make the momentum a parameter

    if normalization_type.lower()=='batch':
        return DimBatchNorm(dim)(nr_channels, eps=0.0001, momentum=0.75, affine=True)
    elif normalization_type.lower()=='layer':
        int_im_sz = [int(elem) for elem in im_sz]
        layer_sz = [int(nr_channels)] + int_im_sz
        return nn.LayerNorm(layer_sz)
    elif normalization_type.lower()=='instance':
        return DimInstanceNorm(dim)(nr_channels, eps=0.0001, momentum=0.75, affine=True)
    elif normalization_type.lower()=='group':
        channels_per_group = nr_channels # just create one channel here
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
                 noisy_convolution_std=0.25,
                 noisy_convolution_optimize_over_std=False,
                 use_noise_layer=False,
                 noise_layer_std=0.25,
                 start_reducing_from_iter=0
                 ):

        super(conv_norm_in_rel, self).__init__()
        self.use_noisy_convolution = use_noisy_convolution
        self.use_noise_layer = use_noise_layer

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

        self.noisy_layer = nc.NoisyLayer(std_init=noise_layer_std,start_reducing_from_iter=start_reducing_from_iter) if self.use_noise_layer else None

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
        if self.use_noise_layer:
            x = self.noisy_layer(x,iter=iter)
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

        self.normalization_type = self.params[('normalization_type', 'group',
                                               "Normalization type between layers: ['batch'|'layer'|'instance'|'group'|'none']")]
        if self.normalization_type.lower() == 'none':
            self.normalization_type = None
        self.normalize_last_layer_type = self.params[('normalize_last_layer_type', 'group',
                                               "Normalization type between layers: ['batch'|'layer'|'instance'|'group'|'none']")]

        self.use_noisy_convolution = self.params[
            ('use_noisy_convolution', False, 'when true then the convolution layers will be replaced by noisy convolution layer')]

        # these settings are only relevant if we do a noisy convolution
        if self.use_noisy_convolution:
            self.noisy_convolution_std = self.params[('noisy_convolution_std', 0.25, 'Standard deviation for the noise')]
            self.noisy_convolution_optimize_over_std = self.params[
                ('noisy_convolution_optimize_over_std', False, 'If set to True, noise standard deviations are optimized')]
        else:
            self.noisy_convolution_std = None
            self.noisy_convolution_optimize_over_std = False

        self.use_noise_layers = self.params[('use_noise_layers',False,'If set to true noise is injected before the nonlinear activation function and *after* potential normalization')]
        if self.use_noise_layers:
            self.noise_layer_std = self.params[('noise_layers_std', 0.25, 'Standard deviation for the noise')]
            self.last_noise_layer_std = self.params[('last_noise_layer_std',0.025,'Standard deviation of noise for the last noise layer')]
        else:
            self.noise_layer_std = None
            self.last_noise_layer_std = None

        if self.use_noise_layers or self.use_noisy_convolution:
            self.start_reducing_from_iter = self.params[('start_reducing_from_iter',10,'After which iteration the noise is reduced')]
        else:
            self.start_reducing_from_iter = None

        if self.use_noise_layers and self.use_noisy_convolution:
            raise ValueError('Noise layers and noisy convolution are not intended to be used together. Pick one or the other!')

        self.normalize_last_layer = self.params[('normalize_last_layer',True,'If set to true normalization is also used for the last layer')]
        self.normalize_last_layer_initial_affine_slope = self.params[('normalize_last_layer_initial_affine_slope',0.025,'initial slope of affine transformation for batch and group normalization')]

    def _find_last_layer_of_type(self, layer_types):
        ln = None
        for m in self.modules():
            for t in layer_types:
                if isinstance(m, t):
                    ln = m

        return ln

    def _find_last_normalization_layer(self):
        ln = self._find_last_layer_of_type([DimBatchNorm(self.dim),DimInstanceNorm(self.dim),nn.LayerNorm,nn.GroupNorm])
        return ln

    def _find_last_noisy_layer(self):
        ln = self._find_last_layer_of_type([nc.NoisyLayer])
        return ln

    @abstractmethod
    def get_last_kernel_size(self):
        """
        Returns the size of the kernel along one direction (needs to be taken to the power of the dimension) for the last convolutional layer.
        This allows for example to scale numerical algorithms with respect to it.
        :return:
        """
        pass

    def initialize_network_weights(self):


        last_norm_layer = self._find_last_normalization_layer()

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
                # only the last normalization layer gets smaller weights (to make sure the output can stay close to the original weights)
                if self.normalize_last_layer:
                    if m == last_norm_layer:
                        m.weight.data.fill_(self.normalize_last_layer_initial_affine_slope)
            elif isinstance(m, nn.GroupNorm):
                # only the last normalization layer gets smaller weights (to make sure the output can stay close to the original weights)
                if self.normalize_last_layer:
                    if m==last_norm_layer:
                        m.weight.data.fill_(self.normalize_last_layer_initial_affine_slope)
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
        self.down_path_1 = conv_norm_in_rel(dim, n_in_channel, 16, kernel_size=3, im_sz=im_sz, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )
        self.down_path_2_1 = conv_norm_in_rel(dim, 16, 32, kernel_size=3, im_sz=im_sz_down_1, stride=2, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_2_2 = conv_norm_in_rel(dim, 32, 32, kernel_size=3, im_sz=im_sz_down_1, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_4_1 = conv_norm_in_rel(dim, 32, 32, kernel_size=3, im_sz=im_sz_down_2, stride=2, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_4_2 = conv_norm_in_rel(dim, 32, 32, kernel_size=3, im_sz=im_sz_down_2, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_8_1 = conv_norm_in_rel(dim, 32, 64, kernel_size=3, im_sz=im_sz_down_3, stride=2, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_8_2 = conv_norm_in_rel(dim, 64, 64, kernel_size=3, im_sz=im_sz_down_3, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )

        self.up_path_4_1 = conv_norm_in_rel(dim, 64, 64, kernel_size=2, im_sz=im_sz_down_2, stride=2, active_unit='leaky_relu', same_padding=False, normalization_type=self.normalization_type, reverse=True,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )
        self.up_path_4_2 = conv_norm_in_rel(dim, 96, 32, kernel_size=3, im_sz=im_sz_down_2, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )
        self.up_path_2_1 = conv_norm_in_rel(dim, 32, 32, kernel_size=2, im_sz=im_sz_down_1, stride=2, active_unit='leaky_relu', same_padding=False, normalization_type=self.normalization_type,reverse=True,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )
        self.up_path_2_2 = conv_norm_in_rel(dim, 64, 32, kernel_size=3, im_sz=im_sz_down_1, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )
        self.up_path_1_1 = conv_norm_in_rel(dim, 32, 14, kernel_size=2, im_sz=im_sz, stride=2, active_unit='None', same_padding=False, normalization_type=self.normalization_type, reverse=True,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )

        # we do not want to normalize the last layer as it will create the output
        if self.normalize_last_layer:
            current_normalization_type = self.normalize_last_layer_type
        else:
            current_normalization_type = None

        self.up_path_1_2 = conv_norm_in_rel(dim, 30, n_out_channel, kernel_size=3, im_sz=im_sz, stride=1, active_unit='None', same_padding=True,
                                            normalization_type=current_normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.last_noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )

    def get_last_kernel_size(self):
        return 3

    def forward(self, x, iter=0):
        d1 = self.down_path_1(x, iter=iter)
        d2_1 = self.down_path_2_1(d1, iter=iter)
        d2_2 = self.down_path_2_2(d2_1, iter=iter)
        d4_1 = self.down_path_4_1(d2_2, iter=iter)
        d4_2 = self.down_path_4_2(d4_1, iter=iter)
        d8_1 = self.down_path_8_1(d4_2, iter=iter)
        d8_2 = self.down_path_8_2(d8_1, iter=iter)

        u4_1 = self.up_path_4_1(d8_2, iter=iter)
        u4_2 = self.up_path_4_2(torch.cat((d4_2,u4_1),1), iter=iter)
        u2_1 = self.up_path_2_1(u4_2, iter=iter)
        u2_2 = self.up_path_2_2(torch.cat((d2_2, u2_1), 1), iter=iter)
        u1_1 = self.up_path_1_1(u2_2, iter=iter)
        output = self.up_path_1_2(torch.cat((d1, u1_1), 1), iter=iter)

        return output




class Simple_Unet(DeepNetwork):
    """
    unet include 4 down path (1/16)  and 4 up path (16)
    """
    def __init__(self, dim, n_in_channel, n_out_channel, im_sz, params):

        super(Simple_Unet, self).__init__(dim, n_in_channel, n_out_channel, im_sz, params)

        im_sz_down_1 = [elem//2 for elem in im_sz]
        im_sz_down_2 = [elem//2 for elem in im_sz_down_1]
        im_sz_down_3 = [elem//2 for elem in im_sz_down_2]
        im_sz_down_4 = [elem//2 for elem in im_sz_down_3]

        # each dimension of the input should be 16x
        self.down_path_1 = conv_norm_in_rel(dim, n_in_channel, 8, kernel_size=3, im_sz=im_sz, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )
        self.down_path_2_1 = conv_norm_in_rel(dim, 8, 16, kernel_size=3, im_sz=im_sz_down_1, stride=2, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_2_2 = conv_norm_in_rel(dim, 16, 16, kernel_size=3, im_sz=im_sz_down_1, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_4_1 = conv_norm_in_rel(dim, 16, 16, kernel_size=3, im_sz=im_sz_down_2, stride=2, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_4_2 = conv_norm_in_rel(dim, 16, 16, kernel_size=3, im_sz=im_sz_down_2, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_8_1 = conv_norm_in_rel(dim, 16, 32, kernel_size=3, im_sz=im_sz_down_3, stride=2, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_8_2 = conv_norm_in_rel(dim, 32, 32, kernel_size=3, im_sz=im_sz_down_3, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )

        self.up_path_4_1 = conv_norm_in_rel(dim, 32, 32, kernel_size=2, im_sz=im_sz_down_2, stride=2, active_unit='leaky_relu', same_padding=False, normalization_type=self.normalization_type, reverse=True,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )
        self.up_path_4_2 = conv_norm_in_rel(dim, 48, 16, kernel_size=3, im_sz=im_sz_down_2, stride=1, active_unit='leaky_relu', same_padding=True, normalization_type=self.normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )
        self.up_path_2_1 = conv_norm_in_rel(dim, 16, 8, kernel_size=2, im_sz=im_sz_down_1, stride=2, active_unit='None', same_padding=False, normalization_type=self.normalization_type,reverse=True,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )
        if self.normalize_last_layer:
            current_normalization_type = self.normalize_last_layer_type
        else:
            current_normalization_type = None
        self.up_path_2_2 = conv_norm_in_rel(dim, 24, n_out_channel, kernel_size=3, im_sz=im_sz_down_1, stride=1, active_unit='None', same_padding=True,
                                            normalization_type = current_normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )




    def get_last_kernel_size(self):
        return 3

    def forward(self, x, iter=0):
        d1 = self.down_path_1(x, iter=iter)
        d2_1 = self.down_path_2_1(d1, iter=iter)
        d2_2 = self.down_path_2_2(d2_1, iter=iter)
        d4_1 = self.down_path_4_1(d2_2, iter=iter)
        d4_2 = self.down_path_4_2(d4_1, iter=iter)
        d8_1 = self.down_path_8_1(d4_2, iter=iter)
        d8_2 = self.down_path_8_2(d8_1, iter=iter)

        u4_1 = self.up_path_4_1(d8_2, iter=iter)
        u4_2 = self.up_path_4_2(torch.cat((d4_2,u4_1),1), iter=iter)
        u2_1 = self.up_path_2_1(u4_2, iter=iter)
        u2_2 = self.up_path_2_2(torch.cat((d2_2, u2_1), 1), iter=iter)
        output =F.interpolate(u2_2, scale_factor=2, mode='trilinear')
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

    def get_last_kernel_size(self):
        return 3

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

        self.kernel_sizes = self.params[('kernel_sizes', [5, 5], 'size of the convolution kernels')]

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

        self.use_dropout = self.params[('use_dropout', True, 'If set to true, dropout witll be used')]
        if self.use_dropout:
            self.dropout_prob = self.params[('dropout_prob',0.25,'dropout probability')]
            self.dropout = nn.Dropout(self.dropout_prob)
        else:
            self.dropout = None

        self.conv_layers = None
        self.normalizations = None
        self.noise_layers = None

        # needs to be initialized here, otherwise the optimizer won't see the modules from ModuleList
        # todo: figure out how to do ModuleList initialization not in __init__
        # todo: this would allow removing dim and nr_of_image_channels from interface
        # todo: because it could be compute on the fly when forward is executed
        self._init(self.nr_of_image_channels, dim=self.dim, im_sz=self.im_sz)

    def apply_dropout(self, input):
        if self.use_dropout:
            return self.dropout(input)
        else:
            return input

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

        if self.normalize_last_layer:
            nr_normalizations = self.nr_of_layers
        else:
            nr_normalizations = self.nr_of_layers-1

        if self.normalization_type is not None:

            normalizations = [None] * nr_normalizations
            for b in range(nr_normalizations):
                normalizations[b] = DimNormalization(self.dim,normalization_type=self.normalization_type,nr_channels=self.nr_of_features_per_layer[b],im_sz=im_sz)

            self.normalizations = nn.ModuleList()
            for b in normalizations:
                self.normalizations.append(b)

        if self.use_noise_layers:
            noise_layers = [None] * self.nr_of_layers
            for nl in range(self.nr_of_layers-1):
                noise_layers[nl] = nc.NoisyLayer(std_init=self.noise_layer_std,start_reducing_from_iter=self.start_reducing_from_iter)
            noise_layers[-1] = nc.NoisyLayer(std_init=self.last_noise_layer_std,start_reducing_from_iter=self.start_reducing_from_iter)

            self.noise_layers = nn.ModuleList()
            for nl in noise_layers:
                self.noise_layers.append(nl)


    def get_last_kernel_size(self):
        return self.kernel_sizes[-1]

    def forward(self, x, iter=0):

        # now let's apply all the convolution layers, until the last
        # (because the last one is not relu-ed

        if self.normalization_type is not None:
            for l in range(len(self.conv_layers) - 1):
                if self.use_noisy_convolution:
                    y = self.normalizations[l](self.conv_layers[l](x,iter=iter))
                    if self.use_noise_layers:
                        x = self.apply_dropout(self.active_unit(self.noise_layers[l](y,iter=iter)))
                    else:
                        x = self.apply_dropout(self.active_unit(y))
                else:
                    y = self.normalizations[l](self.conv_layers[l](x))
                    if self.use_noise_layers:
                        x = self.apply_dropout(self.active_unit(self.noise_layers[l](y,iter=iter)))
                    else:
                        x = self.apply_dropout(self.active_unit(y))
        else:
            for l in range(len(self.conv_layers) - 1):
                if self.use_noisy_convolution:
                    if self.use_noise_layers:
                        x = self.apply_dropout(self.active_unit(self.noise_layers[l](self.conv_layers[l](x,iter=iter),iter=iter)))
                    else:
                        x = self.apply_dropout(self.active_unit(self.conv_layers[l](x,iter=iter)))
                else:
                    if self.use_noise_layers:
                        x = self.apply_dropout(self.active_unit(self.noise_layers[l](self.conv_layers[l](x),iter=iter)))
                    else:
                        x = self.apply_dropout(self.active_unit(self.conv_layers[l](x)))


        # and now apply the last one without an activation for now
        # because we want to have the ability to smooth *before* the softmax
        # this is similar to smoothing in the logit domain for an active mean field approach
        if self.normalize_last_layer:
            if self.use_noisy_convolution:
                x = self.normalizations[-1](self.conv_layers[-1](x,iter=iter))
            else:
                x = self.normalizations[-1](self.conv_layers[-1](x))
        else:
            if self.use_noisy_convolution:
                x = self.conv_layers[-1](x,iter=iter)
            else:
                x = self.conv_layers[-1](x)

        # does not go through a nonlinearity at the end, so just add some noise if desired
        if self.use_noise_layers:
            x = self.noise_layers[-1](x,iter=iter)

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

        self.down_path_1 = conv_norm_in_rel(dim, n_in_channel, 16, kernel_size=3, im_sz=im_sz, stride=1, active_unit='leaky_relu', same_padding=True,
                                            normalization_type=self.normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )
        self.down_path_2_1 = conv_norm_in_rel(dim, 16, 32, kernel_size=3, im_sz=im_sz_down_1, stride=2, active_unit='leaky_relu', same_padding=True,
                                              normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_2_2 = conv_norm_in_rel(dim, 32, 32, kernel_size=3, im_sz=im_sz_down_1, stride=1, active_unit='leaky_relu', same_padding=True,
                                              normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_4_1 = conv_norm_in_rel(dim, 32, 32, kernel_size=3, im_sz=im_sz_down_2, stride=2, active_unit='leaky_relu', same_padding=True,
                                              normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_4_2 = conv_norm_in_rel(dim, 32, 32, kernel_size=3, im_sz=im_sz_down_2, stride=1, active_unit='leaky_relu', same_padding=True,
                                              normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_8_1 = conv_norm_in_rel(dim, 32, 64, kernel_size=3, im_sz=im_sz_down_3, stride=2, active_unit='leaky_relu', same_padding=True,
                                              normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_8_2 = conv_norm_in_rel(dim, 64, 64, kernel_size=3, im_sz=im_sz_down_3, stride=1, active_unit='leaky_relu', same_padding=True,
                                              normalization_type=self.normalization_type,
                                              use_noisy_convolution=self.use_noisy_convolution,
                                              noisy_convolution_std=self.noisy_convolution_std,
                                              noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                              use_noise_layer=self.use_noise_layers,
                                              noise_layer_std=self.noise_layer_std,
                                              start_reducing_from_iter=self.start_reducing_from_iter
                                              )
        self.down_path_16 = conv_norm_in_rel(dim, 64, 64, kernel_size=3, im_sz=im_sz_down_4, stride=2, active_unit='leaky_relu', same_padding=True,
                                             normalization_type=self.normalization_type,
                                             use_noisy_convolution=self.use_noisy_convolution,
                                             noisy_convolution_std=self.noisy_convolution_std,
                                             noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                             use_noise_layer=self.use_noise_layers,
                                             noise_layer_std=self.noise_layer_std,
                                             start_reducing_from_iter=self.start_reducing_from_iter
                                             )


        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.up_path_8_1 = conv_norm_in_rel(dim, 64, 64, kernel_size=2, im_sz=im_sz_down_3, stride=2, active_unit='leaky_relu', same_padding=False,
                                            normalization_type=self.normalization_type,reverse=True,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )
        self.up_path_8_2 = conv_norm_in_rel(dim, 64, 64, kernel_size=3, im_sz=im_sz_down_3, stride=1, active_unit='leaky_relu', same_padding=True,
                                            normalization_type=self.normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )
        self.up_path_4_1 = conv_norm_in_rel(dim, 64, 64, kernel_size=2, im_sz=im_sz_down_2, stride=2, active_unit='leaky_relu', same_padding=False,
                                            normalization_type=self.normalization_type,reverse=True,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )
        self.up_path_4_2 = conv_norm_in_rel(dim, 64, 32, kernel_size=3, im_sz=im_sz_down_2, stride=1, active_unit='leaky_relu', same_padding=True,
                                            normalization_type=self.normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )
        self.up_path_2_1 = conv_norm_in_rel(dim, 32, 32, kernel_size=2, im_sz=im_sz_down_1, stride=2, active_unit='leaky_relu', same_padding=False,
                                            normalization_type=self.normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )
        self.up_path_1_1 = conv_norm_in_rel(dim, 8, 8, kernel_size=2, im_sz=im_sz, stride=2, active_unit='None', same_padding=False,
                                            normalization_type=self.normalization_type, reverse=True,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )

        if self.normalize_last_layer:
            current_normalization_type = self.normalization_type
        else:
            current_normalization_type = None

        self.up_path_1_2 = conv_norm_in_rel(dim, 8, n_out_channel, kernel_size=3, im_sz=im_sz, stride=1, active_unit='None', same_padding=True,
                                            normalization_type=current_normalization_type,
                                            use_noisy_convolution=self.use_noisy_convolution,
                                            noisy_convolution_std=self.noisy_convolution_std,
                                            noisy_convolution_optimize_over_std=self.noisy_convolution_optimize_over_std,
                                            use_noise_layer=self.use_noise_layers,
                                            noise_layer_std=self.last_noise_layer_std,
                                            start_reducing_from_iter=self.start_reducing_from_iter
                                            )

    def get_last_kernel_size(self):
        return 3

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

class WeightRangeLoss(nn.Module):
    def __init__(self,dim,decay_factor,weight_type):
        super(WeightRangeLoss,self).__init__()
        self.dim = dim
        self.decay_factor = decay_factor
        self.is_w_K_w = weight_type=='w_K_w'

    def forward(self,x, spacing, weights):
        weights = weights if not self.is_w_K_w else torch.sqrt(weights)
        view_sz = [1] + [len(weights)] + [1]*self.dim
        init_weights = weights.view(*view_sz)
        diff = x - init_weights
        volumeElement = spacing.prod()
        loss = utils.remove_infs_from_variable(diff ** 2).sum() * volumeElement
        return loss

    def cal_weights_for_weightrange(self,epoch):
        def sigmoid_decay(ep, static=5, k=5):
            static = static
            if ep < static:
                return float(1.)
            else:
                ep = ep - static
                factor = k / (k + np.exp(ep / k))

            return float(factor)

        cur_weight = max(sigmoid_decay(epoch, static=10, k=self.decay_factor),0.1)
        return cur_weight



class WeightInputRangeLoss(nn.Module):
    def __init__(self):
        super(WeightInputRangeLoss, self).__init__()

    def forward(self, x, spacing, use_weighted_linear_softmax=False, weights=None, min_weight=0.0, max_weight=1.0, dim=None):

        if spacing is not None:
            volumeElement = spacing.prod()
        else:
            volumeElement = 1.0

        if not use_weighted_linear_softmax:
            # checks what is hit by the clamping
            xd = x-torch.clamp(x,min_weight,max_weight)
            loss = utils.remove_infs_from_variable((xd**2)).sum()*volumeElement
            #loss = torch.abs(xd).sum() * volumeElement
        else:
            # weights are in dimension 1; assumes that weighted linear softmax is used
            # Here, we account for the fact that the input is modulated by the global weights

            if (weights is None) or (dim is None):
                raise ValueError('Weights and dim need to be defined to use the weighted linear softmax')

            sz = x.size()
            input_offset = x.sum(dim=dim)/sz[dim]

            loss = MyTensor(1).zero_()
            print(weights.shape, x.shape, input_offset.shape)

            for c in range(sz[dim]):
                if dim==0:
                    eff_input = weights[c] + x[c, ...] - input_offset
                elif dim==1:
                    eff_input = weights[c] + x[:, c, ...] - input_offset
                elif dim==2:
                    eff_input = weights[c] + x[:, :, c, ...] - input_offset
                elif dim==3:
                    eff_input = weights[c] + x[:, :, :, c, ...] - input_offset
                elif dim==4:
                    eff_input = weights[c] + x[:, :, :, :, c, ...] - input_offset
                else:
                    raise ValueError('Only dimensions {0,1,2,3,4} are supported')
                eff_input_d = eff_input-torch.clamp(eff_input,min_weight,max_weight)
                loss += utils.remove_infs_from_variable(eff_input_d**2).sum()*volumeElement
                #loss += torch.abs(eff_input_d).sum() * volumeElement

        return loss

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, spacing):
        volumeElement = spacing.prod()
        b = x*torch.log(x)
        #F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        #b = -1.0 * b.sum()*volumeElement
        b = -1.0*b.sum()*volumeElement
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

class OMTLoss(nn.Module):
    """
    OMT Loss function
    """

    def __init__(self,spacing,desired_power,use_log_transform,params,img_sz):
        super(OMTLoss, self).__init__()

        self.params = params
        self.spacing = spacing
        self.desired_power = desired_power
        self.use_log_transform = use_log_transform
        self.img_sz =img_sz
        self.use_boundary_mask = False
        self.mask=None
        if self.use_boundary_mask:
            print("ATTENTION, THE BOUNDARY MASK IS USED, CURRENT SETTING IS ONLY FOR OAI DATASET")
            self.mask =utils.omt_boundary_weight_mask(img_sz, spacing, mask_range=3, mask_value=10, smoother_std=0.04)
        self.volume_element = self.spacing.prod()

    def compute_omt_penalty(self, weights, multi_gaussian_stds):

        # weights: B x weights x X x Y

        if weights.size()[1] != len(multi_gaussian_stds):
            raise ValueError(
                'Number of weights need to be the same as number of Gaussians. Format recently changed for weights to B x weights x X x Y')

        penalty = MyTensor(1).zero_()

        max_std = multi_gaussian_stds.max()
        min_std = multi_gaussian_stds.min()

        if self.desired_power == 2:
            for i, s in enumerate(multi_gaussian_stds):
                if self.use_log_transform:
                    penalty += ((weights[:, i, ...]).sum() if self.mask is None else weights[:, i]*self.mask[:,0]) * ((torch.log(max_std / s)) ** self.desired_power)
                else:
                    penalty += ((weights[:, i, ...]).sum() if self.mask is None else weights[:, i]*self.mask[:,0]) * ((s - max_std) ** self.desired_power)

            if self.use_log_transform:
                penalty /= (torch.log(max_std / min_std)) ** self.desired_power
            else:
                penalty /= (max_std - min_std) ** self.desired_power
        else:
            for i, s in enumerate(multi_gaussian_stds):
                if self.use_log_transform:
                    penalty += ((weights[:, i, ...] if self.mask is None else weights[:, i]*self.mask[:,0]).sum()) * (torch.abs(torch.log(max_std / s)) ** self.desired_power)
                else:
                    penalty += ((weights[:, i, ...] if self.mask is None else weights[:, i]*self.mask[:,0]).sum()) * (torch.abs(s - max_std) ** self.desired_power)

            if self.use_log_transform:
                penalty /= torch.abs(torch.log(max_std / min_std)) ** self.desired_power
            else:
                penalty /= torch.abs(max_std - min_std) ** self.desired_power

        penalty *= self.volume_element

        return penalty
    def cal_weights_for_omt(self,epoch):
        def sigmoid_decay(ep, static=5, k=5):
            static = static
            if ep < static:
                return float(1.)
            else:
                ep = ep - static
                factor = k / (k + np.exp(ep / k))

            return float(factor)

        cur_weight = max(sigmoid_decay(epoch, static=30, k=10),0.001)
        return cur_weight


    def forward(self, weights, gaussian_stds):
        return self.compute_omt_penalty(weights=weights,multi_gaussian_stds=gaussian_stds)

class TotalVariationLoss(nn.Module):
    """
    Loss function to penalize total variation
    """

    def __init__(self, dim, im_sz, spacing,
                 use_omt_weighting=False,
                 gaussian_stds=None,
                 omt_power=1.0,
                 omt_use_log_transformed_std=True,
                 params=None):
        """

        :param params: ParameterDict() object to hold and keep track of general parameters
        """
        super(TotalVariationLoss, self).__init__()

        self.params = params
        """ParameterDict() parameters"""
        self.dim = dim
        # dimension
        self.im_sz = im_sz
        self.spacing = spacing

        self.use_omt_weighting = use_omt_weighting
        self.gaussian_stds = gaussian_stds
        self.omt_power = omt_power
        self.omt_use_log_transformed_std = omt_use_log_transformed_std

        self.smooth_image_for_edge_detection = self.params[('smooth_image_for_edge_detection',True,'Smooth image for edge detection')]
        self.smooth_image_for_edge_detection_std = self.params[('smooth_image_for_edge_detection_std',0.01,'Standard deviation for edge detection')]

        self.tv_weights = None
        if self.use_omt_weighting:
            self.tv_weights = self._compute_tv_weights()

        if self.smooth_image_for_edge_detection:
            from . import smoother_factory as sf

            s_m_params = pars.ParameterDict()
            s_m_params['smoother']['type'] = 'gaussian'
            s_m_params['smoother']['gaussian_std'] = self.smooth_image_for_edge_detection_std

            self.image_smoother = sf.SmootherFactory(im_sz, spacing=spacing).create_smoother(s_m_params)
        else:
            self.image_smoother = None

    def _compute_tv_weights(self):
        multi_gaussian_stds = self.gaussian_stds.detach().cpu().numpy()

        max_std = max(multi_gaussian_stds)
        min_std = min(multi_gaussian_stds)

        tv_weights = MyTensor(len(multi_gaussian_stds))

        desired_power = self.omt_power
        use_log_transform = self.omt_use_log_transformed_std

        for i, s in enumerate(multi_gaussian_stds):
            if use_log_transform:
                tv_weights[i] = abs(np.log(max_std / s)) ** desired_power
            else:
                tv_weights[i] = abs(s - max_std) ** desired_power

        if use_log_transform:
            tv_weights /= abs(np.log(max_std / min_std)) ** desired_power
        else:
            tv_weights /= abs(max_std - min_std) ** desired_power

        return tv_weights


    # todo: merge this with the code in deep_smoothers.py
    def compute_local_weighted_tv_norm(self, I, weights, spacing, nr_of_gaussians, use_color_tv, pnorm=2):

        import mermaid.deep_smoothers as deep_smoothers

        volumeElement = spacing.prod()

        individual_sum_of_total_variation_penalty = MyTensor(nr_of_gaussians).zero_()
        # first compute the edge map, based on a smoothed image

        if self.smooth_image_for_edge_detection:
            I_edge = self.image_smoother.smooth(I)
        else:
            I_edge = I

        g_I = deep_smoothers.compute_localized_edge_penalty(I_edge[:, 0, ...], spacing, self.params)

        # now computed weighted TV norm channel-by-channel, square it and then take the square root (this is like in color TV)
        for g in range(nr_of_gaussians):
            c_local_norm_grad = deep_smoothers._compute_local_norm_of_gradient(weights[:, g, ...], spacing, pnorm)

            to_sum = g_I * c_local_norm_grad * volumeElement
            current_tv = (to_sum).sum()

            individual_sum_of_total_variation_penalty[g] = current_tv

        if use_color_tv:
            if self.use_omt_weighting:
                total_variation_penalty = torch.norm(self.tv_weights*individual_sum_of_total_variation_penalty,p=2)
            else:
                total_variation_penalty = torch.norm(individual_sum_of_total_variation_penalty,p=2)
        else:
            if self.use_omt_weighting:
                total_variation_penalty = (self.tv_weights*individual_sum_of_total_variation_penalty).sum()
            else:
                total_variation_penalty = individual_sum_of_total_variation_penalty.sum()

        return total_variation_penalty


    def forward(self, input_images, label_probabilities, use_color_tv=False):
        # first compute the weighting functions

        nr_of_gaussians = label_probabilities.size()[1]
        current_penalty = self.compute_local_weighted_tv_norm(input_images, label_probabilities, self.spacing, nr_of_gaussians, use_color_tv)
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

    def get_last_kernel_size(self):
        return 1

    def forward(self, input_images, spacing, label_probabilities):
        import mermaid.deep_smoothers as deep_smoothers

        # first compute the weighting functions
        localized_edge_penalty = deep_smoothers.compute_localized_edge_penalty(input_images[:,0,...],spacing,self.params)

        batch_size = label_probabilities.size()[0]
        nr_of_clusters = label_probabilities.size()[1]
        current_penalties = AdaptVal(torch.ones(batch_size)*nr_of_clusters)

        for k in range(nr_of_clusters):
            current_penalties -= self._compute_cut_cost_for_label_k(w_edge=localized_edge_penalty,p=label_probabilities[:,k,...])

        current_penalty = current_penalties.sum()
        return current_penalty
