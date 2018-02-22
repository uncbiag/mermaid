import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from data_wrapper import USE_CUDA, MyTensor, AdaptVal

import math
import pyreg.finite_differences as fd

batch_norm_momentum_val = 0.1

def compute_omt_penalty(weights, multi_gaussian_stds,volume_element,desired_power=2.0):

    # weights: B x weights x X x Y

    if weights.size()[1] != len(multi_gaussian_stds):
        raise ValueError('Number of weights need to be the same as number of Gaussians. Format recently changed for weights to B x weights x X x Y')

    penalty = Variable(MyTensor(1).zero_(), requires_grad=False)
    batch_size = weights.size()[0]
    max_std = max(multi_gaussian_stds)
    for i,s in enumerate(multi_gaussian_stds):
        penalty += ((weights[:,i,...]).sum())*((s-max_std)**desired_power)

    penalty /= batch_size
    penalty *= volume_element

    return penalty

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

def DimConvTranspose(dim):
    if dim==1:
        return nn.ConvTranspose1d
    elif dim==2:
        return nn.ConvTranspose2d
    elif dim==3:
        return nn.ConvTranspose3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')

def weighted_softmax(input, dim=None, weights=None ):
    r"""Applies a softmax function.

    Weighted_softmax is defined as:

    :math:`weighted_softmax(x) = \frac{w_i exp(x_i)}{\sum_j w_j exp(x_j)}`

    It is applied to all slices along dim, and will rescale them so that the elements
    lie in the range `(0, 1)` and sum to 1.

    See :class:`~torch.nn.WeightedSoftmax` for more details.

    Arguments:
        input (Variable): input
        dim (int): A dimension along which weighted_softmax will be computed.

    """
    if dim is None:
        raise ValueError('dimension needs to be defined!')

    sz = input.size()
    if weights is None: # just make them all one; this is the default softmax
        weights = [1.]*sz[dim]

    nr_of_weights = len(weights)
    assert( sz[dim]==nr_of_weights )

    ret = torch.zeros_like(input)

    if dim==0:
        norm = torch.zeros_like(input[0,...])
        for c in range(sz[0]):
            norm += weights[c]*torch.exp(input[c,...])
        for c in range(sz[0]):
            ret[c,...] = weights[c]*torch.exp(input[c,...])/norm
    elif dim==1:
        norm = torch.zeros_like(input[:,0, ...])
        for c in range(sz[1]):
            norm += weights[c] * torch.exp(input[:,c, ...])
        for c in range(sz[1]):
            ret[:,c, ...] = weights[c] * torch.exp(input[:,c, ...]) / norm
    elif dim==2:
        norm = torch.zeros_like(input[:,:,0, ...])
        for c in range(sz[2]):
            norm += weights[c] * torch.exp(input[:,:,c, ...])
        for c in range(sz[2]):
            ret[:,:,c, ...] = weights[c] * torch.exp(input[:,:,c, ...]) / norm
    elif dim==3:
        norm = torch.zeros_like(input[:,:,:,0, ...])
        for c in range(sz[3]):
            norm += weights[c] * torch.exp(input[:,:,:,c, ...])
        for c in range(sz[3]):
            ret[:,:,:,c, ...] = weights[c] * torch.exp(input[:,:,:,c, ...]) / norm
    elif dim==4:
        norm = torch.zeros_like(input[:,:,:,:,0, ...])
        for c in range(sz[4]):
            norm += weights[c] * torch.exp(input[:,:,:,:,c, ...])
        for c in range(sz[4]):
            ret[:,:,:,:,c, ...] = weights[c] * torch.exp(input[:,:,:,:,c, ...]) / norm
    else:
        raise ValueError('weighted_softmax is only supported for dimensions 0, 1, 2, 3, and 4.')

    return ret


class WeightedSoftmax(nn.Module):
    r"""Applies the WeightedSoftmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range (0,1) and sum to 1

    WeightedSoftmax is defined as
    :math:`f_i(x) = \frac{w_i\exp(x_i)}{\sum_j w_j\exp(x_j)}`

    It is assumed that w_i>0 and that the weights sum up to one.
    The effect of this weighting is that for a zero input (x=0) the output for f_i(x) will be w_i.
    I.e., we can obtain a default output which is not 1/n.

    Shape:
        - Input: any shape
        - Output: same as input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Arguments:
        dim (int): A dimension along which WeightedSoftmax will be computed (so every slice
            along dim will sum to 1).

    Examples::

        >>> m = nn.WeightedSoftmax()
        >>> input = autograd.Variable(torch.randn(2, 3))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, dim=None, weights=None):
        super(WeightedSoftmax, self).__init__()
        self.dim = dim
        self.weights = weights

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None
        if not hasattr(self, 'weights'):
            self.weights = None

    def forward(self, input):

        return weighted_softmax(input, self.dim, self.weights, _stacklevel=5)

    def __repr__(self):

        return self.__class__.__name__ + '()'



class DeepSmoothingModel(nn.Module):
    """
    Base class for mini neural network which takes as an input a set of smoothed velocity field as
    well as input images and predicts weights for a multi-Gaussian smoothing from this
    Enforces the same weighting for all the dimensions of the vector field to be smoothed

    """

    def __init__(self, nr_of_gaussians, gaussian_stds, dim, spacing, nr_of_image_channels=1, omt_power=2.0,params=None):
        super(DeepSmoothingModel, self).__init__()

        self.nr_of_image_channels = nr_of_image_channels
        self.dim = dim
        self.omt_power = omt_power
        self.pnorm = 2

        self.spacing = spacing
        self.fdt = fd.FD_torch(self.spacing)
        self.volumeElement = self.spacing.prod()

        # check that the largest standard deviation is the largest one
        if max(gaussian_stds) > gaussian_stds[-1]:
            raise ValueError('The last standard deviation needs to be the largest')

        self.params = params

        self.omt_weight_penalty = self.params[('omt_weight_penalty', 25.0, 'Penalty for the optimal mass transport')]
        self.total_variation_weight_penalty = self.params[('total_variation_weight_penalty', 1.0, 'Penalize the total variation of the weights if desired')]
        self.diffusion_weight_penalty = self.params[('diffusion_weight_penalty',0.0,'Penalized the squared gradient of the weights')]

        self.omt_power = self.params[('omt_power', 2.0, 'Power for the optimal mass transport (i.e., to which power distances are penalized')]
        """optimal mass transport power"""

        self.do_input_standardization = self.params[('do_input_standardization',True,'if true, subtracts the mean from any image input and leaves momentum as is')]
        """if true subtracts the mean from all image input before running it through the network"""

        self.nr_of_gaussians = nr_of_gaussians
        self.gaussian_stds = gaussian_stds
        self.min_weight = 0.0001

        self.computed_weights = None
        """stores the computed weights if desired"""
        self.current_penalty = None
        """to stores the current penalty (for example OMT) after running through the model"""

    def get_omt_weight_penalty(self):
        return self.omt_weight_penalty

    def get_omt_power(self):
        return self.omt_power

    def _initialize_weights(self):

        print('WARNING: weight initialization disabled')
        return

        for m in self.modules():
            if isinstance(m, DimConv(self.dim)):
                n = m.out_channels
                for d in range(self.dim):
                    n *= m.kernel_size[d]
                m.weight.data.normal_(0, math.sqrt(0.05 / n))
            elif isinstance(m, DimBatchNorm(self.dim)):
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

    def compute_diffusion(self, d):
        # just do the standard component-wise Euclidean squared norm of the gradient

        if self.dim == 1:
            return self._compute_diffusion_1d(d)
        elif self.dim == 2:
            return self._compute_diffusion_2d(d)
        elif self.dim == 3:
            return self._compute_diffusion_3d(d)
        else:
            raise ValueError('Diffusion computation is currently only supported in dimensions 1 to 3')

    def _compute_diffusion_1d(self, d):

        # need to use torch.abs here to make sure the proper subgradient is computed at zero
        batch_size = d.size()[0]
        t0 = (self.fdt.dXc(d))**2

        return (t0).sum() * self.volumeElement / batch_size

    def _compute_diffusion_2d(self, d):

        # need to use torch.norm here to make sure the proper subgradient is computed at zero
        batch_size = d.size()[0]
        t0 = self.fdt.dXc(d)**2+self.fdt.dYc(d)**2

        return t0.sum() * self.volumeElement / batch_size

    def _compute_diffusion_3d(self, d):

        # need to use torch.norm here to make sure the proper subgradient is computed at zero
        batch_size = d.size()[0]
        t0 = self.fdt.dXc(d)**2 + self.fdt.dYc(d)**2 + self.fdt.dZc(d)**2

        return t0.sum() * self.volumeElement / batch_size


    def compute_total_variation(self, d):
        # just do the standard component-wise Euclidean norm of the gradient

        if self.dim == 1:
            return self._compute_total_variation_1d(d)
        elif self.dim == 2:
            return self._compute_total_variation_2d(d)
        elif self.dim == 3:
            return self._compute_total_variation_3d(d)
        else:
            raise ValueError('Total variation computation is currently only supported in dimensions 1 to 3')

    def _compute_total_variation_1d(self, d):

        # need to use torch.abs here to make sure the proper subgradient is computed at zero
        batch_size = d.size()[0]
        t0 = torch.abs(self.fdt.dXc(d))

        return (t0).sum()*self.volumeElement/batch_size

    def _compute_total_variation_2d(self, d):

        # need to use torch.norm here to make sure the proper subgradient is computed at zero
        batch_size = d.size()[0]
        t0 = torch.norm(torch.stack((self.fdt.dXc(d),self.fdt.dYc(d))),self.pnorm,0)

        return t0.sum()*self.volumeElement/batch_size

    def _compute_total_variation_3d(self, d):

        # need to use torch.norm here to make sure the proper subgradient is computed at zero
        batch_size = d.size()[0]
        t0 = torch.norm(torch.stack((self.fdt.dXc(d),
                                     self.fdt.dYc(d),
                                     self.fdt.dZc(d))), self.pnorm, 0)

        return t0.sum()*self.volumeElement/batch_size

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


class encoder_block_2d(nn.Module):
    def __init__(self, input_feature, output_feature, use_dropout, use_batch_normalization):
        super(encoder_block_2d, self).__init__()

        if use_batch_normalization:
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

        self.use_dropout = use_dropout
        self.use_batch_normalization = use_batch_normalization
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

    def forward_without_batch_normalization(self,x):
        output = self.conv_input(x)
        output = self.apply_dropout(self.prelu1(output))
        output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)))
        output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)))
        return self.prelu4(self.conv_pooling(output))

    def forward(self, x):
        if self.use_batch_normalization:
            return self.forward_with_batch_normalization(x)
        else:
            return self.forward_without_batch_normalization(x)

class decoder_block_2d(nn.Module):
    def __init__(self, input_feature, output_feature, pooling_filter,use_dropout, use_batch_normalization,last_block=False):
        super(decoder_block_2d, self).__init__()
        # todo: check padding here, not sure if it is the right thing to do

        if use_batch_normalization:
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

        self.use_dropout = use_dropout
        self.use_batch_normalization = use_batch_normalization
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
        else:
            return self.forward_without_batch_normalization(x)

class EncoderDecoderSmoothingModel(DeepSmoothingModel):
    """
    Similar to the model used in Quicksilver
    """
    def __init__(self, nr_of_gaussians, gaussian_stds, dim, spacing, nr_of_image_channels=1, omt_power=2.0, params=None ):
        super(EncoderDecoderSmoothingModel, self).__init__(nr_of_gaussians=nr_of_gaussians,\
                                                                     gaussian_stds=gaussian_stds,\
                                                                     dim=dim,\
                                                                     spacing=spacing,\
                                                                     nr_of_image_channels=nr_of_image_channels,\
                                                                     omt_power=omt_power,\
                                                                     params=params)

        # is 64 in quicksilver paper
        feature_num = self.params[('number_of_features_in_first_layer', 32, 'number of features in the first encoder layer (64 in quicksilver)')]
        use_dropout= self.params[('use_dropout',False,'use dropout for the layers')]
        self.use_separate_decoders_per_gaussian = self.params[('use_separate_decoders_per_gaussian',True,'if set to true separte decoder branches are used for each Gaussian')]
        self.use_momentum_as_input = self.params[('use_momentum_as_input',True,'If true, uses the image and the momentum as input')]
        use_batch_normalization = self.params[('use_batch_normalization',True,'If true, uses batch normalization between layers')]
        self.use_one_encoder_decoder_block = self.params[('use_one_encoder_decoder_block',True,'If False, using two each as in the quicksilver paper')]
        self.estimate_around_global_weights = self.params[('estimate_around_global_weights', True, 'If true, a weighted softmax is used so the default output (for input zero) are the global weights')]

        self.use_source_image_as_input = self.params[('use_source_image_as_input',True,'If true, uses the source image as additional input')]
        self.use_target_image_as_input = self.params[('use_target_image_as_input', True, 'If true, uses the target image as additional input')]

        if self.use_momentum_as_input or self.use_target_image_as_input or self.use_source_image_as_input:
            self.encoder_1 = encoder_block_2d(self.get_number_of_input_channels(nr_of_image_channels,dim), feature_num,
                                              use_dropout, use_batch_normalization)
        else:
            self.encoder_1 = encoder_block_2d(1, feature_num, use_dropout, use_batch_normalization)

        if not self.use_one_encoder_decoder_block:
            self.encoder_2 = encoder_block_2d(feature_num, feature_num * 2, use_dropout, use_batch_normalization)

        # todo: maybe have one decoder for each Gaussian here.
        # todo: the current version seems to produce strange gridded results

        if self.use_separate_decoders_per_gaussian:
            if not self.use_one_encoder_decoder_block:
                self.decoder_1 = nn.ModuleList()
            self.decoder_2 = nn.ModuleList()
            # create two decoder blocks for each Gaussian
            for g in range(nr_of_gaussians):
                if not self.use_one_encoder_decoder_block:
                    self.decoder_1.append( decoder_block_2d(feature_num * 2, feature_num, 2, use_dropout, use_batch_normalization) )
                self.decoder_2.append( decoder_block_2d(feature_num, 1, 2, use_dropout, use_batch_normalization, last_block=True) )
        else:
            if not self.use_one_encoder_decoder_block:
                self.decoder_1 = decoder_block_2d(feature_num * 2, feature_num, 2, use_dropout, use_batch_normalization)
            self.decoder_2 = decoder_block_2d(feature_num, nr_of_gaussians, 2, use_dropout, use_batch_normalization, last_block=True)  # 3?

        self._initialize_weights()

    def get_number_of_input_channels(self, nr_of_image_channels, dim):
        """
        legacy; to support velocity fields as input channels
        currently only returns the number of image channels, but if something else would be used as
        the network input, would need to return the total number of inputs
        """
        add_channels = 0
        if self.use_momentum_as_input:
            add_channels+=dim
        if self.use_source_image_as_input:
            add_channels+=1
        if self.use_target_image_as_input:
            add_channels+=1

        return self.nr_of_image_channels + add_channels

    def forward(self, multi_smooth_v, I, additonal_inputs, global_multi_gaussian_weights,
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
        if self.do_input_standardization:
            # network input
            x = I - I.mean()
            if self.use_momentum_as_input:
                # let's not remove the mean from the momentum
                x = torch.cat([x, additional_inputs['m']], dim=1)
            if self.use_source_image_as_input:
                x = torch.cat([x, additional_inputs['I0'] - additional_inputs['I0'].mean()], dim=1)
            if self.use_target_image_as_input:
                x = torch.cat([x, additional_inputs['I1'] - additional_inputs['I1'].mean()], dim=1)
        else:
            # network input
            x = I
            if self.use_momentum_as_input:
                x = torch.cat([x, additional_inputs['m']], dim=1)
            if self.use_source_image_as_input:
                x = torch.cat([x, additional_inputs['I0']], dim=1)
            if self.use_target_image_as_input:
                x = torch.cat([x, additional_inputs['I1']], dim=1)

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
                    decoder_output_individual.append( self.decoder_2[g](self.decoder_1[g](encoder_output)) )
                decoder_output = torch.cat(decoder_output_individual, dim=1);
            else:
                decoder_output = self.decoder_2(self.decoder_1(encoder_output))

        # now we are ready for the softmax
        if self.estimate_around_global_weights:
            weights = weighted_softmax(decoder_output, dim=1, weights=global_multi_gaussian_weights)
        else:
            weights = F.softmax(decoder_output, dim=1)

        # compute tht total variation penalty
        total_variation_penalty = Variable(MyTensor(1).zero_(), requires_grad=False)
        if self.total_variation_weight_penalty > 0:
            for g in range(self.nr_of_gaussians):
                total_variation_penalty += self.compute_total_variation(weights[:, g, ...])

        diffusion_penalty = Variable(MyTensor(1).zero_(), requires_grad=False)
        if self.diffusion_weight_penalty > 0:
            for g in range(self.nr_of_gaussians):
                diffusion_penalty += self.compute_diffusion(weights[:, g, ...])

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

        current_omt_penalty = self.omt_weight_penalty * compute_omt_penalty(weights, self.gaussian_stds,
                                                                            self.volumeElement, self.omt_power)
        self.current_penalty = current_omt_penalty
        if self.total_variation_weight_penalty > 0:
            current_tv_penalty = self.total_variation_weight_penalty * total_variation_penalty
            print('TV_penalty = ' + str(current_tv_penalty.data.cpu().numpy()) + \
                  'OMT_penalty = ' + str(current_omt_penalty.data.cpu().numpy()))
            self.current_penalty += current_tv_penalty

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
    def __init__(self, nr_of_gaussians, gaussian_stds, dim, spacing, nr_of_image_channels=1, omt_power=2.0, params=None ):
        super(SimpleConsistentWeightedSmoothingModel, self).__init__(nr_of_gaussians=nr_of_gaussians,\
                                                                     gaussian_stds=gaussian_stds,\
                                                                     dim=dim,\
                                                                     spacing=spacing,\
                                                                     nr_of_image_channels=nr_of_image_channels,\
                                                                     omt_power=omt_power,
                                                                     params=params)

        self.kernel_sizes = self.params[('kernel_sizes',[7,7],'size of the convolution kernels')]
        self.use_momentum_as_input = self.params[('use_momentum_as_input',True,'If true, uses the momentum as an additional input')]
        self.use_source_image_as_input = self.params[('use_source_image_as_input',True,'If true, uses the target image as additional input')]
        self.use_target_image_as_input = self.params[('use_target_image_as_input',True,'If true, uses the target image as additional input')]
        self.estimate_around_global_weights = self.params[('estimate_around_global_weights', True, 'If true, a weighted softmax is used so the default output (for input zero) are the global weights')]
        self.use_batch_normalization = self.params[('use_batch_normalization', True, 'If true, uses batch normalization between layers')]

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
        self.batch_normalizations = None

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
        add_channels = 0
        if self.use_momentum_as_input:
            add_channels+=dim
        if self.use_target_image_as_input:
            add_channels+=1
        if self.use_source_image_as_input:
            add_channels+=1

        return self.nr_of_image_channels + add_channels

    def _init(self,nr_of_image_channels,dim):
        """
        Initalizes all the conv layers
        :param nr_of_image_channels:
        :param dim:
        :return:
        """

        assert(self.nr_of_layers>0)

        nr_of_input_channels = self.get_number_of_input_channels(nr_of_image_channels,dim)

        convs = [None]*self.nr_of_layers

        # first layer
        convs[0] = DimConv(self.dim)(nr_of_input_channels,
                                  self.nr_of_features_per_layer[0],
                                  self.kernel_sizes[0], padding=(self.kernel_sizes[0]-1)//2)

        # all the intermediate layers and the last one
        for l in range(self.nr_of_layers-1):
            convs[l+1] = DimConv(self.dim)(self.nr_of_features_per_layer[l],
                                        self.nr_of_features_per_layer[l+1],
                                        self.kernel_sizes[l+1],
                                        padding=(self.kernel_sizes[l+1]-1)//2)

        self.conv_layers = nn.ModuleList()
        for c in convs:
            self.conv_layers.append(c)

        if self.use_batch_normalization:

            batch_normalizations = [None]*(self.nr_of_layers-1) # not on the last layer
            for b in range(self.nr_of_layers-1):
                batch_normalizations[b] = DimBatchNorm(self.dim)(self.nr_of_features_per_layer[b],momentum=batch_norm_momentum_val)

            self.batch_normalizations = nn.ModuleList()
            for b in batch_normalizations:
                self.batch_normalizations.append(b)

        self._initialize_weights()

    def forward(self, multi_smooth_v, I, additional_inputs, global_multi_gaussian_weights,
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

        if self.do_input_standardization:
            # network input
            x = I-I.mean()
            if self.use_momentum_as_input:
                # let's not remove the mean from the momentum
                x = torch.cat([x, additional_inputs['m']], dim=1)
            if self.use_source_image_as_input:
                x = torch.cat([x, additional_inputs['I0']-additional_inputs['I0'].mean()], dim=1)
            if self.use_target_image_as_input:
                x = torch.cat([x, additional_inputs['I1']-additional_inputs['I1'].mean()], dim=1)
        else:
            # network input
            x = I
            if self.use_momentum_as_input:
                x = torch.cat([x,additional_inputs['m']],dim=1)
            if self.use_source_image_as_input:
                x = torch.cat([x, additional_inputs['I0']], dim=1)
            if self.use_target_image_as_input:
                x = torch.cat([x,additional_inputs['I1']],dim=1)


        # now let's apply all the convolution layers, until the last
        # (because the last one is not relu-ed

        if self.batch_normalizations:
            for l in range(len(self.conv_layers) - 1):
                if self.use_relu:
                    x = F.relu(self.batch_normalizations[l](self.conv_layers[l](x)))
                else:
                    x = F.sigmoid(self.batch_normalizations[l](self.conv_layers[l](x)))
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

        # now do the smoothing if desired
        y = torch.zeros_like(x)

        if encourage_spatial_weight_consistency:
            # now we do local averaging for the weights and force the boundaries to zero
            for g in range(self.nr_of_gaussians):
                y[:, g, ...] = self.spatially_average(x[:, g, ...])
        else:
            y = x

        # now we are ready for the weighted softmax (will be like softmax if no weights are specified)
        if self.estimate_around_global_weights:
            weights = weighted_softmax(y, dim=1, weights=global_multi_gaussian_weights)
        else:
            weights = F.softmax(y, dim=1)

        # compute tht total variation penalty
        total_variation_penalty = Variable(MyTensor(1).zero_(), requires_grad=False)
        if self.total_variation_weight_penalty > 0:
            for g in range(self.nr_of_gaussians):
                total_variation_penalty += self.compute_total_variation(weights[:,g,...])

        diffusion_penalty = Variable(MyTensor(1).zero_(), requires_grad=False)
        if self.diffusion_weight_penalty > 0:
            for g in range(self.nr_of_gaussians):
                diffusion_penalty += self.compute_diffusion(weights[:, g, ...])

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

        current_omt_penalty = self.omt_weight_penalty*compute_omt_penalty(weights,self.gaussian_stds,self.volumeElement,self.omt_power)
        current_tv_penalty = self.total_variation_weight_penalty * total_variation_penalty
        current_diffusion_penalty = self.diffusion_weight_penalty * diffusion_penalty

        print('TV_penalty = ' + str(current_tv_penalty.data.cpu().numpy()) + \
              '; OMT_penalty = ' + str(current_omt_penalty.data.cpu().numpy()) + \
              '; diffusion_penalty = ' + str(current_diffusion_penalty.data.cpu().numpy()))

        self.current_penalty = current_omt_penalty + current_tv_penalty + current_diffusion_penalty

        if retain_weights:
            # todo: change visualization to work with this new format:
            # B x weights x X x Y instead of weights x B x X x Y
            self.computed_weights[:] = weights.data

        return ret


class DeepSmootherFactory(object):
    """
    Factory to quickly create different types of deep smoothers.
    """

    def __init__(self, nr_of_gaussians, gaussian_stds, dim, spacing, nr_of_image_channels=1 ):
        self.nr_of_gaussians = nr_of_gaussians
        """number of Gaussians as input"""
        self.gaussian_stds = gaussian_stds
        """stds of the Gaussians"""
        self.dim = dim
        """dimension of input image"""
        self.nr_of_image_channels = nr_of_image_channels
        """number of channels the image has (currently only one is supported)"""
        self.spacing = spacing
        """Spacing of the image"""

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
                                                          spacing=self.spacing,
                                                          nr_of_image_channels=self.nr_of_image_channels,
                                                          params=cparams)
        elif smootherType=='encoder_decoder':
            return EncoderDecoderSmoothingModel(nr_of_gaussians=self.nr_of_gaussians,
                                                  gaussian_stds=self.gaussian_stds,
                                                  dim=self.dim,
                                                  spacing=self.spacing,
                                                  nr_of_image_channels=self.nr_of_image_channels,
                                                  params=cparams)
        else:
            raise ValueError('Deep smoother: ' + smootherType + ' not known')