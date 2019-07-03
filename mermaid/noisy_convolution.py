import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn

from torch.nn.modules.utils import _single, _pair, _triple

from .data_wrapper import MyTensor, USE_CUDA

device = torch.device("cuda:0" if (USE_CUDA and torch.cuda.is_available()) else "cpu")


class NoisyLinear(nn.Module):
    """Applies a noisy linear transformation to the incoming data: :math:`y = (mu_w + sigma_w \cdot epsilon_w)x + mu_b + sigma_b \cdot epsilon_b`
    More details can be found in the paper `ZZ` _ .
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True
        factorised: whether or not to use factorised noise. Default: True
        std_init: initialization constant for standard deviation component of weights. If None, \
            defaults to 0.017 for independent and 0.4 for factorised. Default: None

    Shape:
        - Input: (N, in_features)
        - Output:(N, out_features)

    Attributes:
        weight: the learnable weights of the module of shape (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)

    Examples::
        >>> m = nn.NoisyLinear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True, factorised=True, std_init=None):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.factorised = factorised
        self.weight_mu = Parameter(MyTensor(out_features, in_features))
        self.weight_sigma = Parameter(MyTensor(out_features, in_features))
        if bias:
            self.bias_mu = Parameter(MyTensor(out_features))
            self.bias_sigma = Parameter(MyTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if std_init is None:
            if self.factorised:
                self.std_init = 0.4
            else:
                self.std_init = 0.017
        else:
            self.std_init = std_init
        self.reset_parameters(bias)

    def reset_parameters(self, bias):
        if self.factorised:
            mu_range = 1. / math.sqrt(self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
        else:
            mu_range = math.sqrt(3. / self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init)
            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init)

    def scale_noise(self, size):
        x = MyTensor(size).normal_()
        x = x.sign().mul(x.abs().sqrt())
        return x

    def forward(self, input):
        if self.factorised:
            epsilon_in = self.scale_noise(self.in_features)
            epsilon_out = self.scale_noise(self.out_features)
            weight_epsilon = epsilon_out.ger(epsilon_in)
            bias_epsilon = self.scale_noise(self.out_features)
        else:
            weight_epsilon = MyTensor(*(self.out_features, self.in_features)).normal_()
            bias_epsilon = MyTensor(self.out_features).normal_()
        return F.linear(input,
                        self.weight_mu + self.weight_sigma.mul(weight_epsilon),
                        self.bias_mu + self.bias_sigma.mul(bias_epsilon))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class NoisyLayer(nn.Module):
    def __init__(self, std_init=None, start_reducing_from_iter=25):
        super(NoisyLayer, self).__init__()

        self.std_init = std_init

        if self.std_init is None:
            self.std_init = 0.25
        else:
            self.std_init = std_init

        self.start_reducing_from_iter = start_reducing_from_iter

    def forward(self, input, iter=0):

        noise_epsilon = MyTensor(input.size()).normal_()

        if self.training:
            effective_iter = max(0,iter-self.start_reducing_from_iter)
            output = input + 1. / (effective_iter + 1) * self.std_init * noise_epsilon
        else:
            output = input

        return output


class _NoisyConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, scalar_sigmas=True, optimize_sigmas=False, std_init=None, start_reducing_from_iter=25):
        super(_NoisyConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.scalar_sigmas = scalar_sigmas
        self.optimize_sigmas = optimize_sigmas
        self.std_init = std_init

        self.start_reducing_from_iter = start_reducing_from_iter

        if self.std_init is None:
            self.std_init = 0.25
        else:
            self.std_init = std_init

        if transposed:
            self.weight = Parameter(MyTensor(
                in_channels, out_channels // groups, *kernel_size))
            if self.scalar_sigmas:
                if self.optimize_sigmas:
                    self.weight_sigma = Parameter(MyTensor(1))
                else:
                    self.weight_sigma = MyTensor(1)
            else:
                if self.optimize_sigmas:
                    self.weight_sigma = Parameter(MyTensor(in_channels, out_channels//groups))
                else:
                    self.weight_sigma = MyTensor(in_channels, out_channels//groups)
        else:
            self.weight = Parameter(MyTensor(
                out_channels, in_channels // groups, *kernel_size))
            if self.scalar_sigmas:
                if self.optimize_sigmas:
                    self.weight_sigma = Parameter(MyTensor(1))
                else:
                    self.weight_sigma = MyTensor(1)
            else:
                if self.optimize_sigmas:
                    self.weight_sigma = Parameter(MyTensor(out_channels, in_channels//groups))
                else:
                    self.weight_sigma = MyTensor(out_channels, in_channels // groups)
        if bias:
            self.bias = Parameter(MyTensor(out_channels))
            if self.scalar_sigmas:
                if self.optimize_sigmas:
                    self.bias_sigma = Parameter(MyTensor(1))
                else:
                    self.bias_sigma = MyTensor(1)
            else:
                if self.optimize_sigmas:
                    self.bias_sigma = Parameter(MyTensor(out_channels))
                else:
                    self.bias_sigma = MyTensor(out_channels)
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_sigma', None)

        self.reset_parameters(bias)

    def reset_parameters(self, bias):

        # todo: adapt this to the used type of nonlinearity
        nn.init.kaiming_normal_(self.weight.data)
        self.weight_sigma.data.fill_(self.std_init)
        if bias:
            self.bias.data.fill_(0)
            self.bias_sigma.data.fill_(self.std_init)

        #mu_range = math.sqrt(3. / self.weight.size(1))
        #self.weight.data.uniform_(-mu_range, mu_range)
        #self.weight_sigma.data.fill_(self.std_init)
        #if bias:
        #    self.bias.data.uniform_(-mu_range, mu_range)
        #    self.bias_sigma.data.fill_(self.std_init)

        #n = self.in_channels
        #for k in self.kernel_size:
        #    n *= k
        #stdv = 1. / math.sqrt(n)
        #self.weight.data.uniform_(-stdv, stdv)
        #if self.bias is not None:
        #    self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class NoisyConv1d(_NoisyConvNd):
    r"""Applies a 1D noisy convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, L)` and output :math:`(N, C_{out}, L_{out})` can be
    precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_{out_j}) = \text{bias}(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{out_j}, k) \star \text{input}(N_i, k)
        \end{equation*},

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both sides
      for :attr:`padding` number of points.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor \frac{\text{out_channels}}{\text{in_channels}} \right\rfloor`).

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid
         `cross-correlation`_, and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, L_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\text{in_channels}=C_{in}, \text{out_channels}=C_{in} * K, ..., \text{groups}=C_{in})`

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            (out_channels, in_channels, kernel_size)
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels)

    Examples::

        >>> m = nn.NoisyConv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 scalar_sigmas=True, optimize_sigmas=False, std_init=None, start_reducing_from_iter=25):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(NoisyConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, scalar_sigmas, optimize_sigmas, std_init, start_reducing_from_iter)

    def forward(self, input, iter=0):

        weight_epsilon = MyTensor(*(self.out_channels, self.in_channels, *self.kernel_size)).normal_()
        bias_epsilon = MyTensor(self.out_channels).normal_()

        if self.bias is not None:
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_bias_value = self.bias + 1./(effective_iter+1)*self.bias_sigma * bias_epsilon
            else:
                new_bias_value = self.bias
        else:
            new_bias_value = None

        if self.scalar_sigmas:

            if self.optimize_sigmas:
                if self.bias is not None:
                    print('Noisy convolution: sigma_conv={:2.4f}, sigma_bias={:2.4f}'.format(self.weight_sigma.item(),
                                                                                         self.bias_sigma.item()))
                else:
                    print('Noisy convolution: sigma_conv={:2.4f}'.format(self.weight_sigma.item()))

            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1./(effective_iter+1)*self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight

            return F.conv1d(input,
                            new_weight_value,
                            new_bias_value,
                            self.stride,
                            self.padding, self.dilation, self.groups)
        else:

            if self.training:
                delta_weight = weight_epsilon
                sz = self.weight_sigma.size()
                for i in range(sz[0]):
                    delta_weight[i, ...] *= self.weight_sigma[i]

                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1./(effective_iter+1)*delta_weight
            else:
                new_weight_value = self.weight


            return F.conv1d(input,
                            new_weight_value,
                            new_bias_value,
                            self.stride,
                            self.padding, self.dilation, self.groups)


class NoisyConv2d(_NoisyConvNd):
    r"""Applies a 2D noisy convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_{out_j}) = \text{bias}(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{out_j}, k) \star \text{input}(N_i, k)
        \end{equation*},

    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\text{in_channels}=C_{in}, \text{out_channels}=C_{in} * K, ..., \text{groups}=C_{in})`

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.NoisyConv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.NoisyConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.NoisyConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 scalar_sigmas=True, optimize_sigmas=False, std_init=None, start_reducing_from_iter=25):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(NoisyConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, scalar_sigmas, optimize_sigmas, std_init, start_reducing_from_iter)

    def forward(self, input, iter=0):

        weight_epsilon = MyTensor(*(self.out_channels, self.in_channels, *self.kernel_size)).normal_()
        bias_epsilon = MyTensor(self.out_channels).normal_()

        if self.bias is not None:
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_bias_value = self.bias + 1./(effective_iter+1)*self.bias_sigma * bias_epsilon
            else:
                new_bias_value = self.bias
        else:
            new_bias_value = None

        if self.scalar_sigmas:

            if self.optimize_sigmas:
                if self.bias is not None:
                    print('Noisy convolution: sigma_conv={:2.4f}, sigma_bias={:2.4f}'.format(self.weight_sigma.item(),self.bias_sigma.item()))
                else:
                    print('Noisy convolution: sigma_conv={:2.4f}'.format(self.weight_sigma.item()))

            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1./(effective_iter+1)*self.weight_sigma*weight_epsilon
            else:
                new_weight_value = self.weight

            return F.conv2d(input,
                            new_weight_value,
                            new_bias_value,
                            self.stride,
                            self.padding, self.dilation, self.groups)
        else:

            if self.training:

                delta_weight = weight_epsilon
                sz = self.weight_sigma.size()
                for i in range(sz[0]):
                    for j in range(sz[1]):
                        delta_weight[i,j,...] *= self.weight_sigma[i,j]

                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1./(effective_iter+1)*delta_weight
            else:
                new_weight_value = self.weight

            return F.conv2d(input,
                            new_weight_value,
                            new_bias_value,
                            self.stride,
                            self.padding, self.dilation, self.groups)



class NoisyConv3d(_NoisyConvNd):
    r"""Applies a 3D noisy convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`
    and output :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_{out_j}) = \text{bias}(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{out_j}, k) \star \text{input}(N_i, k)
        \end{equation*},

    where :math:`\star` is the valid 3D `cross-correlation`_ operator

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\text{in_channels}=C_{in}, \text{out_channels}=C_{in} * K, ..., \text{groups}=C_{in})`

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
                    \times (\text{kernel_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.NoisyConv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.NoisyConv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 scalar_sigmas=True, optimize_sigmas=False, std_init=None, start_reducing_from_iter=25):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(NoisyConv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, scalar_sigmas, optimize_sigmas, std_init, start_reducing_from_iter)

    def forward(self, input):

        weight_epsilon = MyTensor(*(self.out_channels, self.in_channels, *self.kernel_size)).normal_()
        bias_epsilon = MyTensor(self.out_channels).normal_()

        if self.bias is not None:
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_bias_value = self.bias + 1./(effective_iter+1)*self.bias_sigma * bias_epsilon
            else:
                new_bias_value = self.bias
        else:
            new_bias_value = None

        if self.scalar_sigmas:

            if self.optimize_sigmas:
                if self.bias is not None:
                    print('Noisy convolution: sigma_conv={:2.4f}, sigma_bias={:2.4f}'.format(self.weight_sigma.item(),
                                                                                             self.bias_sigma.item()))
                else:
                    print('Noisy convolution: sigma_conv={:2.4f}'.format(self.weight_sigma.item()))

            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1./(effective_iter+1)*self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight

            return F.conv3d(input,
                            new_weight_value,
                            new_bias_value,
                            self.stride,
                            self.padding, self.dilation, self.groups)
        else:

            if self.training:

                delta_weight = weight_epsilon
                sz = self.weight_sigma.size()
                for i in range(sz[0]):
                    for j in range(sz[1]):
                        for k in range(sz[2]):
                            delta_weight[i, j, k, ...] *= self.weight_sigma[i, j, k]

                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1./(effective_iter+1)*delta_weight
            else:
                new_weight_value = self.weight

            return F.conv3d(input,
                            new_weight_value,
                            new_bias_value,
                            self.stride,
                            self.padding, self.dilation, self.groups)


class _NoisyConvTransposeMixin(object):

    def forward(self, input, output_size=None, iter=0):
        output_padding = self._output_padding(input, output_size)
        func = self._backend.ConvNd(
            self.stride, self.padding, self.dilation, self.transposed,
            output_padding, self.groups)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)

    def _output_padding(self, input, output_size):
        if output_size is None:
            return self.output_padding

        output_size = list(output_size)
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[-2:]
        if len(output_size) != k:
            raise ValueError(
                "output_size must have {} or {} elements (got {})"
                .format(k, k + 2, len(output_size)))

        def dim_size(d):
            return ((input.size(d + 2) - 1) * self.stride[d] -
                    2 * self.padding[d] + self.kernel_size[d])

        min_sizes = [dim_size(d) for d in range(k)]
        max_sizes = [min_sizes[d] + self.stride[d] - 1 for d in range(k)]
        for size, min_size, max_size in zip(output_size, min_sizes, max_sizes):
            if size < min_size or size > max_size:
                raise ValueError((
                    "requested an output size of {}, but valid sizes range "
                    "from {} to {} (for an input of {})").format(
                        output_size, min_sizes, max_sizes, input.size()[2:]))

        return tuple([output_size[d] - min_sizes[d] for d in range(k)])


class NoisyConvTranspose1d(_NoisyConvTransposeMixin, _NoisyConvNd):
    r"""Applies a 1D noisy transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv1d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for ``kernel_size - 1 - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::
        The :attr:`padding` argument effectively adds ``kernel_size - 1 - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv1d` and a :class:`~torch.nn.ConvTranspose1d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv1d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``kernel_size - 1 - padding`` zero-padding
            will be added to both sides of the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = (L_{in} - 1) \times \text{stride} - 2 \times \text{padding}
                    + \text{kernel_size} + \text{output_padding}

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, out_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 scalar_sigmas=True, optimize_sigmas=False, std_init=None, start_reducing_from_iter=25):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        super(NoisyConvTranspose1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias,
            scalar_sigmas=scalar_sigmas, optimize_sigmas=optimize_sigmas, std_init=std_init,
            start_reducing_from_iter=start_reducing_from_iter)

    def forward(self, input, output_size=None, iter=0):
        output_padding = self._output_padding(input, output_size)

        weight_epsilon = MyTensor(*(self.out_channels, self.in_channels, *self.kernel_size)).normal_()
        bias_epsilon = MyTensor(self.out_channels).normal_()

        if self.bias is not None:
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_bias_value = self.bias + 1./(effective_iter+1)*self.bias_sigma * bias_epsilon
            else:
                new_bias_value = self.bias
        else:
            new_bias_value = None

        if self.scalar_sigmas:

            if self.optimize_sigmas:
                if self.bias is not None:
                    print('Noisy convolution: sigma_conv={:2.4f}, sigma_bias={:2.4f}'.format(self.weight_sigma.item(),
                                                                                             self.bias_sigma.item()))
                else:
                    print('Noisy convolution: sigma_conv={:2.4f}'.format(self.weight_sigma.item()))

            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1./(1+effective_iter)*self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight

            return F.conv_transpose1d(input,
                                      new_weight_value,
                                      new_bias_value,
                                      self.stride,
                                      self.padding, output_padding, self.groups, self.dilation)
        else:

            if self.training:

                delta_weight = weight_epsilon
                sz = self.weight_sigma.size()
                for i in range(sz[0]):
                    delta_weight[i, ...] *= self.weight_sigma[i]

                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1./(1+effective_iter)*self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight

            return F.conv_transpose1d(input,
                                      new_weight_value,
                                      new_bias_value,
                                      self.stride,
                                      self.padding, output_padding, self.groups, self.dilation)


class NoisyConvTranspose2d(_NoisyConvTransposeMixin, _NoisyConvNd):
    r"""Applies a 2D noisy transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv2d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for ``kernel_size - 1 - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimensions
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::
        The :attr:`padding` argument effectively adds ``kernel_size - 1 - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv2d` and a :class:`~torch.nn.ConvTranspose2d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv2d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``kernel_size - 1 - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0]
                    + \text{kernel_size}[0] + \text{output_padding}[0]

              W_{out} = (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1]
                    + \text{kernel_size}[1] + \text{output_padding}[1]

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, out_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.NoisyConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.NoisyConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12)
        >>> downsample = nn.NoisyConv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nn.NoisyConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 scalar_sigmas=True, optimize_sigmas=False, std_init=None, start_reducing_from_iter=25):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(NoisyConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias,
            scalar_sigmas=scalar_sigmas, optimize_sigmas=optimize_sigmas, std_init=std_init,
            start_reducing_from_iter=start_reducing_from_iter)

    def forward(self, input, output_size=None, iter=0):
        output_padding = self._output_padding(input, output_size)

        weight_epsilon = MyTensor(*(self.out_channels, self.in_channels, *self.kernel_size)).normal_()
        bias_epsilon = MyTensor(self.out_channels).normal_()

        if self.bias is not None:
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_bias_value = self.bias + 1./(1+effective_iter)*self.bias_sigma * bias_epsilon
            else:
                new_bias_value = self.bias
        else:
            new_bias_value = None

        if self.scalar_sigmas:

            if self.optimize_sigmas:
                if self.bias is not None:
                    print('Noisy convolution: sigma_conv={:2.4f}, sigma_bias={:2.4f}'.format(self.weight_sigma.item(),
                                                                                             self.bias_sigma.item()))
                else:
                    print('Noisy convolution: sigma_conv={:2.4f}'.format(self.weight_sigma.item()))

            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1./(1+effective_iter)*self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight

            return F.conv_transpose2d(input,
                                      new_weight_value,
                                      new_bias_value,
                                      self.stride,
                                      self.padding, output_padding, self.groups, self.dilation)
        else:

            if self.training:

                delta_weight = weight_epsilon
                sz = self.weight_sigma.size()
                for i in range(sz[0]):
                    for j in range(sz[1]):
                        delta_weight[i, j, ...] *= self.weight_sigma[i, j]

                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1./(1+effective_iter)*self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight

            return F.conv_transpose2d(input,
                                      new_weight_value,
                                      new_bias_value,
                                      self.stride,
                                      self.padding, output_padding, self.groups, self.dilation)



class NoisyConvTranspose3d(_NoisyConvTransposeMixin, _NoisyConvNd):
    r"""Applies a 3D noisy transposed convolution operator over an input image composed of several input
    planes.
    The transposed convolution operator multiplies each input value element-wise by a learnable kernel,
    and sums over the outputs from all input feature planes.

    This module can be seen as the gradient of Conv3d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for ``kernel_size - 1 - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimensions
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::
        The :attr:`padding` argument effectively adds ``kernel_size - 1 - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv3d` and a :class:`~torch.nn.ConvTranspose3d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv3d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``kernel_size - 1 - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where

          .. math::
              D_{out} = (D_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0]
                    + \text{kernel_size}[0] + \text{output_padding}[0]

              H_{out} = (H_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1]
                    + \text{kernel_size}[1] + \text{output_padding}[1]

              W_{out} = (W_{in} - 1) \times \text{stride}[2] - 2 \times \text{padding}[2]
                    + \text{kernel_size}[2] + \text{output_padding}[2]

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.NoisyConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.NoisyConv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 scalar_sigmas=True, optimize_sigmas=False, std_init=None, start_reducing_from_iter=25):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        super(NoisyConvTranspose3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias,
            scalar_sigmas=scalar_sigmas, optimize_sigmas=optimize_sigmas, std_init=std_init,
            start_reducing_from_iter=start_reducing_from_iter)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)

        weight_epsilon = MyTensor(*(self.out_channels, self.in_channels, *self.kernel_size)).normal_()
        bias_epsilon = MyTensor(self.out_channels).normal_()

        if self.bias is not None:
            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_bias_value = self.bias + 1./(effective_iter+1)*self.bias_sigma * bias_epsilon
            else:
                new_bias_value = self.bias
        else:
            new_bias_value = None

        if self.scalar_sigmas:

            if self.optimize_sigmas:
                if self.bias is not None:
                    print('Noisy convolution: sigma_conv={:2.4f}, sigma_bias={:2.4f}'.format(self.weight_sigma.item(),
                                                                                             self.bias_sigma.item()))
                else:
                    print('Noisy convolution: sigma_conv={:2.4f}'.format(self.weight_sigma.item()))

            if self.training:
                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1./(effective_iter+1)*self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight

            return F.conv_transpose3d(input,
                            new_weight_value,
                            new_bias_value,
                            self.stride,
                            self.padding, output_padding, self.groups, self.dilation)
        else:

            if self.training:

                delta_weight = weight_epsilon
                sz = self.weight_sigma.size()
                for i in range(sz[0]):
                    for j in range(sz[1]):
                        for k in range(sz[2]):
                            delta_weight[i, j, k, ...] *= self.weight_sigma[i, j, k]

                effective_iter = max(0, iter - self.start_reducing_from_iter)
                new_weight_value = self.weight + 1./(effective_iter+1)*self.weight_sigma * weight_epsilon
            else:
                new_weight_value = self.weight

            return F.conv_transpose3d(input,
                                      new_weight_value,
                                      new_bias_value,
                                      self.stride,
                                      self.padding, output_padding, self.groups, self.dilation)
