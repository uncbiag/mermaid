"""
This package implements pytorch functions for Fourier-based convolutions.
While this may not be relevant for GPU-implementations, convolutions in the spatial domain are slow on CPUs. Hence, this function should be useful for memory-intensive models that need to be run on the CPU or CPU-based computations involving convolutions in general.

.. todo::
  Create a CUDA version of these convolutions functions. There is already a CUDA based FFT implementation available which could be built upon. Alternatively, spatial smoothing may be sufficiently fast on the GPU.
"""
# TODO

import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from torch.autograd import gradcheck
from dataWarper import USE_CUDA, FFTVal,AdaptVal
if USE_CUDA:
    import pytorch_fft.fft as fft

import utils


def are_indices_close(loc):
    """
    This function takes a set of indices (as produced by np.where) and determines 
    if they are roughly closeby. If not it returns *False* otherwise *True*.
    
    :param loc: Index locations as outputted by np.where
    :return: Returns if the indices are roughly closeby or not

    .. todo::
       There should be a better check for closeness of points. The implemented one is very crude.
    """

    # TODO: potentially do a better check here, this one is very crude
    for cloc in loc:
        cMaxDist = (abs(cloc - cloc.max())).max()
        if cMaxDist > 2:
            return False
    return True


def create_complex_fourier_filter(spatial_filter, sz, enforceMaxSymmetry=True):
    """
    Creates a filter in the Fourier domain given a spatial array defining the filter
    
    :param spatial_filter: Array defining the filter.
    :param sz: Desired size of the filter in the Fourier domain.
    :param enforceMaxSymmetry: If set to *True* (default) forces the filter to be real and hence forces the filter
        in the spatial domain to be symmetric
    :return: Returns the complex coefficients for the filter in the Fourier domain
    """
    # we assume this is a spatial filter, F, hence conj(F(w))=F(-w)
    sz = np.array(sz)
    if enforceMaxSymmetry:
        maxIndex = np.unravel_index(np.argmax(spatial_filter), spatial_filter.shape)
        maxValue = spatial_filter[maxIndex]
        loc = np.where(spatial_filter == maxValue)
        nrOfMaxValues = len(loc[0])
        if nrOfMaxValues > 1:
            # now need to check if they are close to each other
            if not are_indices_close(loc):
                raise ValueError('Cannot enforce max symmetry as maximum is not unique')

        spatial_filter_max_at_zero = np.roll(spatial_filter, -np.array(maxIndex),
                                             range(len(spatial_filter.shape)))

        # we assume this is symmetric and hence take the absolute value
        # as the FT of a symmetric kernel has to be real
        if USE_CUDA:
            f_filter =  create_cuda_filter(spatial_filter_max_at_zero, sz)
            abs_filter = torch.sqrt(f_filter[0]**2 + f_filter[1]**2)
            abs_filter = abs_filter
        else:
            f_filter = create_numpy_filter(spatial_filter_max_at_zero, sz)
            abs_filter = np.absolute(f_filter)

        return abs_filter
    else:
        if USE_CUDA:
            return create_cuda_filter(spatial_filter)
        else:
            return create_numpy_filter(spatial_filter, sz)


def create_cuda_filter(spatial_filter, sz):
    """
    create cuda version filter, another one dimension is added for computational convenient
    :param spatial_filter:
    :param sz:
    :return: cuda filter
    """
    fftn = sel_fftn(sz.size)
    spatial_filter_th = torch.from_numpy(spatial_filter).float().cuda()
    spatial_filter_th = spatial_filter_th[None, ...]
    spatial_filter_th_cell = fftn(spatial_filter_th)
    return spatial_filter_th_cell


def create_numpy_filter(spatial_filter, sz):
    return np.fft.fftn(spatial_filter, s=sz)

def sel_fftn(dim):
    """
    sel the gpu and cpu version of the fft
    :param dim:
    :return: function pointer
    """
    if USE_CUDA:
        if dim == 1:
            f = fft.rfft
        elif dim == 2:
            f = fft.rfft2
        elif dim == 3:
            f = fft.rfft3
        else:
            raise ValueError('Only 3D cuda fft supported')
        return f
    else:
        if dim == 1:
            f = np.fft.fft
        elif dim == 2:
            f = np.fft.fft2
        elif dim == 3:
            f = np.fft.fftn
        else:
            raise ValueError('Only 3D cpu fft supported')
        return f
def sel_ifftn(dim):
    """
    select the cpu and gpu version of the ifft
    :param dim:
    :return: function pointer
    """
    if USE_CUDA:
        if dim == 1:
            f = fft.irfft
        elif dim == 2:
            f = fft.irfft2
        elif dim == 3:
            f = fft.irfft3
        else:
            raise ValueError('Only 3D cuda fft supported')
    else:
        if dim == 1:
            f = np.fft.ifft
        elif dim == 2:
            f = np.fft.ifft2
        elif dim == 3:
            f = np.fft.ifftn
        else:
            raise ValueError('Only 3D cpu ifft supported')
    return f

class FourierConvolution(Function):
    """
    pyTorch function to compute convolutions in the Fourier domain: f = g*h
    """

    def __init__(self, complex_fourier_filter):
        """
        Constructor for the Fouier-based convolution
        
        :param complex_fourier_filter: Filter in the Fourier domain as created by *createComplexFourierFilter*
        """
        # we assume this is a spatial filter, F, hence conj(F(w))=F(-w)
        super(FourierConvolution, self).__init__()
        self.complex_fourier_filter = complex_fourier_filter
        if USE_CUDA:
            self.dim = complex_fourier_filter.dim() - 1
        else:
            self.dim = len(complex_fourier_filter.shape)
        self.fftn = sel_fftn(self.dim)
        self.ifftn = sel_ifftn(self.dim)

        """The filter in the Fourier domain"""

    def forward(self, input):
        """
        Performs the Fourier-based filtering
        the 3d cpu fft is implemented in fftn, to avoid fusing with batch and channel, here 3d is calcuated in loop
        1d 2d cpu works well because fft and fft2 is inbuilt, similarly , 1d 2d 3d gpu fft also is inbuilt

        in gpu implementation, the rfft is used for efficiency, which means the filter should be symmetric
        :param input: Image
        :return: Filtered-image
        """
        # (a+bi)(c+di) = (ac-bd) + (bc+ad)i
        # filter_imag =0, then get  ac + bci

        if USE_CUDA:
            input = FFTVal(input, ini=1)
            f_input_real, f_input_imag = self.fftn(input)
            f_filter_real = self.complex_fourier_filter
            f_filter_real.expand_as(f_input_real)
            f_conv_real = f_input_real * f_filter_real
            f_conv_imag = f_input_imag * f_filter_real
            conv_ouput_real = self.ifftn(f_conv_real, f_conv_imag)
            result = conv_ouput_real

            return FFTVal(result, ini=-1)
        else:
            if self.dim <3:
                conv_output = self.ifftn(self.fftn(input.numpy()) * self.complex_fourier_filter)
                result = conv_output.real  # should in principle be real
            elif self.dim == 3:
                result = np.zeros(input.shape)
                for batch in range(input.size()[0]):
                    for ch in range(input.size()[1]):
                        conv_output = self.ifftn(self.fftn(input[batch,ch].numpy()) * self.complex_fourier_filter)
                        result[batch,ch] = conv_output.real
            else:
                raise ValueError("cpu fft smooth should be 1d-3d")



            return torch.FloatTensor(result)
            # print( 'max(imag) = ' + str( (abs( conv_output.imag )).max() ) )
            # print( 'max(real) = ' + str( (abs( conv_output.real )).max() ) )




    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        """
        Computes the gradient
        the 3d cpu ifft is implemented in ifftn, to avoid fusing with batch and channel, here 3d is calcuated in loop
        1d 2d cpu works well because ifft and ifft2 is inbuilt, similarly , 1d 2d 3d gpu fft also is inbuilt

        in gpu implementation, the irfft is used for efficiency, which means the filter should be symmetric
        :param grad_output: Gradient output of previous layer
        :return: Gradient including the Fourier-based convolution
        """

        # Initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.

        grad_input = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.


        # (a+bi)(c+di) = (ac-bd) + (bc+ad)i
        # input_imag =0, then get  ac + bci
        if USE_CUDA:
            grad_output = FFTVal(grad_output, ini=1)
            #print grad_output.view(-1,1).sum()
            f_go_real, f_go_imag = self.fftn(grad_output)
            f_filter_real = self.complex_fourier_filter
            f_filter_real.expand_as(f_go_real)
            f_conv_real = f_go_real * f_filter_real
            f_conv_conj_imag = f_go_imag * f_filter_real
            grad_input = self.ifftn(f_conv_real, f_conv_conj_imag)
            # print(grad_input)
            # print((grad_input[0,0,12:15]))

            return FFTVal(grad_input, ini=-1)
        else:
            # if self.needs_input_grad[0]:
            numpy_go = grad_output.numpy()
            # we use the conjugate because the assumption was that the spatial filter is real
            # THe following two lines should be correct
            if self.dim < 3:
                grad_input_c = (self.ifftn(np.conjugate(self.complex_fourier_filter) * self.fftn(numpy_go)))
                grad_input = grad_input_c.real
            elif self.dim == 3:
                grad_input = np.zeros(numpy_go.shape)
                assert grad_output.dim() == 5
                for batch in range(grad_output.size()[0]):
                    for ch in range(grad_output.size()[1]):
                        grad_input_c = (self.ifftn(np.conjugate(self.complex_fourier_filter) *self.fftn(numpy_go[batch,ch])))
                        grad_input[batch,ch] = grad_input_c.real
            else:
                raise ValueError("cpu fft smooth should be 1d-3d")
            # print(grad_input)

            # print((grad_input[0,0,12:15]))
            return torch.FloatTensor(grad_input)
            # print( 'grad max(imag) = ' + str( (abs( grad_input_c.imag )).max() ) )
            # print( 'grad max(real) = ' + str( (abs( grad_input_c.real )).max() ) )




class InverseFourierConvolution(Function):
    """
    pyTorch function to compute convolutions in the Fourier domain: f = g*h
    But uses the inverse of the smoothing filter
    """

    def __init__(self, complex_fourier_filter):
        """
        Constructor for the Fouier-based convolution (WARNING: EXPERIMENTAL)

        :param complex_fourier_filter: Filter in the Fourier domain as created by *createComplexFourierFilter*
        """
        # we assume this is a spatial filter, F, hence conj(F(w))=F(-w)
        super(InverseFourierConvolution, self).__init__()
        self.complex_fourier_filter = complex_fourier_filter
        if USE_CUDA:
            self.dim = complex_fourier_filter.dim() - 1
        else:
            self.dim = len(complex_fourier_filter.shape)
        self.fftn = sel_fftn(self.dim)
        self.ifftn = sel_ifftn(self.dim)
        """Fourier filter"""
        self.alpha = 0.1
        """Regularizing weight"""

    def set_alpha(self, alpha):
        """
        Sets the regularizing weight
        
        :param alpha: regularizing weight
        """
        self.alpha = alpha

    def get_alpha(self):
        """
        Returns the regularizing weight
        
        :return: regularizing weight 
        """
        return self.alpha

    def forward(self, input):
        """
        Performs the Fourier-based filtering

        :param input: Image
        :return: Filtered-image
        """
        # do the filtering in the Fourier domain
        # (a+bi)/(c) = (a/c) + (b/c)i

        if USE_CUDA:
            input = FFTVal(input, ini=1)
            f_input_real, f_input_imag = self.fftn(input)
            f_filter_real = self.complex_fourier_filter
            f_filter_real += self.alpha
            f_filter_real.expand_as(f_input_real)
            f_conv_real = f_input_real / f_filter_real
            f_conv_imag = f_input_imag / f_filter_real
            conv_ouput_real = self.ifftn(f_conv_real, f_conv_imag)
            result = conv_ouput_real
            return FFTVal(result, ini=-1)
        else:
            result = np.zeros(input.shape)
            if self.dim <3:
                conv_output = self.ifftn(self.fftn(input.numpy()) / (self.alpha + self.complex_fourier_filter))
                # result = abs(conv_output) # should in principle be real
                result = conv_output.real
            elif self.dim == 3:
                result = np.zeros(input.shape)
                for batch in range(input.size()[0]):
                    for ch in range(input.size()[1]):
                        conv_output = self.ifftn(
                            self.fftn(input[batch,ch].numpy()) / (self.alpha + self.complex_fourier_filter))
                        result[batch, ch] = conv_output.real
            else:
                raise ValueError("cpu fft smooth should be 1d-3d")
            return torch.FloatTensor(result)


    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        """
        Computes the gradient

        :param grad_output: Gradient output of previous layer
        :return: Gradient including the Fourier-based convolution
        """

        # Initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.

        grad_input = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        # if self.needs_input_grad[0]:

        if USE_CUDA:
            grad_output =FFTVal(grad_output, ini=1)
            f_go_real, f_go_imag = self.fftn(grad_output)
            f_filter_real = self.complex_fourier_filter
            f_filter_real += self.alpha
            f_filter_real.expand_as(f_go_real)
            f_conv_real = f_go_real / f_filter_real
            f_conv_conj_imag = f_go_imag / f_filter_real
            grad_input = self.ifftn(f_conv_real, f_conv_conj_imag)
            return FFTVal(grad_input, ini=-1)
        else:
            # if self.needs_input_grad[0]:
            numpy_go = grad_output.numpy()
            # we use the conjugate because the assumption was that the spatial filter is real
            # THe following two lines should be correct
            if self.dim<3:
                grad_input_c = (self.ifftn(self.fftn(numpy_go) / (self.alpha + np.conjugate(self.complex_fourier_filter))))
                grad_input = grad_input_c.real
            elif self.dim == 3:
                grad_input = np.zeros(numpy_go.shape)
                for batch in range(grad_output.size()[0]):
                    for ch in range(grad_output.size()[1]):
                        grad_input_c = (
                            self.ifftn(self.fftn(numpy_go[batch,ch]) / (self.alpha + np.conjugate(self.complex_fourier_filter))))
                        grad_input[batch, ch] = grad_input_c.real
            else:
                raise ValueError("cpu fft smooth should be 1d-3d")
            return torch.FloatTensor(grad_input)



def fourier_convolution(input, complex_fourier_filter):
    """
    Convenience function for Fourier-based convolutions. Make sure to use this one (instead of directly
    using the class FourierConvolution). This will assure that each call generates its own instance
    and hence autograd will work properly
    
    :param input: Input image
    :param complex_fourier_filter: Filter in Fourier domain as generated by *createComplexFourierFilter* 
    :return: 
    """
    # First braces create a Function object. Any arguments given here
    # will be passed to __init__. Second braces will invoke the __call__
    # operator, that will then use forward() to compute the result and
    # return it.
    return FourierConvolution(complex_fourier_filter)(input)


def inverse_fourier_convolution(input, complex_fourier_filter):
    # just filtering with inverse filter
    return InverseFourierConvolution(complex_fourier_filter)(input)


def check_fourier_conv():
    """
    Convenience function to check the gradient. Fails, as pytorch's check appears to have difficulty
    
    :return: True if analytical and numerical gradient are the same

    .. todo::
       The current check seems to fail in pyTorch. However, the gradient appears to be correct. Potentially an issue with the numerical gradient approximiaton.
    """
    # gradcheck takes a tuple of tensor as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    # TODO: Seems to fail at the moment, check why if there are issues with the gradient
    sz = np.array([20, 20],dtype='int64')
    # f = np.ones(sz)
    f = 1 / 400. * np.ones(sz)
    dim = len(sz)

    mus = np.zeros(dim)
    stds = np.ones(dim)
    id = utils.identity_map(sz)
    g = 100 * utils.compute_normalized_gaussian(id, mus, stds)
    FFilter = create_complex_fourier_filter(g, sz)
    input = AdaptVal(Variable(torch.randn([1, 1] + list(sz)), requires_grad=True))
    test = gradcheck(FourierConvolution(FFilter), input, eps=1e-6, atol=1e-4)
    print(test)

def check_run_forward_and_backward():
    """
    Convenience function to check running the function forward and backward
    s
    :return: 
    """
    sz = [20, 20]
    f = 1 / 400. * np.ones(sz)
    FFilter = create_complex_fourier_filter(f, sz, False)
    input = Variable(torch.randn(sz).float(), requires_grad=True)
    fc = FourierConvolution(FFilter)(input)
    # print( fc )
    fc.backward(torch.randn(sz).float())
    print(input.grad)

