"""
This package implements pytorch functions for Fourier-based convolutions.
While this may not be relevant for GPU-implementations, convolutions in the spatial 
domain are slow on CPUs. Hence, this function should be useful for memory-intensive
models that need to be run on the CPU or CPU-based computations involving convolutions in general.
"""

import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from torch.autograd import gradcheck

import utils

def are_indices_close(loc):
    """
    This function takes a set of indices (as produced by np.where) and determines 
    if they are roughly closeby. If not it returns *False* otherwise *True*.
    
    :param loc: Index locations as outputted by np.where
    :return: Returns if the indices are roughly closeby or not
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
        F = np.fft.fftn(spatial_filter_max_at_zero, s=sz)
        return np.absolute(F)
    else:
        return np.fft.fftn(spatial_filter, s=sz)

class FourierConvolution(Function):
    """
    pyTorch function to compute convolutions in the Fourier domain: f = g*h
    """

    def __init__(self,complex_fourier_filter):
        """
        Constructor for the Fouier-based convolution
        
        :param complex_fourier_filter: Filter in the Fourier domain as created by *createComplexFourierFilter*
        """
        # we assume this is a spatial filter, F, hence conj(F(w))=F(-w)
        super(FourierConvolution,self).__init__()
        self.complex_fourier_filter = complex_fourier_filter
        """The filter in the Fourier domain"""

    def forward(self, input):
        """
        Performs the Fourier-based filtering
        
        :param input: Image
        :return: Filtered-image
        """
        # do the filtering in the Fourier domain
        conv_output = np.fft.ifftn(np.fft.fftn(input.numpy())*self.complex_fourier_filter)
        #result = abs(conv_output) # should in principle be real
        result = conv_output.real
        #print( 'max(imag) = ' + str( (abs( conv_output.imag )).max() ) )
        #print( 'max(real) = ' + str( (abs( conv_output.real )).max() ) )

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
        #if self.needs_input_grad[0]:
        numpy_go = grad_output.numpy()
        # we use the conjugate because the assumption was that the spatial filter is real

        # THe following two lines should be correct
        grad_input_c = (np.fft.ifftn(np.conjugate(self.complex_fourier_filter) * np.fft.fftn(numpy_go)))
        grad_input = grad_input_c.real

        #print( 'grad max(imag) = ' + str( (abs( grad_input_c.imag )).max() ) )
        #print( 'grad max(real) = ' + str( (abs( grad_input_c.real )).max() ) )

        return torch.FloatTensor(grad_input)


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
        """Fourier filter"""
        self.alpha = 0.1
        """Regularizing weight"""

    def set_alpha(self,alpha):
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
        conv_output = np.fft.ifftn(np.fft.fftn(input.numpy()) / (self.alpha + self.complex_fourier_filter))
        # result = abs(conv_output) # should in principle be real
        result = conv_output.real

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
        numpy_go = grad_output.numpy()
        # we use the conjugate because the assumption was that the spatial filter is real

        # THe following two lines should be correct
        grad_input_c = (np.fft.ifftn( np.fft.fftn(numpy_go) / (self.alpha + np.conjugate(self.complex_fourier_filter))))
        grad_input = grad_input_c.real

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
    """
    # gradcheck takes a tuple of tensor as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    # TODO: Seems to fail at the moment, check why if there are issues with the gradient
    sz = [20,20]
    #f = np.ones(sz)
    f = 1/400.*np.ones(sz)
    dim = len(sz)

    mus = np.zeros(dim)
    stds = np.ones(dim)
    id = utils.identity_map(sz)
    g = 10000*utils.compute_normalized_gaussian(id, mus, stds)
    FFilter = create_complex_fourier_filter(g, sz)
    input = (Variable(torch.randn(sz).float(), requires_grad=True),)
    test = gradcheck(FourierConvolution(FFilter), input, eps=1e-6, atol=1e-4)
    print(test)

def check_run_forward_and_backward():
    """
    Convenience function to check running the function forward and backward
    
    :return: 
    """
    sz = [20,20]
    f = 1/400.*np.ones(sz)
    FFilter = create_complex_fourier_filter(f, sz, False)
    input = Variable( torch.randn(sz).float(), requires_grad=True )
    fc = FourierConvolution(FFilter)(input)
    #print( fc )
    fc.backward(torch.randn(sz).float())
    print(input.grad)
