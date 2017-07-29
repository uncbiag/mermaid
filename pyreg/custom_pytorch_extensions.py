import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from torch.autograd import gradcheck

import utils

# Inherit from Function

def are_indices_close(loc):
    # TODO: potentially do a better check here, this one is very crude
    for cloc in loc:
        cMaxDist = (abs(cloc - cloc.max())).max()
        if cMaxDist > 2:
            return False
    return True

def createComplexFourierFilter(spatial_filter,sz,enforceMaxSymmetry=True):
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

    def __init__(self,complex_fourier_filter):
        # we assume this is a spatial filter, F, hence conj(F(w))=F(-w)
        super(FourierConvolution,self).__init__()
        self.complex_fourier_filter = complex_fourier_filter

    def forward(self, input):
        # do the filtering in the Fourier domain
        conv_output = np.fft.ifftn(np.fft.fftn(input.numpy())*self.complex_fourier_filter)
        #result = abs(conv_output) # should in principle be real
        result = conv_output.real
        #print( 'max(imag) = ' + str( (abs( conv_output.imag )).max() ) )
        #print( 'max(real) = ' + str( (abs( conv_output.real )).max() ) )

        return torch.FloatTensor(result)

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
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

def fourierConvolution(input, complex_fourier_filter):
    # First braces create a Function object. Any arguments given here
    # will be passed to __init__. Second braces will invoke the __call__
    # operator, that will then use forward() to compute the result and
    # return it.
    return FourierConvolution(complex_fourier_filter)(input)

def checkFourierConv():
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
    id = utils.identityMap(sz)
    g = 10000*utils.computeNormalizedGaussian(id, mus, stds)
    FFilter = createComplexFourierFilter(g, sz)
    input = (Variable(torch.randn(sz).float(), requires_grad=True),)
    test = gradcheck(FourierConvolution(FFilter), input, eps=1e-6, atol=1e-4)
    print(test)

def checkRunForwardAndBackward():
    sz = [20,20]
    f = 1/400.*np.ones(sz)
    FFilter = createComplexFourierFilter(f, sz, False)
    input = Variable( torch.randn(sz).float(), requires_grad=True )
    fc = FourierConvolution(FFilter)(input)
    #print( fc )
    fc.backward(torch.randn(sz).float())
    print(input.grad)

#checkRunForwardAndBackward()
#checkFourierConv()