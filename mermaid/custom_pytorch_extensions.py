"""
This package implements pytorch functions for Fourier-based convolutions.
While this may not be relevant for GPU-implementations, convolutions in the spatial domain are slow on CPUs. Hence, this function should be useful for memory-intensive models that need to be run on the CPU or CPU-based computations involving convolutions in general.

.. todo::
  Create a CUDA version of these convolutions functions. There is already a CUDA based FFT implementation available which could be built upon. Alternatively, spatial smoothing may be sufficiently fast on the GPU.
"""
from __future__ import print_function
from __future__ import absolute_import
# TODO

from builtins import range
from builtins import object
import torch
from torch.autograd import Function
import numpy as np
from torch.autograd import gradcheck
from .data_wrapper import USE_CUDA, FFTVal,AdaptVal, MyTensor
# if USE_CUDA:
#     import pytorch_fft.fft as fft

from . import utils

def _symmetrize_filter_center_at_zero_1D(filter):
    sz = filter.shape
    if sz[0] % 2 == 0:
        # symmetrize if it is even
        filter[1:sz[0] // 2] = filter[sz[0]:sz[0] // 2:-1]
    else:
        # symmetrize if it is odd
        filter[1:sz[0] // 2 + 1] = filter[sz[0]:sz[0] // 2:-1]

def _symmetrize_filter_center_at_zero_2D(filter):
    sz = filter.shape
    if sz[0] % 2 == 0:
        # symmetrize if it is even
        filter[1:sz[0] // 2,:] = filter[sz[0]:sz[0] // 2:-1,:]
    else:
        # symmetrize if it is odd
        filter[1:sz[0] // 2 + 1,:] = filter[sz[0]:sz[0] // 2:-1,:]

    if sz[1] % 2 == 0:
        # symmetrize if it is even
        filter[:,1:sz[1] // 2] = filter[:,sz[1]:sz[1] // 2:-1]
    else:
        # symmetrize if it is odd
        filter[:,1:sz[1] // 2 + 1] = filter[:,sz[1]:sz[1] // 2:-1]

def _symmetrize_filter_center_at_zero_3D(filter):
    sz = filter.shape
    if sz[0] % 2 == 0:
        # symmetrize if it is even
        filter[1:sz[0] // 2,:,:] = filter[sz[0]:sz[0] // 2:-1,:,:]
    else:
        # symmetrize if it is odd
        filter[1:sz[0] // 2 + 1,:,:] = filter[sz[0]:sz[0] // 2:-1,:,:]

    if sz[1] % 2 == 0:
        # symmetrize if it is even
        filter[:,1:sz[1] // 2,:] = filter[:,sz[1]:sz[1] // 2:-1,:]
    else:
        # symmetrize if it is odd
        filter[:,1:sz[1] // 2 + 1,:] = filter[:,sz[1]:sz[1] // 2:-1,:]

    if sz[2] % 2 == 0:
        # symmetrize if it is even
        filter[:,:,1:sz[2] // 2] = filter[:,:,sz[2]:sz[2] // 2:-1]
    else:
        # symmetrize if it is odd
        filter[:,:,1:sz[2] // 2 + 1] = filter[:,:,sz[2]:sz[2] // 2:-1]

def symmetrize_filter_center_at_zero(filter,renormalize=False):
    """
    Symmetrizes filter. The assumption is that the filter is already in the format for input to an FFT.
    I.e., that it has been transformed so that the center of the pixel is at zero.

    :param filter: Input filter (in spatial domain). Will be symmetrized (i.e., will change its value)
    :param renormalize: (bool) if true will normalize so that the sum is one
    :return: n/a (returns via call by reference)
    """
    sz = filter.shape
    dim = len(sz)
    if dim==1:
        _symmetrize_filter_center_at_zero_1D(filter)
    elif dim==2:
        _symmetrize_filter_center_at_zero_2D(filter)
    elif dim==3:
        _symmetrize_filter_center_at_zero_3D(filter)
    else:
        raise ValueError('Only implemented for dimensions 1,2, and 3 so far')

    if renormalize:
        filter = filter / filter.sum()

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


def create_complex_fourier_filter(spatial_filter, sz, enforceMaxSymmetry=True, maxIndex=None, renormalize=False):
    """
    Creates a filter in the Fourier domain given a spatial array defining the filter
    
    :param spatial_filter: Array defining the filter.
    :param sz: Desired size of the filter in the Fourier domain.
    :param enforceMaxSymmetry: If set to *True* (default) forces the filter to be real and hence forces the filter
        in the spatial domain to be symmetric
    :param maxIndex: specifies the index of the maximum which will be used to enforceMaxSymmetry. If it is not
        defined, the maximum is simply computed
    :param renormalize: (bool) if true, the filter is renormalized to sum to one (useful for Gaussians for example)
    :return: Returns the complex coefficients for the filter in the Fourier domain and the maxIndex 
    """
    # we assume this is a spatial filter, F, hence conj(F(w))=F(-w)
    sz = np.array(sz)
    if enforceMaxSymmetry:
        if maxIndex is None:
            maxIndex = np.unravel_index(np.argmax(spatial_filter), spatial_filter.shape)
            maxValue = spatial_filter[maxIndex]
            loc = np.where(spatial_filter == maxValue)
            nrOfMaxValues = len(loc[0])
            if nrOfMaxValues > 1:
                # now need to check if they are close to each other
                if not are_indices_close(loc):
                    raise ValueError('Cannot enforce max symmetry as maximum is not unique')

        spatial_filter_max_at_zero = np.roll(spatial_filter, -np.array(maxIndex),
                                             list(range(len(spatial_filter.shape))))

        symmetrize_filter_center_at_zero(spatial_filter_max_at_zero,renormalize=renormalize)

        # we assume this is symmetric and hence take the absolute value
        # as the FT of a symmetric kernel has to be real
        if USE_CUDA:
            f_filter =  create_cuda_filter(spatial_filter_max_at_zero, sz)
            ret_filter = f_filter[...,0] # only the real part
        else:
            f_filter = create_numpy_filter(spatial_filter_max_at_zero, sz)
            ret_filter = f_filter.real

        return ret_filter,maxIndex
    else:
        if USE_CUDA:
            return create_cuda_filter(spatial_filter),maxIndex
        else:
            return create_numpy_filter(spatial_filter, sz),maxIndex


def create_cuda_filter(spatial_filter, sz):
    """
    create cuda version filter, another one dimension is added to the output for computational convenient
    besides the output will not be full complex result of shape (∗,2),
    where ∗ is the shape of input, but instead the last dimension will be halfed as of size ⌊Nd/2⌋+1.
    :param spatial_filter: N1 x...xNd, no batch dimension, no channel dimension
    :param sz: [N1,..., Nd]
    :return: filter, with size [1,N1,..Nd-1,⌊Nd/2⌋+1,2⌋
    """
    fftn = torch.rfft
    spatial_filter_th = torch.from_numpy(spatial_filter).float().cuda()
    spatial_filter_th = spatial_filter_th[None, ...]
    spatial_filter_th_fft = fftn(spatial_filter_th, len(sz))
    return spatial_filter_th_fft


def create_numpy_filter(spatial_filter, sz):
    return np.fft.fftn(spatial_filter, s=sz)

# todo: maybe check if we can use rfft's here for better performance
def sel_fftn(dim):
    """
    sel the gpu and cpu version of the fft
    :param dim:
    :return: function pointer
    """
    if USE_CUDA:
        if dim in[1,2,3]:
            f= torch.rfft
        else:
            print('Warning, fft more than 3d is supported but not tested')
        return f
    else:
        if dim == 1:
            f = np.fft.fft
        elif dim == 2:
            f = np.fft.fft2
        elif dim == 3:
            f = np.fft.fftn
        else:
            raise ValueError('Only 3D cpu ifft supported')
        return f

def sel_ifftn(dim):
    """
    select the cpu and gpu version of the ifft
    :param dim:
    :return: function pointer
    """
    if USE_CUDA:
        if dim in [1,2,3]:
            f = torch.irfft
        else:
            print('Warning, fft more than 3d is supported but not tested')
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
            self.dim = complex_fourier_filter.dim() -1
        else:
            self.dim = len(complex_fourier_filter.shape)
        self.fftn = sel_fftn(self.dim)
        self.ifftn = sel_ifftn(self.dim)

        """The filter in the Fourier domain"""

    def forward(self, input):
        """
        Performs the Fourier-based filtering
        the 3d cpu fft is not implemented in fftn, to avoid fusing with batch and channel, here 3d is calcuated in loop
        1d 2d cpu works well because fft and fft2 is inbuilt, similarly , 1d 2d 3d gpu fft also is inbuilt

        in gpu implementation, the rfft is used for efficiency, which means the filter should be symmetric
        (input_real+input_img)(filter_real+filter_img) = (input_real*filter_real-input_img*filter_img) + (input_img*filter_real+input_real*filter_img)i
        filter_img =0, then get input_real*filter_real + (input_img*filter_real)i ac + bci

        :param input: Image
        :return: Filtered-image
        """

        if USE_CUDA:
            input = FFTVal(input,ini=1)
            f_input = self.fftn(input,self.dim,onesided=True)
            f_filter_real = self.complex_fourier_filter[0]
            f_filter_real=f_filter_real.expand_as(f_input[...,0])
            f_filter_real = torch.stack((f_filter_real,f_filter_real),-1)
            f_conv = f_input * f_filter_real
            dim_input = len(input.shape)
            dim_input_batch = dim_input-self.dim
            conv_ouput_real = self.ifftn(f_conv, self.dim,onesided=True,signal_sizes=input.shape[dim_input_batch::])
            result = conv_ouput_real

            return FFTVal(result, ini=-1)
        else:
            if self.dim <3:
                conv_output = self.ifftn(self.fftn(input.detach().cpu().numpy()) * self.complex_fourier_filter)
                result = conv_output.real  # should in principle be real
            elif self.dim==3:
                result = np.zeros(input.shape)
                for batch in range(input.size()[0]):
                    for ch in range(input.size()[1]):
                        conv_output = self.ifftn(self.fftn(input[batch,ch].detach().cpu().numpy()) * self.complex_fourier_filter)
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
        the 3d cpu ifft is not implemented in ifftn, to avoid fusing with batch and channel, here 3d is calcuated in loop
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
            f_go = self.fftn(grad_output,self.dim,onesided=True)
            f_filter_real = self.complex_fourier_filter[0]
            f_filter_real = f_filter_real.expand_as(f_go[..., 0])
            f_filter_real = torch.stack((f_filter_real, f_filter_real), -1)
            f_conv = f_go * f_filter_real
            dim_input = len(grad_output.shape)
            dim_input_batch = dim_input - self.dim
            grad_input = self.ifftn(f_conv,self.dim,onesided=True,signal_sizes=grad_output.shape[dim_input_batch::])

            # print(grad_input)
            # print((grad_input[0,0,12:15]))

            return FFTVal(grad_input, ini=-1)
        else:
            # if self.needs_input_grad[0]:
            numpy_go = grad_output.detach().cpu().numpy()
            # we use the conjugate because the assumption was that the spatial filter is real
            # THe following two lines should be correct
            if self.dim < 3:
                grad_input_c = (self.ifftn(np.conjugate(self.complex_fourier_filter) * self.fftn(numpy_go)))
                grad_input = grad_input_c.real
            elif self.dim == 3:
                grad_input = np.zeros(numpy_go.shape)
                assert grad_output.dim() == 5  # to ensure the behavior correct, we avoid more than 3 dimension fftn method
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
            f_input =  self.fftn(input,self.dim,onesided=True)
            f_filter_real = self.complex_fourier_filter[0]
            f_filter_real += self.alpha
            f_filter_real = f_filter_real.expand_as(f_input[..., 0])
            f_filter_real = torch.stack((f_filter_real, f_filter_real), -1)
            f_conv = f_input/f_filter_real
            dim_input = len(input.shape)
            dim_input_batch = dim_input - self.dim
            conv_ouput_real = self.ifftn(f_conv,self.dim,onesided=True,signal_sizes=input.shape[dim_input_batch::])
            result = conv_ouput_real
            return FFTVal(result, ini=-1)


        else:
            result = np.zeros(input.shape)
            if self.dim <3:
                conv_output = self.ifftn(self.fftn(input.detach().cpu().numpy()) / (self.alpha + self.complex_fourier_filter))
                # result = abs(conv_output) # should in principle be real
                result = conv_output.real
            elif self.dim == 3:
                result = np.zeros(input.shape)
                for batch in range(input.size()[0]):
                    for ch in range(input.size()[1]):
                        conv_output = self.ifftn(
                            self.fftn(input[batch,ch].detach().cpu().numpy()) / (self.alpha + self.complex_fourier_filter))
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

            f_go = self.fftn(grad_output, self.dim, onesided=True)
            f_filter_real = self.complex_fourier_filter[0]
            f_filter_real += self.alpha
            f_filter_real = f_filter_real.expand_as(f_go[..., 0])
            f_filter_real = torch.stack((f_filter_real, f_filter_real), -1)
            f_conv = f_go / f_filter_real
            dim_input = len(grad_output.shape)
            dim_input_batch = dim_input - self.dim
            grad_input = self.ifftn(f_conv, self.dim, onesided=True, signal_sizes=grad_output.shape[dim_input::])

            return FFTVal(grad_input, ini=-1)
        else:
            # if self.needs_input_grad[0]:
            numpy_go = grad_output.detach().cpu().numpy()
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


class GaussianFourierFilterGenerator(object):
    def __init__(self, sz, spacing, nr_of_slots=1):
        self.sz = sz
        """image size"""
        self.spacing = spacing
        """image spacing"""
        self.volumeElement = self.spacing.prod()
        """volume of pixel/voxel"""
        self.dim = len(spacing)
        """dimension"""
        self.nr_of_slots = nr_of_slots
        """number of slots to hold Gaussians (to be able to support multi-Gaussian); this is related to storage"""
        """typically should be set to the number of total desired Gaussians (so that none of them need to be recomputed)"""

        self.mus = np.zeros(self.dim)
        # TODO: storing the identity map may be a little wasteful
        self.centered_id = utils.centered_identity_map(self.sz,self.spacing)

        self.complex_gaussian_fourier_filters = [None] * self.nr_of_slots
        self.max_indices = [None]*self.nr_of_slots
        self.sigmas_complex_gaussian_fourier_filters = [None]*self.nr_of_slots
        self.complex_gaussian_fourier_xsqr_filters = [None]*self.nr_of_slots
        self.sigmas_complex_gaussian_fourier_xsqr_filters = [None]*self.nr_of_slots
        self.sigmas_complex_gaussian_fourier_filters_np=[]

    def get_number_of_slots(self):
        return self.nr_of_slots

    def get_number_of_currently_stored_gaussians(self):
        nr_of_gaussians = 0
        for s in self.sigmas_complex_gaussian_fourier_filters:
            if s is not None:
                nr_of_gaussians += 1
        return  nr_of_gaussians

    def get_dimension(self):
        return self.dim

    def _compute_complex_gaussian_fourier_filter(self,sigma):

        stds = sigma.detach().cpu().numpy() * np.ones(self.dim)
        gaussian_spatial_filter = utils.compute_normalized_gaussian(self.centered_id, self.mus, stds)
        complex_gaussian_fourier_filter,max_index = create_complex_fourier_filter(gaussian_spatial_filter,self.sz,True)
        return complex_gaussian_fourier_filter,max_index

    def _compute_complex_gaussian_fourier_xsqr_filter(self,sigma,max_index=None):

        if max_index is None:
            raise ValueError('A Gaussian filter needs to be generated / requested *before* any other filter')

        # TODO: maybe compute this jointly with the gaussian filter itself to avoid computing the spatial filter twice
        stds = sigma.detach().cpu().numpy() * np.ones(self.dim)
        gaussian_spatial_filter = utils.compute_normalized_gaussian(self.centered_id, self.mus, stds)
        gaussian_spatial_xsqr_filter = gaussian_spatial_filter*(self.centered_id**2).sum(axis=0)

        complex_gaussian_fourier_xsqr_filter,max_index = create_complex_fourier_filter(gaussian_spatial_xsqr_filter,self.sz,True,max_index)
        return complex_gaussian_fourier_xsqr_filter,max_index

    def _find_closest_sigma_index(self, sigma, available_sigmas):
        """
        For a given sigma, finds the closest one in a list of available sigmas
        - If a sigma is already computed it finds its index
        - If the sigma has not been computed (it finds the next empty slot (None)
        - If no empty slots are available it replaces the closest
        :param available_sigmas: a list of sigmas that have already been computed (or None if they have not)
        :return: returns the index for the closest sigma among the available_sigmas
        """
        closest_i = None
        same_i = None
        empty_slot_i = None
        current_dist_sqr = None


        for i,s in enumerate(available_sigmas):
            if s is not None:

                # keep track of the one with the closest distance
                new_dist_sqr = (s-sigma)**2
                if current_dist_sqr is None:
                    current_dist_sqr = new_dist_sqr
                    closest_i = i
                else:
                    if new_dist_sqr<current_dist_sqr:
                        current_dist_sqr = new_dist_sqr
                        closest_i = i

                # also check if this is the same
                # if it is records the first occurrence
                if torch.isclose(sigma,s):
                    if same_i is None:
                        same_i = i
            else:
                # found an empty slot, record it if it is the first one that was found
                if empty_slot_i is None:
                    empty_slot_i = i

        # if we found the same we return it
        if same_i is not None:
            # we found the same; i.e., already computed
            return same_i
        elif empty_slot_i is not None:
            # it was not already computed, but we found an empty slot to put it in
            return empty_slot_i
        elif closest_i is not None:
            # no empty slot, so just overwrite the closest one if there is one
            return closest_i
        else:
            # nothing has been computed yet, so return the 0 index (this should never execute, as it should be taken care of by the empty slot
            return 0


    def get_gaussian_xsqr_filters(self,sigmas):
        """
        Returns complex Gaussian Fourier filter multiplied with x**2 with standard deviation sigma. 
        Only recomputes the filter if sigma has changed.
        :param sigmas: standard deviation of the filter as a list
        :return: Returns the complex Gaussian Fourier filters as a list (in the same order as requested)
        """

        current_complex_gaussian_fourier_xsqr_filters = []

        # only recompute the ones that need to be recomputed
        for sigma in sigmas:

            # now find the index that corresponds to this
            i = self._find_closest_sigma_index(sigma, self.sigmas_complex_gaussian_fourier_xsqr_filters)

            if self.sigmas_complex_gaussian_fourier_xsqr_filters[i] is None:
                need_to_recompute = True
            elif self.complex_gaussian_fourier_xsqr_filters[i] is None:
                need_to_recompute = True
            elif torch.isclose(sigma,self.sigmas_complex_gaussian_fourier_xsqr_filters[i]):
                need_to_recompute = False
            else:
                need_to_recompute = True

            if need_to_recompute:
                print('INFO: Recomputing gaussian xsqr filter for sigma={:.2f}'.format(sigma))
                self.sigmas_complex_gaussian_fourier_xsqr_filters[i] = sigma #.clone()
                self.complex_gaussian_fourier_xsqr_filters[i],_ = self._compute_complex_gaussian_fourier_xsqr_filter(sigma,self.max_indices[i])

            current_complex_gaussian_fourier_xsqr_filters.append(self.complex_gaussian_fourier_xsqr_filters[i])

        return current_complex_gaussian_fourier_xsqr_filters


    def get_gaussian_filters(self,sigmas):
        """
        Returns a complex Gaussian Fourier filter with standard deviation sigma. 
        Only recomputes the filter if sigma has changed.
        :param sigma: standard deviation of filter.
        :return: Returns the complex Gaussian Fourier filter
        """

        current_complex_gaussian_fourier_filters = []

        # only recompute the ones that need to be recomputed
        for sigma in sigmas:

            # now find the index that corresponds to this
            sigma_value = sigma.item()
            if sigma_value in self.sigmas_complex_gaussian_fourier_filters_np:
                i = self.sigmas_complex_gaussian_fourier_filters_np.index(sigma_value)
            else:
                i = self._find_closest_sigma_index(sigma,self.sigmas_complex_gaussian_fourier_filters)

                if self.sigmas_complex_gaussian_fourier_filters[i] is None:
                    need_to_recompute = True
                elif self.complex_gaussian_fourier_filters[i] is None:
                    need_to_recompute = True
                elif torch.isclose(sigma,self.sigmas_complex_gaussian_fourier_filters[i]):
                    need_to_recompute = False
                else:
                    need_to_recompute = True

                if need_to_recompute: # todo  not comment this warning
                    print('INFO: Recomputing gaussian filter for sigma={:.2f}'.format(sigma))
                    self.sigmas_complex_gaussian_fourier_filters[i] = sigma #.clone()
                    self.sigmas_complex_gaussian_fourier_filters_np.append(sigma_value)
                    self.complex_gaussian_fourier_filters[i], self.max_indices[i] = self._compute_complex_gaussian_fourier_filter(sigma)

            current_complex_gaussian_fourier_filters.append(self.complex_gaussian_fourier_filters[i])

        return current_complex_gaussian_fourier_filters

class FourierGaussianConvolution(Function):
    """
    pyTorch function to compute Gaussian convolutions in the Fourier domain: f = g*h.
    Also allows to differentiate through the Gaussian standard deviation.
    """

    def __init__(self, gaussian_fourier_filter_generator):
        """
        Constructor for the Fouier-based convolution
        :param sigma: standard deviation for the filter
        """
        # we assume this is a spatial filter, F, hence conj(F(w))=F(-w)
        super(FourierGaussianConvolution, self).__init__()

        self.gaussian_fourier_filter_generator = gaussian_fourier_filter_generator
        self.dim = self.gaussian_fourier_filter_generator.get_dimension()

        self.fftn = sel_fftn(self.dim)
        self.ifftn = sel_ifftn(self.dim)

    def _compute_convolution_CUDA(self,input,complex_fourier_filter):
        input = FFTVal(input, ini=1)
        f_input = self.fftn(input, self.dim, onesided=True)
        f_filter_real = complex_fourier_filter[0]
        f_filter_real = f_filter_real.expand_as(f_input[..., 0])
        f_filter_real = torch.stack((f_filter_real, f_filter_real), -1)
        f_conv = f_input * f_filter_real
        dim_input = len(input.shape)
        dim_input_batch = dim_input - self.dim
        conv_ouput_real = self.ifftn(f_conv, self.dim, onesided=True, signal_sizes=input.shape[dim_input_batch::])
        result = conv_ouput_real

        return FFTVal(result, ini=-1)

    def _compute_convolution_CPU(self,input,complex_fourier_filter):
        if self.dim < 3:
            conv_output = self.ifftn(self.fftn(input.detach().cpu().numpy()) * complex_fourier_filter)
            result = conv_output.real  # should in principle be real
        elif self.dim == 3:
            result = np.zeros(input.shape)
            for batch in range(input.size()[0]):
                for ch in range(input.size()[1]):
                    conv_output = self.ifftn(self.fftn(input[batch, ch].detach().cpu().numpy()) * complex_fourier_filter)
                    result[batch, ch] = conv_output.real
        else:
            raise ValueError("cpu fft smooth should be 1d-3d")

        return torch.FloatTensor(result)
        # print( 'max(imag) = ' + str( (abs( conv_output.imag )).max() ) )
        # print( 'max(real) = ' + str( (abs( conv_output.real )).max() ) )


    def _compute_input_gradient_CUDA(self,grad_output,complex_fourier_filter):
        grad_output = FFTVal(grad_output, ini=1)
        # print grad_output.view(-1,1).sum()
        f_go = self.fftn(grad_output, self.dim, onesided=True)
        f_filter_real = complex_fourier_filter[0]
        f_filter_real = f_filter_real.expand_as(f_go[..., 0])
        f_filter_real = torch.stack((f_filter_real, f_filter_real), -1)
        f_conv = f_go * f_filter_real
        dim_input = len(grad_output.shape)
        dim_input_batch = dim_input - self.dim
        grad_input = self.ifftn(f_conv, self.dim, onesided=True, signal_sizes=grad_output.shape[dim_input_batch::])

        return FFTVal(grad_input, ini=-1)

    def _compute_input_gradient_CPU(self,grad_output,complex_fourier_filter):
        numpy_go = grad_output.detach().cpu().numpy()
        # we use the conjugate because the assumption was that the spatial filter is real
        # THe following two lines should be correct
        if self.dim < 3:
            grad_input_c = (self.ifftn(np.conjugate(complex_fourier_filter) * self.fftn(numpy_go)))
            grad_input = grad_input_c.real
        elif self.dim == 3:
            grad_input = np.zeros(numpy_go.shape)
            assert grad_output.dim() == 5 # to ensure the behavior correct, we avoid more than 3 dimension fftn method
            for batch in range(grad_output.size()[0]):
                for ch in range(grad_output.size()[1]):
                    grad_input_c = (
                        self.ifftn(np.conjugate(complex_fourier_filter) * self.fftn(numpy_go[batch, ch])))
                    grad_input[batch, ch] = grad_input_c.real
        else:
            raise ValueError("cpu fft smooth should be 1d-3d")

        return torch.FloatTensor(grad_input)

    def _compute_sigma_gradient_CUDA(self,input,sigma,grad_output,complex_fourier_filter,complex_fourier_xsqr_filter):
        convolved_input = self._compute_convolution_CUDA(input, complex_fourier_filter)
        grad_sigma = -1. / sigma * self.dim * (grad_output.detach().cpu().numpy() * convolved_input).sum()
        convolved_input_xsqr = self._compute_convolution_CUDA(input, complex_fourier_xsqr_filter)
        grad_sigma += 1. / (sigma ** 3) * (grad_output.detach().cpu().numpy() * convolved_input_xsqr).sum()

        return grad_sigma

    # TODO: gradient appears to be incorrect
    def _compute_sigma_gradient_CPU(self,input,sigma,grad_output,complex_fourier_filter,complex_fourier_xsqr_filter):
        convolved_input = self._compute_convolution_CPU(input,complex_fourier_filter)
        grad_sigma = -1./sigma*self.dim*(grad_output.detach().cpu().numpy()*convolved_input).sum()
        convolved_input_xsqr = self._compute_convolution_CPU(input,complex_fourier_xsqr_filter)
        grad_sigma += 1./(sigma**3)*(grad_output.detach().cpu().numpy()*convolved_input_xsqr).sum()

        return grad_sigma


class FourierSingleGaussianConvolution(FourierGaussianConvolution):
    """
    pyTorch function to compute Gaussian convolutions in the Fourier domain: f = g*h.
    Also allows to differentiate through the Gaussian standard deviation.
    """

    def __init__(self, gaussian_fourier_filter_generator, compute_std_gradient):
        """
        Constructor for the Fouier-based convolution
        :param sigma: standard deviation for the filter
        :param compute_std_gradient: if True computes the gradient with respect to the std, otherwise set to 0
        """
        # we assume this is a spatial filter, F, hence conj(F(w))=F(-w)
        super(FourierSingleGaussianConvolution, self).__init__(gaussian_fourier_filter_generator)

        self.gaussian_fourier_filter_generator = gaussian_fourier_filter_generator

        self.complex_fourier_filter = None
        self.complex_fourier_xsqr_filter = None

        self.input = None
        self.sigma = None

        self.compute_std_gradient = compute_std_gradient

    def forward(self, input, sigma):
        """
        Performs the Fourier-based filtering
        the 3d cpu fft is not implemented in fftn, to avoid fusing with batch and channel, here 3d is calcuated in loop
        1d 2d cpu works well because fft and fft2 is inbuilt, similarly , 1d 2d 3d gpu fft also is inbuilt

        in gpu implementation, the rfft is used for efficiency, which means the filter should be symmetric
        :param input: Image
        :return: Filtered-image
        """

        self.input = input
        self.sigma = sigma

        self.complex_fourier_filter = self.gaussian_fourier_filter_generator.get_gaussian_filters(self.sigma)[0]
        self.complex_fourier_xsqr_filter = self.gaussian_fourier_filter_generator.get_gaussian_xsqr_filters(self.sigma)[0]

        # (a+bi)(c+di) = (ac-bd) + (bc+ad)i
        # filter_imag =0, then get  ac + bci

        if USE_CUDA:
            return self._compute_convolution_CUDA(input,self.complex_fourier_filter)
        else:
            return self._compute_convolution_CPU(input,self.complex_fourier_filter)

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        """
        Computes the gradient
        the 3d cpu ifft is not implemented in ifftn, to avoid fusing with batch and channel, here 3d is calcuated in loop
        1d 2d cpu works well because ifft and ifft2 is inbuilt, similarly , 1d 2d 3d gpu fft also is inbuilt

        in gpu implementation, the irfft is used for efficiency, which means the filter should be symmetric
        :param grad_output: Gradient output of previous layer
        :return: Gradient including the Fourier-based convolution
        """

        # Initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.

        grad_input = grad_sigma = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.

        # first compute the gradient with respect to the input
        if self.needs_input_grad[0]:
            # (a+bi)(c+di) = (ac-bd) + (bc+ad)i
            # input_imag =0, then get  ac + bci
            if USE_CUDA:
                grad_input = self._compute_input_gradient_CUDA(grad_output,self.complex_fourier_filter)
            else:
                grad_input = self._compute_input_gradient_CPU(grad_output,self.complex_fourier_filter)

        # now compute the gradient with respect to the standard deviation of the filter
        if self.compute_std_gradient:
            if self.needs_input_grad[1]:
                if USE_CUDA:
                    grad_sigma = self._compute_sigma_gradient_CUDA(self.input,self.sigma,grad_output,self.complex_fourier_filter,self.complex_fourier_xsqr_filter)
                else:
                    grad_sigma = self._compute_sigma_gradient_CPU(self.input,self.sigma,grad_output,self.complex_fourier_filter,self.complex_fourier_xsqr_filter)
        else:
            grad_sigma = torch.zeros_like(self.sigma)

        # now return the computed gradients
        return grad_input, grad_sigma


def fourier_single_gaussian_convolution(input, gaussian_fourier_filter_generator,sigma,compute_std_gradient):
    """
    Convenience function for Fourier-based Gaussian convolutions. Make sure to use this one (instead of directly
    using the class FourierGaussianConvolution). This will assure that each call generates its own instance
    and hence autograd will work properly

    :param input: Input image
    :param gaussian_fourier_filter_generator: generator which will create Gaussian Fourier filter (and caches them)
    :param sigma: standard deviation for the Gaussian filter
    :param compute_std_gradient: if set to True computes the gradient otherwise sets it to 0
    :return: 
    """
    # First braces create a Function object. Any arguments given here
    # will be passed to __init__. Second braces will invoke the __call__
    # operator, that will then use forward() to compute the result and
    # return it.
    return FourierSingleGaussianConvolution(gaussian_fourier_filter_generator,compute_std_gradient)(input,sigma)


class FourierMultiGaussianConvolution(FourierGaussianConvolution):
    """
    pyTorch function to compute multi Gaussian convolutions in the Fourier domain: f = g*h.
    Also allows to differentiate through the Gaussian standard deviation.
    """

    def __init__(self, gaussian_fourier_filter_generator,compute_std_gradients,compute_weight_gradients):
        """
        Constructor for the Fouier-based convolution

        :param gaussian_fourier_filter_generator: class instance that creates and caches the Gaussian filters
        :param compute_std_gradients: if set to True the gradients for std are computed, otherwise they are filled w/ zero
        :param compute_weight_gradients: if set to True the gradients for weights are computed, otherwise they are filled w/ zero
        """
        # we assume this is a spatial filter, F, hence conj(F(w))=F(-w)
        super(FourierMultiGaussianConvolution, self).__init__(gaussian_fourier_filter_generator)

        self.gaussian_fourier_filter_generator = gaussian_fourier_filter_generator

        self.complex_fourier_filters = None
        self.complex_fourier_xsqr_filters = None

        self.input = None
        self.weights = None
        self.sigmas = None
        self.nr_of_gaussians = None

        self.compute_std_gradients = compute_std_gradients
        self.compute_weight_gradients = compute_weight_gradients

    def forward(self, input, sigmas, weights):
        """
        Performs the Fourier-based filtering
        the 3d cpu fft is not implemented in fftn, to avoid fusing with batch and channel, here 3d is calcuated in loop
        1d 2d cpu works well because fft and fft2 is inbuilt, similarly , 1d 2d 3d gpu fft also is inbuilt

        in gpu implementation, the rfft is used for efficiency, which means the filter should be symmetric
        :param input: Image
        :return: Filtered-image
        """

        self.input = input
        self.sigmas = sigmas
        self.weights = weights

        self.nr_of_gaussians = len(self.sigmas)
        nr_of_weights = len(self.weights)

        assert(self.nr_of_gaussians==nr_of_weights)

        self.complex_fourier_filters = self.gaussian_fourier_filter_generator.get_gaussian_filters(self.sigmas)
        self.complex_fourier_xsqr_filters = self.gaussian_fourier_filter_generator.get_gaussian_xsqr_filters(self.sigmas)

        # (a+bi)(c+di) = (ac-bd) + (bc+ad)i
        # filter_imag =0, then get  ac + bci

        ret = torch.zeros_like(input)

        for i in range(self.nr_of_gaussians):
            if USE_CUDA:
                ret += self.weights[i]*self._compute_convolution_CUDA(input,self.complex_fourier_filters[i])
            else:
                ret+= self.weights[i]*self._compute_convolution_CPU(input,self.complex_fourier_filters[i])

        return ret

    def _compute_input_gradient_CUDA_multi_gaussian(self,grad_output,complex_fourier_filters):
        grad_input = torch.zeros_like(self.input)
        for i in range(self.nr_of_gaussians):
            grad_input += self.weights[i]*self._compute_input_gradient_CUDA(grad_output, complex_fourier_filters[i])
        return grad_input

    def _compute_input_gradient_CPU_multi_gaussian(self,grad_output,complex_fourier_filters):
        grad_input = torch.zeros_like(self.input)
        for i in range(self.nr_of_gaussians):
            grad_input += self.weights[i] * self._compute_input_gradient_CPU(grad_output,complex_fourier_filters[i])
        return grad_input

    def _compute_sigmas_gradient_CUDA_multi_gaussian(self,input,sigmas,grad_output,complex_fourier_filters,complex_fourier_xsqr_filters):
        grad_sigmas = torch.zeros_like(sigmas)
        for i in range(self.nr_of_gaussians):
            grad_sigmas[i] = self.weights[i] * self._compute_sigma_gradient_CUDA(input,sigmas[i],grad_output,
                                                                                complex_fourier_filters[i],
                                                                                complex_fourier_xsqr_filters[i])
        return grad_sigmas

    def _compute_sigmas_gradient_CPU_multi_gaussian(self,input,sigmas,grad_output,complex_fourier_filters,complex_fourier_xsqr_filters):
        grad_sigmas = torch.zeros_like(sigmas)
        for i in range(self.nr_of_gaussians):
            grad_sigmas[i] = self.weights[i] * self._compute_sigma_gradient_CPU(input,sigmas[i],grad_output,
                                                                                complex_fourier_filters[i],
                                                                                complex_fourier_xsqr_filters[i])
        return grad_sigmas

    def _compute_weights_gradient_CUDA_multi_gaussian(self,input,weights,grad_output,complex_fourier_filters):
        grad_weights = torch.zeros_like(weights)
        for i in range(self.nr_of_gaussians):
            grad_weights[i] = (grad_output*self._compute_convolution_CUDA(input,complex_fourier_filters[i])).sum()
        return grad_weights

    def _compute_weights_gradient_CPU_multi_gaussian(self,input,weights,grad_output,complex_fourier_filters):
        grad_weights = torch.zeros_like(weights)
        for i in range(self.nr_of_gaussians):
            grad_weights[i] = (grad_output * self._compute_convolution_CPU(input, complex_fourier_filters[i])).sum()
        return grad_weights

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        """
        Computes the gradient
        the 3d cpu ifft is not implemented in ifftn, to avoid fusing with batch and channel, here 3d is calcuated in loop
        1d 2d cpu works well because ifft and ifft2 is inbuilt, similarly , 1d 2d 3d gpu fft also is inbuilt

        in gpu implementation, the irfft is used for efficiency, which means the filter should be symmetric
        :param grad_output: Gradient output of previous layer
        :return: Gradient including the Fourier-based convolution
        """

        # Initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.

        grad_input = grad_sigmas = grad_weights = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.

        # first compute the gradient with respect to the input
        if self.needs_input_grad[0]:
            # (a+bi)(c+di) = (ac-bd) + (bc+ad)i
            # input_imag =0, then get  ac + bci
            if USE_CUDA:
                grad_input = self._compute_input_gradient_CUDA_multi_gaussian(grad_output,self.complex_fourier_filters)
            else:
                grad_input = self._compute_input_gradient_CPU_multi_gaussian(grad_output,self.complex_fourier_filters)

        # now compute the gradient with respect to the standard deviation of the filter
        if self.needs_input_grad[1]:
            if self.compute_std_gradients:
                if USE_CUDA:
                    grad_sigmas = self._compute_sigmas_gradient_CUDA_multi_gaussian(self.input,self.sigmas,grad_output,self.complex_fourier_filters,self.complex_fourier_xsqr_filters)
                else:
                    grad_sigmas = self._compute_sigmas_gradient_CPU_multi_gaussian(self.input,self.sigmas,grad_output,self.complex_fourier_filters,self.complex_fourier_xsqr_filters)
            else:
                grad_sigmas = torch.zeros_like(self.sigmas)

        if self.needs_input_grad[2]:
            if self.compute_weight_gradients:
                if USE_CUDA:
                    grad_weights = self._compute_weights_gradient_CUDA_multi_gaussian(self.input,self.weights,grad_output,self.complex_fourier_filters)
                else:
                    grad_weights = self._compute_weights_gradient_CPU_multi_gaussian(self.input,self.weights,grad_output,self.complex_fourier_filters)
            else:
                grad_weights = torch.zeros_like(self.weights)

        # now return the computed gradients

        #print('gsigmas: min=' + str(grad_sigmas.min()) + '; max=' + str(grad_sigmas.max()))
        #print('gweight: min=' + str(grad_weights.min()) + '; max=' + str(grad_weights.max()))
        #print( 'gsigmas = ' + str( grad_sigmas))
        #print( 'gweight = ' + str( grad_weights))


        return grad_input, grad_sigmas, grad_weights

def fourier_multi_gaussian_convolution(input, gaussian_fourier_filter_generator,sigma,weights,compute_std_gradients=True,compute_weight_gradients=True):
    """
    Convenience function for Fourier-based multi Gaussian convolutions. Make sure to use this one (instead of directly
    using the class FourierGaussianConvolution). This will assure that each call generates its own instance
    and hence autograd will work properly

    :param input: Input image
    :param gaussian_fourier_filter_generator: generator which will create Gaussian Fourier filter (and caches them)
    :param sigma: standard deviations for the Gaussian filter (need to be positive)
    :param weights: weights for the multi-Gaussian kernel (need to sum up to one and need to be positive)
    :param compute_std_gradients: if set to True computes the gradients with respect to the standard deviation
    :param compute_weight_gradients: if set to True then gradients for weight are computed, otherwise they are replaced w/ zero
    :return: 
    """
    # First braces create a Function object. Any arguments given here
    # will be passed to __init__. Second braces will invoke the __call__
    # operator, that will then use forward() to compute the result and
    # return it.
    return FourierMultiGaussianConvolution(gaussian_fourier_filter_generator,compute_std_gradients,compute_weight_gradients)(input,sigma,weights)


class FourierSetOfGaussianConvolutions(FourierGaussianConvolution):
    """
    pyTorch function to compute a set of Gaussian convolutions (as in the multi-Gaussian) in the Fourier domain: f = g*h.
    Also allows to differentiate through the standard deviations. THe output is not a smoothed field, but the
    set of all of them. This can then be fed into a subsequent neural network for further processing.
    """

    def __init__(self, gaussian_fourier_filter_generator,compute_std_gradients):
        """
        Constructor for the Fouier-based convolution

        :param gaussian_fourier_filter_generator: class instance that creates and caches the Gaussian filters
        :param compute_std_gradients: if set to True the gradients for the stds are computed, otherwise they are filled w/ zero
        """
        # we assume this is a spatial filter, F, hence conj(F(w))=F(-w)
        super(FourierSetOfGaussianConvolutions, self).__init__(gaussian_fourier_filter_generator)

        self.gaussian_fourier_filter_generator = gaussian_fourier_filter_generator

        self.complex_fourier_filters = None
        self.complex_fourier_xsqr_filters = None

        self.input = None
        self.sigmas = None
        self.nr_of_gaussians = None

        self.compute_std_gradients = compute_std_gradients

    def forward(self, input, sigmas):
        """
        Performs the Fourier-based filtering
        the 3d cpu fft is not implemented in fftn, to avoid fusing with batch and channel, here 3d is calculated in loop
        1d 2d cpu works well because fft and fft2 is inbuilt, similarly , 1d 2d 3d gpu fft also is inbuilt

        in gpu implementation, the rfft is used for efficiency, which means the filter should be symmetric
        :param input: Image
        :return: Filtered-image
        """

        self.input = input
        self.sigmas = sigmas

        self.nr_of_gaussians = len(self.sigmas)

        self.complex_fourier_filters = self.gaussian_fourier_filter_generator.get_gaussian_filters(self.sigmas)
        if self.compute_std_gradients:
            self.complex_fourier_xsqr_filters = self.gaussian_fourier_filter_generator.get_gaussian_xsqr_filters(self.sigmas)
        # TODO check if the xsqr should be put into an if statement here

        # (a+bi)(c+di) = (ac-bd) + (bc+ad)i
        # filter_imag =0, then get  ac + bci

        sz = input.size()
        new_sz = [self.nr_of_gaussians] + list(sz)

        ret = AdaptVal(MyTensor(*new_sz))

        for i in range(self.nr_of_gaussians):
            if USE_CUDA:
                ret[i,...] = self._compute_convolution_CUDA(input,self.complex_fourier_filters[i])
            else:
                ret[i,...] = self._compute_convolution_CPU(input,self.complex_fourier_filters[i])

        return ret

    def _compute_input_gradient_CUDA_multi_gaussian(self,grad_output,complex_fourier_filters):
        grad_input = torch.zeros_like(self.input)
        for i in range(self.nr_of_gaussians):
            grad_input += self._compute_input_gradient_CUDA(grad_output[i,...], complex_fourier_filters[i])
        return grad_input

    def _compute_input_gradient_CPU_multi_gaussian(self,grad_output,complex_fourier_filters):
        grad_input = torch.zeros_like(self.input)
        for i in range(self.nr_of_gaussians):
            grad_input += self._compute_input_gradient_CPU(grad_output[i,...],complex_fourier_filters[i])
        return grad_input

    def _compute_sigmas_gradient_CUDA_multi_gaussian(self,input,sigmas,grad_output,complex_fourier_filters,complex_fourier_xsqr_filters):
        grad_sigmas = torch.zeros_like(sigmas)
        for i in range(self.nr_of_gaussians):
            grad_sigmas[i] = self._compute_sigma_gradient_CUDA(input,sigmas[i],grad_output[i,...],
                                                                                complex_fourier_filters[i],
                                                                                complex_fourier_xsqr_filters[i])
        return grad_sigmas

    def _compute_sigmas_gradient_CPU_multi_gaussian(self,input,sigmas,grad_output,complex_fourier_filters,complex_fourier_xsqr_filters):
        grad_sigmas = torch.zeros_like(sigmas)
        for i in range(self.nr_of_gaussians):
            grad_sigmas[i] = self._compute_sigma_gradient_CPU(input,sigmas[i],grad_output[i,...],
                                                                                complex_fourier_filters[i],
                                                                                complex_fourier_xsqr_filters[i])
        return grad_sigmas

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        """
        Computes the gradient
        the 3d cpu ifft is not implemented in ifftn, to avoid fusing with batch and channel, here 3d is calcuated in loop
        1d 2d cpu works well because ifft and ifft2 is inbuilt, similarly , 1d 2d 3d gpu fft also is inbuilt

        in gpu implementation, the irfft is used for efficiency, which means the filter should be symmetric
        :param grad_output: Gradient output of previous layer
        :return: Gradient including the Fourier-based convolution
        """

        # Initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.

        grad_input = grad_sigmas = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.

        # first compute the gradient with respect to the input
        if self.needs_input_grad[0]:
            # (a+bi)(c+di) = (ac-bd) + (bc+ad)i
            # input_imag =0, then get  ac + bci
            if USE_CUDA:
                grad_input = self._compute_input_gradient_CUDA_multi_gaussian(grad_output,self.complex_fourier_filters)
            else:
                grad_input = self._compute_input_gradient_CPU_multi_gaussian(grad_output,self.complex_fourier_filters)

        # now compute the gradient with respect to the standard deviation of the filter
        if self.needs_input_grad[1]:
            if self.compute_std_gradients:
                if USE_CUDA:
                    grad_sigmas = self._compute_sigmas_gradient_CUDA_multi_gaussian(self.input,self.sigmas,grad_output,self.complex_fourier_filters,self.complex_fourier_xsqr_filters)
                else:
                    grad_sigmas = self._compute_sigmas_gradient_CPU_multi_gaussian(self.input,self.sigmas,grad_output,self.complex_fourier_filters,self.complex_fourier_xsqr_filters)
            else:
                grad_sigmas = torch.zeros_like(self.sigmas)


        # now return the computed gradients
        return grad_input, grad_sigmas

def fourier_set_of_gaussian_convolutions(input, gaussian_fourier_filter_generator,sigma,compute_std_gradients=False):
    """
    Convenience function for Fourier-based multi Gaussian convolutions. Make sure to use this one (instead of directly
    using the class FourierGaussianConvolution). This will assure that each call generates its own instance
    and hence autograd will work properly

    :param input: Input image
    :param gaussian_fourier_filter_generator: generator which will create Gaussian Fourier filter (and caches them)
    :param sigma: standard deviations for the Gaussian filter (need to be positive)
    :param compute_weight_std_gradients: if set to True then gradients for standard deviation are computed, otherwise they are replaced w/ zero
    :return:
    """
    # First braces create a Function object. Any arguments given here
    # will be passed to __init__. Second braces will invoke the __call__
    # operator, that will then use forward() to compute the result and
    # return it.
    return FourierSetOfGaussianConvolutions(gaussian_fourier_filter_generator,compute_std_gradients)(input,sigma)


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
    sz = np.array([20, 20], dtype='int64')
    # f = np.ones(sz)
    f = 1 / 400. * np.ones(sz)
    dim = len(sz)

    mus = np.zeros(dim)
    stds = np.ones(dim)
    spacing = np.ones(dim)
    centered_id = utils.centered_identity_map(sz,spacing)
    g = 100 * utils.compute_normalized_gaussian(centered_id, mus, stds)
    FFilter,_ = create_complex_fourier_filter(g, sz)
    input = AdaptVal(torch.randn([1, 1] + list(sz)))
    input.requires_grad = True
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
    FFilter,_ = create_complex_fourier_filter(f, sz, False)
    input = torch.randn(sz).float()
    input.requires_grad = True
    fc = FourierConvolution(FFilter)(input)
    # print( fc )
    fc.backward(torch.randn(sz).float())
    print(input.grad)



