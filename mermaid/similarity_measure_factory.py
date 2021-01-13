"""
Similarity measures for the registration methods and factory to create similarity measures.
"""
from __future__ import print_function
from __future__ import absolute_import

from builtins import range
from builtins import object
from abc import ABCMeta, abstractmethod
import torch
from .data_wrapper import AdaptVal,USE_CUDA
from .data_wrapper import MyTensor
from . import utils
from math import floor
from .similarity_helper_omt import *
import torch.nn.functional as F

import numpy as np
from future.utils import with_metaclass


class SimilarityMeasure(with_metaclass(ABCMeta, object)):
    """Abstract base class for a similarity measure.
    """

    def __init__(self, spacing, params):
        self.spacing = spacing
        """pixel/voxel spacing"""

        self.volumeElement = self.spacing.prod()
        """volume element"""
        self.dim = len(spacing)
        """image dimension"""
        self.params = params
        """external parameters"""

        self.sigma = params['similarity_measure'][('sigma', 0.1, '1/sigma^2 is the weight in front of the similarity measure')]
        """1/sigma^2 is a balancing constant"""

    @abstractmethod
    def compute_similarity_multiNC(self, I0, I1, I0Source=None, phi=None):
        """
        Abstract method to compute the image similarity between two images of format BxCxXxYxZ.

        :param I0: first image (the warped source image)
        :param I1: second image (target image)
        :param I0Source: source image (will typically not be used)
        :param phi: map in the target image to warp the source image (will typically not be used)
        :return: returns similarity measure
        """
        pass

    def set_sigma(self, sigma):
        """
        Set balancing constant :math:`\\sigma`

        :param sigma: balancing constant
        """
        self.sigma = sigma
        self.params['similarity_measure']['sigma']=sigma

    def get_sigma(self):
        """
        Get balancing constant

        :return: balancing constant
        """
        return self.sigma

class SimilarityMeasureSingleImage(SimilarityMeasure):
    """Abstract base class for a single image similarity measure.
    """

    def __init__(self, spacing, params):
        self.spacing = spacing
        """pixel/voxel spacing"""

        self.volumeElement = self.spacing.prod()
        """volume element"""
        self.dim = len(spacing)
        """image dimension"""
        self.params = params
        """external parameters"""

        self.sigma = params['similarity_measure'][('sigma', 0.1, '1/sigma^2 is the weight in front of the similarity measure')]
        """1/sigma^2 is a balancing constant"""

    def compute_similarity_multiNC(self, I0, I1, I0Source=None, phi=None):
        """
        Compute the multi-image multi-channel image similarity between two images of format BxCxXxYzZ

        :param I0: first image (the warped source image)
        :param I1: second image (target image)
        :param I0Source: source image (will typically not be used)
        :param phi: map in the target image to warp the source image (will typically not be used)
        :return: returns similarity measure
        """
        n_batch = I0.size()[0]
        sim = torch.zeros(1).type_as(I0)

        for nrI in range(n_batch):  # loop over all the images
            sim = sim + self.compute_similarity_multiC(I0[nrI, ...], I1[nrI, ...], None, None )

        return sim[0]


    def compute_similarity_multiC(self, I0, I1, I0Source=None, phi=None):
        """
        Compute the multi-channel image similarity between two images of format CxXxYzZ

        :param I0: first image (the warped source image)
        :param I1: second image (target image)
        :param I0Source: source image (will typically not be used)
        :param phi: map in the target image to warp the source image (will typically not be used)
        :return: returns similarity measure
        """
        n_channel = I0.size()[0]
        sim = torch.zeros(1).type_as(I0)

        if I0Source is None:
            for nrC in range(n_channel):  
                # loop over all the channels, just advect them all the same; 
                # if available map is the same for all channels
                sim = sim + self.compute_similarity(I0[nrC, ...], I1[nrC, ...], None, phi)

        else:
            for nrC in range(n_channel): 
                # loop over all the channels, just advect them all the same; 
                # if available map is the same for all channels
                sim = sim + self.compute_similarity(I0[nrC, ...], I1[nrC, ...],I0Source[nrC, ...],phi)

        return sim[0]/n_channel # needs to be normalized based on the number of channels


    @abstractmethod
    def compute_similarity(self, I0, I1, I0Source=None, phi=None):
        """
        Abstract method to compute the *single*-channel image similarity between two images of format XxYzZ.
        This is the only method that should be overwritten by a specific implemented similarity measure. 
        The multi-channel variants then come for free.

        For proper implementation it is important that the similarity measure is muliplied by
        1./(self.sigma ** 2)  and also by self.volumeElement if it is a volume integral
        (and not a correlation measure for example)

        :param I0: first image (the warped source image)
        :param I1: second image (target image)
        :param I0Source: source image (will typically not be used)
        :param phi: map in the target image to warp the source image (will typically not be used)
        :return: returns similarity measure
        """
        pass


class SSDSingleImageSimilarity(SimilarityMeasureSingleImage):
    """
    Sum of squared differences (SSD) similarity measure.
    
    :math:`1/sigma^2||I_0-I_1||^2`
    """

    def __init__(self, spacing, params):
        super(SSDSimilarity,self).__init__(spacing,params)

    def compute_similarity(self, I0, I1, I0Source=None, phi=None):
        """
        Computes the SSD measure between two images
        
        :param I0: first image 
        :param I1: second image
        :param I0Source: not used
        :param phi: not used
        :return: SSD/sigma^2
        """

        # TODO: This is to avoid a current pytorch bug 0.3.0 which cannot properly deal with infinity or NaN
        return AdaptVal(utils.remove_infs_from_variable((I0 - I1) ** 2).sum() / (self.sigma ** 2) * self.volumeElement)

        #return AdaptVal(((I0 - I1) ** 2).sum() / (self.sigma ** 2) * self.volumeElement)


class SSDSimilarity(SimilarityMeasure):
    """
    Sum of squared differences (SSD) similarity measure. The result is normalized over channel dimension and sum up over batch dimension.
    
    :math:`1/sigma^2||I_0-I_1||^2`
    """

    def __init__(self, spacing, params):
        super(SSDSimilarity,self).__init__(spacing,params)

    def compute_similarity_multiNC(self, I0, I1, I0Source=None, phi=None):
        """
        Computes the SSD measure between two images
        
        :param I0: first image 
        :param I1: second image
        :param I0Source: not used
        :param phi: not used
        :return: SSD/sigma^2
        """

        # TODO: This is to avoid a current pytorch bug 0.3.0 which cannot properly deal with infinity or NaN
        return AdaptVal(utils.remove_infs_from_variable((I0 - I1) ** 2).sum(dim=2).mean(dim=1).sum() / (self.sigma ** 2) * self.volumeElement)

        #return AdaptVal(((I0 - I1) ** 2).sum() / (self.sigma ** 2) * self.volumeElement)


class OptimalMassTransportSimilarity(SimilarityMeasureSingleImage):
    """
    Constructor
        :param spacing: spacing of the grid (as in parameters structure)
        :param std_dev: similarity measure parameter
        :param sinkhorn_iterations: number of iterations in sinkhorn algorithm
        :param std_sinkhorn: standard deviation of the entropic regularization (not too small to avoid nan)

    """

    def __init__(self, spacing, params,sinkhorn_iterations = 200,std_sinkhorn = 0.08):
        super(OptimalMassTransportSimilarity, self).__init__(spacing, params)
        self.spacing = spacing
        #self.params = params
        self.std_dev = self.sigma
        self.std_sinkhorn = std_sinkhorn
        self.sinkhorn_iterations = sinkhorn_iterations

        self.spline_order = params[('spline_order', 1, 'Spline interpolation order; 1 is linear interpolation (default); 3 is cubic spline')]
        """order of spline for interpolation (if needed)"""

    def compute_similarity(self, I0, I1, I0Source, phi):
        """
        Computes the SSD measure between two images

        :param I0: first image (not used)
        :param I1: second image (target image)
        :param I0Source: source image (not warped)
        :param phi: map to warp the source image to the target
        :return: OMTSimilarity/sigma^2
        """

        # todo: using the map here is not very efficient. It is much more efficient to apply the map
        # todo: to an entire batch of images.

        # FX: put your OMT code here; this is just a placeholder for now which is simple SSD (but using the source image and the map)
        if phi is None:
            raise ValueError('OptimalMassTransportSimiliary can only be computed for map-based models.')

        # warp the source image (would be more efficient if we process a batch of images at once;
        # but okay for now and no overhead if you only use one image pair at a time)
        I1_warped = utils.compute_warped_image(I0Source, phi, self.spacing,self.spline_order)

        # Encapsulate the data in tensor Variables
        multiplier0 = torch.zeros(I0.size())
        multiplier1 = torch.zeros(I1.size())
        nr_iterations_sinkhorn = torch.Tensor([self.sinkhorn_iterations])
        std_sink = torch.Tensor([self.std_sinkhorn])

        # Compute the actual similarity
        result = OTSimilarityHelper.apply(phi,I1_warped,I1,multiplier0,multiplier1,torch.Tensor(self.spacing),nr_iterations_sinkhorn,std_sink)
        return result/(self.std_dev**2)

class NCCSimilarity(SimilarityMeasure):
    """
    Computes a normalized-cross correlation based similarity measure between two images.
    The result of each pair of image is normalized over channel dimension and sum up over batch dimension.
    :math:`sim = (1-ncc^2)/(\\sigma^2)`
    """
    def __init__(self, spacing, params):
        super(NCCSimilarity,self).__init__(spacing,params)

    def compute_similarity_multiNC(self, I0, I1, I0Source=None, phi=None):
        """
       Computes the NCC-based image similarity measure between two images

       :param I0: first image 
       :param I1: second image
       :param I0Source: not used
       :param phi: not used
       :return: (1-NCC^2)/sigma^2
       """
        # TODO: may require a safeguard against infinity

        # this way of computing avoids the square root of the standard deviation
        # I0mean = I0.mean()
        # I1mean = I1.mean()
        # I0_m_mean = I0-I0mean
        # I1_m_mean = I1-I1mean
        # nccSqr = (((I0_m_mean)*(I1_m_mean)).mean()**2)/\
        #          (((I0_m_mean)**2).mean()*((I1_m_mean)**2).mean())
        n_batch = I0.shape[0]
        dim = len(I0.shape[2:])
        input_shape = [I0.shape[0], I0.shape[1], -1]+[1]*dim
        I0 = I0.view(*input_shape)
        I1 = I1.view(*input_shape)
        I0mean = I0.mean(dim=2, keepdim=True)
        I1mean = I1.mean(dim=2, keepdim=True)
        I0_m_mean = I0-I0mean
        I1_m_mean = I1-I1mean
        nccSqr = (((I0_m_mean)*(I1_m_mean)).mean(dim=2)**2)/\
                 (((I0_m_mean)**2).mean(dim=2)*((I1_m_mean)**2).mean(dim=2))
        nccSqr =nccSqr.mean(dim=1).sum()
        return AdaptVal((n_batch*1.-nccSqr)/self.sigma**2)

        #ncc = ((I0-I0.mean().expand_as(I0))*(I1-I1.mean().expand_as(I1))).mean()/(I0.std()*I1.std())
        # does not need to be multiplied by self.volumeElement (as we are dealing with a correlation measure)
        #return AdaptVal((1.0-ncc**2) / (self.sigma ** 2))

class NCCPositiveSimilarity(SimilarityMeasure):
    """
    Computes a normalized-cross correlation based similarity measure between two images. Only allows positive correlations.
    The result of each pair of image is normalized over channel dimension and sum up over batch dimension.
    :math:`sim = (1-ncc)/(\\sigma^2)`
    """
    def __init__(self, spacing, params):
        super(NCCPositiveSimilarity,self).__init__(spacing,params)

    def compute_similarity_multiNC(self, I0, I1, I0Source=None, phi=None):
        """
       Computes the NCC-based image similarity measure between two images

       :param I0: first image
       :param I1: second image
       :param I0Source: not used
       :param phi: not used
       :return: (1-NCC)/sigma^2
       """
        # TODO: may require a safeguard against infinity

        # this way of computing avoids the square root of the standard deviation
        n_batch = I0.shape[0]
        dim = len(I0.shape[2:])
        input_shape = [I0.shape[0], I0.shape[1], -1]+[1]*dim
        I0 = I0.view(*input_shape)
        I1 = I1.view(*input_shape)
        I0mean = I0.mean(dim=2, keepdim=True)
        I1mean = I1.mean(dim=2, keepdim=True)
        I0_m_mean = I0 - I0mean
        I1_m_mean = I1 - I1mean
        ncc = (((I0_m_mean)*(I1_m_mean)).mean(dim=2))/\
                 (torch.sqrt(((I0_m_mean)**2).mean(dim=2))*torch.sqrt(((I1_m_mean)**2).mean(dim=2)))
        ncc = ncc.mean(dim=1).sum()

        return AdaptVal((n_batch*1.-ncc)/self.sigma**2)

class NCCNegativeSimilarity(SimilarityMeasure):
    """
    Computes a normalized-cross correlation based similarity measure between two images. Only allows negative correlations.
    The result of each pair of image is normalized over channel dimension and sum up over batch dimension.
    :math:`sim = (ncc)/(\\sigma^2)`
    """
    def __init__(self, spacing, params):
        super(NCCNegativeSimilarity,self).__init__(spacing,params)

    def compute_similarity_multiNC(self, I0, I1, I0Source=None, phi=None):
        """
       Computes the NCC-based image similarity measure between two images

       :param I0: first image
       :param I1: second image
       :param I0Source: not used
       :param phi: not used
       :return: (NCC)/sigma^2
       """
        # TODO: may require a safeguard against infinity

        # this way of computing avoids the square root of the standard deviation
        dim = len(I0.shape[2:])
        input_shape = [I0.shape[0], I0.shape[1], -1] + [1] * dim
        I0 = I0.view(*input_shape)
        I1 = I1.view(*input_shape)
        I0mean = I0.mean(dim=2, keepdim=True)
        I1mean = I1.mean(dim=2, keepdim=True)
        I0_m_mean = I0 - I0mean
        I1_m_mean = I1 - I1mean
        ncc = (((I0_m_mean) * (I1_m_mean)).mean(dim=2)) / \
              (torch.sqrt((I0_m_mean ** 2).mean(dim=2)) * torch.sqrt((I1_m_mean ** 2).mean(dim=2)))
        ncc = ncc.mean(dim=1).sum()

        return AdaptVal((ncc)/self.sigma**2)

class LNCCSimilarity(SimilarityMeasure):
    """This is an generalized LNCC; we implement multi-scale (means resolution)
    multi kernel (means size of neighborhood) LNCC. The result of each pair of image 
    is normalized over channel dimension and sum up over batch dimension.

    :param: resol_bound : type list,  resol_bound[0]> resol_bound[1] >... resol_bound[end]
    :param: kernel_size_ratio: type list,  the ratio of the current input size
    :param: kernel_weight_ratio: type list,  the weight ratio of each kernel size, should sum to 1
    :param: stride: type_list, the stride between each pixel that would compute its lncc
    :param: dilation: type_list

    Settings in json::

        "similarity_measure": {
                "develop_mod_on": false,
                "sigma": 0.5,
                "type": "lncc",
                "lncc":{
                    "resol_bound":[-1],
                    "kernel_size_ratio":[[0.25]],
                    "kernel_weight_ratio":[[1.0]],
                    "stride":[0.25,0.25,0.25],
                    "dilation":[1]
                }

    For multi-scale multi kernel, e.g.,::

        "resol_bound":[64,32],
        "kernel_size_ratio":[[0.0625,0.125, 0.25], [0.25,0.5], [0.5]],
        "kernel_weight_ratio":[[0.1,0.3,0.6],[0.3,0.7],[1.0]],
        "stride":[0.25,0.25,0.25],
        "dilation":[1,2,2] #[2,1,1]

    or for single-scale single kernel, e.g.,::

        "resol_bound":[-1],
        "kernel_size_ratio":[[0.25]],
        "kernel_weight_ratio":[[1.0]],
        "stride":[0.25],
        "dilation":[1]


    Multi-scale is controlled by "resol_bound", e.g resol_bound = [128, 64], it means if input size>128, then it would compute multi-kernel
    lncc designed for large image size,  if 64<input_size<128, then it would compute multi-kernel lncc desiged for mid-size image, otherwise,
    it would compute the multi-kernel lncc designed for small image.
    Attention! we call it multi-scale just because it is designed for multi-scale registration or segmentation problem.
    ONLY ONE scale would be activated during computing the similarity, which depends on the current input size.

    At each scale, corresponding multi-kernel lncc is implemented, here multi-kernel means lncc with different window sizes
    Loss = w1*lncc_win1 + w2*lncc_win2 ... + wn*lncc_winn, where /sum(wi) =1
    for example. when (image size) S>128, three windows sizes can be used, namely S/16, S/8, S/4.
    for easy notation, we use img_ratio to refer window size, the example here use the parameter [1./16,1./8,1.4]

    In implementation, we compute lncc by calling convolution function, so in this case, the [S/16, S/8, S/4] refers
    to the kernel size of convolution function.  Intuitively,  we would have another two parameters,
    stride and dilation. For each window size (W), we recommend using W/4 as stride. In extreme case the stride can be 1, but
    can large increase computation.   The dilation expand the reception field, set dilation as 2 would physically twice the window size.
    """

    def __init__(self, spacing, params):
        super(LNCCSimilarity,self).__init__(spacing,params)
        self.dim = len(spacing)
        self.resol_bound = params['similarity_measure']['lncc'][('resol_bound',[128,64], "resolution bound for using different strategy")]
        self.kernel_size_ratio = params['similarity_measure']['lncc'][('kernel_size_ratio',[[1./16,1./8,1./4],[1./4,1./2],[1./2]], "kernel size, ratio of input size")]
        self.kernel_weight_ratio = params['similarity_measure']['lncc'][('kernel_weight_ratio',[[0.1, 0.3, 0.6],[0.3,0.7],[1.]], "kernel size, ratio of input size")]
        self.strides = params['similarity_measure']['lncc'][('stride',[[1./4,1./4,1./4],[1./4,1./4],[1./4]], "step size, responded with ratio of kernel size")]
        self.dilations = params['similarity_measure']['lncc'][('dilation',[[2,2,2],[2,2],[1]], "dilation param, responded with ratio of kernel size")]
        if self.resol_bound[0] >-1:
            assert len(self.resol_bound)+1 == len(self.kernel_size_ratio)
            assert len(self.resol_bound)+1 == len(self.kernel_weight_ratio)
            assert len(self.resol_bound)+1 == len(self.strides)
            assert len(self.resol_bound)+1 == len(self.dilations)

    def __stepup(self,img_sz):
        max_scale  = min(img_sz)
        for i, bound in enumerate(self.resol_bound):
            if max_scale >= bound:
                self.kernel = [int(max_scale*kz) for kz in self.kernel_size_ratio[i]]
                self.weight = self.kernel_weight_ratio[i]
                self.stride = self.strides[i]
                self.dilation = self.dilations[i]
                break
        if max_scale < self.resol_bound[-1]:
            self.kernel =  [int(max_scale*kz) for kz in self.kernel_size_ratio[-1]]
            self.weight = self.kernel_weight_ratio[-1]
            self.stride = self.strides[-1]
            self.dilation = self.dilations[-1]

        self.num_scale = len(self.kernel)
        self.kernel_sz = [[k for _ in range(self.dim)] for k in self.kernel]
        self.step = [[max(int((ksz + 1) * self.stride[scale_id]),1) for ksz in self.kernel_sz[scale_id]] for scale_id in range(self.num_scale)]
        if USE_CUDA:
            self.filter = [torch.ones([1, 1] + self.kernel_sz[scale_id]).cuda() for scale_id in range(self.num_scale)]
        else:
            self.filter = [torch.ones([1, 1] + self.kernel_sz[scale_id]) for scale_id in range(self.num_scale)]
        if self.dim==1:
            self.conv= F.conv1d
        elif self.dim ==2:
            self.conv= F.conv2d
        elif self.dim ==3:
            self.conv = F.conv3d
        else:
            raise ValueError(" Only 1-3d support")


    def compute_similarity_multiNC(self, I0, I1, I0Source=None, phi=None):
        """
       Computes the NCC-based image similarity measure between two images

       :param I0: first image
       :param I1: second image
       :param I0Source: not used
       :param phi: not used

       """
        n_batch = I0.shape[0]
        n_channel = I0.shape[1]
        input = I0.view([n_batch*n_channel,1]+list(I0.shape[2:])) #.view([1,1]+ list(I0.shape))
        target =I1.view([n_batch*n_channel,1]+list(I0.shape[2:])) #.view([1,1]+ list(I1.shape))
        self.__stepup(img_sz=list(I0.shape[2:]))

        input_2 = input ** 2
        target_2 = target ** 2
        input_target = input * target
        lncc_total = 0.
        for scale_id in range(self.num_scale):
            input_local_sum = self.conv(input, self.filter[scale_id], padding=0, 
                                        dilation=self.dilation[scale_id],
                                        stride=self.step[scale_id]).view(n_batch, n_channel, -1)
            target_local_sum = self.conv(target, self.filter[scale_id], padding=0, 
                                         dilation=self.dilation[scale_id],
                                         stride=self.step[scale_id]).view(n_batch, n_channel, -1)
            input_2_local_sum = self.conv(input_2, self.filter[scale_id], padding=0, 
                                          dilation=self.dilation[scale_id],
                                          stride=self.step[scale_id]).view(n_batch, n_channel, -1)
            target_2_local_sum = self.conv(target_2, self.filter[scale_id], padding=0, 
                                           dilation=self.dilation[scale_id],
                                           stride=self.step[scale_id]).view(n_batch, n_channel, -1)
            input_target_local_sum = self.conv(input_target, self.filter[scale_id], padding=0,
                                               dilation=self.dilation[scale_id], 
                                               stride=self.step[scale_id]).view(n_batch, n_channel, -1)

            input_local_sum = input_local_sum.contiguous()
            target_local_sum = target_local_sum.contiguous()
            input_2_local_sum = input_2_local_sum.contiguous()
            target_2_local_sum = target_2_local_sum.contiguous()
            input_target_local_sum = input_target_local_sum.contiguous()

            numel = float(np.array(self.kernel_sz[scale_id]).prod())

            input_local_mean = input_local_sum / numel
            target_local_mean = target_local_sum / numel

            cross = input_target_local_sum - target_local_mean * input_local_sum - \
                    input_local_mean * target_local_sum + target_local_mean * input_local_mean * numel
            input_local_var = input_2_local_sum - 2 * input_local_mean * input_local_sum + input_local_mean ** 2 * numel
            target_local_var = target_2_local_sum - 2 * target_local_mean * target_local_sum + target_local_mean ** 2 * numel

            lncc = cross * cross / (input_local_var * target_local_var + 1e-5)
            lncc = 1 - lncc.mean()
            lncc = n_batch* lncc
            lncc_total += lncc * self.weight[scale_id]

        return lncc_total / (self.sigma ** 2)


class SimilarityMeasureFactory(object):
    """
    Factory to quickly generate similarity measures that can then be used by the different registration algorithms.
    """

    def __init__(self,spacing):
        self.spacing = spacing
        """image spacing"""
        self.dim = len( spacing )
        """dimension of image"""
        self.similarity_measure_default_type = 'ssd'
        """default image similarity measure"""

        self.simMeasures = {
            'ssd': SSDSimilarity,
            'ncc': NCCSimilarity,
            'ncc_positive': NCCPositiveSimilarity,
            'ncc_negative': NCCNegativeSimilarity,
            'lncc': LNCCSimilarity,#LocalizedNCCSimilarity,
            'omt': OptimalMassTransportSimilarity,
            'ssd_single': SSDSingleImageSimilarity
        }
        """currently implemented similiarity measures"""

    def add_similarity_measure(self,simName,simClass):
        """
        Adds a new custom similarity measure
        
        :param simName: desired name of the similarity measure
        :param simClass: similiarity measure class (whcih can be instantiated)
        """
        print('Registering new similarity measure ' + simName + ' in factory')
        self.simMeasures[simName]=simClass

    def print_available_similarity_measures(self):
        """
        Prints all the available similarity measures
        """
        print(self.simMeasures)

    def set_similarity_measure_default_type_to_ssd(self):
        """
        Set the default similarity measure to SSD
        """
        self.similarity_measure_default_type = 'ssd'

    def set_similarity_measure_default_type_to_omt(self):
        """
        Set the default similarity measure to OMT (optimal mass transport)
        """
        self.similarity_measure_default_type = 'omt'

    def set_similarity_measure_default_type_to_ncc(self):
        """
        Set the default similarity measure to NCC
        """
        self.similarity_measure_default_type = 'ncc'

    def set_similarity_measure_default_type_to_ncc_positive(self):
        """
        Set the default similarity measure to positive NCC (i.e., only positive correlations allowed)
        """
        self.similarity_measure_default_type = 'ncc_positive'

    def set_similarity_measure_default_type_to_ncc_negative(self):
        """
        Set the default similarity measure to positive NCC (i.e., only negative correlations allowed)
        """
        self.similarity_measure_default_type = 'ncc_negative'

    def set_similarity_measure_default_type_to_lncc(self):
        """
        Set the default similarity measure to *localized* NCC
        """
        self.similarity_measure_default_type = 'lncc'

    def create_similarity_measure(self, params):
        """
        Create the actual similarity measure
        
        :param params: ParameterDict() object holding the parameters which can contol similarity measure settings 
        :return: returns a similarity measure (which can then be used to evaluate similarities)
        """

        cparams = params[('similarity_measure',{},'settings for the similarity measure')]
        similarityMeasureType = cparams[('type', self.similarity_measure_default_type, 'type of similarity measure (ssd/ncc)')]

        if similarityMeasureType in self.simMeasures:
            print('Using ' + similarityMeasureType + ' similarity measure')
            return self.simMeasures[similarityMeasureType](self.spacing,params)
        else:
            raise ValueError( 'Similarity measure: ' + similarityMeasureType + ' not known')
