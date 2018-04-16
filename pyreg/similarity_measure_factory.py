"""
Similarity measures for the registration methods and factory to create similarity measures.
"""

from abc import ABCMeta, abstractmethod
import torch
from torch.autograd import Variable
from data_wrapper import AdaptVal
from data_wrapper import MyTensor
import utils
from math import floor
from similarity_helper_omt import *

import numpy as np

class SimilarityMeasure(object):
    """
    Abstract base class for a similarity measure
    """
    __metaclass__ = ABCMeta

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
        sz = I0.size()
        sim = Variable(torch.zeros(1), requires_grad=False).type_as(I0)

        if I0Source is None and phi is None:
            for nrI in range(sz[0]):  # loop over all the images
                sim = sim + self.compute_similarity_multiC(I0[nrI, ...], I1[nrI, ...], None, None )

        elif I0Source is not None and phi is None:
            for nrI in range(sz[0]):  # loop over all the images
                sim = sim + self.compute_similarity_multiC(I0[nrI, ...], I1[nrI, ...], I0Source[nrI,...], None)

        elif I0Source is None and phi is not None:
            for nrI in range(sz[0]):  # loop over all the images
                sim = sim + self.compute_similarity_multiC(I0[nrI, ...], I1[nrI, ...], None, phi[nrI,...])

        else:
            for nrI in range(sz[0]):  # loop over all the images
                sim = sim + self.compute_similarity_multiC(I0[nrI, ...], I1[nrI, ...], I0Source[nrI,...], phi[nrI,...])

        return sim/sz[0] # needs to be normalized based on batch size


    def compute_similarity_multiC(self, I0, I1, I0Source=None, phi=None):
        """
        Compute the multi-channel image similarity between two images of format CxXxYzZ

        :param I0: first image (the warped source image)
        :param I1: second image (target image)
        :param I0Source: source image (will typically not be used)
        :param phi: map in the target image to warp the source image (will typically not be used)
        :return: returns similarity measure
        """
        sz = I0.size()
        sim = Variable(torch.zeros(1), requires_grad=False).type_as(I0)

        if I0Source is None:
            for nrC in range(sz[0]):  # loop over all the channels, just advect them all the same; if available map is the same for all channels
                sim = sim + self.compute_similarity(I0[nrC, ...], I1[nrC, ...], None, phi)

        else:
            for nrC in range(sz[0]):  # loop over all the channels, just advect them all the same; if available map is the same for all channels
                sim = sim + self.compute_similarity(I0[nrC, ...], I1[nrC, ...],I0Source[nrC, ...],phi)

        return sim/sz[0] # needs to be normalized based on the number of channels

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


class SSDSimilarity(SimilarityMeasure):
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


class OptimalMassTransportSimilarity(SimilarityMeasure):
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
        I1_warped = utils.compute_warped_image(I0Source, phi, self.spacing,self.params)

        # Encapsulate the data in tensor Variables
        multiplier0 = Variable(torch.zeros(I0.size()))
        multiplier1 = Variable(torch.zeros(I1.size()))
        nr_iterations_sinkhorn = Variable(torch.Tensor([self.sinkhorn_iterations]))
        std_sink = Variable(torch.Tensor([self.std_sinkhorn]))

        # Compute the actual similarity
        result = OTSimilarityHelper.apply(phi,I1_warped,I1,multiplier0,multiplier1,Variable(torch.Tensor(self.spacing)),nr_iterations_sinkhorn,std_sink)
        return result/(self.std_dev**2)

class NCCSimilarity(SimilarityMeasure):
    """
    Computes a normalized-cross correlation based similarity measure between two images.
    :math:`sim = (1-ncc^2)/(\\sigma^2)`
    """
    def __init__(self, spacing, params):
        super(NCCSimilarity,self).__init__(spacing,params)

    def compute_similarity(self, I0, I1, I0Source=None, phi=None):
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
        I0mean = I0.mean()
        I1mean = I1.mean()
        nccSqr = (((I0-I0mean.expand_as(I0))*(I1-I1mean.expand_as(I1))).mean()**2)/\
                 (((I0-I0mean)**2).mean()*((I1-I1mean)**2).mean())

        return AdaptVal((1-nccSqr)/self.sigma**2)

        #ncc = ((I0-I0.mean().expand_as(I0))*(I1-I1.mean().expand_as(I1))).mean()/(I0.std()*I1.std())
        # does not need to be multiplied by self.volumeElement (as we are dealing with a correlation measure)
        #return AdaptVal((1.0-ncc**2) / (self.sigma ** 2))



class LocalizedNCCSimilarity(SimilarityMeasure):
    """
    Computes a normalized-cross correlation based similarity measure between two images.
    :math:`sim = (1-ncc^2)/(\\sigma^2)`
    """

    def __init__(self, spacing, params):
        super(LocalizedNCCSimilarity,self).__init__(spacing,params)
        #todo: maybe add some form of Gaussian weighing and tie it to the real image dimensions
        self.gaussian_std = params['similarity_measure'][('gaussian_std', 0.025, 'standard deviation of Gaussian that will be used for local NCC computations')]
        """half the side length of the cube over which lNCC is computed"""

        self.nr_of_elements_in_direction = None
        self.weighting_coefficients = None
        self.mask = None

        self._create_gaussian_weighting(self.gaussian_std)


    def _get_shifted_1d(self, I, x):
        ret = torch.zeros_like(I)
        sz = ret.size()

        if x >= 0:
            ret[0:sz[0] - x] = I[x:]
        else: # x < 0
            ret[-x:] = I[0:sz[0] + x]

        return ret

    def _get_shifted_2d(self,I,x,y):
        ret = torch.zeros_like(I)
        sz = ret.size()

        if x>=0 and y>=0:
            ret[0:sz[0]-x,0:sz[1]-y]=I[x:,y:]
        elif x<0 and y>=0:
            ret[-x:,0:sz[1]-y]=I[0:sz[0]+x,y:]
        elif x>=0 and y<0:
            ret[0:sz[0]-x,-y:] = I[x:, 0:sz[1]+y]
        else: # x<0 and y<0
            ret[-x:,-y:]= I[0:sz[0]+x,0:sz[1]+y]

        return ret

    def _get_shifted_3d(self, I, x, y, z):
        ret = torch.zeros_like(I)
        sz = ret.size()

        if x >= 0 and y >= 0 and z>=0:
            ret[0:sz[0] - x, 0:sz[1] - y,0:sz[2]-z] = I[x:, y:, z:]
        elif x < 0 and y >= 0 and z>=0:
            ret[-x:, 0:sz[1] - y,0:sz[2]-z] = I[0:sz[0] + x, y:, z:]
        elif x >= 0 and y < 0 and z>=0:
            ret[0:sz[0] - x, -y:,0:sz[2]-z] = I[x:, 0:sz[1] + y, z:]
        elif x<0 and y<0 and z>=0:
            ret[-x:, -y:,0:sz[2]-z] = I[0:sz[0] + x, 0:sz[1] + y, z:]
        elif x >= 0 and y >= 0 and z<0:
            ret[0:sz[0] - x, 0:sz[1] - y, -z:] = I[x:, y:, 0:sz[2]+z]
        elif x < 0 and y >= 0 and z<0:
            ret[-x:, 0:sz[1] - y, -z:] = I[0:sz[0] + x, y:, 0:sz[2]+z]
        elif x >= 0 and y < 0 and z<0:
            ret[0:sz[0] - x, -y:, -z:] = I[x:, 0:sz[1] + y, 0:sz[2]+z]
        else: # x < 0 and y < 0 and z < 0:
            ret[-x:, -y:, -z:] = I[0:sz[0] + x, 0:sz[1] + y, 0:sz[2]+z]

        return ret

    def _create_gaussian_weighting(self,sigma):

        radiusSqr = (3.*sigma)**2
        self.nr_of_elements_in_direction = MyTensor(self.dim).zero_()
        for i in range(self.dim):
            self.nr_of_elements_in_direction[i] = 3*sigma/self.spacing[i]
        self.nr_of_elements_in_direction = torch.ceil(self.nr_of_elements_in_direction).int()

        # now create the precomputed weights
        self.weighting_coefficients = MyTensor(*list((2*self.nr_of_elements_in_direction+1).int())).zero_()
        self.mask = torch.zeros_like(self.weighting_coefficients)

        if self.dim==1:
            for x in range(-self.nr_of_elements_in_direction[0],self.nr_of_elements_in_direction[0]+1):
                currentRSqr = (x*self.spacing[0])**2
                if currentRSqr<= radiusSqr:
                    self.weighting_coefficients[x+self.nr_of_elements_in_direction[0]]=np.exp(-currentRSqr/(2*sigma**2))
                    self.mask[x+self.nr_of_elements_in_direction[0]] = 1
            # now normalize it
            self.weighting_coefficients /= self.weighting_coefficients.sum()

        elif self.dim==2:
            for x in range(-self.nr_of_elements_in_direction[0],self.nr_of_elements_in_direction[0]+1):
                for y in range(-self.nr_of_elements_in_direction[1],self.nr_of_elements_in_direction[1]+1):
                    currentRSqr = (x*self.spacing[0])**2 + (y*self.spacing[1])**2
                    if currentRSqr<= radiusSqr:
                        self.weighting_coefficients[x+self.nr_of_elements_in_direction[0],y+self.nr_of_elements_in_direction[1]]=\
                            np.exp(-currentRSqr/(2*sigma**2))
                        self.mask[x+self.nr_of_elements_in_direction[0],y+self.nr_of_elements_in_direction[1]] = 1
            # now normalize it
            self.weighting_coefficients /= self.weighting_coefficients.sum()
        elif self.dim==3:
            for x in range(-self.nr_of_elements_in_direction[0],self.nr_of_elements_in_direction[0]+1):
                for y in range(-self.nr_of_elements_in_direction[1],self.nr_of_elements_in_direction[1]+1):
                    for z in range(-self.nr_of_elements_in_direction[2], self.nr_of_elements_in_direction[2] + 1):
                        currentRSqr = (x*self.spacing[0])**2 + (y*self.spacing[1])**2 + (z*self.spacing[2])**2
                        if currentRSqr<= radiusSqr:
                            self.weighting_coefficients[x+self.nr_of_elements_in_direction[0],
                                                        y+self.nr_of_elements_in_direction[1],
                                                        z+self.nr_of_elements_in_direction[2]]= np.exp(-currentRSqr/(2*sigma**2))
                            self.mask[x+self.nr_of_elements_in_direction[0],
                                      y+self.nr_of_elements_in_direction[1],
                                      z+self.nr_of_elements_in_direction[2]] = 1
            # now normalize it
            self.weighting_coefficients /= self.weighting_coefficients.sum()
        else:
            raise ValueError('Only dimensions 1,2, and 3 are supported.')


    def _compute_local_squared_cross_correlation(self,I0,I1):

        ones = torch.ones_like(I0)

        sumOnes = torch.zeros_like(I0)
        sumI0 = torch.zeros_like(I0)
        sumI1 = torch.zeros_like(I0)
        sumI0I1 = torch.zeros_like(I0)
        sumI0I0 = torch.zeros_like(I0)
        sumI1I1 = torch.zeros_like(I0)

        I0I0 = I0*I0
        I1I1 = I1*I1
        I0I1 = I0*I1

        if self.dim==1:
            for x in range(-self.nr_of_elements_in_direction[0],self.nr_of_elements_in_direction[0]+1):
                if self.mask[self.nr_of_elements_in_direction[0] + x] > 0:
                    current_weight = self.weighting_coefficients[self.nr_of_elements_in_direction[0]+x]
                    sumOnes += current_weight*self._get_shifted_1d(ones, x)
                    sumI0 += current_weight*self._get_shifted_1d(I0, x)
                    sumI1 += current_weight*self._get_shifted_1d(I1, x)
                    sumI0I1 += current_weight*self._get_shifted_1d(I0I1, x)
                    sumI0I0 += current_weight*self._get_shifted_1d(I0I0, x)
                    sumI1I1 += current_weight*self._get_shifted_1d(I1I1, x)

        elif self.dim==2:
            for x in range(-self.nr_of_elements_in_direction[0],self.nr_of_elements_in_direction[0]+1):
                for y in range(-self.nr_of_elements_in_direction[1],self.nr_of_elements_in_direction[1]+1):
                    if self.mask[self.nr_of_elements_in_direction[0]+x,self.nr_of_elements_in_direction[1]+y]>0:
                        current_weight = self.weighting_coefficients[self.nr_of_elements_in_direction[0]+x,self.nr_of_elements_in_direction[1]+y]
                        sumOnes += current_weight*self._get_shifted_2d(ones,x,y)
                        sumI0 += current_weight*self._get_shifted_2d(I0,x,y)
                        sumI1 += current_weight*self._get_shifted_2d(I1,x,y)
                        sumI0I1 += current_weight*self._get_shifted_2d(I0I1,x,y)
                        sumI0I0 += current_weight*self._get_shifted_2d(I0I0,x,y)
                        sumI1I1 += current_weight*self._get_shifted_2d(I1I1,x,y)

        elif self.dim==3:
            for x in range(-self.nr_of_elements_in_direction[0], self.nr_of_elements_in_direction[0] + 1):
                for y in range(-self.nr_of_elements_in_direction[1], self.nr_of_elements_in_direction[1] + 1):
                    for z in range(-self.nr_of_elements_in_direction[2], self.nr_of_elements_in_direction[2] + 1):
                        if self.mask[
                            self.nr_of_elements_in_direction[0] + x,
                            self.nr_of_elements_in_direction[1] + y,
                            self.nr_of_elements_in_direction[2] + z] > 0:

                            current_weight = self.weighting_coefficients[
                                self.nr_of_elements_in_direction[0] + x,
                                self.nr_of_elements_in_direction[1] + y,
                                self.nr_of_elements_in_direction[2] + z]

                            sumOnes += current_weight*self._get_shifted_3d(ones, x, y, z)
                            sumI0 += current_weight*self._get_shifted_3d(I0, x, y, z)
                            sumI1 += current_weight*self._get_shifted_3d(I1, x, y, z)
                            sumI0I1 += current_weight*self._get_shifted_3d(I0I1, x, y, z)
                            sumI0I0 += current_weight*self._get_shifted_3d(I0I0, x, y, z)
                            sumI1I1 += current_weight*self._get_shifted_3d(I1I1, x, y, z)

        else:
            raise ValueError('Only supported in dimensions 1, 2, and 3')

        # 1/n\sum_i (I0-mean(I0))(I1-mean(I1)) = 1/n \sum_i (I0I1 -I0 mean(I1) - mean(I0)I1 + mean(I0)mean(I1) )
        # ... = ( 1/n \sum_i I0 I1 ) -  mean(I0)mean(I1)

        # \sigma_0 = 1/n \sum_i (I0-mean(I0))^2 = 1/n \sum_i I0^2 - 2I0 mean(I0) + mean(I0)^2
        # ... = (1/n \sum_i I0^2 ) - mean(I0)^2

        meanI0 = sumI0/sumOnes
        meanI1 = sumI1/sumOnes
        nom = sumI0I1/sumOnes - meanI0*meanI1
        sig0Sqr = (sumI0I0/sumOnes - meanI0**2)
        sig1Sqr = (sumI1I1/sumOnes - meanI1**2)

        # todo: maybe find a little less hacky solution to deal with division by zero
        # we are returning the square here, because it is squared later anyway
        # and taking the sub-gradient for the square root at zero is not well defined
        eps = 1e-6 # to avoid division by zero
        lnccSqr = (nom*nom+eps)/(sig0Sqr*sig1Sqr+eps)

        return lnccSqr

    def compute_similarity(self, I0, I1, I0Source=None, phi=None):
        """
       Computes the NCC-based image similarity measure between two images

       :param I0: first image
       :param I1: second image
       :param I0Source: not used
       :param phi: not used
       :return: (1-NCC^2)/sigma^2
       """

        # TODO: may require a safeguard against infinity
        lnccSqr = self._compute_local_squared_cross_correlation(I0,I1)
        # does not need to be multiplied by self.volumeElement (as we are dealing with a correlation measure)

        sim_measure = AdaptVal((1.0-lnccSqr).sum() / (I0.numel()*self.sigma ** 2))
        #print( 'sim_measure = ' + str( sim_measure.data.numpy()))
        return sim_measure

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
            'lncc': LocalizedNCCSimilarity,
            'omt': OptimalMassTransportSimilarity
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

        if self.simMeasures.has_key( similarityMeasureType ):
            print('Using ' + similarityMeasureType + ' similarity measure')
            return self.simMeasures[similarityMeasureType](self.spacing,params)
        else:
            raise ValueError( 'Similarity measure: ' + similarityMeasureType + ' not known')
