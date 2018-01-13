"""
Similarity measures for the registration methods and factory to create similarity measures.
"""

from abc import ABCMeta, abstractmethod
import torch
from torch.autograd import Variable
from data_wrapper import AdaptVal
import utils

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

        self.sigma = params[('sigma', 0.1, '1/sigma^2 is the weight in front of the similarity measure')]
        """1/sigma^2 is a balancing constant"""

    def compute_similarity_multiNC(self, I0, I1):
        """
        Compute the multi-image multi-channel image similarity between two images of format BxCxXxYzZ
        
        :param I0: first image 
        :param I1: second image
        :return: returns similarity measure
        """
        sz = I0.size()
        sim = Variable(torch.zeros(1), requires_grad=False).type_as(I0)
        for nrI in range(sz[0]):  # loop over all the images
            sim = sim + self.compute_similarity_multiC(I0[nrI, ...], I1[nrI, ...])
        return sim

    def compute_similarity_multiC(self, I0, I1):
        """
        Compute the multi-channel image similarity between two images of format CxXxYzZ

        :param I0: first image 
        :param I1: second image
        :return: returns similarity measure
        """
        sz = I0.size()
        sim = Variable(torch.zeros(1), requires_grad=False).type_as(I0)
        for nrC in range(sz[0]):  # loop over all the channels, just advect them all the same
            sim = sim + self.compute_similarity(I0[nrC, ...], I1[nrC, ...])
        return sim

    @abstractmethod
    def compute_similarity(self, I0, I1):
        """
        Abstract method to compute the *single*-channel image similarity between two images of format XxYzZ.
        This is the only method that should be overwritten by a specific implemented similarity measure. 
        The multi-channel variants then come for free.

        For proper implementation it is important that the similarity measure is muliplied by
        1./(self.sigma ** 2)  and also by self.volumeElement if it is a volume integral
        (and not a correlation measure for example)

        :param I0: first image 
        :param I1: second image
        :return: returns similarity measure
        """
        pass

    def set_sigma(self, sigma):
        """
        Set balancing constant :math:`\\sigma`

        :param sigma: balancing constant
        """
        self.sigma = sigma
        self.params['sigma']=sigma

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

    def compute_similarity(self, I0, I1):
        """
        Computes the SSD measure between two images
        
        :param I0: first image 
        :param I1: second image
        :return: SSD
        """

        # TODO: This is to avoid a current pytorch bug 0.3.0 which cannot properly deal with infinity or NaN
        return AdaptVal((utils.remove_infs_from_variable((I0- I1) ** 2)).sum() / (self.sigma ** 2) * self.volumeElement)
        #return AdaptVal(((I0 - I1) ** 2).sum() / (self.sigma ** 2) * self.volumeElement)

class NCCSimilarity(SimilarityMeasure):
    """
    Computes a normalized-cross correlation based similarity measure between two images.
    :math:`sim = (1-ncc^2)/(\\sigma^2)`
    """

    def __init__(self, spacing, params):
        super(NCCSimilarity,self).__init__(spacing,params)

    def compute_similarity(self, I0, I1):
        """
       Computes the NCC-based image similarity measure between two images

       :param I0: first image 
       :param I1: second image
       :return: (1-NCC^2)/sigma^2
       """

        # TODO: may require a safeguard against infinity
        ncc = ((cI0-cI0.mean().expand_as(cI0))*(cI1-cI1.mean().expand_as(cI1))).mean()/(cI0.std()*cI1.std())
        # does not need to be multiplied by self.volumeElement (as we are dealing with a correlation measure)
        return (1-ncc**2) / (self.sigma ** 2)


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
            'ncc': NCCSimilarity
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

    def set_similarity_measure_default_type_to_ncc(self):
        """
        Set the default similarity measure to NCC
        """
        self.similarity_measure_default_type = 'ncc'

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
            return self.simMeasures[similarityMeasureType](self.spacing,cparams)
        else:
            raise ValueError( 'Similarity measure: ' + similarityMeasureType + ' not known')
