'''
General purpose similarity measures that can be used
'''

from abc import ABCMeta, abstractmethod
import torch
from torch.autograd import Variable

class SimilarityMeasure(object):
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

    def compute_similarity_multiNC(self, I0, I1):
        sz = I0.size()
        sim = Variable(torch.zeros(1), requires_grad=False)
        for nrI in range(sz[0]):  # loop over all the images
            sim = sim + self.compute_similarity_multiC(I0[nrI, ...], I1[nrI, ...])
        return sim

    def compute_similarity_multiC(self, I0, I1):
        sz = I0.size()
        sim = Variable(torch.zeros(1), requires_grad=False)
        for nrC in range(sz[0]):  # loop over all the channels, just advect them all the same
            sim = sim + self.compute_similarity(I0[nrC, ...], I1[nrC, ...])
        return sim

    @abstractmethod
    def compute_similarity(self, I0, I1):
        pass


class SSDSimilarity(SimilarityMeasure):

    def __init__(self, spacing, params):
        super(SSDSimilarity,self).__init__(spacing,params)
        self.sigma = params[('sigma', 0.1,
                                        '1/sigma^2 is the weight in front of the similarity measure')]

    def set_sigma(self,sigma):
        self.sigma = sigma

    def get_sigma(self):
        return self.sigma

    def compute_similarity(self, I0, I1):
        # sums over all images and channels
        return ((I0 - I1) ** 2).sum() / (self.sigma ** 2) * self.volumeElement

class NCCSimilarity(SimilarityMeasure):

    def __init__(self, spacing, params):
        super(NCCSimilarity,self).__init__(spacing,params)
        self.sigma = params[('sigma', 0.1,
                                        '1/sigma^2 is the weight in front of the similarity measure')]

    def set_sigma(self,sigma):
        self.sigma = sigma

    def get_sigma(self):
        return self.sigma

    def compute_similarity(self, I0, I1):

        ncc = ((cI0-cI0.mean().expand_as(cI0))*(cI1-cI1.mean().expand_as(cI1))).mean()/(cI0.std()*cI1.std())
        # does not need to be multiplied by self.volumeElement (as we are dealing with a correlation measure)
        return (1-ncc**2) / (self.sigma ** 2)


class SimilarityMeasureFactory(object):

    def __init__(self,spacing):
        self.spacing = spacing
        self.dim = len( spacing )
        self.similarity_measure_default_type = 'ssd'

        self.simMeasures = {
            'ssd': SSDSimilarity,
            'ncc': NCCSimilarity
        }

    def add_similarity_measure(self,simName,simClass):
        print('Registering new similarity measure ' + simName + ' in factory')
        self.simMeasures[simName]=simClass

    def print_available_similarity_measures(self):
        print(self.simMeasures)

    def set_similarity_measure_default_type_to_ssd(self):
        self.similarity_measure_default_type = 'ssd'

    def set_similarity_measure_default_type_to_ncc(self):
        self.similarity_measure_default_type = 'ncc'

    def create_similarity_measure(self, params):

        cparams = params[('similarity_measure',{},'settings for the similarity measure')]
        similarityMeasureType = cparams[('type', self.similarity_measure_default_type, 'type of similarity measure (ssd/ncc)')]

        if self.simMeasures.has_key( similarityMeasureType ):
            print('Using ' + similarityMeasureType + ' similarity measure')
            return self.simMeasures[similarityMeasureType](self.spacing,cparams)
        else:
            raise ValueError( 'Similarity measure: ' + similarityMeasureType + ' not known')
