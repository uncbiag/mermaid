'''
General purpose similarity measures that can be used
'''

from abc import ABCMeta, abstractmethod

import utils


class SimilarityMeasure(object):
    __metaclass__ = ABCMeta

    def __init__(self, spacing, params):
        self.spacing = spacing
        self.volumeElement = self.spacing.prod()
        self.dim = len(spacing)
        self.params = params

    @abstractmethod
    def computeSimilarity(self, I0, I1):
        pass


class SSDSimilarity(SimilarityMeasure):

    def __init__(self, spacing, params):
        super(SSDSimilarity,self).__init__(spacing,params)
        self.sigma = utils.getpar(params, 'sigma', 0.1)

    def computeSimilarity(self,I0,I1):
        return ((I0 - I1) ** 2).sum() / (self.sigma ** 2) * self.volumeElement

class NCCSimilarity(SimilarityMeasure):

    def __init__(self, spacing, params):
        super(NCCSimilarity,self).__init__(spacing,params)
        self.sigma = utils.getpar(params, 'sigma', 0.1)

    def computeSimilarity(self,I0,I1):
        ncc = ((I0-I0.mean().expand_as(I0))*(I1-I1.mean().expand_as(I1))).mean()/(I0.std()*I1.std())
        # does not need to be multipled by self.volumeElement (as we are dealing with a correlation measure)
        return (1-ncc**2) / (self.sigma ** 2)


class SimilarityMeasureFactory(object):

    __metaclass__ = ABCMeta

    def __init__(self,spacing):
        self.spacing = spacing
        self.dim = len( spacing )

    def createSimilarityMeasure(self,similarityMeasureName='ssd',params=None):
        if similarityMeasureName=='ssd':
            print('Using SSD similarity measure')
            return SSDSimilarity(self.spacing,params)
        elif similarityMeasureName=='ncc':
            print('Using NCC similarity measure')
            return NCCSimilarity(self.spacing, params)
        else:
            raise ValueError( 'Similarity measure: ' + similarityMeasureName + ' not known')