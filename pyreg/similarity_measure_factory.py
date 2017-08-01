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
        # sums over all images and channels
        return ((I0 - I1) ** 2).sum() / (self.sigma ** 2) * self.volumeElement

class NCCSimilarity(SimilarityMeasure):

    def __init__(self, spacing, params):
        super(NCCSimilarity,self).__init__(spacing,params)
        self.sigma = utils.getpar(params, 'sigma', 0.1)

    def computeSimilarity(self,I0,I1):
        # compute mean over all channels and images
        szI = I0.size()
        nrOfI = szI[0]
        nrOfC = szI[2]

        nccS = 0
        for i in range(nrOfI):
            for c in range(nrOfC):
                cI0 = I0[i,c,...]
                cI1 = I1[i,c,...]
                nccS = nccS + ((cI0-cI0.mean().expand_as(cI0))*(cI1-cI1.mean().expand_as(cI1))).mean()/(cI0.std()*cI1.std())
        # does not need to be multiplied by self.volumeElement (as we are dealing with a correlation measure)

        # compute overall NCC as average over all the individual nccS
        ncc = nccs/(nrOfI*nrOfC)

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