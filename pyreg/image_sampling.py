from scipy import ndimage as nd
import torch
from torch.autograd import Variable
import numpy as np
#from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('./libraries'))
sys.path.insert(0, os.path.abspath('./libraries/modules'))
sys.path.insert(0, os.path.abspath('./libraries/functions'))

sys.path.insert(0, os.path.abspath('./pyreg'))
sys.path.insert(0, os.path.abspath('./pyreg/libraries'))


import smoother_factory as SF
import module_parameters as MP
#TODO: finish this class

#TODO: skimage in version 0.14 (not officially released yet, support 3D up- and downsampling
#TODO: should replace this class with the skimage functionality once officially available
#TODO: We convert here from torch to numpy and back to torch, hence it is not possible to autograd through these transformations

class ResampleImage(object):
#    __metaclass__ = ABCMeta

    def __init__(self,scalingFactor=0.5):
        self.params = MP.ParameterDict()
        self.params['iter']=10
        self.scalingFactor = scalingFactor

    def _zoom_image_singleC(self,I,spacing,scaling):
        sz = np.array(list(I.size()))  # we assume this is a pytorch tensor
        resSz = sz * scaling
        resSzInt = resSz.round().astype('int')  # zoom works with rounding

        # TODO: replace zoom, so we can backprop through it if necessary
        Iz = nd.zoom(I.data.numpy(),scaling,None,order=1,mode='reflect')
        newSpacing = spacing*(sz.astype('float')/resSzInt.astype('float'))
        Iz_t = Variable(torch.from_numpy(Iz),requires_grad=False)

        return Iz_t,newSpacing

    def _zoom_image_multiC(self,I,spacing,scaling):
        sz = np.array(list(I.size()))  # we assume this is a pytorch tensor
        resSz = sz[1::] * scaling
        resSzInt = resSz.round().astype('int')  # zoom works with rounding

        Iz = Variable(torch.zeros([sz[0]]+list(resSzInt)), requires_grad=False)
        for nrC in range(sz[0]):  # loop over all channels
            Iz[nrC, ...], newSpacing = self._zoom_image_singleC(I[nrC, ...], spacing, scaling)

        return Iz,newSpacing

    def _zoom_image_multiNC(self,I,spacing,scaling):
        sz = np.array(list(I.size())) # we assume this is a pytorch tensor
        resSz = sz[2::]*scaling
        resSzInt = resSz.round().astype('int') # zoom works with rounding

        Iz = Variable(torch.zeros([sz[0],sz[1]]+list(resSzInt)), requires_grad=False)
        for nrI in range(sz[0]): # loop over images
            Iz[nrI,...],newSpacing = self._zoom_image_multiC(I[nrI,...],spacing,scaling)

        return Iz,newSpacing

    def downsample_image(self,I,spacing):

        # assume we are dealing with a pytorch tensor
        sz = np.array(list(I.size()))
        dim = len(spacing)
        scaling = np.tile( self.scalingFactor, dim )

        smoother = SF.DiffusionSmoother(sz,spacing,self.params)
        smoothedImage_multiNC = smoother.smoothScalarField_multiNC(I)

        return self._zoom_image_multiNC(smoothedImage_multiNC,spacing,scaling)


    def upsample_image(self,I,spacing):

        # assume we are dealing with a pytorch tensor
        sz = np.array(list(I.size()))
        dim = len(spacing)
        scaling = 1./(np.tile(self.scalingFactor, dim))

        IZ,newspacing = self._zoom_image_multiNC(I, spacing, scaling)
        newSz = IZ.size()[-1-dim+1::]

        smoother = SF.DiffusionSmoother(newSz, newspacing, self.params)
        smoothedImage_multiNC = smoother.smoothScalarField_multiNC(IZ)

        return smoothedImage_multiNC,newspacing

    def downsample_vector_field(self,v,spacing):

        # assume we are dealing with a pytorch tensor
        sz = np.array(list(v.size()))
        dim = len(spacing)
        scaling = np.tile(self.scalingFactor, dim)

        smoother = SF.DiffusionSmoother(sz, spacing, self.params)
        smoothedV_multiN = smoother.smoothVectorField_multiN(v)

        # for zooming purposes we can just treat it as a multi-channel image
        return self._zoom_image_multiNC(smoothedV_multiN,spacing,scaling)

    def upsample_vector_field(self,v,spacing):

        sz = np.array(list(v.size()))
        dim = len(spacing)
        scaling = 1. / (np.tile(self.scalingFactor, dim))

        # for zooming purposes we can just treat it as a multi-channel image
        vZ, newspacing = self._zoom_image_multiNC(v, spacing, scaling)
        newSz = vZ.size()[-1 - dim + 1::]

        smoother = SF.DiffusionSmoother(newSz, newspacing, self.params)
        smoothedImage_multiNC = smoother.smoothVectorField_multiN(vZ)

        return smoothedImage_multiNC,newspacing

def testMe():


    I = Variable( torch.zeros([1,1,100,100]), requires_grad = False )
    I[0,0,30:70,30:70]=1

    ri = ResampleImage()
    spacing = np.array([1,1])
    ID,spacing_down = ri.downsample_image(I,spacing)
    IU,spacing_up = ri.upsample_image(ID,spacing_down)

    print(spacing)
    print(spacing_down)
    print(spacing_up)

    plt.subplot(131)
    plt.imshow(I[0,0,:,:].data.numpy().squeeze())

    plt.subplot(132)
    plt.imshow(ID[0, 0, :, :].data.numpy().squeeze())

    plt.subplot(133)
    plt.imshow(IU[0, 0, :, :].data.numpy().squeeze())

    plt.show()

#testMe()