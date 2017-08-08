"""
Package to allow for resampling of images, for example to support multi-scale solvers.
"""

from scipy import ndimage as nd
import torch
from torch.autograd import Variable
import numpy as np

import smoother_factory as SF
import module_parameters as MP

import utils

#TODO: skimage in version 0.14 (not officially released yet, support 3D up- and downsampling
#TODO: should replace this class with the skimage functionality once officially available
#TODO: We convert here from torch to numpy and back to torch, hence it is not possible to autograd through these transformations

class ResampleImage(object):
    """
    This class supports image resampling, both based on a scale factor (implemented via skimage's zoom
    functionality) and to a fixed size (via custom interpolation). For multi-scaling the fixed size
    option is preferred as it gives better control over the resulting image sizes. In particular using
    the scaling factors consistent image sizes cannot be guaranteed when down-/up-sampling multiple times.
    """

    def __init__(self):
        self.params = MP.ParameterDict()
        self.params['iter']=0

    def set_iter(self,nrIter):
        """
        Sets the number of smoothing iterations done after upsampling and before downsampling.
        The default is 0, i.e., no smoothing at all.
        
        :param nrIter: number of iterations
        :return: no return arguments
        """
        self.params['iter'] = nrIter

    def get_iter(self):
        """
        Returns the number of iterations
        
        :return: number of smoothing iterations after upsampling and before downsampling 
        """
        return self.params['iter']

    def _compute_scaled_size(self, sz, scaling):
        resSz = sz*scaling
        resSzInt = np.zeros(len(scaling),dtype='int')
        for v in enumerate(resSz):
            resSzInt[v[0]]=int(round(v[1])) # zoom works with rounding
        return resSzInt


    def _zoom_image_singleC(self,I,spacing,scaling):
        sz = np.array(list(I.size()))  # we assume this is a pytorch tensor
        resSzInt = self._compute_scaled_size(sz, scaling)

        # TODO: replace zoom, so we can backprop through it if necessary
        Iz = nd.zoom(I.data.numpy(),scaling,None,order=1,mode='reflect')
        newSpacing = spacing*(sz.astype('float')/resSzInt.astype('float'))
        Iz_t = Variable(torch.from_numpy(Iz),requires_grad=False)

        return Iz_t,newSpacing

    def _zoom_image_multiC(self,I,spacing,scaling):
        sz = np.array(list(I.size()))  # we assume this is a pytorch tensor
        resSzInt = self._compute_scaled_size(sz[1::], scaling)

        Iz = Variable(torch.zeros([sz[0]]+list(resSzInt)), requires_grad=False)
        for nrC in range(sz[0]):  # loop over all channels
            Iz[nrC, ...], newSpacing = self._zoom_image_singleC(I[nrC, ...], spacing, scaling)

        return Iz,newSpacing

    def _zoom_image_multiNC(self,I,spacing,scaling):
        sz = np.array(list(I.size())) # we assume this is a pytorch tensor
        resSzInt = self._compute_scaled_size(sz[2::], scaling)

        Iz = Variable(torch.zeros([sz[0],sz[1]]+list(resSzInt)), requires_grad=False)
        for nrI in range(sz[0]): # loop over images
            Iz[nrI,...],newSpacing = self._zoom_image_multiC(I[nrI,...],spacing,scaling)

        return Iz,newSpacing

    def upsample_image_to_size(self,I,spacing,desiredSize):
        """
        Upsamples an image to a given desired size
        
        :param I: Input image (expected to be of BxCxXxYxZ format) 
        :param spacing: array describing the spatial spacing
        :param desiredSize: array for the desired size (excluding B and C, i.e, 1 entry for 1D, 2 for 2D, and 3 for 3D)
        :return: returns a tuple: the upsampled image, the new spacing after upsampling
        """

        sz = np.array(list(I.size()))
        dim = len(spacing)

        # check that the batch size and the number of channels is the same
        nrOfI = sz[0]
        nrOfC = sz[1]

        desiredSizeNC = np.array([nrOfI,nrOfC]+list(desiredSize))

        if (sz>desiredSizeNC).any():
            print(sz)
            print(desiredSizeNC)
            raise('For upsampling sizes need to increase')

        idDes = Variable(torch.from_numpy(utils.identity_map_multiN(desiredSizeNC)))
        newspacing = spacing*(sz[2::].astype('float')/desiredSizeNC[2::].astype('float'))

        # now use this map for resampling
        IZ = utils.compute_warped_image_multiNC(I, idDes)
        newSz = IZ.size()[-1 - dim + 1::]

        smoother = SF.DiffusionSmoother(newSz, newspacing, self.params)
        smoothedImage_multiNC = smoother.smooth_scalar_field_multiNC(IZ)

        return smoothedImage_multiNC,newspacing

    def downsample_image_to_size(self,I,spacing,desiredSize):
        """
        Downsamples an image to a given desired size

        :param I: Input image (expected to be of BxCxXxYxZ format) 
        :param spacing: array describing the spatial spacing
        :param desiredSize: array for the desired size (excluding B and C, i.e, 1 entry for 1D, 2 for 2D, and 3 for 3D)
        :return: returns a tuple: the downsampled image, the new spacing after downsampling
        """

        sz = np.array(list(I.size()))
        # check that the batch size and the number of channels is the same
        nrOfI = sz[0]
        nrOfC = sz[1]

        desiredSizeNC = np.array([nrOfI,nrOfC]+list(desiredSize))

        if (sz<desiredSizeNC).any():
            raise('For downsampling sizes need to decrease')

        smoother = SF.DiffusionSmoother(sz, spacing, self.params)
        smoothedImage_multiNC = smoother.smooth_scalar_field_multiNC(I)

        idDes = Variable(torch.from_numpy(utils.identity_map_multiN(desiredSizeNC)))
        newspacing = spacing*(sz[2::].astype('float')/desiredSizeNC[2::].astype('float'))

        # now use this map for resampling
        ID = utils.compute_warped_image_multiNC(smoothedImage_multiNC, idDes)

        return ID,newspacing


    def upsample_image_by_factor(self, I, spacing, scalingFactor=0.5):
        """
        Upsamples an image based on a given scale factor

        :param I: Input image (expected to be of BxCxXxYxZ format) 
        :param spacing: array describing the spatial spacing
        :param scalingFactor: scaling factor, e.g., 2 scales all dimensions by two
        :return: returns a tuple: the upsampled image, the new spacing after upsampling
        """

        # assume we are dealing with a pytorch tensor
        sz = np.array(list(I.size()))
        dim = len(spacing)
        scaling = 1./(np.tile(scalingFactor, dim))

        IZ,newspacing = self._zoom_image_multiNC(I, spacing, scaling)
        newSz = IZ.size()[-1-dim+1::]

        smoother = SF.DiffusionSmoother(newSz, newspacing, self.params)
        smoothedImage_multiNC = smoother.smooth_scalar_field_multiNC(IZ)

        return smoothedImage_multiNC,newspacing

    def downsample_image_by_factor(self, I, spacing, scalingFactor=0.5):
        """
        Downsamples an image based on a given scale factor
        
        :param I: Input image (expected to be of BxCxXxYxZ format) 
        :param spacing: array describing the spatial spacing
        :param scalingFactor: scaling factor, e.g., 0.5 scales all dimensions by half
        :return: returns a tuple: the downsampled image, the new spacing after downsampling
        """

        # assume we are dealing with a pytorch tensor
        sz = np.array(list(I.size()))
        dim = len(spacing)
        scaling = np.tile( scalingFactor, dim )

        smoother = SF.DiffusionSmoother(sz,spacing,self.params)
        smoothedImage_multiNC = smoother.smooth_scalar_field_multiNC(I)

        return self._zoom_image_multiNC(smoothedImage_multiNC,spacing,scaling)

    def upsample_vector_field_by_factor(self, v, spacing, scalingFactor=0.5):
        """
        Upsamples a vector field based on a given scale factor
        
        :param v: Input vector field (expected to be of BxCxXxYxZ format)
        :param spacing: array describing the spatial spacing
        :param scalingFactor: scaling factor, e.g., 2 scales all dimensions by two
        :return: returns a tuple: the upsampled vector field, the new spacing after upsampling
        """

        sz = np.array(list(v.size()))
        dim = len(spacing)
        scaling = 1. / (np.tile(scalingFactor, dim))

        # for zooming purposes we can just treat it as a multi-channel image
        vZ, newspacing = self._zoom_image_multiNC(v, spacing, scaling)
        newSz = vZ.size()[-1 - dim + 1::]

        smoother = SF.DiffusionSmoother(newSz, newspacing, self.params)
        smoothedImage_multiNC = smoother.smooth_vector_field_multiN(vZ)

        return smoothedImage_multiNC, newspacing

    def downsample_vector_field_by_factor(self, v, spacing, scalingFactor=0.5):
        """
        Downsamples a vector field based on a given scale factor

        :param v: Input vector field (expected to be of BxCxXxYxZ format)
        :param spacing: array describing the spatial spacing
        :param scalingFactor: scaling factor, e.g., 0.5 scales all dimensions by half
        :return: returns a tuple: the downsampled vector field, the new spacing after downsampling
        """

        # assume we are dealing with a pytorch tensor
        sz = np.array(list(v.size()))
        dim = len(spacing)
        scaling = np.tile(scalingFactor, dim)

        smoother = SF.DiffusionSmoother(sz, spacing, self.params)
        smoothedV_multiN = smoother.smooth_vector_field_multiN(v)

        # for zooming purposes we can just treat it as a multi-channel image
        return self._zoom_image_multiNC(smoothedV_multiN,spacing,scaling)



def test_me():
    """
    Convenience testing function (to be converted to a test)
    """
    I = Variable( torch.zeros([1,1,100,100]), requires_grad = False )
    I[0,0,30:70,30:70]=1

    ri = ResampleImage()
    spacing = np.array([1,1])
    ID,spacing_down = ri.downsample_image_by_factor(I, spacing)
    IU,spacing_up = ri.upsample_image_by_factor(ID, spacing_down)

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

def test_me_2():
    """
    Convenience testing function (to be converted to a test)
    """
    I = Variable(torch.zeros([1, 1, 100, 100]), requires_grad=False)
    I[0, 0, 30:70, 30:70] = 1

    ri = ResampleImage()
    desiredSizeDown = np.array([1,1,32,78])
    desiredSizeUp = np.array([1,1,100,100])

    spacing = np.array([1.,1.])
    ID, spacing_down = ri.downsample_image_to_size(I, spacing, desiredSizeDown)
    IU, spacing_up = ri.upsample_image_to_size(ID, spacing_down, desiredSizeUp)

    print(I.size())
    print(ID.size())
    print(IU.size())
    print(spacing)
    print(spacing_down)
    print(spacing_up)

    plt.subplot(131)
    plt.imshow(I[0, 0, :, :].data.numpy().squeeze())

    plt.subplot(132)
    plt.imshow(ID[0, 0, :, :].data.numpy().squeeze())

    plt.subplot(133)
    plt.imshow(IU[0, 0, :, :].data.numpy().squeeze())

    plt.show()