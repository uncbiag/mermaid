"""
Helper functions to take care of all the file IO
"""

import itk
import os
import nrrd
import utils
import torch
import pyreg.image_manipulations as IM
import numpy as np

from abc import ABCMeta, abstractmethod

class FileIO(object):
    """
    Abstract base class for file i/o.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Constructor
        """

    # check if we are dealing with a nrrd file
    def _is_nrrd_filename(self,filename):
        sf = os.path.splitext(filename)
        ext = sf[1].lower()

        if ext == '.nrrd':
            return True
        elif ext == '.nhdr':
            return True
        else:
            return False

    def _convert_itk_vector_to_numpy(self,v):
        return itk.GetArrayFromVnlVector(v.Get_vnl_vector())

    def _convert_itk_matrix_to_numpy(self,M):
        return itk.GetArrayFromVnlMatrix(M.GetVnlMatrix().as_matrix())

    def _convert_data_to_numpy_if_needed(self,data):
        if ( type( data ) == torch.autograd.variable.Variable ) or \
                (type(data) == torch.torch.nn.parameter.Parameter) or \
                (type(data) == torch.FloatTensor) or \
                (type(data) == torch.DoubleTensor) or \
                (type(data) == torch.HalfTensor) or \
                (type(data) == torch.ByteTensor) or \
                (type(data) == torch.CharTensor) or \
                (type(data) == torch.ShortTensor) or \
                (type(data) == torch.IntTensor) or \
                (type(data) == torch.LongTensor) or \
                (type(data) == torch.cuda.FloatTensor) or \
                (type(data) == torch.cuda.DoubleTensor) or \
                (type(data) == torch.cuda.HalfTensor) or \
                (type(data) == torch.cuda.ByteTensor) or \
                (type(data) == torch.cuda.CharTensor) or \
                (type(data) == torch.cuda.ShortTensor) or \
                (type(data) == torch.cuda.IntTensor) or \
                (type(data) == torch.cuda.LongTensor):
            return utils.t2np(data)
        else:
            return data

    @abstractmethod
    def read(self, filename):
        """
        Abstract method to read a file

        :param filename: file to be read 
        :return: Will return the read file and its header information (as a tuple; im,hdr)
        """
        pass

    @abstractmethod
    def write(self, filename, data, hdr):
        """
        Abstract method to write a file
        
        :param filename: filename to write the data to
        :param data: data array that should be written (will be converted to numpy on the fly if necessary) 
        :param hdr: hdr information for the file (in form of a dictionary)
        :return: n/a
        """
        pass

class ImageIO(FileIO):
    """
    Class to read images
    """
    def __init__(self):
        super(ImageIO, self).__init__()
        self.intensity_normalize_image = False
        """Intensity normalizes an image after reading (default: False)"""
        self.squeeze_image = False
        """squeezes the image when reading; e.g., dimension 1x256x256 becomes 256x256"""
        self.adaptive_padding = -1
        """ padding the img to favorable size default img.shape%adaptive_padding = 0"""

    def turn_intensity_normalization_on(self):
        """
        Turns on intensity normalization on when loading an image
        """
        self.intensity_normalize_image = True

    def turn_intensity_normalization_off(self):
        """
        Turns on intensity normalization off when loading an image
        """
        self.intensity_normalize_image = False

    def set_intensity_normalization(self, int_norm):
        """
        Set if intensity normalization should be on (True) or off (False)

        :param int_norm: image intensity normalization on (True) or off (False) 
        """
        self.intensity_normalize_image = int_norm

    def set_adaptive_padding(self, adaptive_padding):
        """
        if adaptive_padding != -1 adaptive padding on
        padding the img to favorable size  e.g  img.shape%adaptive_padding = 0
        padding size should be bigger than 3, to avoid confused with channel
        :param adaptive_padding:
        :return:
        """
        if adaptive_padding<4:
            raise ValueError,"may confused with channel, adaptive padding must bigger than 4"
        self.adaptive_padding = adaptive_padding

    def get_intensity_normalization(self):
        """
        Returns if image will be intensity normalized when loading

        :return: Returns True if image will be intensity normalized when loading
        """
        return self.intensity_normalize_image

    def turn_squeeze_image_on(self):
        """
        Squeezes the image when loading
        """
        self.squeeze_image = True

    def turn_squeeze_image_off(self):
        """
        Does not squeeze image when loading
        """
        self.squeeze_image = False

    def set_squeeze_image(self, squeeze_im):
        """
        Set if image should be squeezed (True) or not (False)

        :param squeeze_im: squeeze image on (True) or off (False) 
        """
        self.squeeze_image = squeeze_im

    def get_squeeze_image(self):
        """
        Returns if image will be squeezed when loading

        :return: Returns True if image will be squeezed when loading
        """
        return self.squeeze_image


    def _compute_squeezed_spacing(self,spacing0, dim0, sz0, dimSqueezed):
        """
        Extracts the spacing information for non-trivial dimensions (i.e., with more than one entry)
        :param spacing0: original spacing information
        :param dim0: original dimension
        :param sz0: original size
        :param dimSqueezed: dimension after squeezing
        :return: returns only the spacing information for the dimensions with more than one entry
        """
        spacing = np.zeros(dimSqueezed)
        j = 0
        for i in range(dim0):
            if sz0[i] != 1:
                spacing[j] = spacing0[i]
                j += 1
        return spacing

    def _transform_image_to_NC_image_format(self, I):
        '''
        Takes an input image and returns it in the format which is typical for torch.
        I.e., two dimensions are added (the first one for number of images and the second for the 
        number of channels). As were are dealing with individual single-channel intensity images here, these
        dimensions will be 1x1

        :param I: input image of size, sz
        :return: input image, reshaped to size [1,1] + sz
        '''
        return I.reshape([1, 1] + list(I.shape))

    def _try_fixing_image_dimension(self, im, map):

        im_fixed = None  # default, if we cannot fix it

        # try to detect cases such as 128x128x1 and convert them to 1x1x128x128
        si = im.shape
        sm = map.shape

        # get the image dimension from the map (always has to be the second entry)
        dim = sm[1]

        if len(si) != len(sm):
            # most likely we are not dealing with a batch of images and have a dimension that needs to be removed
            im_s = im.squeeze()
            dim_s = len(im_s.shape)
            if dim_s == dim:
                # we can likely fix it, because this is an individual image
                im_fixed = self._transform_image_to_NC_image_format(im_s)
                print('Attempted to fix image dimensions for compatibility with map.')
                print('Modified shape from  ' + str(si) + ' to ' + str(im_fixed.shape))

        return im_fixed

    def _map_is_compatible_with_image(self,im, map):
        si = im.shape
        sm = map.shape

        if len(si) != len(sm):
            return False
        else:
            if si[0] != sm[0]:
                return False
            else:
                for i in range(2, len(si)):
                    if si[i] != sm[i]:
                        return False
        return True

    def _convert_itk_image_to_numpy(self,I0_itk):
        I0 = itk.GetArrayViewFromImage(I0_itk)
        image_meta_data = dict()
        image_meta_data['space origin'] = self._convert_itk_vector_to_numpy(I0_itk.GetOrigin())
        image_meta_data['spacing'] = self._convert_itk_vector_to_numpy(I0_itk.GetSpacing())
        image_meta_data['space directions'] = self._convert_itk_matrix_to_numpy(I0_itk.GetDirection())
        image_meta_data['sizes'] = I0.shape
        image_meta_data['dimension'] = I0_itk.GetImageDimension()
        image_meta_data['space'] = 'left-posterior-superior'
        """
        NRRD format
        {u'dimension': 3,
         u'encoding': 'gzip',
         u'endian': 'little',
         u'keyvaluepairs': {},
         u'kinds': ['domain', 'domain', 'domain'],
         u'sizes': [128, 128, 1],
         u'space': 'left-posterior-superior',
         u'space directions': [['2', '0', '0'], ['0', '2', '0'], ['0', '0', '2']],
         u'space origin': ['0', '0', '0'],
         u'type': 'float'}
        """
        return I0, image_meta_data

    def _do_adaptive_padding(self, im):
        im_sz = list(im.shape)
        dim = len(im_sz)
        dim_to_pad = [dim_sz%self.adaptive_padding!=0 and dim_sz>3 for dim_sz in im_sz]
        dim_rem = [dim_sz//self.adaptive_padding for dim_sz in im_sz]
        new_dim_sz = [(dim_rem[i]+1)*self.adaptive_padding if dim_to_pad[i] else im_sz[i] for i in range(dim)]
        before_id = [(new_dim_sz[i] -im_sz[i]+1)//2 for i in range(dim)]
        after_id = [new_dim_sz[i] - im_sz[i] - before_id[i] for i in range(dim)]
        padding_loc = tuple([(before_id[i],after_id[i]) for i in range(dim)])
        new_img = np.lib.pad(im, padding_loc, 'edge')
        return new_img

    def read(self, filename, intensity_normalize=False, squeeze_image=False, adaptive_padding=-1, verbose=False):
        """
        Reads the image assuming and converts it to NxCxXxYxC format if needed 
        :param filename: filename to be read
        :param intensity_normalize: uses image intensity normalization
        :param squeeze_image: squeezes image first (e.g, from 1x128x128 to 128x128)
        :return: Will return the read file, its header information, the spacing, and the normalized spacing \
         (as a tuple: im,hdr,spacing,normalized_spacing)
        """
        self.set_intensity_normalization(intensity_normalize)
        self.set_squeeze_image(squeeze_image)
        self.set_adaptive_padding(adaptive_padding)
        if verbose:
            print('Reading image: ' + filename)

        if self._is_nrrd_filename(filename):
            # load with the dedicated nrrd reader (can also read higher dimensional files)
            im, hdr = nrrd.read(filename)
        else:
            # read with the itk reader (can also read other file formats)
            im_itk = itk.imread(filename)
            im, hdr = self._convert_itk_image_to_numpy(im_itk)


        if not hdr.has_key('spacing'):
            print('Image does not seem to have spacing information.')
            if hdr.has_key('sizes'):
                dim_guess = len( hdr['sizes'] )
            else:
                dim_guess = len( im.shape )
            print('Guessed dimension to be dim = ' + str( dim_guess ))
            spacing = np.ones( dim_guess )
            hdr['spacing'] = spacing
            print('Using guessed spacing of ' + str(spacing))

        spacing = hdr['spacing']
        normalized_spacing = spacing # will be changed if image is squeezed

        if self.squeeze_image==True:
            if verbose:
                print('Squeezing image')
            dim = len(im.shape)
            sz = im.shape
            im = im.squeeze()
            dimSqueezed = len(im.shape)
            sz_squeezed = im.shape
            if dim!=dimSqueezed:
                if verbose:
                    print('Squeezing changed dimension from ' + str(dim) + ' -> ' + str(dimSqueezed))
            squeezed_spacing = self._compute_squeezed_spacing(spacing,dim,sz,dimSqueezed)
            if verbose:
                print( 'squeezed_spacing = ' + str(squeezed_spacing))
            normalized_spacing = squeezed_spacing / (np.array(sz_squeezed) - 1)
            if verbose:
                print('Normalized spacing = ' + str(normalized_spacing))

        if adaptive_padding>0:
            im = self._do_adaptive_padding(im)

        if self.intensity_normalize_image==True:
            im = IM.IntensityNormalizeImage().defaultIntensityNormalization(im)
        else:
            print('WARNING: Image was NOT intensity normalized when loading.')

        return im,hdr,spacing,normalized_spacing

    def read_to_nc_format(self,filename,intensity_normalize=False,squeeze_image=False ):
        """
        Reads the image assuming it is single channel and of XxYxZ format and convert it to NxCxXxYxC format 
        :param filename: filename to be read
        :param squeeze_image: squeezes image first (e.g, from 1x128x128 to 128x128)
        :return: Will return the read file, its header information, the spacing, and the normalized spacing \
         (as a tuple: im,hdr,spacing,normalized_spacing)
        """
        im,hdr,spacing,normalized_spacing = self.read(filename,intensity_normalize,squeeze_image)
        im = self._transform_image_to_NC_image_format(im)
        return im,hdr,spacing,normalized_spacing

    def read_to_map_compatible_format(self,filename,map,intensity_normalize=False,squeeze_image=False):
        """
        Reads the image and makes sure it is compatiblle with the map. If it is not it tries to fix the format.
        :param filename: filename to be read
        :param map: map which is used to determine the format
        :return: Will return the read file, its header information, the spacing, and the normalized spacing \
         (as a tuple: im,hdr,spacing,normalized_spacing)
        """

        if map is None:
            print('Map needs to be specified. Currently set to None. Aborting.')
            return None,None,None,None

        im,hdr,spacing,normalized_spacing = self.read(filename,intensity_normalize,squeeze_image)
        if not self._map_is_compatible_with_image(im, map):
            im_fixed = self._try_fixing_image_dimension(im, map)

            if im_fixed is None:
                print('Cannot apply map to image due to dimension mismatch')
                print('Attempt at automatically fixing dimensions failed')
                print('Image dimension:')
                print(im.shape)
                print('Map dimension:')
                print(map.shape)
                return None,None.None,None
            else:
                im = im_fixed

        return im,hdr,spacing,normalized_spacing

    def write(self, filename, data, hdr):
        if not self._is_nrrd_filename(filename):
            print('Sorry, currently only nrrd files are supported as output. Aborting.')
            return
        # now write it out
        print('Writing image: ' + filename)
        nrrd.write(filename, self._convert_data_to_numpy_if_needed( data ), hdr)


class GenericIO(FileIO):
    """
    Class to read images
    """

    def __init__(self):
        super(GenericIO, self).__init__()

    def read(self, filename):
        if not self._is_nrrd_filename(filename):
            print('Sorry, currently only nrrd files are supported when reading. Aborting.')
            return None, None
        else:
            print('Reading: ' + filename)
            map, map_hdr = nrrd.read(filename)
            return map, map_hdr

    def write(self, filename, data, hdr):
        if not self._is_nrrd_filename(filename):
            print('Sorry, currently only nrrd files are supported when writing. Aborting.')
            return
        else:
            print('Writing: ' + filename)
            nrrd.write(filename, self._convert_data_to_numpy_if_needed( data ), hdr)


class MapIO(GenericIO):
    """
    Class to read images
    """

    def __init__(self):
        super(MapIO, self).__init__()




