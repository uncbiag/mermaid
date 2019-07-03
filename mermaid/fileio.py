"""
Helper functions to take care of all the file IO
"""
from __future__ import print_function
from __future__ import absolute_import

from future.utils import native_str

# from builtins import str
# from builtins import range
# from builtins import object
import itk
# needs to be imported after itk to overwrite itk's incorrect error handling
from . import fixwarnings

import os
import nrrd
from . import utils
import torch
from . import image_manipulations as IM
import numpy as np
import glob

import copy

from .config_parser import USE_FLOAT16

from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass


class FileIO(with_metaclass(ABCMeta, object)):
    """
    Abstract base class for file i/o.
    """

    def __init__(self):
        """
        Constructor
        """
        if USE_FLOAT16:
            self.default_datatype = 'float16'
        else:
            self.default_datatype = 'float32'

        self.datatype_conversion = True
        """Automatically convers the datatype to the default_data_type when loading or writing"""

        self.replace_nans_with_zeros = True
        """If NaNs are detected they are automatically replaced by zeroes"""

    def turn_nan_to_zero_conversion_on(self):
        self.replace_nans_with_zeros = True

    def turn_nan_to_zero_conversion_off(self):
        self.replace_nans_with_zeros = False

    def is_turn_nan_to_zero_conversion_on(self):
        return self.replace_nans_with_zeros

    def turn_datatype_conversion_on(self):
        """
        Turns the automatic datatype conversion on
        :return: n/a
        """
        self.datatype_conversion = True

    def turn_datatype_conversion_off(self):
        """
        Turns the automatic datatype conversion off
        :return: n/a
        """
        self.datatype_conversion = False

    def is_datatype_conversion_on(self):
        """
        Returns True if automatic datatype conversion is on (default), otherwise False
        :return: True if dataype conversion is on otherwise False
        """
        return self.datatype_conversion

    def set_default_datatype(self,dtype):
        """
        Sets the default data type that will be used if data_type_conversion is on
        :param dtype: standard dataype (e.g., 'float16', 'float32')
        :return: n/a
        """
        self.default_datatype = dtype

    def get_default_datatype(self):
        """
        Get the default data type
        :return: default data type
        """
        return self.default_datatype

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
        return itk.GetArrayFromVnlVector(v.GetVnlVector())

    def _convert_itk_matrix_to_numpy(self,M):
        return itk.GetArrayFromVnlMatrix(M.GetVnlMatrix().as_matrix())

    def _convert_numpy_vector_to_itk(self,v):
        v_itk = itk.Vector[itk.D, len(v)]()
        vnl_vec = itk.PyVnl[itk.D].GetVnlVectorFromArray(v.astype('float64'))
        v_itk.SetVnlVector(vnl_vec)
        return v_itk

    def _convert_numpy_matrix_to_itk(self,M):
        s = M.shape
        vnl_mat = itk.PyVnl[itk.D].GetVnlMatrixFromArray(M.astype('float64'))
        m_itk = itk.Matrix[itk.D,s[0],s[1]](vnl_mat)
        return m_itk

    def _convert_data_to_numpy_if_needed(self,data):
        if type( data ) == torch.Tensor:
            datar = utils.t2np(data)
        else:
            datar = data

        if self.datatype_conversion:
            return datar.astype(self.default_datatype)
        else:
            return datar

    @abstractmethod
    def read(self, filename):
        """
        Abstract method to read a file

        :param filename: file to be read 
        :return: Will return the read file and its header information (as a tuple; im,hdr)
        """
        pass

    @abstractmethod
    def write(self, filename, data, hdr=None):
        """
        Abstract method to write a file
        
        :param filename: filename to write the data to
        :param data: data array that should be written (will be converted to numpy on the fly if necessary) 
        :param hdr: optional header information for the file (in form of a dictionary)
        :return: n/a
        """
        pass

class ImageIO(FileIO):
    """
    Class to read images. All the image reading should be performed via this class. In this way everything
    will be read and processed consistently.
    """
    def __init__(self):
        super(ImageIO, self).__init__()
        self.intensity_normalize_image = False
        """Intensity normalizes an image after reading (default: False)"""
        self.squeeze_image = False
        """squeezes the image when reading; e.g., dimension 1x256x256 becomes 256x256"""
        self.adaptive_padding = -1
        """ padding the img to favorable size default img.shape%adaptive_padding = 0"""
        self.normalize_spacing = True
        """normalized spacing so that the aspect ratio remains and the largest extent is in [0,1]"""
        self.scale_vectors_on_read_and_write = True
        """
        When writing vector fields (for example maps), the vectors in the field are scaled back to original world coordinates.
        When reading they are scaled if the spacing is being normalized. Should be turned off when trying to read or write 
        vector-valued images that do not represent maps or displacement fields.
        """

    def turn_scale_vectors_on_read_and_write_on(self):
        self.scale_vectors_on_read_and_write = True

    def turn_scale_vectors_on_read_and_write_off(self):
        self.scale_vectors_on_read_and_write = False

    def set_scale_vectors_on_read_and_write(self,val):
        self.scale_vectors_on_read_and_write = val

    def get_scale_vectors_on_read_and_write(self):
        return self.scale_vectors_on_read_and_write

    def turn_normalize_spacing_on(self):
        """
        Turns the normalized spacing (to interval [0,1]) on.
        :return: n/a
        """
        self.normalize_spacing = True

    def turn_normalize_spacing_off(self):
        """
        Turns the normalized spacing (to interval [0,1]) off.
        :return: n/a
        """
        self.normalize_spacing = False

    def set_normalize_spacing(self, normalize_spacing):
        """
        Sets if spacing should be normalized [0,1] or not.
        :param norm_spacing: True/False
        :return: n/a
        """
        self.normalize_spacing = normalize_spacing

    def get_normalize_spacing(self):
        """
        Returns the setting if spacing should be normalized [0,1] or not.
        :return: n/a
        """
        return self.normalize_spacing

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
        :return: n/a
        """
        if adaptive_padding<4 and adaptive_padding != -1:
            raise ValueError("may confused with channel, adaptive padding must bigger than 4")
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

    def _normalize_spacing(self,spacing,sz,silent_mode=False):
        """
        Normalizes spacing.
        :param spacing: Vector with spacing info, in XxYxZ format
        :param sz: size vector in XxYxZ format
        :param silent_mode: if True suppresses output
        :return: vector with normalized spacings in XxYxZ format
        """
        dim = len(spacing)
        # first determine the largest extent
        current_largest_extent = -1
        extent = np.zeros_like(spacing)
        for d in range(dim):
            current_extent = spacing[d]*(sz[d]-1)
            extent[d] = current_extent
            if current_extent>current_largest_extent:
                current_largest_extent = current_extent

        scalingFactor = 1./current_largest_extent
        normalized_spacing = spacing*scalingFactor

        normalized_extent = extent*scalingFactor

        if not silent_mode:
            print('Normalize spacing: ' + str(spacing) + ' -> ' + str(normalized_spacing))
            print('Normalize spacing, extent: ' + str(extent) + ' -> ' + str(normalized_extent))

        return normalized_spacing

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

    def _get_scalar_itk_image_from_numpy(self,np_im,hdr=None):
        if hdr is not None:
            if 'sizes' not in hdr:
                raise ValueError('Expected size information in header')
            im = itk.GetImageFromArray(np_im.view().reshape(list(hdr['sizes'])))

            if 'original_spacing' in hdr:
                im.SetSpacing(self._convert_numpy_vector_to_itk(np.flipud(hdr['original_spacing'])))
            else:
                im.SetSpacing(self._convert_numpy_vector_to_itk(np.flipud(hdr['spacing'])))
            im.SetOrigin(self._convert_numpy_vector_to_itk(hdr['space origin']))
            im.SetDirection(self._convert_numpy_matrix_to_itk(hdr['space directions']))
        else:
            im = itk.GetImageFromArray(np_im)

        return im

    def _get_vector_itk_image_from_numpy(self,np_vec_im_in,hdr=None):
        s = np_vec_im_in.shape

        if hdr is not None:
            if 'dimension' in hdr:
                dim = hdr['dimension']
            else:
                dim = len(s)-1
            if 'squeezed_dim' in hdr:
                squeezed_dim = hdr['squeezed_dim']
            else:
                squeezed_dim = dim
        else:
            dim = len(s)-1
            squeezed_dim = dim

        if self.scale_vectors_on_read_and_write:
            # scale it so that we have the original spacing
            print_warning = True
            if hdr is not None:
                if 'vector_was_scaled' in hdr:
                    print_warning = not hdr['vector_was_scaled']
            if print_warning:
                print('WARNING: scaling vector image that may not have been scaled when reading')

            scaling = np.array(hdr['original_spacing']) / np.array(hdr['spacing'])

            np_vec_im = np.copy(np_vec_im_in)
            for d in range(squeezed_dim):
                np_vec_im[d, ...] *= scaling[d]

        else:
            # just keep it as is
            np_vec_im = np_vec_im_in

        nv_vec_im_rf = np.moveaxis(np_vec_im, 0, -1)

        VectorType = itk.Vector[itk.F, dim]
        FieldType = itk.Image[VectorType, dim]

        size = itk.Size[dim]()

        if squeezed_dim!=dim:
            if 'sizes' in hdr:
                current_shape = list(hdr['sizes'])
                nv_vec_im_rf = nv_vec_im_rf.view().reshape(current_shape+[squeezed_dim])
            else:
                raise ValueError('Expected sizes filed in header')

        reverse_shape = list(nv_vec_im_rf.shape[0:-1])
        reverse_shape.reverse()

        for d in range(dim):
            size[d] = reverse_shape[d]

        region = itk.ImageRegion[dim](size)

        vec_im = FieldType.New()
        vec_im.SetRegions(region)
        vec_im.Allocate()

        vec_view_np = itk.GetArrayViewFromImage(vec_im)

        for d in range(squeezed_dim):
            vec_view_np[...,d] = nv_vec_im_rf[...,d]
        for d in range(squeezed_dim,dim):
            vec_view_np[...,d]=0

        if 'original_spacing' in hdr:
            vec_im.SetSpacing(self._convert_numpy_vector_to_itk(hdr['original_spacing']))
        else:
            vec_im.SetSpacing(self._convert_numpy_vector_to_itk(hdr['spacing']))

        vec_im.SetOrigin(self._convert_numpy_vector_to_itk(hdr['space origin']))
        vec_im.SetDirection(self._convert_numpy_matrix_to_itk(hdr['space directions']))

        return vec_im

    def _get_itk_image_from_numpy(self,np_im, hdr=None):

        if hdr is not None:
            if 'squeezed_dim' in hdr:
                dim = hdr['squeezed_dim']
            elif 'dimension' in hdr:
                dim = hdr['dimension']
            else:
                dim = len(np_im.shape)
        else:
            dim = len(np_im.shape)

        if len(np_im.shape)>dim:
            treat_as_scalar_image = False
        else:
            treat_as_scalar_image = True

        if treat_as_scalar_image:
            return self._get_scalar_itk_image_from_numpy(np_im,hdr)
        else:
            return self._get_vector_itk_image_from_numpy(np_im,hdr)

    def _convert_itk_image_to_numpy(self,I0_itk):
        if self.datatype_conversion:
            I0 = itk.GetArrayViewFromImage(I0_itk).astype(self.default_datatype)
        else:
            I0 = itk.GetArrayViewFromImage(I0_itk)

        if len(I0.shape)>I0_itk.GetImageDimension():
            is_vector_image = True
            # itk has a different convention, there the vector component is the last dimension, so we need to swap it
            I0 = np.moveaxis(I0,-1,0)
        else:
            is_vector_image = False

        image_meta_data = dict()
        image_meta_data['space origin'] = self._convert_itk_vector_to_numpy(I0_itk.GetOrigin())
        image_meta_data['spacing'] = self._convert_itk_vector_to_numpy(I0_itk.GetSpacing())
        image_meta_data['space directions'] = self._convert_itk_matrix_to_numpy(I0_itk.GetDirection())

        image_meta_data['dimension'] = I0_itk.GetImageDimension()
        image_meta_data['space'] = 'left-posterior-superior'

        image_meta_data['is_vector_image'] = is_vector_image

        if is_vector_image:
            image_meta_data['sizes'] = I0.shape[1:] # first dimension is vector dimension here
        else:
            image_meta_data['sizes'] = I0.shape


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
        """
        padding the img to favored size, (divided by certain number, here is 4), here using default 4 , favored by cuda fft
        :param im:
        :return:
        """
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

    def read(self, filename, intensity_normalize=False, squeeze_image=False, normalize_spacing=True, adaptive_padding=-1, verbose=False, silent_mode=False):
        """
        Reads the image assuming it is a single channel 

        :param filename: filename to be read
        :param intensity_normalize: uses image intensity normalization
        :param squeeze_image: squeezes image first (e.g, from 1x128x128 to 128x128)
        :param normalize_spacing: normalizes spacing so largest extent is in [0,1]
        :param silent_mode: if True, suppresses output
        :return: Will return the read file, its header information, the spacing, and the normalized spacing \
         (as a tuple: im,hdr,spacing,squeezed_spacing)
        """
        self.set_intensity_normalization(intensity_normalize)
        self.set_squeeze_image(squeeze_image)
        self.set_adaptive_padding(adaptive_padding)
        self.set_normalize_spacing(normalize_spacing)

        if verbose and not silent_mode:
            print('Reading image: ' + filename)

        # read with the itk reader (can also read other file formats)
        im_itk = itk.imread(native_str(filename))
        im, hdr = self._convert_itk_image_to_numpy(im_itk)

        if self.replace_nans_with_zeros:
            im[np.isnan(im)]=0

        if self.datatype_conversion:
            im = im.astype(self.default_datatype)

        if 'spacing' not in hdr:
            if not silent_mode:
                print('Image does not seem to have spacing information.')
            if 'sizes' in hdr:
                dim_guess = len( hdr['sizes'] )
            else:
                dim_guess = len( im.shape )
            if not silent_mode:
                print('Guessed dimension to be dim = ' + str( dim_guess ))
            spacing = np.ones( dim_guess )
            hdr['spacing'] = spacing
            if not silent_mode:
                print('Using guessed spacing of ' + str(spacing))

        spacing = np.flipud( hdr['spacing'])

        squeezed_spacing = spacing # will be changed if image is squeezed
        sz = im.shape
        sz_squeezed = sz

        if self.squeeze_image==True:
            if verbose and not silent_mode:
                print('Squeezing image')
            if hdr['is_vector_image']:
                dim = len(im.shape[1:])
            else:
                dim = len(im.shape)

            im = im.squeeze()

            if hdr['is_vector_image']:
                dimSqueezed = len(im.shape[1:])
            else:
                dimSqueezed = len(im.shape)

            hdr['squeezed_dim'] = dimSqueezed

            sz_squeezed = im.shape
            if dim!=dimSqueezed:
                if verbose and not silent_mode:
                    print('Squeezing changed dimension from ' + str(dim) + ' -> ' + str(dimSqueezed))

            if hdr['is_vector_image']:
                squeezed_spacing = self._compute_squeezed_spacing(spacing, dim, sz[1:], dimSqueezed)
            else:
                squeezed_spacing = self._compute_squeezed_spacing(spacing,dim,sz,dimSqueezed)

            if verbose and not silent_mode:
                print( 'squeezed_spacing = ' + str(squeezed_spacing))

            #squeezed_spacing = squeezed_spacing / (np.array(sz_squeezed) - 1)
            #if verbose and not silent_mode:
            #    print('Normalized spacing = ' + str(squeezed_spacing))

        if adaptive_padding>0:
            im = self._do_adaptive_padding(im)

        if self.intensity_normalize_image==True:
            im = IM.IntensityNormalizeImage().default_intensity_normalization(im)
            if not silent_mode:
                print('INFO: Image WAS intensity normalized when loading:' \
                      + ' [' + str(im.min()) + ',' + str(im.max()) + ']')
        else:
            if not silent_mode:
                print('WARNING: Image was NOT intensity normalized when loading:' \
                      + ' [' + str(im.min()) + ',' + str(im.max()) + ']')




        if self.normalize_spacing:
            if not silent_mode:
                print('INFO: Normalizing the spacing to [0,1] in the largest dimension. (Turn normalize_spacing off if this is not desired.)')
            hdr['original_spacing'] = spacing


            if hdr['is_vector_image']:
                spacing = self._normalize_spacing(spacing, sz[1:], silent_mode)
                squeezed_spacing = self._normalize_spacing(squeezed_spacing, sz_squeezed[1:], silent_mode)
            else:
                spacing = self._normalize_spacing(spacing,sz,silent_mode)
                squeezed_spacing = self._normalize_spacing(squeezed_spacing,sz_squeezed,silent_mode)

            hdr['spacing'] = spacing

            if verbose and not silent_mode:
                print('Normalized spacing = ' + str(spacing))
                print('Normalized squeezed spacing = ' + str(squeezed_spacing))

            if hdr['is_vector_image']:
                if self.scale_vectors_on_read_and_write:
                    hdr['vector_image_was_scaled'] = True
                    if not silent_mode:
                        print('Scaling the vector image to conform to the scaled spacing')
                    # we also need to normalize the vector components in this case
                    vector_scaling = np.array(hdr['spacing']) / np.array(hdr['original_spacing'])
                    for d in range(dim):
                        im[d, ...] *= vector_scaling[d]

        return im,hdr,spacing,squeezed_spacing

    def read_batch_to_nc_format(self,filenames,intensity_normalize=False,squeeze_image=False, normalize_spacing=True, silent_mode=False ):
        """
        Wrapper around read_to_nc_format which allows to read a whole batch of images at once (as specified
        in filenames) and returns the image in format NxCxXxYxZ. An individual image is assumed to have a single intensity channel.

        :param filenames: list of filenames to be read or expression with wildcard
        :param intensity_normalize: if set to True uses image intensity normalization
        :param squeeze_image: squeezed individual image first (e.g, from 1x128x128 to 128x128)
        :param normalize_spacing: normalizes the spacing so the largest extent is [0,1]
        :param silent_mode: if True, suppresses output
        :return Will return the read files, their header information, their spacing, and their normalized spacing \
         (as a tuple: im,hdr,spacing,squeezed_spacing). The assumption is that all files have the same
         header and spacing. So only one is returned for the entire batch.
        """

        ims = None
        hdr = None
        spacing = None
        squeezed_spacing = None

        if type(filenames)!=list:
            # this is a glob expression
            filenames = glob.glob(filenames)

        nr_of_files = len(filenames)

        for counter,filename in enumerate(filenames):
            if not os.path.isfile(filename):
                raise ValueError( 'File: ' + filename + ' does not exist.')
            if counter==0:
                # simply load the file (this will determine the headers size and dimension)
                im,hdr,spacing,squeezed_spacing = self.read_to_nc_format(filename,
                                                                         intensity_normalize=intensity_normalize,
                                                                         squeeze_image=squeeze_image,
                                                                         normalize_spacing=normalize_spacing,
                                                                         silent_mode=silent_mode)
                sz = list(im.shape)
                sz[0] = nr_of_files
                if not silent_mode:
                    print('Size:')
                    print(sz)
                ims = np.zeros(sz,dtype=im.dtype)
                ims[0,...] = im
            else:
                im, _, _, _ = self.read_to_nc_format(filename,
                                                     intensity_normalize=intensity_normalize,
                                                     squeeze_image=squeeze_image,
                                                     normalize_spacing=normalize_spacing,
                                                     silent_mode=silent_mode)
                ims[counter,...] = im

        return ims, hdr, spacing, squeezed_spacing

    def read_to_nc_format(self,filename,intensity_normalize=False,squeeze_image=False,normalize_spacing=True, silent_mode=False ):
        """
        Reads the image assuming it is single channel and of XxYxZ format and convert it to NxCxXxYxC format 

        :param filename: filename to be read
        :param intensity_normalize: if set to True uses image intensity normalization
        :param squeeze_image: squeezes image first (e.g, from 1x128x128 to 128x128)
        :param normalize_spacing: normalizes the spacing to [0,1] in largest dimension
        :param silent_mode: if True, suppresses output
        :return: Will return the read file, its header information, the spacing, and the normalized spacing \
         (as a tuple: im,hdr,spacing,squeezed_spacing)
        """
        im,hdr,spacing,squeezed_spacing = self.read(filename,intensity_normalize,squeeze_image,normalize_spacing,silent_mode=silent_mode)
        im = self._transform_image_to_NC_image_format(im)
        return im,hdr,spacing,squeezed_spacing

    def read_to_map_compatible_format(self,filename,map,intensity_normalize=False,squeeze_image=False,normalize_spacing=True):
        """
        Reads the image and makes sure it is compatible with the map. If it is not it tries to fix the format.

        :param filename: filename to be read
        :param map: map which is used to determine the format
        :param intensity_normalize: if set to True uses image intensity normalization
        :param squeeze_image: squeezes image first (e.g, from 1x128x128 to 128x128)
        :param normalize_spacing: normalizes the spacing to [0,1] in largest dimension
        :return: Will return the read file, its header information, the spacing, and the normalized spacing \
         (as a tuple: im,hdr,spacing,squeezed_spacing)
        """

        if map is None:
            print('Map needs to be specified. Currently set to None. Aborting.')
            return None,None,None,None

        im,hdr,spacing,squeezed_spacing = self.read(filename,intensity_normalize,squeeze_image,normalize_spacing)
        if not self._map_is_compatible_with_image(im, map):
            im_fixed = self._try_fixing_image_dimension(im, map)

            if im_fixed is None:
                print('Cannot apply map to image due to dimension mismatch')
                print('Attempt at automatically fixing dimensions failed')
                print('Image dimension:')
                print(im.shape)
                print('Map dimension:')
                print(map.shape)
                return None,None,None,None
            else:
                im = im_fixed

        return im,hdr,spacing,squeezed_spacing

    def write_batch_to_individual_files(self,filenames,data,hdr=None):
        """
        Takes a batch of images in the NxCxXxYxZ format and writes them out as individual files.
        Currently an image can only have one channel, i,e., C=1.
        :param filenames: either a list of filenames (one for each N) or one filename which will then be written out with different indices.
        :param data: image data in NxCxXxYxZ format 
        :param hdr: optional hrd, all images will get the same
        :return: n/a
        """

        npd = self._convert_data_to_numpy_if_needed(data)
        sz = npd.shape
        nr_of_images = sz[0]
        nr_of_channels = sz[1]

        if type(filenames)==list:
            nr_of_filenames = len(filenames)
            if nr_of_filenames!=nr_of_images:
                raise ValueError('Error: a filename needs to be specified for each image in the batch')
            # filenames were specified separately
            for counter,filename in enumerate(filenames):
                if nr_of_channels==1: # this is a scalar image
                    self.write(filename,npd[counter,0,...].squeeze(),hdr)
                else: # this is a vector image
                    self.write(filename, npd[counter, ...], hdr)
        else:
            # there is one filename specified as a pattern
            filenamepattern, file_extension = os.path.splitext(filenames)
            for counter in range(nr_of_images):
                current_filename = filenamepattern + '_' + str(counter).zfill(4) + file_extension
                if nr_of_channels==1: # this is a scalar image
                    self.write(current_filename,npd[counter,0,...].squeeze(),hdr)
                else: # this is a vector image
                    self.write(current_filename,npd[counter,...].squeeze(),hdr)

    def write(self, filename, data, hdr=None):

       # now write it out
        print('Writing: ' + filename)
        npd = self._convert_data_to_numpy_if_needed(data)
        data_itk = self._get_itk_image_from_numpy(npd, hdr)
        itk.imwrite(data_itk, native_str(filename))


class GenericIO(FileIO):
    """
    Generic class to read nrrd images. More or less legacy. Use should be avoided if at all possible
    """

    def __init__(self):
        super(GenericIO, self).__init__()

    def read(self, filename):
        if not self._is_nrrd_filename(filename):
            print('Sorry, currently only nrrd files are supported when reading. Aborting.')
            return None, None
        else:
            print('Reading: ' + filename)
            data, data_hdr = nrrd.read(filename)
            if self.replace_nans_with_zeros:
                data[np.isnan(data)]=0
            if self.datatype_conversion:
                data = data.astype(self.default_datatype)
            return data, data_hdr

    def write(self, filename, data, hdr=None):
        if not self._is_nrrd_filename(filename):
            print('Sorry, currently only nrrd files are supported when writing. Aborting.')
            return
        else:
            print('Writing: ' + filename)
            if hdr is not None:
                hdr_modified = copy.deepcopy(hdr)
                if 'original_spacing' in hdr_modified:
                    hdr_modified['spacing'] = hdr_modified['original_spacing']
                    hdr_modified.pop('original_spacing')
                    if 'is_vector_image' in hdr_modified:
                        hdr_modified.pop('is_vector_image')
                    if 'vector_image_was_scaled' in hdr_modified:
                        hdr_modified.pop('vector_image_was_scaled')
                if 'squeezed_dim' in hdr_modified:
                    hdr_modified.pop('squeezed_dim')
                nrrd.write(filename, self._convert_data_to_numpy_if_needed( data ), hdr_modified)
            else:
                nrrd.write(filename, self._convert_data_to_numpy_if_needed( data ) )

class MapIO(ImageIO):
    """
    To read and write maps or displacement fields.
    """

    def __init__(self):
        super(MapIO, self).__init__()

    def read_from_validation_map_format(self, filename):
        current_scale_mode = self.get_scale_vectors_on_read_and_write()
        self.turn_scale_vectors_on_read_and_write_off()

        # now that we are sure this is a map (or dispalcement field), let's write it with the standard image IO
        im, hdr, spacing, squeezed_spacing = super(MapIO, self).read(filename,silent_mode=True,squeeze_image=True)

        self.set_scale_vectors_on_read_and_write(current_scale_mode)

        return im, hdr, spacing, squeezed_spacing


    def write(self, filename, data, hdr):

        if hdr is None:
            raise ValueError('hdr needs to be specified to keep track of spacing')

        if 'squeezed_dim' in hdr:
            dim = hdr['squeezed_dim']
        else:
            dim = hdr['dimension']

        data_new = copy.deepcopy(self._convert_data_to_numpy_if_needed(data))
        sz = data_new.shape

        if len(sz) != dim + 1:
            raise ValueError('Expected a dim x X x Y x Z format; i.e., cannot write an entire batch at once')

        nr_of_dim = sz[0]
        if nr_of_dim > 3:
            raise ValueError(
                'Only dimensions up to 3 are supported. Make sure the data is in dim x X x Y x Z format')

        # now that we are sure this is a map (or dispalcement field), let's write it with the standard image IO
        super(MapIO, self).write(filename,data,hdr)

    def write_to_validation_map_format(self, filename, data, hdr):

        if hdr is None:
            raise ValueError('hdr needs to be specified to keep track of spacing')

        if 'squeezed_dim' in hdr:
            dim = hdr['squeezed_dim']
        else:
            dim = hdr['dimension']

        data_new = copy.deepcopy(self._convert_data_to_numpy_if_needed(data))
        sz = data_new.shape

        if len(sz)!=dim+1:
            raise ValueError('Expected a dim x X x Y x Z format; i.e., cannot write an entire batch at once')

        nr_of_dim = sz[0]
        if nr_of_dim > 3:
            raise ValueError(
                'Only dimensions up to 3 are supported. Make sure the data is in dim x X x Y x Z format')

        # since this is map we need to convert the values so we get [0,szx-1]x[0,szy-1]x[0,szz-1] format
        for d in range(nr_of_dim):
            data_new[d, ...] = data_new[d, ...] / hdr['spacing'][d]

        hdr_modified = copy.deepcopy(hdr)
        if 'original_spacing' in hdr_modified:
            hdr_modified.pop('original_spacing')
        if 'is_vector_image' in hdr_modified:
            hdr_modified.pop('is_vector_image')
        if 'vector_image_was_scaled' in hdr_modified:
            hdr_modified.pop('vector_image_was_scaled')

        #hdr_modified['spacing'] = [1] * dim # keep the original spacing; just need to be aware of the convention

        current_scale_mode = self.get_scale_vectors_on_read_and_write()
        self.turn_scale_vectors_on_read_and_write_off()

        # now that we are sure this is a map (or dispalcement field), let's write it with the standard image IO
        super(MapIO, self).write(filename, data_new, hdr_modified)

        self.set_scale_vectors_on_read_and_write(current_scale_mode)

