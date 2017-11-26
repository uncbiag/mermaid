"""
Applying a pre-computed map to warp an image.
Contributors:
  Marc Niethammer: mn@cs.unc.edu
"""

import set_pyreg_paths
import torch
from torch.autograd import Variable
import pyreg.utils as utils
import nrrd
import itk
import os
from pyreg.data_wrapper import AdaptVal

# check if we are dealing with a nrrd file
def is_nrrd_filename( filename ):
    sf = os.path.splitext( filename )
    ext = sf[1].lower()

    if ext=='.nrrd':
        return True
    elif ext=='.nhdr':
        return True
    else:
        return False

def try_fixing_image_dimension(im,map):

    im_fixed = None # default, if we cannot fix it

    # try to detect cases such as 128x128x1 and convert them to 1x1x128x128
    si = im.shape
    sm = map.shape

    # get the image dimension from the map (always has to be the second entry)
    dim = sm[1]

    if len(si)!=len(sm):
        # most likely we are not dealing with a batch of images and have a dimension that needs to be removed
        im_s = im.squeeze()
        dim_s = len( im_s.shape )
        if dim_s==dim:
            # we can likely fix it, because this is an individual image
            print('Attempted to fix image dimensions for compatibility with map.')
            im_fixed = utils.transform_image_to_NC_image_format(im_s)

    return im_fixed

def map_is_compatible_with_image(im,map):
    si = im.shape
    sm = map.shape

    if len(si)!=len(sm):
        return False
    else:
        if si[0]!=sm[0]:
            return False
        else:
            for i in range(2,len(si)):
                if si[i]!=sm[i]:
                    return False
    return True

def read_image_and_map_and_apply_map(image_filename,map_filename):

    im_warped = None
    if not is_nrrd_filename(map_filename):
        print('Sorry, currently only nrrd files are supported for maps. Aborting.')
        return im_warped

    map, map_hdr = nrrd.read(map_filename)

    if is_nrrd_filename( image_filename ):
        # load with the dedicated nrrd reader (can also read higher dimensional files)
        im,hdr = nrrd.read(image_filename)
    else:
        # read with the itk reader (can also read other file formats)
        im_itk = itk.imread(image_filename)
        im,hdr = utils.convert_itk_image_to_numpy(im_itk)

    if not map_is_compatible_with_image(im,map):
        im_fixed = try_fixing_image_dimension(im,map)

        if im_fixed is None:
            print('Cannot apply map to image due to dimension mismatch')
            print('Attempt at automatically fixing dimensions failed')
            print('Image dimension:')
            print(im.shape)
            print('Map dimension:')
            print(map.shape)
            return im_warped
        else:
            im = im_fixed

    # make pytorch arrays for subsequent processing
    im_t = AdaptVal(Variable(torch.from_numpy(im), requires_grad=False))
    map_t = AdaptVal(Variable(torch.from_numpy(map), requires_grad=False))
    im_warped = utils.t2np( utils.compute_warped_image_multiNC(im_t,map_t) )

    return im_warped,hdr


if __name__ == "__main__":
    # execute this as a script

    import argparse

    parser = argparse.ArgumentParser(description='Apply map to warp image')

    required = parser.add_argument_group('required arguments')
    required.add_argument('--map', required=True, help='Map that should be applied [need to be in [0,1]^d format currently]')
    required.add_argument('--image', required=True, help='Image to which the map should be applied')
    required.add_argument('--warped_image', required=True, help='Warped image after applying the map')

    args = parser.parse_args()
    image_filename = args.image
    map_filename = args.map
    im_warped_filename = args.warped_image

    if not is_nrrd_filename(im_warped_filename):
        print('Sorry, currently only nrrd files are supported as output. Aborting.')

    im_warped,hdr = read_image_and_map_and_apply_map(image_filename, map_filename)

    # now write it out
    print( 'Writing warped image to file: ' + im_warped_filename )
    nrrd.write(im_warped_filename, im_warped, hdr)