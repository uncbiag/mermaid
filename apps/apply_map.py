"""
Applying a pre-computed map to warp an image.
Contributors:
  Marc Niethammer: mn@cs.unc.edu
"""
from __future__ import print_function

import torch
from torch.autograd import Variable
import mermaid.utils as utils
from mermaid.data_wrapper import AdaptVal
import mermaid.fileio as fileio

def read_image_and_map_and_apply_map(image_filename,map_filename):
    """
    Reads an image and a map and applies the map to an image
    :param image_filename: input image filename
    :param map_filename: input map filename
    :return: the warped image and its image header as a tupe (im,hdr)
    """

    im_warped = None
    map,map_hdr = fileio.MapIO().read(map_filename)
    im,hdr,_,_ = fileio.ImageIO().read_to_map_compatible_format(image_filename,map)

    spacing = hdr['spacing']
    #TODO: check that the spacing is compatible with the map

    if (im is not None) and (map is not None):
        # make pytorch arrays for subsequent processing
        im_t = AdaptVal(torch.from_numpy(im))
        map_t = AdaptVal(torch.from_numpy(map))
        im_warped = utils.t2np( utils.compute_warped_image_multiNC(im_t,map_t,spacing) )

        return im_warped,hdr
    else:
        print('Could not read map or image')
        return None,None

if __name__ == "__main__":
    # execute this as a script

    import argparse

    print('WARNING: TODO: need to add support for different spline orders for image warping!! (I.e., support for params for compute_warped_image_multiNC')

    parser = argparse.ArgumentParser(description='Apply map to warp image')

    required = parser.add_argument_group('required arguments')
    required.add_argument('--map', required=True, help='Map that should be applied [need to be in [-1,1]^d format currently]')
    required.add_argument('--image', required=True, help='Image to which the map should be applied')
    required.add_argument('--warped_image', required=True, help='Warped image after applying the map')

    args = parser.parse_args()
    image_filename = args.image
    map_filename = args.map
    im_warped_filename = args.warped_image

    im_warped,hdr = read_image_and_map_and_apply_map(image_filename, map_filename)

    # now write it out
    print( 'Writing warped image to file: ' + im_warped_filename )
    fileio.ImageIO().write(im_warped_filename, im_warped, hdr)
