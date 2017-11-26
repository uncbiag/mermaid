"""
Applying a pre-computed map to warp an image.
Contributors:
  Marc Niethammer: mn@cs.unc.edu
"""

import set_pyreg_paths
import torch
from torch.autograd import Variable
import pyreg.utils as utils
from pyreg.data_wrapper import AdaptVal
import pyreg.fileio as fileio

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

    if (im is not None) and (map is not None):
        # make pytorch arrays for subsequent processing
        im_t = AdaptVal(Variable(torch.from_numpy(im), requires_grad=False))
        map_t = AdaptVal(Variable(torch.from_numpy(map), requires_grad=False))
        im_warped = utils.t2np( utils.compute_warped_image_multiNC(im_t,map_t) )

        return im_warped,hdr
    else:
        print('Could not read map or image')
        return None,None

if __name__ == "__main__":
    # execute this as a script

    import argparse

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