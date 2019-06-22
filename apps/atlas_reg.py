"""
Building an atlas from given images.
Contributors:
  Marc Niethammer: mn@cs.unc.edu
"""
from __future__ import print_function

from builtins import str
from builtins import range

# first do the torch imports
import torch
from torch.autograd import Variable
from mermaid.data_wrapper import AdaptVal
import mermaid.fileio as FIO
import mermaid.simple_interface as SI

import matplotlib.pyplot as plt
import os
import fnmatch

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def compute_average_image(images):
    im_io = FIO.ImageIO()
    Iavg = None
    for nr,im_name in enumerate(images):
        Ic,hdrc,spacing,_ = im_io.read_to_nc_format(filename=im_name)
        if nr==0:
            Iavg = AdaptVal(torch.from_numpy(Ic))
        else:
            Iavg += AdaptVal(torch.from_numpy(Ic))
    Iavg = Iavg/len(images)
    return Iavg,spacing

def build_atlas(images, nr_of_cycles, warped_images, temp_folder, visualize):
    si = SI.RegisterImagePair()
    im_io = FIO.ImageIO()

    # compute first average image
    Iavg, sp = compute_average_image(images)
    Iavg = Iavg.data

    if visualize:
        plt.imshow(AdaptVal(Iavg[0, 0, ...]).detach().cpu().numpy(), cmap='gray')
        plt.title('Initial average based on ' + str(len(images)) + ' images')
        plt.colorbar()
        plt.show()

    # initialize list to save model parameters in between cycles
    mp = []

    # register all images to the average image and while doing so compute a new average image
    for c in range(nr_of_cycles):
        print('Starting cycle ' + str(c+1) + '/' + str(nr_of_cycles))
        for i, im_name in enumerate(images):
            print('Registering image ' + str(i) + '/' + str(len(images)))
            Ic, hdrc, spacing, _ = im_io.read_to_nc_format(filename=im_name)

            # set former model parameters if available
            if c != 0:
                si.set_model_parameters(mp[i])

            # register current image to average image
            si.register_images(Ic, AdaptVal(Iavg).detach().cpu().numpy(), spacing,
                               model_name='svf_scalar_momentum_map',
                               map_low_res_factor=0.5,
                               nr_of_iterations=5,
                               visualize_step=None,
                               similarity_measure_sigma=0.5)
            wi = si.get_warped_image()

            # save current model parametrs for the next circle
            if c == 0:
                mp.append(si.get_model_parameters())
            elif c != nr_of_cycles - 1:
                mp[i] = si.get_model_parameters()

            if c == nr_of_cycles - 1:  # last time this is run, so let's save the image
                current_filename = warped_images + '/atlas_reg_Image' + str(i+1).zfill(4) + '.nrrd'
                print("writing image " + str(i+1))
                im_io.write(current_filename, wi, hdrc)

            if i == 0:
                newAvg = wi.data
            else:
                newAvg += wi.data

        Iavg = newAvg / len(images)

        if visualize:
            plt.imshow(AdaptVal(Iavg[0, 0, ...]).detach().cpu().numpy(), cmap='gray')
            plt.title('Average ' + str(c + 1) + '/' + str(nr_of_cycles))
            plt.colorbar()
            plt.show()
    return Iavg


if __name__ == "__main__":
    # execute this as a script

    import argparse

    parser = argparse.ArgumentParser(description='Builds an atlas from input images')

    parser.add_argument('--input_image_folder', required=True, help='Path to folder containing input images')
    parser.add_argument('--input_image_pattern', required=True, help='Pattern in filenames to choose input images')
    parser.add_argument('--number_input_images', required=True, help='Number of input images used for building atlas',
                        type = int)
    parser.add_argument('--number_of_cycles', required=True, help='Number of cycles for building atlas', type = int)
    parser.add_argument('--warped_images', required=True, help='Path to where to save final warped images')
    parser.add_argument('--temp_folder', required=True, help='Path to the temp folder')
    parser.add_argument('--visualize', required=True, help='Visualize the first, intermediate and final atlas images',
                        type = bool)
    parser.add_argument('--final_atlas', required=True, help='Path to where to save final atlas image')
    args = parser.parse_args()

    input_image_folder = args.input_image_folder
    input_image_pattern = args.input_image_pattern
    number_input_images = args.number_input_images
    number_of_cycles = args.number_of_cycles
    warped_images = args.warped_images
    temp_folder = args.temp_folder
    visualize = args.visualize
    final_atlas = args.final_atlas

    # get files to build the atlas from
    images_list = find(input_image_pattern, input_image_folder)
    images = images_list[0:number_input_images]

    # build the atlas
    Iatlas = build_atlas(images, number_of_cycles, None, None, visualize)

    # save final atlas image
    atlas_filename = final_atlas + '/atlas_reg_finalAtlasImage.nrrd'
    print("Writing final average image")
    FIO.ImageIO().write(atlas_filename,Iatlas)
