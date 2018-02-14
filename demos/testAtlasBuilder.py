# This is a simple atlas builder
# To be used (for now) to create training data for the learned smoother

import set_pyreg_paths

# first do the torch imports
import torch
from torch.autograd import Variable
from pyreg.data_wrapper import AdaptVal
import pyreg.fileio as FIO
import pyreg.simple_interface as SI

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
            Iavg = AdaptVal(Variable(torch.from_numpy(Ic), requires_grad=False))
        else:
            Iavg += AdaptVal(Variable(torch.from_numpy(Ic), requires_grad=False))
    Iavg = Iavg/len(images)
    return Iavg,spacing

def build_atlas(images, nr_of_cycles):
    si = SI.RegisterImagePair()
    im_io = FIO.ImageIO()

    # compute first average image
    Iavg, sp = compute_average_image(images)
    Iavg = Iavg.data

    plt.imshow(AdaptVal(Iavg[0, 0, ...]).cpu().numpy(), cmap='gray')
    plt.title('Initial average based on ' + str(len(images)) + ' images')
    plt.colorbar()
    plt.show()

    # initialize list to save model parameters in between cycles
    mp = []

    # register all images to the average image and while doing so compute a new average image
    for c in range(nr_of_cycles):
        print('Starting cycle ' + str(c+1) + '/' + str(nr_of_cycles))
        for i, im_name in enumerate(images):
            print('Registering image ' + str(i+1) + '/' + str(len(images)))
            Ic, hdrc, spacing, _ = im_io.read_to_nc_format(filename=im_name)

            # set former model parameters if available
            if c != 0:
                si.set_model_parameters(mp[i])

            # register current image to average image
            si.register_images(Ic, AdaptVal(Iavg).cpu().numpy(), spacing,
                               model_name='svf_scalar_momentum_map',
                               map_low_res_factor=0.5,
                               nr_of_iterations=5,
                               visualize_step=None,
                               similarity_measure_sigma=0.5)
            # si.register_images(Ic, AdaptVal(Iavg).cpu().numpy(), spacing,
            #                    model_name='svf_scalar_momentum_map',
            #                    smoother_type='adaptive_multiGaussian',
            #                    # optimize_over_smoother_parameters=True,
            #                    map_low_res_factor=1.0,
            #                    visualize_step=None,
            #                    nr_of_iterations=15,
            #                    rel_ftol=1e-4,
            #                    similarity_measure_sigma=1)
            # si.register_images(Ic, AdaptVal(Iavg).cpu().numpy(), spacing,
            #                    model_name='affine_map', nr_of_iterations=15,
            #                    visualize_step=None, rel_ftol=1e-4)
            wi = si.get_warped_image()

            # save current model parametrs for the next circle
            if c == 0:
                mp.append(si.get_model_parameters())
            elif c != nr_of_cycles - 1:
                mp[i] = si.get_model_parameters()

            if c == nr_of_cycles - 1:  # last time this is run, so let's save the image
                # current_filename = './reg_oasis2d_' + str(i).zfill(4) + '.nrrd'
                current_filename = './atlasTest_Label1_regImage' + str(i+1).zfill(4) + '.nrrd'
                print("Writing image " + str(i+1))
                im_io.write(current_filename, wi, hdrc)

            if i == 0:
                newAvg = wi.data
            else:
                newAvg += wi.data

        Iavg = newAvg / len(images)

        plt.imshow(AdaptVal(Iavg[0, 0, ...]).cpu().numpy(), cmap='gray')
        plt.title('Average ' + str(c + 1) + '/' + str(nr_of_cycles))
        plt.colorbar()
        plt.show()
    return Iavg

# first get a few files to build the atlas from
images_list = find('*Label1.sliced.nii.gz', '../test_data/label_slices/')
nr_of_images = 6
images = images_list[0:nr_of_images]

# build atlas
nr_of_cycles = 3
Iatlas = build_atlas(images, nr_of_cycles)

# save final average image
atlas_filename = './atlasTest_finalAtlasImage.nrrd'
print("Writing final atlas image")
FIO.ImageIO().write(atlas_filename,Iatlas)