# This is a simple atlas builder
# To be used (for now) to create training data for the learned smoother

import set_pyreg_paths

# first do the torch imports
import torch
from torch.autograd import Variable
from pyreg.data_wrapper import AdaptVal, MyTensor
import pyreg.fileio as FIO
import pyreg.simple_interface as SI
import pyreg.smoother_factory as SF
import numpy as np
import pyreg.module_parameters as pars
import pyreg.utils as utils
import h5py
import SimpleITK as sitk

import matplotlib.pyplot as plt
import os
import fnmatch

# keep track of general parameters
params = pars.ParameterDict()

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return sorted(result)

def find_central_segmentation(images):
    im_io = FIO.ImageIO()
    ssd_list = []
    for i ,im_name_i in enumerate(images):
        Ic_i, hdrc, spacing, _ = im_io.read_to_nc_format(filename=im_name_i)
        Ic_i_flat = Ic_i.flatten()
        for j, im_name_j in enumerate(images):
            Ic_j, hdrc, spacing, _ = im_io.read_to_nc_format(filename=im_name_j)
            Ic_j_flat = Ic_j.flatten()
            ssd = np.sum((Ic_i_flat-Ic_j_flat)*(Ic_i_flat-Ic_j_flat))
            if j==0:
                ssd_list.append(ssd)
            else:
                ssd_list[i]+= ssd
    ssd_connect = ssd_list[0:10]+ssd_list[11:]
    ssd_min = np.argwhere(ssd_connect == np.min(ssd_connect))[0][0]
    print(ssd_min)
    if ssd_min>=10:
        ssd_min+=1
    print(ssd_min, ssd_list[ssd_min])

    filename = './central/Central_Segmentation.nii.gz'
    Central, hdrc, spacing, _ = im_io.read_to_nc_format(filename=images[ssd_min])
    FIO.ImageIO().write(filename, Central[0,0, :])
    #sitk.WriteImage(sitk.GetImageFromArray(Central), filename)

    plt.imshow(Central[0, 0, ...], cmap='gray', vmin=0, vmax=1)
    plt.title('Central Segmentation')
    plt.colorbar()
    plt.show()

def build_atlas(images, nr_of_cycles, target_images, set_target_images=True):
    #si = SI.RegisterImagePair()
    im_io = FIO.ImageIO()

    # # compute first average image
    # Iavg, spacing = compute_average_image(images)
    # Iavg = AdaptVal(Iavg).data.cpu().numpy()

    # read in average image
    Iavg, hdrc, spacing, _ = im_io.read_to_nc_format('./central/Central_Segmentation.nrrd')

    # filename = './central/woSmooth_last8_predMom_Central_initial.nrrd'
    # FIO.ImageIO().write(filename, Iavg[0,0, :])

    # plt.imshow(Iavg[0, 0,0, ...], cmap='gray', vmin=0, vmax=1)
    # plt.title('Initial average based on ' + str(len(images)) + ' images')
    # plt.colorbar()
    # plt.show()


    create_binary = False
    if create_binary:
        Iavg = create_binary_image(Iavg,600)

        filename = './results/Reg_initial_bin.nrrd'
        FIO.ImageIO().write(filename, Iavg[0, :])

        plt.imshow(Iavg[0, 0, ...], cmap='gray', vmin=0, vmax=1)
        plt.title('Initial binary average based on ' + str(len(images)) + ' images')
        plt.colorbar()
        plt.show()

    do_smoothing = True
    if do_smoothing:
        # create the target image as pyTorch variable
        Iavg_pt = AdaptVal(Variable(torch.from_numpy(Iavg), requires_grad=False))

        # smooth a little bit
        params[('image_smoothing', {}, 'image smoothing settings')]
        params['image_smoothing'][
            ('smooth_images', True, '[True|False]; smoothes the images before registration')]
        params['image_smoothing'][('smoother', {}, 'settings for the image smoothing')]
        params['image_smoothing']['smoother'][('gaussian_std', 0.01, 'how much smoothing is done')]
        params['image_smoothing']['smoother'][
            ('type', 'gaussian', "['gaussianSpatial'|'gaussian'|'diffusion']")]

        sz = Iavg.shape
        cparams = params['image_smoothing']
        s = SF.SmootherFactory(sz[2::], spacing).create_smoother(cparams)
        Iavg_pt = s.smooth(Iavg_pt)
        Iavg = Iavg_pt.data.numpy()


    # initialize lists to save maps and momentums of last cycle
    wp_list = []
    mom_list = []

    # register all images to the average image and while doing so compute a new average image
    for c in range(nr_of_cycles):
        print('Starting cycle ' + str(c+1) + '/' + str(nr_of_cycles))
        for i, im_name in enumerate(images):
            print('Registering image ' + str(i+1) + '/' + str(len(images)))
            si = SI.RegisterImagePair()
            Ic, hdrc, spacing, _ = im_io.read_to_nc_format(filename=images[i])

            if do_smoothing:
                # create the source image as pyTorch variable
                Ic_pt = AdaptVal(Variable(torch.from_numpy(Ic), requires_grad=False))
                Ic_beforeSmoothing = AdaptVal(Variable(torch.from_numpy(Ic), requires_grad=False))

                # smoth a little bit
                Ic_pt = s.smooth(Ic_pt)
                Ic = Ic_pt.data.numpy()


            if set_target_images:
                Iavg,_,_, _ = im_io.read_to_nc_format(filename=target_images[i])
                if do_smoothing:
                    # create the source image as pyTorch variable
                    Iavg_pt = AdaptVal(Variable(torch.from_numpy(Iavg), requires_grad=False))

                    # smoth a little bit
                    Iavg_pt = s.smooth(Iavg_pt)
                    Iavg = Iavg_pt.data.numpy()

            # register current image to average image
            si.register_images(Ic, Iavg[0,:], spacing, model_name='svf_vector_momentum_map',
                               smoother_type='multiGaussian',
                               compute_similarity_measure_at_low_res=False,
                               map_low_res_factor=1.0,
                               visualize_step=None,
                               nr_of_iterations=100,
                               rel_ftol=1e-8,
                               similarity_measure_type="ncc",
                               similarity_measure_sigma=0.1,
                               # params='findTheBug_test.json',
                               params='findTheBug_GaussianWeights.json',
                               # json_config_out_filename='findTheBug_test.json'
                               json_config_out_filename='findTheBug_GaussianWeights.json'
                               )
            # si.register_images(Ic, AdaptVal(Iavg).cpu().numpy(), spacing,


            wi = si.get_warped_image()
            wp = si.get_map()
            if do_smoothing:
                wi = utils.compute_warped_image_multiNC(Ic_beforeSmoothing, wp, si.spacing)

            if c == nr_of_cycles - 1:  # last time this is run, so let's save the map, momentum and image
                mom = si.get_model_parameters()
                mom_list.append(mom['m'].data.numpy().squeeze())
                wp_list.append(wp.data.numpy())

                # current_filename = './central/woSmooth_last8_predMom30_lr1e4_CentralReg_Label1_regImage' + str(i+1).zfill(4) + '.nrrd'
                # print("Writing image " + str(i+1))
                # wi_data = wi.data
                # im_io.write(current_filename, wi_data[0,:])

            if i == 0:
                newAvg = wi.data.cpu().numpy()
            else:
                newAvg += wi.data.cpu().numpy()

            # if i == 0:
            #     newAvg = create_binary_image(wi.data.cpu().numpy(),500)
            # else:
            #     newAvg += create_binary_image(wi.data.cpu().numpy(),500)

        Iavg = newAvg / len(images)
        Iavg = AdaptVal(Variable(torch.from_numpy(Iavg), requires_grad=False)).data.cpu().numpy()

        # filename = './central/woSmooth_last8_predMom30_lr1e4_Average_' + str(c + 1) + 'of' + str(nr_of_cycles)+'_ncc.nrrd'
        # FIO.ImageIO().write(filename, Iavg[0, :])

        create_binary = False
        if create_binary:
            Iavg = create_binary_image(Iavg, 500)
            filename = './results/Reg_bin_Average' + str(c + 1) + 'of' + str(nr_of_cycles) + '_ncc.nrrd'
            FIO.ImageIO().write(filename, Iavg[0, :])

        # plt.imshow(Iavg[0, 0, ...], cmap='gray', vmin=0, vmax=1)
        # plt.title('Average ' + str(c + 1) + '/' + str(nr_of_cycles))
        # plt.colorbar()
        # plt.show()

        # if normalizeAverageImage:
        #     Iavg = Iavg / float(np.amax(AdaptVal(Iavg).cpu().numpy()))
        if do_smoothing:
            Iavg_pt = AdaptVal(Variable(torch.from_numpy(Iavg), requires_grad=False))
            Iavg_pt = s.smooth(Iavg_pt)
            Iavg = Iavg_pt.data.numpy()
        else:
            Iavg = AdaptVal(Variable(torch.from_numpy(Iavg), requires_grad=False)).data.cpu().numpy()

    return Iavg, wp_list, mom_list

##LOAD IMAGES
# first get a few files to build the atlas from
images_list = find('*LEFT*Label1.sliced.nii.gz', '../test_data/label_slices/')
nr_of_images = len(images_list)
images = images_list[0:nr_of_images]
print(nr_of_images,len(images))

##CENTRAL SEGMENTATION
find_central_segmentation(images)

##LOAD TARGET IMAGES
# t_images_list = find('*first78*.nii.gz', '../demos/central/GaussianWeights/ITarget_warped/')
# t_nr_of_images = len(t_images_list)
# target_images = t_images_list[0:t_nr_of_images]
# print(t_nr_of_images,len(target_images))

##SAVE TARGET IMAGES IN TRAINING SET
# inv_IT = []
# for nr, im_name in enumerate(target_images):
#     Ic, hdrc, spacing, _ = FIO.ImageIO().read_to_nc_format(filename=im_name)
#     inv_IT.append(Ic.squeeze())
# train_Timages_file = h5py.File("./central/corrMom_train_Itarget.h5", "w")
# trainTdset = train_Timages_file.create_dataset("dataset", data=inv_IT)
# train_Timages_file.close()

##BUILD ATLAS
# # build atlas
# nr_of_cycles = 1
# normalizeAverageImage = False
# Iatlas, wp_list, mom_list = build_atlas(images, nr_of_cycles,  set_target_images=True, target_images=target_images)

##SAVE MOMENTUMS FOR CORRECTION NETWORK
# f = h5py.File("./central/GaussianWeights/train_corrMomentums.h5", "w")
# invMaps_dset = f.create_dataset("dataset", data=mom_list)
# f.close()


##############
##LOAD IMAGES
# # first get a few files to build the atlas from
# images_list = find('*LEFT*Label1.sliced.nii.gz', '../test_data/label_slices/')
# nr_of_images = len(images_list)
# images = images_list[78:nr_of_images]
# print(nr_of_images,len(images))
#
# t_images_list = find('*last8*.nii.gz', '../demos/central/GaussianWeights/ITarget_warped/')
# t_nr_of_images = len(t_images_list)
# target_images = t_images_list[0:t_nr_of_images]
# print(t_nr_of_images,len(target_images))

##BUILD ATLAS
# # build atlas
# nr_of_cycles = 1
# normalizeAverageImage = False
# Iatlas, wp_list, mom_list = build_atlas(images, nr_of_cycles,  set_target_images=True, target_images=target_images)

##SAVE MOMENTUMS FOR CORRECTION NETWORK
# f = h5py.File("./central/GaussianWeights/test_corrMomentums.h5", "w")
# invMaps_dset = f.create_dataset("dataset", data=mom_list)
# f.close()