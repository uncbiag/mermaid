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

def compute_average_image(images):
    im_io = FIO.ImageIO()
    Iavg = None
    for nr,im_name in enumerate(images):
        Ic,hdrc,spacing,_ = im_io.read_to_nc_format(filename=im_name)
        if nr==0:
            Iavg = AdaptVal(Variable(torch.from_numpy(Ic), requires_grad=False))
        else:
            Iavg += AdaptVal(Variable(torch.from_numpy(Ic), requires_grad=False))
    Iavg = Iavg / len(images)
    return Iavg,spacing

def create_binary_image(image, desired_pixel_nr):
    image_values = np.unique(image, return_counts=True)
    if len(image_values[0])==2:
        print('Image is already binary, return image!')
        return image
    highest_intensities = 0
    idx = 0
    threshold_intensity = 0
    while highest_intensities < desired_pixel_nr:
        idx += 1
        highest_intensities += image_values[1][-idx]
        threshold_intensity = image_values[0][-idx]
    if threshold_intensity <= 0:
        print('Desired number of pixels too high!')
        while threshold_intensity <= 0:
            idx -= 1
            highest_intensities -= image_values[1][-(idx+1)]
            threshold_intensity = image_values[0][-idx]
        print('Reduced number of pixels to ' + str(highest_intensities))
    image_bin = (np.ones(image.shape) * (image >= threshold_intensity)).astype('float32')
    print('##### intensity threshold at: '+str(threshold_intensity))
    return image_bin


def build_atlas(images, nr_of_cycles, set_momentums, **momentums):
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
            Ic, hdrc, spacing, _ = im_io.read_to_nc_format(filename=im_name)

            if do_smoothing:
                # create the source image as pyTorch variable
                Ic_pt = AdaptVal(Variable(torch.from_numpy(Ic), requires_grad=False))
                Ic_beforeSmoothing = AdaptVal(Variable(torch.from_numpy(Ic), requires_grad=False))

                # smoth a little bit
                Ic_pt = s.smooth(Ic_pt)
                Ic = Ic_pt.data.numpy()

            if set_momentums:
                si.set_model_parameters({'m':torch.from_numpy(pred_data[i,:])})

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

                # current_filename = './reg_oasis2d_' + str(i).zfill(4) + '.nrrd'
                current_filename = './central/GaussianWeights/CentralReg_Label1_regImage' + str(i+1).zfill(4) + '.nii.gz'
                print("Writing image " + str(i+1))
                wi_data = wi.data
                im_io.write(current_filename, wi_data[0,:])

            if True:
                Iw = wi.data.cpu().numpy()[0,0,160:250, :]
                deff_data = wp.data.numpy()
                plt.clf()
                plt.imshow(Iw, cmap='gray')
                plt.contour(deff_data[0, 0, :][160:250, :], np.linspace(-1, 1, 200), colors='r', linestyles='solid')
                plt.contour(deff_data[0, 1, :][160:250, :], np.linspace(-1, 1, 200), colors='r', linestyles='solid')
                # plt.colorbar()
                plt.title('deformation field %i' % (i+1))
                plt.savefig('./central/GaussianWeights/deformation_field_grid_200_' + str(i + 1) + '.png')
                plt.show()

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

        filename = './central/GaussianWeights/Average_' + str(c + 1) + 'of' + str(nr_of_cycles)+'_ncc.nii.gz'
        FIO.ImageIO().write(filename, Iavg[0, :])

        create_binary = False
        if create_binary:
            Iavg = create_binary_image(Iavg, 500)
            filename = './results/Reg_bin_Average' + str(c + 1) + 'of' + str(nr_of_cycles) + '_ncc.nrrd'
            FIO.ImageIO().write(filename, Iavg[0, :])

        plt.imshow(Iavg[0, 0, ...], cmap='gray', vmin=0, vmax=1)
        plt.title('Average ' + str(c + 1) + '/' + str(nr_of_cycles))
        plt.colorbar()
        plt.show()

        # if normalizeAverageImage:
        #     Iavg = Iavg / float(np.amax(AdaptVal(Iavg).cpu().numpy()))
        if do_smoothing:
            Iavg_pt = AdaptVal(Variable(torch.from_numpy(Iavg), requires_grad=False))
            Iavg_pt = s.smooth(Iavg_pt)
            Iavg = Iavg_pt.data.numpy()
        else:
            Iavg = AdaptVal(Variable(torch.from_numpy(Iavg), requires_grad=False)).data.cpu().numpy()

    return Iavg, wp_list, mom_list


def compare_maps(prediction_path, groundtruth_path, vizualization=False):
    omaps = h5py.File(groundtruth_path, 'r')
    orig_maps = omaps['/dataset'][()]
    orig_maps = orig_maps.squeeze()
    omaps.close()

    pmaps = h5py.File(prediction_path, 'r')
    pred_maps = pmaps['/dataset'][()]
    pred_maps = pred_maps.squeeze()
    pmaps.close()

    for i in range(8):
        print('_______________________image ' + str(i + 1) + '_______________________')
        ssd_map_comparison(orig_maps[i, :], pred_maps[i, :], i + 1)

        if vizualization:
            plt.subplot(2, 2, 1)
            plt.imshow(orig_maps[i, 0, :], vmin=-0.2, vmax=0.2)
            plt.colorbar()
            plt.subplot(2, 2, 2)
            plt.imshow(orig_maps[i, 1, :], vmin=-0.2, vmax=0.2)
            plt.colorbar()
            plt.subplot(2, 2, 3)
            plt.imshow(pred_maps[i, 0, :], vmin=-0.2, vmax=0.2)
            plt.colorbar()
            plt.subplot(2, 2, 4)
            plt.imshow(pred_maps[i, 1, :], vmin=-0.2, vmax=0.2)
            plt.colorbar()
            plt.show()

def ssd_map_comparison(first_maps, second_maps,nr):
    first_maps = first_maps/0.00261097
    second_maps = second_maps/0.00261097
    difference_maps_squared = (first_maps - second_maps)*(first_maps - second_maps)
    ssd_difference = sum(sum(sum(difference_maps_squared)))
    ssd_difference_mm = sum(sum(sum(np.sqrt(difference_maps_squared))))

    id = utils.identity_map(([384, 160]), ([1,1]))
    first_minusID = abs(first_maps[0, :] -id[0, :]) + abs(first_maps[1, :]-id[1, :])
    first_threshold = sorted(first_minusID.flatten())[-16000]
    first_mask = np.ones(first_minusID[0, :].shape) * (first_minusID > first_threshold)
    second_minusID = abs(second_maps[0, :] -id[0, :]) + abs(second_maps[1, :]-id[1, :])
    second_threshold = sorted(second_minusID.flatten())[-16000]
    second_mask = np.ones(second_minusID[0, :].shape) * (second_minusID > second_threshold)
    mask = np.ones(second_minusID[0, :].shape) * ((first_mask+second_mask) > 0)

    mask = np.zeros(second_maps[0,:].shape)
    mask[125:275,:]=1
    #mask = mask.flatten()

    # plt.imshow(mask)
    # plt.title('mask for image #'+str(nr))
    # plt.colorbar()
    # plt.show()

    x_mask = difference_maps_squared[0, :][mask == 1]
    y_mask = difference_maps_squared[1, :][mask == 1]
    difference_vec = sorted((x_mask + y_mask))
    percentile = np.array([0.003, 0.05, 0.25, 0.5, 0.75, 0.95, 0.997])
    perc = (percentile * len(difference_vec)).astype(int)
    ssd_error_perc = [difference_vec[perc[i]] for i in range(len(perc))]
    ssd_error_perc = ["%.4f" % v for v in ssd_error_perc]
    norm_error_perc = [sorted(np.sqrt(difference_vec))[perc[i]] for i in range(len(perc))]
    norm_error_perc = ["%.4f" % v for v in norm_error_perc]
    ssd_in_mask = sum(difference_vec)
    ssd_in_mask_mm = sum(np.sqrt(difference_vec))

    vismaps = []
    error_map = np.sqrt(difference_maps_squared[0,:]+difference_maps_squared[1,:])
    vis_map = np.reshape(error_map, [384,160])
    vismaps.append(vis_map)

    plt.subplot(1, 3, 1)
    plt.imshow(IO[nr-1])
    plt.colorbar()
    plt.title('original image %i' %(nr))
    plt.subplot(1, 3, 2)
    plt.imshow(vis_map)
    plt.gri
    plt.contour(range(160), range(384), vis_map)
    plt.colorbar()
    plt.title('error map image %i' %(nr))
    plt.subplot(1, 3, 3)
    plt.imshow(IW[nr-1])
    plt.colorbar()
    plt.title('warped image %i' %(nr))
    plt.show()

    x_ground_mask = ((first_maps[0, :] -id[0, :])*(first_maps[0, :] -id[0, :]))[mask == 1]
    y_ground_mask = ((first_maps[1, :] - id[1, :])*(first_maps[1, :] - id[1, :]))[mask == 1]
    diff_ground = sorted((x_ground_mask + y_ground_mask))
    ssd_ground_in_mask = sum(diff_ground)
    ssd_ground_in_mask_mm = sum(np.sqrt(diff_ground))
    ground_error_perc = [diff_ground[perc[i]] for i in range(len(perc))]
    ground_error_perc = ["%.4f" % v for v in ground_error_perc]
    ground_norm_error_perc = [sorted(np.sqrt(diff_ground))[perc[i]] for i in range(len(perc))]
    ground_norm_error_perc = ["%.4f" % v for v in ground_norm_error_perc]

    percentile = ['%.4f' % v for v in percentile]

    # print('total ssd      (mm^2): %.1f' % (ssd_difference)+', 2-norm (mm): %.1f' % (ssd_difference_mm))
    # print('ssd    in mask (mm^2): %.1f' % (ssd_in_mask) + ', 2-norm (mm): %.1f' % (ssd_in_mask_mm))
    # print('GT ssd in mask (mm^2): %.1f' % (ssd_ground_in_mask)+', 2-norm (mm): %.1f' % (ssd_ground_in_mask_mm))
    # print('percentage 2-norm in mask/GT 2-norm in mask: '+'%.2f' % ((ssd_in_mask_mm/ssd_ground_in_mask_mm)*100)+ r'%')
    # print('# mask_pixel: '+ str(len(difference_vec))+', avg error per pixel in mask (mm): %.1f' % (ssd_in_mask_mm/(len(difference_vec))))
    # print('--')
    # print('data percentile ' + str(percentile))
    # print('--------------------------------------------------------------------------------------')
    # print('      ssd perc  ' + str(ssd_error_perc))
    # print('   GT ssd perc  ' + str(ground_error_perc))
    # print('   2-norm perc  ' + str(norm_error_perc))
    # print('GT 2-norm perc  ' + str(ground_norm_error_perc))

    return ssd_difference,ssd_error_perc, vismaps


##LOAD IMAGES
# first get a few files to build the atlas from
images_list = find('*LEFT*Label1.sliced.nii.gz', '../test_data/label_slices/')
nr_of_images = len(images_list)
images = images_list[0:nr_of_images]
print(nr_of_images,len(images))

## COMPUTE DEFORMATION FIELD GRIDS
wimages_list = find('*first78*nii.gz', './central/GaussianWeights/CorrectionResults')
wnr_of_images = len(wimages_list)
wimages = wimages_list[0:]
print(wnr_of_images,len(wimages))

IO = []
IW = []
for i in range(78):
    Io = FIO.ImageIO().read_to_nc_format(images[i])
    Iw = FIO.ImageIO().read_to_nc_format(wimages[i])

    Io = Io[0].squeeze()
    Iw = Iw[0].squeeze()
    Ico = Io[160:250, :]
    Icw = Iw[160:250, :]
    IO.append(Ico)
    IW.append(Icw)

# read in average image
Iavg, hdrc, spacing, _ = FIO.ImageIO().read_to_nc_format('./central/Central_Segmentation.nrrd')
Iavg = Iavg.squeeze()[160:250, :]

# load deformation fields
deff = h5py.File("./central/GaussianWeights/CorrectionResults/CorrRegMaps_PredMom30_lr2e4_first78.h5", 'r')
deff_data = deff['/dataset'][()]
deff.close()

X, Y = np.meshgrid(np.arange(0, 160, 3), np.arange(0, 384, 3))
print(X)

for i in range(78):
    # plt.subplot(1, 3, 1)
    # plt.imshow(IO[i],cmap='gray')
    # plt.colorbar()
    # plt.title('original %i' % (i+1))
    # plt.subplot(1, 3, 2)
    # plt.imshow(IW[i],cmap='gray')
    # plt.imshow(Iavg,cmap='Oranges')
    # plt.colorbar()
    # plt.title('warped %i' % (i+1))
    # plt.subplot(1, 4, 3)

    plt.imshow(IW[i],cmap='gray')
    plt.contour(deff_data[i,0,0,:][160:250,:], np.linspace(-1, 1, 300), colors='r', linestyles='solid')
    plt.contour(deff_data[i,0,1,:][160:250,:], np.linspace(-1, 1, 300), colors='r', linestyles='solid')
    plt.title('deformation field %i' % (i+1))
    #
    # plt.subplot(1, 3, 3)
    # plt.imshow(Iavg,cmap='gray')
    # plt.colorbar()
    # plt.title('target image')

    plt.savefig('./central/GaussianWeights/CorrectionResults/DeformationGrids/deformation_field_grid_300_first78_'+str(i+1)+'.png')
    plt.show()
    plt.clf()


#
# ##COMPUTE AVG NR OF PIXELS IN SEGMENTATIONS
# # im_io = FIO.ImageIO()
# # pix =0
# # for i, im_name in enumerate(images):
# #     Ic, hdrc, spacing, _ = im_io.read_to_nc_format(filename=im_name)
# #     pix+= np.sum(Ic)
# # print('Avg nr of pixel: ' + str(pix/nr_of_images))
#
##INITIAL MOMENTUMS
# # load initial maps from the prediction net
# pred = h5py.File("/playpen/shaeger/prediction/knee_test_results/concatenatedMom30_lr2e4_last8.h5", 'r')
# pred_data = pred['/dataset'][()]
# pred.close()

#COMPARE PRED & GROUNDTRUTH
# load and compare the computed and predicted transMaps
# IO = []
# IW = []
# for i in range(8):
#     Io = FIO.ImageIO().read_to_nc_format(images[i])
#     Iw = FIO.ImageIO().read_to_nc_format(wimages[i])
#
#     Io = Io[0].squeeze()
#     Iw = Iw[0].squeeze()
#     IO.append(Io)
#     IW.append(Iw)
# prediction_path = "./central/Last8/RegMaps_corrPredMom30_lr2e4_last8.h5"
# groundtruth_path = "./central/Network_data/test_Maps.h5"
# compare_maps(prediction_path,groundtruth_path,vizualization=False)
#



# ##CHECK NETWORK RESULTS
# pmaps = h5py.File("/playpen/shaeger/prediction/test_results/pv_1.h5", 'r')
# pred_maps = pmaps['/dataset'][()]
# pred_maps = pred_maps.squeeze()
# pmaps.close()
#
# plt.subplot(1, 2, 1)
# plt.imshow(pred_maps[0, :])
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.imshow(pred_maps[1, :])
# plt.colorbar()
# plt.show()
#
##BUILD ATLAS
# build atlas
# nr_of_cycles = 1
# normalizeAverageImage = False
# set_momentums = False
# Iatlas, wp_list, mom_list = build_atlas(images, nr_of_cycles, set_momentums)
#
##SAVE MOMENTUMS IN TRAIN & TEST SETS
# #write transformation maps to h5 file
# f = h5py.File("./central/GaussianWeights/train_Momentums.h5", "w")
# ff = h5py.File("./central/GaussianWeights/test_Momentums.h5", "w")
# train_dset = f.create_dataset("dataset", data=mom_list[0:78])
# test_dset = ff.create_dataset("dataset", data=mom_list[78:])
# f.close()
# ff.close()
#
##SAVE MAPS
# g = h5py.File("./central/GaussianWeights/train_Maps.h5", "w")
# gg = h5py.File("./central/GaussianWeights/test_Maps.h5", "w")
# train_dset = g.create_dataset("dataset", data=wp_list[0:78])
# test_dset = gg.create_dataset("dataset", data=wp_list[78:])
# g.close()
# gg.close()
#
##CREATE IMAGE TRAIN & TEST SETS
# #write source and target images to h5 files for training and testing
# train_source_images = []
# train_target_images = []
# test_source_images = []
# test_target_images = []
# im_io = FIO.ImageIO()
#
# # Iavg, hdrc, spacing, _ = im_io.read_to_nc_format('./Average2of10_ncc.nrrd')
# # Iavg_bin = (np.ones(Iavg.shape) * (Iavg > 0.15)).astype('float32')
# # Iavg, hdrc, spacing, _ = im_io.read_to_nc_format(
# #     '../test_data/label_slices/9587749_20040413_SAG_3D_DESS_LEFT_016610022903_label_all.alignedCOM.Label1.sliced.nii.gz')
# Iavg, hdrc, spacing, _ = im_io.read_to_nc_format('./central/Central_Segmentation.nrrd')
#
# for nr, im_name in enumerate(images):
#     Ic, hdrc, spacing, _ = im_io.read_to_nc_format(filename=im_name)
#     if nr<=77:
#         train_source_images.append(Ic.squeeze())
#         train_target_images.append(Iavg.squeeze())
#     else:
#         test_source_images.append(Ic.squeeze())
#         test_target_images.append(Iavg.squeeze())
# train_Simages_file = h5py.File("./central/Mom_train_Isource.h5", "w")
# train_Timages_file = h5py.File("./central/Mom_train_Itarget.h5", "w")
# test_Simages_file = h5py.File("./central/Mom_test_Isource.h5", "w")
# test_Timages_file = h5py.File("./central/Mom_test_Itarget.h5", "w")
# trainSdset = train_Simages_file.create_dataset("dataset", data=train_source_images)
# trainTdset = train_Timages_file.create_dataset("dataset", data=train_target_images)
# testSdset = test_Simages_file.create_dataset("dataset", data=test_source_images)
# testTdset = test_Timages_file.create_dataset("dataset", data=test_target_images)
# train_Simages_file.close()
# train_Timages_file.close()
# test_Simages_file.close()
# test_Timages_file.close()

##SQUEEZE MOMENTUMS DATASETS
# train = h5py.File("./central/train_Momentums.h5", 'r')
# data = train['/dataset'][()]
# #data = torch.from_numpy(data)
# train.close()
#
# test = h5py.File("./central/test_Momentums.h5", 'r')
# datat = test['/dataset'][()]
# #data = torch.from_numpy(data)
# train.close()
#
#
# train_data = data.squeeze()
# test_data = datat.squeeze()
#
# wp_train = h5py.File("./strain_Momentums.h5", "w")
# wp_test = h5py.File("./stest_Momentums.h5", "w")
# wp_train_dset = wp_train.create_dataset("dataset", data=train_data)
# wp_test_dset = wp_test.create_dataset("dataset", data=test_data)
# wp_train.close()
# wp_test.close()

## 9369649_20050112_SAG_3D_DESS_LEFT_016610322906_label_all.alignedCOM.Label1.s