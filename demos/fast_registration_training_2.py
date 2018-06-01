# This is part 2 of the fast registration training and testing.
# To be used to create train and test data for the correction network.
#NOTE: assumes train data is loaded images[0-nr_of_train_images],
#               test data is loaded images[nr_of_train_images-end]
#
#IN:
# path and pattern: source images [train,test]
# path: target image
# path: prediction net momentum train&test sets
# choose mode: train/test
#
#   algorithm 'stephanie':                                    algorithm 'marc':
## LOAD IMAGES                                              LOAD IMAGES
## INITIALIZE MOMENTUM                                      INITIALIZE MOMENTUM
## --                                                       SUBTRACT MOMENTUM
## APPLY MODEL                                              APPLY MODEL
## -COMPUTE MAP AND INVERSE MAP OF PREDICTED MOMENTUM       -COMPUTE MAP AND INVERSE MAP OF PREDICTED MOMENTUM
## -COMPUTE WARPED IMAGE OF INVERSE MAP                     -COMPUTE WARPED IMAGE OF INVERSE MAP
## SAVE MAPS AND INVERSE MAPS                               SAVE MAPS AND INVERSE MAPS
## SAVE INVERSE WARPED ITARGET IMAGES                       SAVE INVERSE WARPED ITARGET IMAGES
##  if mode 'train':                                            --
## LOAD TARGET IMAGES                                           --
## MULTI REGISTRATION                                           --
## SAVE MOMENTUMS FOR CORRECTION NETWORK                        --
#
#OUT:
# warped source images
# deformation grid plots (if True)
# inverted Maps
# inverse warped target images
# data set: predicted deformation maps
# data set: inverted predicted deformation maps
# data set: inverse warped target images
# data set: Momentum for correction network (if train mode)



import set_pyreg_paths
import fast_registration_tools as FRT
import torch
from torch.autograd import Variable
from pyreg.data_wrapper import AdaptVal, MyTensor
import pyreg.fileio as FIO
import pyreg.module_parameters as pars
import pyreg.utils as utils
import numpy as np
import matplotlib.pyplot as plt

# keep track of general parameters
params = pars.ParameterDict()

#----------------------------------------TO BE SET-----------------------------------------------
# set loading paths and parameters
source_images_path='../test_data/label_slices/'
source_images_pattern='*LEFT*label1.Sliced.nii.gz'
target_image_path='./fast_reg/results/Central_Segmentation.nii.gz'
mode = 'test'
orig_momentum_train_path='./fast_reg/results/train_Momentums.h5'
orig_momentum_test_path='./fast_reg/results/test_Momentums.h5'
predicted_momentum_train_path ='./fast_reg/results/predMom_trainset.h5'
predicted_momentum_test_path ='./fast_reg/results/predMom_testset.h5'
nr_of_train_images = 78
nr_of_test_images  = 8
plot_deformation_maps   = False
plot_inv_warped_IT      = False
do_smoothing            = True
plot_and_save_def_grids = True
###
algorithm = 'marc' #'stephanie'
###

# set saving paths
results_path='./fast_reg/results/'

# CREATE FOLDERS and set saving paths
warped_Isource_path="./fast_reg/results/wImages_prediction/"   #needed in fast_registration_analysis.py
inverse_deformation_maps_path="./fast_reg/results/inv_DeformationMaps/"
inverse_warped_ITarget_path="./fast_reg/results/inverse_ITarget/"
second_registration_results_path ="./fast_reg/results/second_RegImages/"
#----------------------------------------TO BE SET-----------------------------------------------


##LOAD IMAGES
images_list = FRT.find(source_images_pattern, source_images_path)
nr_of_images = len(images_list)
if mode=='train':
    images = images_list[0:nr_of_train_images]
elif mode=='test':
    images = images_list[-nr_of_test_images:]
print(nr_of_images,len(images))

##INITIALIZE MOMENTUM
# load initial momentums from the prediction net
if mode=='train':
    pred_data = FRT.read_h5file(predicted_momentum_train_path)
elif mode=='test':
    pred_data = FRT.read_h5file(predicted_momentum_test_path)

##SUBTRACT MOMENTUM
if algorithm=='marc':
    sub_momentum_path = results_path + mode + "_sub_Mom.h5"
    if mode == 'train':
        sub_momentum = FRT.subtract_momentums(orig_momentum_train_path, predicted_momentum_train_path,
                                              sub_momentum_path)
    elif mode == 'test':
        sub_momentum = FRT.subtract_momentums(orig_momentum_test_path, predicted_momentum_test_path,
                                              sub_momentum_path)

##APPLY MODEL
im_io = FIO.ImageIO()
Iavg, hdrc, spacing, _ = im_io.read_to_nc_format(target_image_path)

size = np.shape(Iavg)
parameters = 'fast_registration_eval_model_params.json'
params = pars.ParameterDict()
params.load_JSON(parameters)
shared_parameters = dict()

maps_from_mom = []
inv_maps_from_mom = []
ITarget_warped = []

for i, im_name in enumerate(images):
    momDict = dict()
    momDict['m'] = torch.from_numpy(pred_data[i, :])
    print('Registering image ' + str(i + 1) + '/' + str(len(images)))
    Ic, hdrc, spacing, _ = im_io.read_to_nc_format(filename=im_name, silent_mode=True)

    ##COMPUTE MAP AND INVERSE MAP OF PREDICTED MOMENTUM
    Iwarped, map_from_mom, inv_map_from_mom = FRT.evaluate_model(
        AdaptVal(Variable(torch.from_numpy(Ic), requires_grad=False)),
        AdaptVal(Variable(torch.from_numpy(Iavg), requires_grad=False)),
        size,
        spacing,
        momDict,
        shared_parameters,
        params,
        visualize=False,
        compute_inverse_map=True)

    maps_from_mom.append(map_from_mom.cpu().data.numpy())
    inv_maps_from_mom.append(inv_map_from_mom.cpu().data.numpy())

    warped_IS_filename = warped_Isource_path + mode + "_predMom_warped_image" + str(i + 1) + ".nii.gz"
    FIO.ImageIO().write(warped_IS_filename, Iwarped.cpu().data.numpy()[0, 0, :], hdrc)

    inverse_defMap_path = inverse_deformation_maps_path + mode + "_invMap_" + str(i + 1).zfill(3) + ".nii.gz"
    FIO.ImageIO().write(inverse_defMap_path, inv_map_from_mom[0, :])

    if plot_deformation_maps:
        wmap = utils.compute_warped_image_multiNC(map_from_mom, inv_map_from_mom, spacing, 3)
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(np.zeros([100, 160]), cmap='gray')  # just for background purpose
        plt.contour(map_from_mom.cpu().data.numpy()[0, 0, :][125:225, :], np.linspace(-1, 1, 200), colors='r',
                    linestyles='solid')
        plt.contour(map_from_mom.cpu().data.numpy()[0, 1, :][125:225, :], np.linspace(-1, 1, 200), colors='r',
                    linestyles='solid')
        plt.title('predicted deformation field img %i' % (i + 1), fontsize=8)
        plt.subplot(1, 2, 2)
        plt.imshow(np.zeros([100, 160]), cmap='gray')  # just for background purpose
        plt.contour(inv_map_from_mom.cpu().data.numpy()[0, 0, :][125:225, :], np.linspace(-1, 1, 200), colors='r',
                    linestyles='solid')
        plt.contour(inv_map_from_mom.cpu().data.numpy()[0, 1, :][125:225, :], np.linspace(-1, 1, 200), colors='r',
                    linestyles='solid')
        plt.title('inverted predicted deformation field iter %i' % (i + 1), fontsize=10)
        plt.show()
        plt.clf()
        plt.imshow(np.zeros([100, 160]), cmap='gray')  # just for background purpose
        plt.contour(wmap.cpu().data.numpy()[0, 0, :][125:225, :], np.linspace(-1, 1, 400), colors='r',
                    linestyles='solid')
        plt.contour(wmap.cpu().data.numpy()[0, 1, :][125:225, :], np.linspace(-1, 1, 400), colors='r',
                    linestyles='solid')
        plt.title('warped (map & inv_map) %i' % (i + 1), fontsize=10)
        plt.show()
        plt.clf()

    ##COMPUTE WARPED IMAGE OF INVERSE MAP
    inv_IT, _ = FRT.read_image_and_map_and_apply_map(target_image_path, inverse_defMap_path)
    ITarget_warped.append(inv_IT.squeeze())

    inverse_warped_IT_filename = inverse_warped_ITarget_path + mode + "_inv_IT_image" + str(i + 1).zfill(
        3) + ".nii.gz"
    FIO.ImageIO().write(inverse_warped_IT_filename, inv_IT[0, 0, :])

    if plot_inv_warped_IT:
        plt.subplot(1, 2, 1)
        plt.imshow(Ic[0, 0, :])
        plt.colorbar()
        plt.title('Image %i' % (i + 1))
        plt.subplot(1, 2, 2)
        plt.imshow(inv_IT[0, 0, :])
        plt.colorbar()
        plt.title('inv warped ITarget %i' % (i + 1))
        plt.show()

##SAVE MAPS AND INVERSE MAPS
FRT.write_h5file(results_path + mode + "_predicted_DeformationMaps.h5", maps_from_mom[0:])
FRT.write_h5file(results_path + mode + "_inv_predicted_DeformationMaps.h5", inv_maps_from_mom[0:])

##SAVE INVERSE WARPED ITARGET IMAGES
FRT.write_h5file(results_path + mode + "_Itarget_warped.h5", ITarget_warped)

if mode == 'train' and algorithm== 'stephanie':
    ##LOAD TARGET IMAGES
    t_images_list = FRT.find(mode + '*.nii.gz', inverse_warped_ITarget_path)
    t_nr_of_images = len(t_images_list)
    target_images = t_images_list[0:t_nr_of_images]
    print(t_nr_of_images, len(target_images))

    ##MULTI REGISTRATION
    _, mom_list = FRT.multi_registration(images, target_images, do_smoothing, second_registration_results_path,
                                         mode,
                                         plot_and_save_def_grids)

    ##SAVE MOMENTUMS FOR CORRECTION NETWORK
    FRT.write_h5file(results_path + mode + "_corrMomentums.h5", mom_list)