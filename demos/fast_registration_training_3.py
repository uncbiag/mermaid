# This is part 3 of the fast registration training and testing.
# To be used to combine the results of the prediction and the correction network.
#NOTE: assumes train data is loaded images[0-nr_of_train_images],
#               test data is loaded images[nr_of_train_images-end]
#
#IN:
# path and pattern: source images [train,test]
# path: target image
# path: prediction net momentum train&test sets
# path: correction net momentum train&test sets
# choose mode: train/test
#
#   algorithm 'stephanie':                                       algorithm 'marc':
## LOAD IMAGES                                                  LOAD IMAGES
## CONCATENATE MOMENTUM                                         CONCATENATE MOMENTUM
## APPLY MODEL                                                  APPLY MODEL
## COMPUTE MAP AND WARPED IMAGE OF CORRECTED PRED MOMENTUM      COMPUTE MAP AND WARPED IMAGE OF CORRECTED PRED MOMENTUM
## SAVE MAPS                                                    SAVE MAPS
#
#OUT:
# warped source images
# deformation maps
# data set: concatenated momentum
# data set: corrected predicted deformation map


import set_pyreg_paths
import fast_registration_tools as FRT
import torch
from torch.autograd import Variable
from pyreg.data_wrapper import AdaptVal, MyTensor
import pyreg.fileio as FIO
import pyreg.module_parameters as pars
import numpy as np
import matplotlib.pyplot as plt


# keep track of general parameters
params = pars.ParameterDict()

#----------------------------------------TO BE SET-----------------------------------------------
# set loading paths and parameters
source_images_path='../test_data/label_slices/'
source_images_pattern='*LEFT*label1.Sliced.nii.gz'
target_image_path='./fast_reg/results/Central_Segmentation.nii.gz'
mode = 'train'#'test'#
predicted_momentum_train_path ='./fast_reg/results/predMom_trainset.h5'
predicted_momentum_test_path ='./fast_reg/results/predMom_testset.h5'
corrected_momentum_train_path='./fast_reg/results/correction_predMom_trainset.h5'
corrected_momentum_test_path='./fast_reg/results/correction_predMom_testset.h5'
sub_momentum_train_path='./fast_reg/results/sub_predMom_trainset.h5'
sub_momentum_test_path='./fast_reg/results/sub_predMom_testset.h5'
nr_of_train_images = 78
nr_of_test_images  = 8
plot_deformation_maps = False
###
algorithm='stephanie'#'marc' #
###

# set saving paths
results_path='./fast_reg/results/'

# CREATE FOLDERS and set saving paths
correction_results_path ="./fast_reg/results/wImages_correction/" #needed in fast_registration_analysis.py
#----------------------------------------TO BE SET-----------------------------------------------


##LOAD IMAGES
images_list = FRT.find(source_images_pattern, source_images_path)
nr_of_images = len(images_list)
if mode=='train':
    images = images_list[0:nr_of_train_images]
elif mode=='test':
    images = images_list[-nr_of_test_images:]
print(nr_of_images,len(images))

##CONCATENATE MOMENTUM
if algorithm=='stephanie':
    conc_momentum_path = results_path+mode+"_concatenated_Mom.h5"
    if mode=='train':
        conc_momentum = FRT.concatenate_momentums(predicted_momentum_train_path, corrected_momentum_train_path, conc_momentum_path)
    elif mode=='test':
        conc_momentum = FRT.concatenate_momentums(predicted_momentum_test_path, corrected_momentum_test_path, conc_momentum_path)
if algorithm == 'marc':
    conc_momentum_path = results_path + mode + "_sub_concatenated_Mom.h5"
    if mode == 'train':
        conc_momentum = FRT.concatenate_momentums(predicted_momentum_train_path, sub_momentum_train_path,
                                                  conc_momentum_path)
    elif mode == 'test':
        conc_momentum = FRT.concatenate_momentums(predicted_momentum_test_path, sub_momentum_test_path,
                                                      conc_momentum_path)

##APPLY MODEL
im_io = FIO.ImageIO()
Iavg, hdrc, spacing, _ = im_io.read_to_nc_format(target_image_path)

size = np.shape(Iavg)
parameters='fast_registration_eval_model_params.json'
params = pars.ParameterDict()
params.load_JSON(parameters)
shared_parameters = dict()

maps_from_mom=[]
ITarget_warped = []
corrected_Images = []

for i, im_name in enumerate(images):
    momDict = dict()
    momDict['m'] = torch.from_numpy(conc_momentum[i, :])
    print('Registering image ' + str(i + 1) + '/' + str(len(images)))
    Ic, hdrc, spacing, _ = im_io.read_to_nc_format(filename=im_name, silent_mode=True)

    ##COMPUTE MAP AND WARPED IMAGE OF CORRECTED PREDICTED MOMENTUM
    Iwarped, map_from_mom, _ = FRT.evaluate_model(AdaptVal(torch.from_numpy(Ic)),
                          AdaptVal(torch.from_numpy(Iavg)),
                          size,
                          spacing,
                          momDict,
                          shared_parameters,
                          params,
                          visualize=False,
                          compute_inverse_map=False)

    maps_from_mom.append(map_from_mom.detach().cpu().numpy())
    corrected_Images.append(Iwarped.detach().cpu().numpy().squeeze())

    if algorithm=='stephanie':
        warped_IS_filename = correction_results_path+mode+"_correction_predMom_warped_image" + str(i+1) + ".nii.gz"
        FIO.ImageIO().write(warped_IS_filename,Iwarped.detach().cpu().numpy()[0,0,:],hdrc)
    elif algorithm=='marc':
        warped_IS_filename = correction_results_path + mode + "_sub_predMom_warped_image" + str(
            i + 1) + ".nii.gz"
        FIO.ImageIO().write(warped_IS_filename, Iwarped.detach().cpu().numpy()[0, 0, :], hdrc)

    if plot_deformation_maps:
        plt.clf()
        plt.imshow(Iwarped[0,0,125:225, :], cmap='gray')  # just for background purpose
        plt.contour(map_from_mom.detach().cpu().numpy()[0,0,:][125:225, :], np.linspace(-1, 1, 200), colors='r', linestyles='solid')
        plt.contour(map_from_mom.detach().cpu().numpy()[0,1,:][125:225, :], np.linspace(-1, 1, 200), colors='r', linestyles='solid')
        plt.title('predicted deformation field img %i' % (i + 1), fontsize=8)
        plt.show()

##SAVE MAPS
if algorithm == 'stephanie':
    FRT.write_h5file(results_path+mode+"_corrected_predicted_DeformationMaps.h5", maps_from_mom[0:])
    FRT.write_h5file(results_path + mode + "_corrected_predicted_Images.h5", corrected_Images)
elif algorithm=='marc':
    FRT.write_h5file(results_path+mode+"_sub_predicted_DeformationMaps.h5", maps_from_mom[0:])
    FRT.write_h5file(results_path + mode + "_sub_predicted_Images.h5", corrected_Images)
