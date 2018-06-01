# This is to be used for comparison of the results of the prediction and the correction network for both algorithms.
#NOTE: assumes train data is loaded images[0-nr_of_train_images],
#               test data is loaded images[nr_of_train_images-end]
#
#IN:
# path and pattern: source images [train,test]
# path: target image
# path: warped images from predicted momentum
# path: warped images from corrected momentum
#
## LOAD IMAGES
## LOAD WARPED IMAGES FROM PREDICTED MOMENTUM
## LOAD WARPED IMAGES FROM CORRECTED PREDICTED MOMENTUM STEPHANIE
## LOAD WARPED IMAGES FROM CORRECTED PREDICTED MOMENTUM MARC
## COMPUTE AVERAGE IMAGES
## PLOT WARPED IMAGES AND DEFORMATION GRIDS FOR TEST IMAGES
#
#OUT:
# plot warped images and deformation grids for test images


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
reg_images_path='./fast_reg/results/RegImages/'
reg_images_pattern='*regImage*.nii.gz'
target_image_path='./fast_reg/results/Central_Segmentation.nii.gz'
Iwarped_predicted_momentum_path ="./fast_reg/results/wImages_prediction/"
Iwarped_corrected_predicted_momentum_path ="./fast_reg/results/wImages_correction/"

nr_of_train_images = 78
nr_of_test_images  = 8
#----------------------------------------TO BE SET-----------------------------------------------


##LOAD REGISTERED IMAGES
images_list = FRT.find(reg_images_pattern, reg_images_path)
nr_of_images = len(images_list)
images = images_list[0:nr_of_images]
print(nr_of_images,len(images))

##LOAD WARPED IMAGES FROM PREDICTED MOMENTUM
pred_images_list_train = FRT.find('train*', Iwarped_predicted_momentum_path)
pred_images_list_test = FRT.find('test*', Iwarped_predicted_momentum_path)
pred_images = pred_images_list_train+pred_images_list_test
print(len(pred_images))

##LOAD WARPED IMAGES FROM CORRECTED PREDICTED MOMENTUM
cor_pred_images_list_train = FRT.find('train_correction_pred*', Iwarped_corrected_predicted_momentum_path)
cor_pred_images_list_test = FRT.find('test_correction_pred*', Iwarped_corrected_predicted_momentum_path)
cor_pred_images = cor_pred_images_list_train+cor_pred_images_list_test
print(len(pred_images))

##LOAD WARPED IMAGES FROM CORRECTED PREDICTED MOMENTUM
sub_pred_images_list_train = FRT.find('train_sub_pred*', Iwarped_corrected_predicted_momentum_path)
sub_pred_images_list_test = FRT.find('test_sub_pred*', Iwarped_corrected_predicted_momentum_path)
sub_pred_images = sub_pred_images_list_train+sub_pred_images_list_test
print(len(pred_images))

##COMPUTE AVERAGE IMAGES
FRT.compute_average_image(images, './fast_reg/results/Average_Reg.nii.gz', visualize=True)
FRT.compute_average_image(pred_images, './fast_reg/results/Average_pred_Mom.nii.gz', visualize=True)
FRT.compute_average_image(cor_pred_images, './fast_reg/results/Average_cor_pred_Mom.nii.gz', visualize=True)
FRT.compute_average_image(sub_pred_images, './fast_reg/results/Average_sub_pred_Mom.nii.gz', visualize=True)

##PLOT WARPED IMAGES AND DEFORMATION GRIDS FOR TEST IMAGES
deff = FRT.read_h5file('./fast_reg/results/test_Maps.h5')
deff_pred = FRT.read_h5file('./fast_reg/results/test_predicted_DeformationMaps.h5')
deff_cor_pred = FRT.read_h5file('./fast_reg/results/test_corrected_predicted_DeformationMaps.h5')
deff_sub = FRT.read_h5file('./fast_reg/results/test_sub_predicted_DeformationMaps.h5')

im_io = FIO.ImageIO()
ITarget,_,_,_ = im_io.read_to_nc_format(target_image_path, silent_mode=True)

for j in range(nr_of_test_images):
    t=j
    i=j+nr_of_train_images
    Ic, _, _, _ = im_io.read_to_nc_format(images[i], silent_mode=True)
    Ic_p, _, _, _ = im_io.read_to_nc_format(pred_images[i], silent_mode=True)
    Ic_cp, _, _, _ = im_io.read_to_nc_format(cor_pred_images[i], silent_mode=True)
    Ic_s, _, _, _ = im_io.read_to_nc_format(sub_pred_images[i], silent_mode=True)

    print('analyzing image: %i' %(i+1))
    plt.clf()
    plt.figure()
    plt.subplot(141)
    plt.imshow(Ic[0,0,:])
    plt.title('reg image %i' %(i+1))
    plt.subplot(142)
    plt.imshow(Ic_p[0,0,:])
    plt.title('predMom')
    plt.subplot(143)
    plt.imshow(Ic_cp[0,0,:])
    plt.title('corPredMom')
    plt.subplot(144)
    plt.imshow(Ic_s[0,0,:])
    plt.title('subMom')
    plt.show()

    plt.clf()
    plt.figure()
    plt.subplot(221)
    plt.imshow(ITarget[0,0,125:225, :], cmap='gray')
    plt.contour(deff[t,0, 0, :][125:225, :], np.linspace(-1, 1, 200), colors='r', linestyles='solid')
    plt.contour(deff[t,0, 1, :][125:225, :], np.linspace(-1, 1, 200), colors='r', linestyles='solid')
    plt.title('reg def %i' % (i + 1))
    plt.subplot(222)
    plt.imshow(ITarget[0,0,125:225, :], cmap='gray')
    plt.contour(deff_pred[t,0, 0, :][125:225, :], np.linspace(-1, 1, 200), colors='r', linestyles='solid')
    plt.contour(deff_pred[t,0, 1, :][125:225, :], np.linspace(-1, 1, 200), colors='r', linestyles='solid')
    plt.title('pred def %i' % (i + 1))
    plt.subplot(223)
    plt.imshow(ITarget[0,0,125:225, :], cmap='gray')
    plt.contour(deff_cor_pred[t,0, 0, :][125:225, :], np.linspace(-1, 1, 200), colors='r', linestyles='solid')
    plt.contour(deff_cor_pred[t,0, 1, :][125:225, :], np.linspace(-1, 1, 200), colors='r', linestyles='solid')
    plt.title('cor def %i' % (i + 1))
    plt.subplot(224)
    plt.imshow(ITarget[0,0,125:225, :], cmap='gray')
    plt.contour(deff_sub[t,0, 0, :][125:225, :], np.linspace(-1, 1, 200), colors='r', linestyles='solid')
    plt.contour(deff_sub[t,0, 1, :][125:225, :], np.linspace(-1, 1, 200), colors='r', linestyles='solid')
    plt.title('sub def %i' % (i + 1))
    plt.show()