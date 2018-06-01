# This is part 1 of the fast registration training and testing.
# To be used to create train and test data for the prediction network.
#NOTE: assumes train data is loaded images[0-nr_of_train_images],
#               test data is loaded images[nr_of_train_images-end]
#
#IN:
# path and pattern: source images [train,test]
# path: target image
#
## LOAD IMAGES
## SET or COMPUTE TARGET IMAGE
## SINGLE REGISTRATION
## SAVE MOMENTUMS IN TRAIN & TEST SETS
## SAVE MAPS IN TRAIN & TEST SETS
## SAVE TARGET AND SOURCE IMAGES IN TRAIN & TEST SETS
#
#OUT:
# warped source images
# deformation grid plots (if True)
# average image
# train&test sets: Momentum
# train&test sets: Maps
# train&test sets: source images
# train&test sets: target image

import set_pyreg_paths
import fast_registration_tools as FRT
import pyreg.fileio as FIO
import pyreg.module_parameters as pars

# keep track of general parameters
params = pars.ParameterDict()

#----------------------------------------TO BE SET-----------------------------------------------
# set loading paths and parameters
source_images_path='../test_data/label_slices/'
source_images_pattern='*LEFT*label1.Sliced.nii.gz'
target_image_path='./fast_reg/results/Central_Segmentation.nii.gz'
set_or_compute_target_image='path' # 'path','central_segmentation','avg_image'
nr_of_train_images = 78
do_smoothing            = True
plot_and_save_def_grids = True

# CREATE FOLDERS and set saving paths
results_path= './fast_reg/results/'                                 # needed in ALL following skripts
registration_results_images_path='./fast_reg/results/RegImages/'    # needed in fast_registration_analysis.py
registration_results_deformations_path='./fast_reg/results/RegDeformationGrids/'
#----------------------------------------TO BE SET-----------------------------------------------


##LOAD IMAGES
images_list = FRT.find(source_images_pattern, source_images_path)
nr_of_images = len(images_list)
images = images_list[0:nr_of_images]
print(nr_of_images,len(images))

##SET or COMPUTE TARGET IMAGE
# use path to a given target image or compute the central segmentation or the average image as target image
target_image = set_or_compute_target_image

if target_image=='central_segmentation':
    # compute central segmentation as target image
    FRT.find_central_segmentation(images, target_image_path, visualize=False)
elif target_image=='avg_image':
    # compute average image as target image
    FRT.compute_average_image(images, target_image_path, visualize=False)

##SINGLE REGISTRATION
# register images to target image
_, wp_list, mom_list = FRT.single_registration(images, target_image_path, do_smoothing, registration_results_images_path,
                                               registration_results_deformations_path, plot_and_save_def_grids)

##SAVE MOMENTUMS IN TRAIN & TEST SETS
FRT.write_h5file(results_path + "train_Momentums.h5", mom_list[0:nr_of_train_images])
FRT.write_h5file(results_path + "test_Momentums.h5", mom_list[nr_of_train_images:])

##SAVE MAPS IN TRAIN & TEST SETS
FRT.write_h5file(results_path + "train_Maps.h5", wp_list[0:nr_of_train_images])
FRT.write_h5file(results_path + "test_Maps.h5", wp_list[nr_of_train_images:])

##SAVE TARGET AND SOURCE IMAGES IN TRAIN & TEST SETS
train_source_images = []
train_target_images = []
test_source_images = []
test_target_images = []
im_io = FIO.ImageIO()

Iavg,_, _, _ = im_io.read_to_nc_format(target_image_path, silent_mode=True)

for nr, im_name in enumerate(images):
    Ic, _, _, _ = im_io.read_to_nc_format(filename=im_name, silent_mode=True)
    if nr<=(nr_of_train_images-1):
        train_source_images.append(Ic.squeeze())
        train_target_images.append(Iavg.squeeze())
    else:
        test_source_images.append(Ic.squeeze())
        test_target_images.append(Iavg.squeeze())
FRT.write_h5file(results_path + "train_Isource.h5", train_source_images)
FRT.write_h5file(results_path + "train_Itarget.h5", train_target_images)
FRT.write_h5file(results_path + "test_Isource.h5", test_source_images)
FRT.write_h5file(results_path + "test_Itarget.h5", test_target_images)