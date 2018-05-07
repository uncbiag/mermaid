import set_pyreg_paths

import torch
from torch.autograd import Variable

import pyreg.image_sampling as IS

import pyreg.model_factory as MF

import pyreg.module_parameters as pars
from pyreg.data_wrapper import USE_CUDA, AdaptVal, MyTensor

import pyreg.fileio as FIO
import pyreg.visualize_registration_results as vizReg

import pyreg.utils as utils

import numpy as np

import matplotlib.pyplot as plt

import os
import fnmatch
import h5py
from torch.nn.parameter import Parameter
import pyreg.custom_optimizers as CO
import SimpleITK as sitk


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return sorted(result)

def _compute_low_res_image(I,spacing,low_res_size):
    sampler = IS.ResampleImage()
    low_res_image, _ = sampler.downsample_image_to_size(I, spacing, low_res_size[2::])
    return low_res_image

def _get_low_res_size_from_size(sz, factor):
    """
    Returns the corresponding low-res size from a (high-res) sz
    :param sz: size (high-res)
    :param factor: low-res factor (needs to be <1)
    :return: low res size
    """
    if (factor is None) or (factor>=1):
        print('WARNING: Could not compute low_res_size as factor was ' + str( factor ))
        return sz
    else:
        lowResSize = np.array(sz)
        lowResSize[2::] = (np.ceil((np.array(sz[2::]) * factor))).astype('int16')

        if lowResSize[-1]%2!=0:
            lowResSize[-1]-=1
            print('\n\nWARNING: forcing last dimension to be even: fix properly in the Fourier transform later!\n\n')

        return lowResSize

def _get_low_res_spacing_from_spacing(spacing, sz, lowResSize):
    """
    Computes spacing for the low-res parameterization from image spacing
    :param spacing: image spacing
    :param sz: size of image
    :param lowResSize: size of low re parameterization
    :return: returns spacing of low res parameterization
    """
    #todo: check that this is the correct way of doing it
    return spacing * (np.array(sz[2::])-1) / (np.array(lowResSize[2::])-1)

def individual_parameters_to_model_parameters(ind_pars):
    model_pars = dict()
    for par in ind_pars:
        model_pars[par['name']] = par['model_params']

    return model_pars

def concatenate_momentums(first_momentum_filepath, second_momentum_filepath, concatenated_filepath):
    """
    Computes the concatenated momentums from 2 input momentumfiles
    :param first_momentum_filepath: path to first momentum file
    :param second_momentum_filepath: path to second momentum file
    :param concatenated_filepath: filepath under which the concatenated momentums will be saved
    :return: returns the concatenated momentums in nparray
    """
    # load first momentums
    mom_1 = h5py.File(first_momentum_filepath, 'r')
    mom_1_data = mom_1['/dataset'][()]
    mom_1.close()
    sz_1 = mom_1_data.shape

    # load second momentums
    mom_2 = h5py.File(second_momentum_filepath, 'r')
    mom_2_data = mom_2['/dataset'][()]
    mom_2.close()
    sz_2 = mom_2_data.shape

    # check if files are compatible
    assert len(sz_1)==len(sz_2)
    for i in range(len(sz_1)):
        assert sz_1[i]==sz_2[i]

    # concatenate momentums
    mom_concatenated = mom_1_data + mom_2_data

    # save concatenated momentums in h5 file
    mom_c = h5py.File(concatenated_filepath, "w")
    mom_c.create_dataset("dataset", data=mom_concatenated)
    mom_c.close()

    return mom_concatenated

def evaluate_model(ISource_in,ITarget_in,sz,spacing,individual_parameters,shared_parameters,params,visualize=True,**inital_map):

    ISource = AdaptVal(ISource_in)
    ITarget = AdaptVal(ITarget_in)

    model_name = params['model']['registration_model']['type']
    use_map = params['model']['deformation']['use_map']
    map_low_res_factor = params['model']['deformation'][('map_low_res_factor', None, 'low_res_factor')]
    compute_similarity_measure_at_low_res = params['model']['deformation'][
        ('compute_similarity_measure_at_low_res', False, 'to compute Sim at lower resolution')]

    lowResSize = None
    lowResSpacing = None
    ##
    if map_low_res_factor == 1.0:
        map_low_res_factor = None
    ##
    if map_low_res_factor is not None:
        lowResSize = _get_low_res_size_from_size(sz, map_low_res_factor)
        lowResSpacing = _get_low_res_spacing_from_spacing(spacing, sz, lowResSize)

        lowResISource = _compute_low_res_image(ISource, spacing, lowResSize)
        # todo: can be removed to save memory; is more experimental at this point
        lowResITarget = _compute_low_res_image(ITarget, spacing, lowResSize)

    if map_low_res_factor is not None:
        # computes model at a lower resolution than the image similarity
        if compute_similarity_measure_at_low_res:
            mf = MF.ModelFactory(lowResSize, lowResSpacing, lowResSize, lowResSpacing)
        else:
            mf = MF.ModelFactory(sz, spacing, lowResSize, lowResSpacing)
    else:
        # computes model and similarity at the same resolution
        mf = MF.ModelFactory(sz, spacing, sz, spacing)

    model, criterion = mf.create_registration_model(model_name, params['model'])
    # set it to evaluation mode
    model.eval()

    print(model)

    if use_map:
        # create the identity map [-1,1]^d, since we will use a map-based implementation
    ##  ##
        if inital_map:
            id = inital_map
        else:
            id = utils.identity_map_multiN(sz, spacing)
    ##  ##
        identityMap = AdaptVal(Variable(torch.from_numpy(id), requires_grad=False))
        if map_low_res_factor is not None:
            # create a lower resolution map for the computations
            lowres_id = utils.identity_map_multiN(lowResSize, lowResSpacing)
            lowResIdentityMap = AdaptVal(Variable(torch.from_numpy(lowres_id), requires_grad=False))

    if USE_CUDA:
        model = model.cuda()

    dictionary_to_pass_to_integrator = dict()

    if map_low_res_factor is not None:
        dictionary_to_pass_to_integrator['I0'] = lowResISource
        dictionary_to_pass_to_integrator['I1'] = lowResITarget
    else:
        dictionary_to_pass_to_integrator['I0'] = ISource
        dictionary_to_pass_to_integrator['I1'] = ITarget

    model.set_dictionary_to_pass_to_integrator(dictionary_to_pass_to_integrator)

    model.set_shared_registration_parameters(shared_parameters)
    ##model_pars = individual_parameters_to_model_parameters(individual_parameters)
    model_pars = individual_parameters
    model.set_individual_registration_parameters(model_pars)

    # now let's run the model
    rec_IWarped = None
    rec_phiWarped = None

    if use_map:
        if map_low_res_factor is not None:
            if compute_similarity_measure_at_low_res:
                rec_phiWarped = model(lowResIdentityMap, lowResISource)
            else:
                rec_tmp = model(lowResIdentityMap, lowResISource)
                # now upsample to correct resolution
                desiredSz = identityMap.size()[2::]
                sampler = IS.ResampleImage()
                rec_phiWarped, _ = sampler.upsample_image_to_size(rec_tmp, spacing, desiredSz)
        else:
            rec_phiWarped = model(identityMap, ISource)

    else:
        rec_IWarped = model(ISource)

    if use_map:
        rec_IWarped = utils.compute_warped_image_multiNC(ISource, rec_phiWarped, spacing)

    if use_map and map_low_res_factor is not None:
        vizImage, vizName = model.get_parameter_image_and_name_to_visualize(lowResISource)
    else:
        vizImage, vizName = model.get_parameter_image_and_name_to_visualize(ISource)

    if use_map:
        phi_or_warped_image = rec_phiWarped
    else:
        phi_or_warped_image = rec_IWarped

    visual_param = {}
    visual_param['visualize'] = visualize
    visual_param['save_fig'] = False

    if use_map:
        if compute_similarity_measure_at_low_res:
            I1Warped = utils.compute_warped_image_multiNC(lowResISource, phi_or_warped_image, lowResSpacing)
            vizReg.show_current_images(iter, lowResISource, lowResITarget, I1Warped, vizImage, vizName,
                                       phi_or_warped_image, visual_param)
        else:
            I1Warped = utils.compute_warped_image_multiNC(ISource, phi_or_warped_image, spacing)
            vizReg.show_current_images(iter, ISource, ITarget, I1Warped, vizImage, vizName,
                                       phi_or_warped_image, visual_param)
    else:
        vizReg.show_current_images(iter, ISource, ITarget, phi_or_warped_image, vizImage, vizName, None, visual_param)

    return rec_IWarped,rec_phiWarped

def invert_map(map,spacing):
    """
    Inverts the map and returns its inverse. Assumes standard map parameterization [-1,1]^d
    :param map: Input map to be inverted
    :return: inverted map
    """
    # make pytorch arrays for subsequent processing
    map_t = AdaptVal(Variable(torch.from_numpy(map), requires_grad=False))

    # identity map
    id = utils.identity_map_multiN(map_t.data.shape,spacing)
    id_t = AdaptVal(Variable(torch.from_numpy(id),requires_grad=False))

    # parameter to store the inverse map
    invmap_t = AdaptVal(Parameter(torch.from_numpy(id.copy())))

    # some optimizer settings, probably too strict
    nr_of_iterations = 100
    rel_ftol = 1e-8
    optimizer = CO.LBFGS_LS([invmap_t],lr=1, max_iter=1, tolerance_grad=rel_ftol * 10, tolerance_change=rel_ftol, max_eval=10,history_size=30, line_search_fn='backtracking')
    # optimizer = torch.optim.SGD([invmap_t], lr=0.0001, momentum=0.9, dampening=0, weight_decay=0,nesterov=True)
    # optimizer = torch.optim.Adam([invmap_t], lr=0.00001, betas=(0.9, 0.999), eps=rel_ftol, weight_decay=0)

    def compute_loss():
        # warps map_t with inv_map, if it is the inverse should result in the identity map
        wmap = utils.compute_warped_image_multiNC(map_t, invmap_t, spacing)
        current_loss = ((wmap-id_t)**2).sum()
        return current_loss

    def _closure():
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        return loss

    last_loss = utils.t2np( compute_loss() )

    lossplot=[]
    for iter in range(nr_of_iterations):
        lossplot.append(last_loss)
        optimizer.step(_closure )
        current_loss = utils.t2np( compute_loss() )
        print( 'Iter = ' + str( iter ) + '; E = ' + str( current_loss ) )
        if ( current_loss >= last_loss ):
            break
        else:
            last_loss = current_loss

    plt.plot(range(len(lossplot)),lossplot)
    plt.show()

    return invmap_t.data.numpy(), id

def read_image_and_map_and_apply_map(image_filename,map_filename):
    """
    Reads an image and a map and applies the map to an image
    :param image_filename: input image filename
    :param map_filename: input map filename
    :return: the warped image and its image header as a tuple (im,hdr)
    """

    im_warped = None
    map,map_hdr,_,_ = FIO.MapIO().read_to_nc_format(map_filename)
    #map,map_hdr,_,_ = FIO.ImageIO().read_to_nc_format(filename=map_filename)
    #im,hdr,_,_ = FIO.ImageIO().read_to_map_compatible_format(image_filename,map)
    im, hdr, _, _ = FIO.ImageIO().read_to_nc_format(image_filename)

    spacing = hdr['spacing']
    #TODO: check that the spacing is compatible with the map

    if (im is not None) and (map is not None):
        # make pytorch arrays for subsequent processing
        im_t = AdaptVal(Variable(torch.from_numpy(im[0,:]), requires_grad=False))
        map_t = AdaptVal(Variable(torch.from_numpy(map[0,:]), requires_grad=False))
        im_warped = utils.t2np( utils.compute_warped_image_multiNC(im_t,map_t,spacing[-2:]) )

        return im_warped,hdr
    else:
        print('Could not read map or image')
        return None,None


##LOAD IMAGES
# first get a few files to build the atlas from
images_list = find('*LEFT*Label1.sliced.nii.gz', '../test_data/label_slices/')
nr_of_images = len(images_list)
images = images_list[78:nr_of_images]
print(nr_of_images,len(images))

##CONCATENATE MOMENTUMS
first_momentums ='/playpen/shaeger/prediction/GW_knee_test_results/Last8/Prediction/GW_CentralMom30_lr2e4_Prediction_results_last8.h5'
second_momentums ='/playpen/shaeger/prediction/GW_knee_test_results/Last8/Correction/GW_CentralMom30_lr2e4_CorrectionPrediction_results_last8.h5'
conc_momentums = '/playpen/shaeger/prediction/GW_knee_test_results/Last8/GW_concatenatedMomomentums_last8.h5'
conc_mom_data = concatenate_momentums(first_momentums, second_momentums, conc_momentums)

##INITIAL MOMENTUMS
# load initial momentums from the prediction net
pred = h5py.File('/playpen/shaeger/prediction/GW_knee_test_results/Last8/GW_concatenatedMomomentums_last8.h5', 'r')
pred_data = pred['/dataset'][()]
pred.close()

##APPLY MODEL
im_io = FIO.ImageIO()
Iavg, hdrc, spacing, _ = im_io.read_to_nc_format('./central/Central_Segmentation.nrrd')

size = np.shape(Iavg[0,:])
parameters='apply_model_params_GW.json'
params = pars.ParameterDict()
params.load_JSON(parameters)
shared_parameters = dict()
maps_from_mom=[]
images_from_mom=[]
for i, im_name in enumerate(images):
    momDict = dict()
    momDict['m'] = torch.from_numpy(pred_data[i, :])
    print('Registering image ' + str(i + 1) + '/' + str(len(images)))
    Ic, hdrc, spacing, _ = im_io.read_to_nc_format(filename=im_name)

    a, b = evaluate_model(AdaptVal(Variable(torch.from_numpy(Ic), requires_grad=False)),
                          AdaptVal(Variable(torch.from_numpy(Iavg[0,:]), requires_grad=False)),
                          size,
                          spacing,
                          momDict,
                          shared_parameters,
                          params,
                          visualize=False)
    # plt.subplot(1,3,1)
    # plt.imshow(a.data.numpy()[0,0,:])
    # plt.colorbar()
    # plt.title('Iwarped '+str(i+1))
    # plt.subplot(1,3,2)
    # plt.imshow(b.data.numpy()[0,0,:])
    # plt.colorbar()
    # plt.title('map x')
    # plt.subplot(1, 3, 3)
    # plt.imshow(b.data.numpy()[0,1, :])
    # plt.colorbar()
    # plt.title('map y')
    # plt.show()

    maps_from_mom.append(b.data.numpy())

    warped_filename = "./central/GaussianWeights/CorrectionResults/IW_CorrPredMom30_lr2e4_last8_image" + str(i+1) + ".nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(a.data.numpy()[0,:]), warped_filename)

#SAVE MAPS
g = h5py.File("./central/GaussianWeights/CorrectionResults/CorrRegMaps_PredMom30_lr2e4_last8.h5", "w")
train_dset = g.create_dataset("dataset", data=maps_from_mom[0:])
g.close()

##COMPUTE INVERSE MAPS
# load initial maps from the momentums from the prediction net
predMaps = h5py.File("./central/GaussianWeights/GW_RegMaps_PredMom30_lr2e4_first78.h5", 'r')
predMaps_data = predMaps['/dataset'][()]
predMaps.close()

inverse_predMaps = []

avg_test_error = 0
for i in range(predMaps_data.shape[0]):
    inv_predMaps, id = invert_map(predMaps_data[i,:],spacing[-2:])
    inverse_predMaps.append(inv_predMaps)


    # inv_min_id=inv_predMaps-id

    # plt.subplot(2, 4, 1)
    # plt.imshow(predMaps_data[i,0,0,:])
    # plt.colorbar()
    # plt.title('map x')
    #
    # plt.subplot(2,4,2)
    # plt.imshow(predMaps_data[i,0,1, :])
    # plt.colorbar()
    # plt.title('map y')
    #
    # plt.subplot(2, 4, 3)
    # plt.imshow(predMaps_data[i,0,0, :]-id[0,0,:])
    # plt.colorbar()
    # plt.title('map x -id')
    #
    # plt.subplot(2,4,4)
    # plt.imshow(predMaps_data[i,0,1, :]-id[0,1,:])
    # plt.colorbar()
    # plt.title('map y -id')
    #
    # plt.subplot(2, 4, 5)
    # plt.imshow(inv_predMaps[0, 0, :])
    # plt.colorbar()
    # plt.title('inv map x')
    #
    # plt.subplot(2, 4, 6)
    # plt.imshow(inv_predMaps[0, 1, :])
    # plt.colorbar()
    # plt.title('inv map y')
    #
    # plt.subplot(2, 4, 7)
    # plt.imshow(inv_min_id[0, 0, :])
    # plt.colorbar()
    # plt.title('inv map x -id')
    #
    # plt.subplot(2, 4, 8)
    # plt.imshow(inv_min_id[0, 1, :])
    # plt.colorbar()
    # plt.title('inv map y -id')
    # plt.show()
    #
    save_path = "./central/GaussianWeights/RegMaps_inverse/GW_RegMaps_inverse_predMom30_lr2e4_first78_image" + str(i+1).zfill(3) + ".nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(inv_predMaps[0,:]), save_path)

    # save = "./central/GaussianWeights/RegMaps/GW_RegMaps_predMom30_lr2e4_first78_image" + str(i+1).zfill(3) + ".nii.gz"
    # sitk.WriteImage(sitk.GetImageFromArray(predMaps_data[i,0,:]), save)


    # concatenate_Map_invMap_part1,_ = read_image_and_map_and_apply_map('./central/Central_Segmentation.nrrd', save)
    # name1 = "./central/GaussianWeights/TestInversion/TestConc1_first78_image" + str(i+1).zfill(3) + ".nrrd"
    # sitk.WriteImage(sitk.GetImageFromArray(concatenate_Map_invMap_part1[0,:]), name1)
    # concatenate_Map_invMap_part2 = read_image_and_map_and_apply_map(name1, save_path)
    # name2 = "./central/GaussianWeights/TestInversion/TestConc2_first78_image" + str(i+1).zfill(3) + ".nrrd"
    # sitk.WriteImage(sitk.GetImageFromArray(concatenate_Map_invMap_part2[0][0,:]), name2)

    # test = (concatenate_Map_invMap_part2[0][0,0,:]-Iavg[0,0,0,:])[150:250,:]
    # test_error = sum(sum(abs(test)))
    # avg_test_error+=test_error
    #
    # plt.imshow(test)
    # plt.colorbar()
    # plt.title('test image %i' %(i+1) +', error %d' %test_error)
    # plt.savefig('./central/GaussianWeights/TestInversion/InversionError_first78_image' + str(i+1).zfill(3) + '.png')
    # plt.show()
    # plt.clf()
# avg_test_error = avg_test_error/predMaps_data.shape[0]
# print('avgerage error when inverting maps: %d' %avg_test_error)

##SAVE INVERSE MAPS
f = h5py.File("./central/GaussianWeights/GW_RegMaps_inverse_predMom30_lr2e4_last8.h5", "w")
invMaps_dset = f.create_dataset("dataset", data=inverse_predMaps[0:])
f.close()

##COMPUTE WARPED IMAGES OF INVERSE MAPS
ITarget_warped = []
for i in range(8):
    map_filename = "./central/GaussianWeights/RegMaps_inverse/GW_RegMaps_inverse_predMom30_lr2e4_last8_image" + str(i+1).zfill(3) + ".nii.gz"
    Iwarped_inverseMap, _ = read_image_and_map_and_apply_map('./central/Central_Segmentation.nrrd', map_filename)

    Ic, hdrc, spacing, _ = FIO.ImageIO().read_to_nc_format(images[i])

    plt.subplot(1,2,1)
    plt.imshow(Ic[0,0,:])
    plt.colorbar()
    plt.title('Image %i' % (i+1))
    plt.subplot(1,2,2)
    plt.imshow(Iwarped_inverseMap[0,0,:])
    plt.colorbar()
    plt.title('Iwarped inv Map')
    plt.show()

    ITarget_warped.append(Iwarped_inverseMap.squeeze())

    warped_filename = "./central/GaussianWeights/ITarget_warped/GW_ITarget_warped_predMom30_lr1e4_last8_image" + str(i+1).zfill(3) + ".nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(Iwarped_inverseMap[0,:]), warped_filename)

##SAVE ITarget warped images in h5 file
ITw = h5py.File("./central/GaussianWeights/corrMom_test_Itarget.h5", "w")
ITw.create_dataset("dataset", data=ITarget_warped)
ITw.close()
