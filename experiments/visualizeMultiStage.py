import set_pyreg_paths
import torch
from torch.autograd import Variable

import pyreg.smoother_factory as SF
import pyreg.deep_smoothers as DS
import pyreg.utils as utils
import pyreg.image_sampling as IS

import pyreg.model_factory as MF

import pyreg.module_parameters as pars
from pyreg.data_wrapper import USE_CUDA, AdaptVal, MyTensor

import pyreg.fileio as FIO
import pyreg.visualize_registration_results as vizReg

import numpy as np

import matplotlib.pyplot as plt

import os


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


def _load_current_source_and_target_images_as_variables(current_source_filename,current_target_filename,params):
    # now load them
    intensity_normalize = params['model']['data_loader'][('intensity_normalize', True, 'normalized image intensities')]
    normalize_spacing = params['model']['data_loader'][('normalize_spacing', True, 'normalized image spacing')]
    squeeze_image = params['model']['data_loader'][('squeeze_image', False, 'squeeze image dimensions')]

    im_io = FIO.ImageIO()

    ISource, hdr, spacing, _ = im_io.read_batch_to_nc_format([current_source_filename],
                                                             intensity_normalize=intensity_normalize,
                                                             squeeze_image=squeeze_image,
                                                             normalize_spacing=normalize_spacing,
                                                             silent_mode=True)
    ITarget, hdr, spacing, _ = im_io.read_batch_to_nc_format([current_target_filename],
                                                             intensity_normalize=intensity_normalize,
                                                             squeeze_image=squeeze_image,
                                                             normalize_spacing=normalize_spacing,
                                                             silent_mode=True)

    sz = np.array(ISource.shape)

    ISource = Variable(torch.from_numpy(ISource), requires_grad=False)
    ITarget = Variable(torch.from_numpy(ITarget), requires_grad=False)

    return ISource,ITarget,sz,spacing

def individual_parameters_to_model_parameters(ind_pars):
    model_pars = dict()
    for par in ind_pars:
        model_pars[par['name']] = par['model_params']

    return model_pars


def evaluate_model(ISource,ITarget,sz,spacing,individual_parameters,shared_parameters,params,visualize=True):

    model_name = params['model']['registration_model']['type']
    use_map = params['model']['deformation']['use_map']
    map_low_res_factor = params['model']['deformation'][('map_low_res_factor', None, 'low_res_factor')]
    compute_similarity_measure_at_low_res = params['model']['deformation'][
        ('compute_similarity_measure_at_low_res', False, 'to compute Sim at lower resolution')]

    lowResSize = None
    lowResSpacing = None

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
    print(model)

    if use_map:
        # create the identity map [-1,1]^d, since we will use a map-based implementation
        id = utils.identity_map_multiN(sz, spacing)
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
    model_pars = individual_parameters_to_model_parameters(individual_parameters)
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

stage = 0 # 0,1,2

json_file = '../experiments/test.json'
output_dir = '../experiments/test_out'

json_path,json_filename = os.path.split(json_file)

json_stage_1_in = json_file
json_stage_2_in = os.path.join(json_path,'out_stage_1_' + json_filename)
json_stage_3_in = os.path.join(json_path,'out_stage_2_' + json_filename)

json_for_stages = []
json_for_stages.append(json_stage_1_in)
json_for_stages.append(json_stage_2_in)
json_for_stages.append(json_stage_3_in)

output_dir_stage_3 = output_dir
output_dir_stage_2 = output_dir_stage_3 + '_after_stage_2'
output_dir_stage_1 = output_dir_stage_3 + '_after_stage_1'

output_dir_for_stages = []
output_dir_for_stages.append(output_dir_stage_1)
output_dir_for_stages.append(output_dir_stage_2)
output_dir_for_stages.append(output_dir_stage_3)

# current case

current_json = json_for_stages[stage]
individual_dir = os.path.join(output_dir_for_stages[stage],'individual')
shared_dir = os.path.join(output_dir_for_stages[stage],'shared')

# load the data

# load the configuration
params = pars.ParameterDict()
params.load_JSON(current_json)

# load the shared parameters
shared_parameters = torch.load(os.path.join(shared_dir,'shared_parameters.pt'))

# load the individual parameters
pair_nr = 0
individual_parameters_filename = os.path.join(individual_dir,'individual_parameter_pair_{:05d}.pt'.format(pair_nr))

individual_parameters = torch.load(individual_parameters_filename)

# load the mapping to the images
used_pairs = torch.load(os.path.join(output_dir,'used_image_pairs.pt'))

# load the image with given pair number
current_source_filename = used_pairs['source_images'][pair_nr]
current_target_filename = used_pairs['target_images'][pair_nr]

ISource,ITarget,sz,spacing = _load_current_source_and_target_images_as_variables(current_source_filename,current_target_filename,params)

visualize = True
IWarped,phi = evaluate_model(ISource,ITarget,sz,spacing,individual_parameters,shared_parameters,params,visualize)


