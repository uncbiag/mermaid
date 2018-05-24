from __future__ import print_function
from builtins import str
from builtins import range

import set_pyreg_paths
import multiprocessing as mp

# needs to be imported before matplotlib to assure proper plotting
import pyreg.visualize_registration_results as vizReg

import torch
from torch.autograd import Variable

import pyreg.utils as utils
import pyreg.image_sampling as IS
import pyreg.finite_differences as FD

import pyreg.model_factory as MF

import pyreg.module_parameters as pars
from pyreg.data_wrapper import USE_CUDA, AdaptVal, MyTensor

import pyreg.fileio as FIO


import pyreg.utils as utils

import numpy as np


import matplotlib.pyplot as plt

import os

def visualize_filter_grid(filter,title=None,print_figures=False,fname=None):
    nr_of_channels = filter.size()[1]
    nr_of_features_1 = filter.size()[0]

    assert( nr_of_channels==1 )

    # determine grid size
    nr_x = np.ceil(np.sqrt(nr_of_features_1)).astype('int')
    nr_y = nr_x

    plt.clf()

    for f in range(nr_of_features_1):
        plt.subplot(nr_y, nr_x, f+1)
        plt.imshow(filter[f, 0, ...], cmap='gray')
        plt.colorbar()
        plt.axis('off')

    plt.subplots_adjust(wspace=0.5, hspace=1)

    if title is not None:
        plt.suptitle( title )

    if print_figures:
        if fname is None:
            fname = 'filters_w1.pdf'
        plt.savefig(fname)
    else:
        plt.show()


def visualize_filter(filter,title=None,print_figures=False):
    nr_of_gaussians = filter.size()[1]
    nr_of_features_1 = filter.size()[0]

    for c in range(nr_of_gaussians):
        for r in range(nr_of_features_1):
            cp = 1 + c * nr_of_features_1 + r
            plt.subplot(nr_of_gaussians, nr_of_features_1, cp)
            plt.imshow(filter[r, c, ...], cmap='gray')
            plt.axis('off')

    plt.subplots_adjust(wspace=0.5, hspace=1)

    if title is not None:
        plt.suptitle( title )

    if print_figures:
        plt.savefig('filters_w2.pdf')
    else:
        plt.show()

def compute_overall_std(weights,stds):
    szw = weights.size()
    ret = torch.zeros(szw[1:])

    for i,s in enumerate(stds):
        ret += (weights[i,...])*(s**2)

    # now we have the variances, so take the sqrt
    return torch.sqrt(ret)

def get_array_from_set_of_lists(dat, nr):
    res = []
    for n in range(len(dat)):
        res.append(dat[n][nr])
    return res

def compute_mask(im):
    '''
    computes a mask by finding all the voxels where the image is exactly zero

    :param im:
    :return:
    '''
    mask = np.zeros_like(im)
    mask[im!=0] = 1
    mask[im==0] = np.nan

    return mask


def _compute_low_res_image(I,spacing,low_res_size,spline_order):
    sampler = IS.ResampleImage()
    low_res_image, _ = sampler.downsample_image_to_size(I, spacing, low_res_size[2::],spline_order)
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
    intensity_normalize = params['data_loader'][('intensity_normalize', True, 'normalized image intensities')]
    normalize_spacing = params['data_loader'][('normalize_spacing', True, 'normalized image spacing')]
    squeeze_image = params['data_loader'][('squeeze_image', False, 'squeeze image dimensions')]

    im_io = FIO.ImageIO()

    ISource, hdr, spacing, normalized_spacing_ = im_io.read_batch_to_nc_format([current_source_filename],
                                                             intensity_normalize=intensity_normalize,
                                                             squeeze_image=squeeze_image,
                                                             normalize_spacing=normalize_spacing,
                                                             silent_mode=True)
    ITarget, hdr, spacing, normalized_spacing = im_io.read_batch_to_nc_format([current_target_filename],
                                                             intensity_normalize=intensity_normalize,
                                                             squeeze_image=squeeze_image,
                                                             normalize_spacing=normalize_spacing,
                                                             silent_mode=True)

    sz = np.array(ISource.shape)

    ISource = Variable(torch.from_numpy(ISource), requires_grad=False)
    ITarget = Variable(torch.from_numpy(ITarget), requires_grad=False)

    return ISource,ITarget,hdr,sz,normalized_spacing

def individual_parameters_to_model_parameters(ind_pars):
    model_pars = dict()
    for par in ind_pars:
        model_pars[par['name']] = par['model_params']

    return model_pars


def evaluate_model(ISource_in,ITarget_in,sz,spacing,individual_parameters,shared_parameters,params,visualize=True):

    ISource = AdaptVal(ISource_in)
    ITarget = AdaptVal(ITarget_in)

    model_name = params['model']['registration_model']['type']
    use_map = params['model']['deformation']['use_map']
    map_low_res_factor = params['model']['deformation'][('map_low_res_factor', None, 'low_res_factor')]
    compute_similarity_measure_at_low_res = params['model']['deformation'][
        ('compute_similarity_measure_at_low_res', False, 'to compute Sim at lower resolution')]

    spline_order = params['model']['registration_model'][('spline_order', 1, 'Spline interpolation order; 1 is linear interpolation (default); 3 is cubic spline')]

    lowResSize = None
    lowResSpacing = None

    if map_low_res_factor is not None:
        lowResSize = _get_low_res_size_from_size(sz, map_low_res_factor)
        lowResSpacing = _get_low_res_spacing_from_spacing(spacing, sz, lowResSize)

        lowResISource = _compute_low_res_image(ISource, spacing, lowResSize,spline_order)
        # todo: can be removed to save memory; is more experimental at this point
        lowResITarget = _compute_low_res_image(ITarget, spacing, lowResSize,spline_order)

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
                rec_phiWarped, _ = sampler.upsample_image_to_size(rec_tmp, spacing, desiredSz,spline_order)
        else:
            rec_phiWarped = model(identityMap, ISource)

    else:
        rec_IWarped = model(ISource)

    if use_map:
        rec_IWarped = utils.compute_warped_image_multiNC(ISource, rec_phiWarped, spacing,spline_order)

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
            I1Warped = utils.compute_warped_image_multiNC(lowResISource, phi_or_warped_image, lowResSpacing, spline_order)
            vizReg.show_current_images(iter, lowResISource, lowResITarget, I1Warped, vizImage, vizName,
                                       phi_or_warped_image, visual_param)
        else:
            I1Warped = utils.compute_warped_image_multiNC(ISource, phi_or_warped_image, spacing, spline_order)
            vizReg.show_current_images(iter, ISource, ITarget, I1Warped, vizImage, vizName,
                                       phi_or_warped_image, visual_param)
    else:
        vizReg.show_current_images(iter, ISource, ITarget, phi_or_warped_image, vizImage, vizName, None, visual_param)

    dictionary_to_pass_to_smoother = dict()
    if map_low_res_factor is not None:
        dictionary_to_pass_to_smoother['I'] = lowResISource
        dictionary_to_pass_to_smoother['I0'] = lowResISource
        dictionary_to_pass_to_smoother['I1'] = lowResITarget
    else:
        dictionary_to_pass_to_smoother['I'] = ISource
        dictionary_to_pass_to_smoother['I0'] = ISource
        dictionary_to_pass_to_smoother['I1'] = ITarget

    variables_from_forward_model = model.get_variables_to_transfer_to_loss_function()
    smoother = variables_from_forward_model['smoother']
    smoother.set_debug_retain_computed_local_weights(True)

    model_pars = model.get_registration_parameters()
    if 'lam' in model_pars:
        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(model_pars['lam'], lowResISource, lowResSize,lowResSpacing)
    elif 'm' in model_pars:
        m = model_pars['m']
    else:
        raise ValueError('Expected a scalar or a vector momentum in model (use SVF for example)')

    v = smoother.smooth(m, None, dictionary_to_pass_to_smoother)

    local_weights = smoother.get_debug_computed_local_weights()
    default_multi_gaussian_weights = smoother.get_default_multi_gaussian_weights()

    model_dict = dict()
    model_dict['use_map'] = use_map
    model_dict['lowResISource'] = lowResISource
    model_dict['lowResITarget'] = lowResITarget
    model_dict['lowResSpacing'] = lowResSpacing
    model_dict['lowResSize'] = lowResSize
    model_dict['local_weights'] = local_weights
    model_dict['default_multi_gaussian_weights'] = default_multi_gaussian_weights
    model_dict['stds'] = smoother.get_gaussian_stds()
    model_dict['model'] = model
    if 'lam' in model_pars:
        model_dict['lam'] = model_pars['lam']
    model_dict['m'] = m
    model_dict['v'] = v
    if use_map:
        model_dict['id'] = identityMap
    if map_low_res_factor is not None:
        model_dict['map_low_res_factor'] = map_low_res_factor
        model_dict['low_res_id'] = lowResIdentityMap

    return rec_IWarped,rec_phiWarped, model_dict

def get_json_and_output_dir_for_stages(json_file,output_dir):

    res_output_dir = os.path.normpath(output_dir)
    json_path, json_filename = os.path.split(json_file)

    json_stage_0_in = os.path.join(res_output_dir, 'out_stage_0_' + json_filename)
    json_stage_1_in = os.path.join(res_output_dir, 'out_stage_1_' + json_filename)
    json_stage_2_in = os.path.join(res_output_dir, 'out_stage_2_' + json_filename)

    json_for_stages = []
    json_for_stages.append(json_stage_0_in)
    json_for_stages.append(json_stage_1_in)
    json_for_stages.append(json_stage_2_in)

    frozen_json_stage_0_in = os.path.join(res_output_dir, 'frozen_out_stage_0_' + json_filename)
    frozen_json_stage_1_in = os.path.join(res_output_dir, 'frozen_out_stage_1_' + json_filename)
    frozen_json_stage_2_in = os.path.join(res_output_dir, 'frozen_out_stage_2_' + json_filename)

    frozen_json_for_stages = []
    frozen_json_for_stages.append(frozen_json_stage_0_in)
    frozen_json_for_stages.append(frozen_json_stage_1_in)
    frozen_json_for_stages.append(frozen_json_stage_2_in)

    output_dir_stage_0 = os.path.join(res_output_dir, 'results_after_stage_0')
    output_dir_stage_1 = os.path.join(res_output_dir, 'results_after_stage_1')
    output_dir_stage_2 = os.path.join(res_output_dir, 'results_after_stage_2')

    output_dir_for_stages = []
    output_dir_for_stages.append(output_dir_stage_0)
    output_dir_for_stages.append(output_dir_stage_1)
    output_dir_for_stages.append(output_dir_stage_2)

    frozen_output_dir_stage_0 = os.path.join(res_output_dir, 'results_frozen_after_stage_0')
    frozen_output_dir_stage_1 = os.path.join(res_output_dir, 'results_frozen_after_stage_1')
    frozen_output_dir_stage_2 = os.path.join(res_output_dir, 'results_frozen_after_stage_2')

    frozen_output_dir_for_stages = []
    frozen_output_dir_for_stages.append(frozen_output_dir_stage_0)
    frozen_output_dir_for_stages.append(frozen_output_dir_stage_1)
    frozen_output_dir_for_stages.append(frozen_output_dir_stage_2)

    return json_for_stages,frozen_json_for_stages,output_dir_for_stages,frozen_output_dir_for_stages

def cond_flip(v,f):
    if f:
        return np.flipud(v)
    else:
        return v

def visualize_weights(I0,I1,Iw,phi,norm_m,local_weights,stds,spacing,lowResSize,print_path=None, print_figure_id = None, slice_mode=0,flip_axes=False,params=None):

    if local_weights is not None:
        osw = compute_overall_std(local_weights[0,...].cpu(), stds.data.cpu())

    plt.clf()

    spline_order = params['model']['registration_model'][('spline_order', 1, 'Spline interpolation order; 1 is linear interpolation (default); 3 is cubic spline')]

    source_mask = compute_mask(I0[:, 0:1, ...].data.cpu().numpy())
    lowRes_source_mask_v, _ = IS.ResampleImage().downsample_image_to_size(
        Variable(torch.from_numpy(source_mask), requires_grad=False), spacing, lowResSize[2:],spline_order)
    lowRes_source_mask = lowRes_source_mask_v.data.cpu().numpy()[0, 0, ...]

    plt.subplot(2, 3, 1)
    plt.imshow(cond_flip(I0[0, 0, ...].data.cpu().numpy(),flip_axes), cmap='gray')
    plt.title('source')

    plt.subplot(2, 3, 2)
    plt.imshow(cond_flip(I1[0, 0, ...].data.cpu().numpy(),flip_axes), cmap='gray')
    plt.title('target')

    plt.subplot(2, 3, 3)
    plt.imshow(cond_flip(Iw[0, 0, ...].data.cpu().numpy(),flip_axes), cmap='gray')
    plt.title('warped')

    plt.subplot(2, 3, 4)
    plt.imshow(cond_flip(Iw[0, 0, ...].data.cpu().numpy(),flip_axes), cmap='gray')
    plt.contour(cond_flip(phi[0, 0, ...].data.cpu().numpy(),flip_axes), np.linspace(-1, 1, 20), colors='r', linestyles='solid')
    plt.contour(cond_flip(phi[0, 1, ...].data.cpu().numpy(),flip_axes), np.linspace(-1, 1, 20), colors='r', linestyles='solid')
    plt.title('warped+grid')

    plt.subplot(2, 3, 5)
    plt.imshow(cond_flip(norm_m[0, 0, ...].data.cpu().numpy(),flip_axes), cmap='gray')
    plt.title('|m|')

    if local_weights is not None:
        plt.subplot(2, 3, 6)
        cmin = osw.cpu().numpy()[lowRes_source_mask == 1].min()
        cmax = osw.cpu().numpy()[lowRes_source_mask == 1].max()
        plt.imshow(cond_flip(osw.cpu().numpy() * lowRes_source_mask,flip_axes), cmap='gray', vmin=cmin, vmax=cmax)
        plt.title('std')

    plt.suptitle('Registration result: pair id {:03d}'.format(print_figure_id))

    if print_figure_id is not None:
        plt.savefig(os.path.join(print_path,'{:0>3d}_sm{:d}'.format(print_figure_id,slice_mode) + '_registration.pdf'))
    else:
        plt.show()

    if local_weights is not None:
        plt.clf()

        nr_of_gaussians = local_weights.size()[1]

        for g in range(nr_of_gaussians):
            plt.subplot(2, 4, g + 1)
            clw = local_weights[0, g, ...].cpu().numpy()
            cmin = clw[lowRes_source_mask == 1].min()
            cmax = clw[lowRes_source_mask == 1].max()
            plt.imshow(cond_flip((local_weights[0, g, ...]).cpu().numpy() * lowRes_source_mask,flip_axes), vmin=cmin, vmax=cmax)
            plt.title("{:.2f}".format(stds.data.cpu()[g]))
            plt.colorbar()

        plt.subplot(2, 4, 8)
        osw = compute_overall_std(local_weights[0, ...].cpu(), stds.data.cpu())

        cmin = osw.cpu().numpy()[lowRes_source_mask == 1].min()
        cmax = osw.cpu().numpy()[lowRes_source_mask == 1].max()
        plt.imshow(cond_flip(osw.cpu().numpy() * lowRes_source_mask,flip_axes), vmin=cmin, vmax=cmax)
        plt.colorbar()
        plt.suptitle('Weights')

        if print_figure_id is not None:
            plt.savefig(os.path.join(print_path,'{:0>3d}_sm{:d}'.format(print_figure_id,slice_mode) + '_weights.pdf'))
        else:
            plt.show()

def compute_determinant_of_jacobian(phi,spacing):
    fdt = FD.FD_torch(spacing)
    dim = len(spacing)

    if dim==2:
        p0x = fdt.dXc(phi[0:1, 0, ...])
        p0y = fdt.dYc(phi[0:1, 0, ...])
        p1x = fdt.dXc(phi[0:1, 1, ...])
        p1y = fdt.dYc(phi[0:1, 1, ...])

        det = p0x * p1y - p0y * p1x
    elif dim==3:
        p0x = fdt.dXc(phi[0:1, 0, ...])
        p0y = fdt.dYc(phi[0:1, 0, ...])
        p0z = fdt.dZc(phi[0:1, 0, ...])
        p1x = fdt.dXc(phi[0:1, 1, ...])
        p1y = fdt.dYc(phi[0:1, 1, ...])
        p1z = fdt.dZc(phi[0:1, 1, ...])
        p2x = fdt.dXc(phi[0:1, 2, ...])
        p2y = fdt.dYc(phi[0:1, 2, ...])
        p2z = fdt.dZc(phi[0:1, 2, ...])

        det = p0x*p1y*p2z + p0y*p1z*p2x + p0z*p1x*p2y -p0z*p1y*p2x -p0y*p1x*p2z -p0x*p1z*p2y
    else:
        raise ValueError('Can only compute the determinant of Jacobina for dimensions 2 and 3')

    det = det.data[0, ...].cpu().numpy()
    return det


def compute_and_visualize_results(json_file,output_dir,stage,compute_from_frozen,pair_nr,slice_proportion_3d=0.5,slice_mode_3d=0,visualize=False,
                                  print_images=False,write_out_images=True,
                                  write_out_source_image=False,write_out_target_image=False,
                                  compute_det_of_jacobian=True,retarget_data_directory=None,
                                  use_sym_links=True):

    # todo: make this data-adaptive
    flip_axes_3d = [False,True,True]


    if write_out_images:
        write_out_warped_image = True
        write_out_map = True
    else:
        write_out_warped_image = False
        write_out_map = False

    # get the used json configuration and the output directories for the different stages
    json_for_stages, frozen_json_for_stages, output_dir_for_stages, frozen_output_dir_for_stages = get_json_and_output_dir_for_stages(json_file,
                                                                                                            output_dir)

    if compute_from_frozen:
        image_and_map_output_dir = os.path.join(os.path.normpath(output_dir), 'model_results_frozen_stage_{:d}'.format(stage))
        print_output_dir = os.path.join(os.path.normpath(output_dir), 'pdf_frozen_stage_{:d}'.format(stage))
    else:
        image_and_map_output_dir = os.path.join(os.path.normpath(output_dir), 'model_results_stage_{:d}'.format(stage))
        print_output_dir = os.path.join(os.path.normpath(output_dir),'pdf_stage_{:d}'.format(stage))

    if write_out_warped_image or write_out_map or compute_det_of_jacobian:
        if not os.path.exists(image_and_map_output_dir):
            print('Creating output directory: ' + image_and_map_output_dir)
            os.makedirs(image_and_map_output_dir)

    if print_images:
        visualize = True

    if visualize and print_images:
        if not os.path.exists(print_output_dir):
            print('Creating output directory: ' + print_output_dir)
            os.makedirs(print_output_dir)

    warped_output_filename = os.path.join(image_and_map_output_dir,'warped_image_{:05d}.nrrd'.format(pair_nr))
    map_output_filename = os.path.join(image_and_map_output_dir,'map_validation_format_{:05}.nrrd'.format(pair_nr))
    det_jac_output_filename = os.path.join(image_and_map_output_dir,'det_of_jacobian_{:05}.nrrd'.format(pair_nr))
    displacement_output_filename = os.path.join(image_and_map_output_dir,'displacement_{:05}.nrrd'.format(pair_nr))
    det_of_jacobian_txt_filename = os.path.join(image_and_map_output_dir,'det_of_jacobian_{:05}.txt'.format(pair_nr))

    source_image_output_filename = os.path.join(image_and_map_output_dir,'source_image_{:05d}.nrrd'.format(pair_nr))
    target_image_output_filename = os.path.join(image_and_map_output_dir,'target_image_{:05d}.nrrd'.format(pair_nr))

    # current case
    if compute_from_frozen:
        current_json = frozen_json_for_stages[stage]
        individual_dir = os.path.join(frozen_output_dir_for_stages[stage], 'individual')
        shared_dir = os.path.join(frozen_output_dir_for_stages[stage], 'shared')
    else:
        current_json = json_for_stages[stage]
        individual_dir = os.path.join(output_dir_for_stages[stage],'individual')
        shared_dir = os.path.join(output_dir_for_stages[stage],'shared')

    # load the configuration
    params = pars.ParameterDict()
    params.load_JSON(current_json)

    # load the mapping to the images
    used_pairs = torch.load(os.path.join(output_dir,'used_image_pairs.pt'))
    nr_of_computed_pairs = len(used_pairs['source_ids'])

    # load the image with given pair number
    current_source_filename = used_pairs['source_images'][pair_nr]
    current_target_filename = used_pairs['target_images'][pair_nr]

    if retarget_data_directory is not None:
        # map them to a different directory
        current_source_filename = os.path.join(retarget_data_directory,os.path.split(current_source_filename)[1])
        current_target_filename = os.path.join(retarget_data_directory,os.path.split(current_target_filename)[1])

    ISource,ITarget,hdr,sz,spacing = _load_current_source_and_target_images_as_variables(current_source_filename,current_target_filename,params)

    image_dim = len(spacing)

    # load the shared parameters (do this so we can load even if this was created on a GPU machine)
    if USE_CUDA:
        shared_parameters = torch.load(os.path.join(shared_dir,'shared_parameters.pt'))
    else:
        shared_parameters = torch.load(os.path.join(shared_dir,'shared_parameters.pt'),map_location=lambda storage, loc:storage)

    # load the individual parameters
    individual_parameters_filename = os.path.join(individual_dir,'individual_parameter_pair_{:05d}.pt'.format(pair_nr))
    if USE_CUDA:
        individual_parameters = torch.load(individual_parameters_filename)
    else:
        individual_parameters = torch.load(individual_parameters_filename,map_location=lambda storage, loc:storage)

    # apply the actual model and get the warped image and the map (if applicable)
    IWarped,phi,model_dict = evaluate_model(ISource,ITarget,sz,spacing,individual_parameters,shared_parameters,params,visualize=False)

    norm_m = ((model_dict['m']**2).sum(dim=1,keepdim=True))**0.5

    if visualize:
        if image_dim==2:
            if print_images:
                visualize_weights(ISource,ITarget,IWarped,phi,
                                  norm_m,model_dict['local_weights'],model_dict['stds'],
                                  spacing,model_dict['lowResSize'],print_output_dir,pair_nr,params=params)
            else:
                visualize_weights(ISource,ITarget,IWarped,phi,
                                  norm_m,model_dict['local_weights'],model_dict['stds'],
                                  spacing,model_dict['lowResSize'],params=params)
        elif image_dim==3:
            sz_I = ISource.size()
            sz_norm_m = norm_m.size()

            if not set(slice_mode_3d)<=set([0,1,2]):
                raise ValueError('slice mode needs to be in {0,1,2}')

            for sm in slice_mode_3d:

                slice_I = (np.ceil(np.array(sz_I[-1-(2-slice_mode_3d[sm])]) * slice_proportion_3d[sm])).astype('int16')
                slice_norm_m = (np.ceil(np.array(sz_norm_m[-1-(2-slice_mode_3d[sm])]) * slice_proportion_3d[sm])).astype('int16')

                if slice_mode_3d[sm]==0:
                    IS_slice = ISource[:, :, slice_I, ...]
                    IT_slice = ITarget[:, :, slice_I, ...]
                    IW_slice = IWarped[:, :, slice_I, ...]
                    phi_slice = phi[:, 1:, slice_I, ...]
                    norm_m_slice = norm_m[:, :, slice_norm_m, ...]
                    lw_slice = None
                    if 'local_weights' in model_dict:
                        if model_dict['local_weights'] is not None:
                            lw_slice = model_dict['local_weights'][:, :, slice_norm_m, ...]
                    spacing_slice = spacing[1:]
                    lowResSize = list(model_dict['lowResSize'])
                    lowResSize_slice = np.array(lowResSize[0:2] + lowResSize[3:])
                elif slice_mode_3d[sm]==1:
                    IS_slice = ISource[:, :, :, slice_I, :]
                    IT_slice = ITarget[:, :, :, slice_I, :]
                    IW_slice = IWarped[:, :, :, slice_I, :]
                    phi_slice = torch.zeros_like(phi[:, 1:, :, slice_I, :])
                    phi_slice[:,0,...] = phi[:,0,:,slice_I,:]
                    phi_slice[:,1,...] = phi[:,2,:,slice_I,:]
                    norm_m_slice = norm_m[:, :, :, slice_norm_m, :]
                    lw_slice = None
                    if 'local_weights' in model_dict:
                        if model_dict['local_weights'] is not None:
                            lw_slice = model_dict['local_weights'][:, :, :, slice_norm_m, :]
                    spacing_slice = np.array([spacing[0],spacing[2]])
                    lowResSize = list(model_dict['lowResSize'])
                    lowResSize_slice = np.array(lowResSize[0:3] + [lowResSize[-1]])
                elif slice_mode_3d[sm]==2:
                    IS_slice = ISource[:,:,:,:,slice_I]
                    IT_slice = ITarget[:,:,:,:,slice_I]
                    IW_slice = IWarped[:,:,:,:,slice_I]
                    phi_slice = phi[:,0:2,:,:,slice_I]
                    norm_m_slice = norm_m[:,:,:,:,slice_norm_m]
                    lw_slice = None
                    if 'local_weights' in model_dict:
                        if model_dict['local_weights'] is not None:
                            lw_slice = model_dict['local_weights'][:,:,:,:,slice_norm_m]
                    spacing_slice = spacing[0:-1]
                    lowResSize_slice = model_dict['lowResSize'][0:-1]

                if print_images:
                    visualize_weights(IS_slice,IT_slice,IW_slice,phi_slice,
                                      norm_m_slice,lw_slice,model_dict['stds'],
                                      spacing_slice,lowResSize_slice,print_output_dir,pair_nr,slice_mode_3d[sm],flip_axes_3d[sm],params=params)
                else:
                    visualize_weights(IS_slice, IT_slice, IW_slice, phi_slice,
                                      norm_m_slice, lw_slice, model_dict['stds'],
                                      spacing_slice, lowResSize_slice,flip_axes=flip_axes_3d[sm],params=params)

        else:
            raise ValueError('I do not know how to visualize results with dimensions other than 2 or 3')


    # save the images
    if write_out_warped_image:
        im_io = FIO.ImageIO()
        im_io.write(warped_output_filename, IWarped[0,0,...], hdr)

    if write_out_map:
        map_io = FIO.MapIO()
        map_io.write_to_validation_map_format(map_output_filename, phi[0,...], hdr)

        if 'id' in model_dict:
            displacement = phi[0,...]-model_dict['id'][0,...]
            map_io.write(displacement_output_filename, displacement, hdr)

    if write_out_source_image:
        if use_sym_links:
            utils.create_symlink_with_correct_ext(current_source_filename,source_image_output_filename)
        else:
            im_io = FIO.ImageIO()
            im_io.write(source_image_output_filename, ISource[0,0,...], hdr)

    if write_out_target_image:
        if use_sym_links:
            utils.create_symlink_with_correct_ext(current_target_filename,target_image_output_filename)
        else:
            im_io = FIO.ImageIO()
            im_io.write(target_image_output_filename, ITarget[0, 0, ...], hdr)

    # compute determinant of Jacobian of map
    if compute_det_of_jacobian:
        det = compute_determinant_of_jacobian(phi,spacing)

        im_io = FIO.ImageIO()
        im_io.write(det_jac_output_filename, det, hdr)

        det_min = np.min(det)
        det_max = np.max(det)
        det_mean = np.mean(det)
        det_median = np.median(det)
        det_1_perc = np.percentile(det,1)
        det_5_perc = np.percentile(det,5)
        det_95_perc = np.percentile(det,95)
        det_99_perc = np.percentile(det,99)

        f = open(det_of_jacobian_txt_filename, 'w')
        f.write('min, max, mean, median, 1p, 5p, 95p, 99p\n')
        out_str = str(det_min) + ', '
        out_str += str(det_max) + ', '
        out_str += str(det_mean) + ', '
        out_str += str(det_median) + ', '
        out_str += str(det_1_perc) + ', '
        out_str += str(det_5_perc) + ', '
        out_str += str(det_95_perc) + ', '
        out_str += str(det_99_perc) + '\n'

        f.write(out_str)
        f.close()

if __name__ == "__main__":

    torch.set_num_threads(mp.cpu_count())

    import argparse

    parser = argparse.ArgumentParser(description='Computes registration results from batch optimization output')

    parser.add_argument('--config', required=True, help='The main json configuration file that was used to create the results')
    parser.add_argument('--output_directory', required=True, help='Where the output was stored (now this will be the input directory)')
    parser.add_argument('--stage_nr', required=True, type=int, help='stage number for which the computations should be performed {0,1,2}')

    parser.add_argument('--compute_only_pair_nr', required=False, type=int, default=None, help='When specified only this pair is computed; otherwise all of them')
    parser.add_argument('--slice_proportion_3d', required=False, type=str, default=None, help='Where to slice for 3D visualizations [0,1] for each mode, as a comma separated list')
    parser.add_argument('--slice_mode_3d', required=False, type=str, default=None, help='Which visualization mode {0,1,2} as a comma separated list')

    parser.add_argument('--compute_from_frozen', action='store_true', help='computes the results from optimization results with frozen parameters')

    parser.add_argument('--do_not_write_source_image', action='store_true', help='otherwise also writes the source image for easy visualization')
    parser.add_argument('--do_not_write_target_image', action='store_true', help='otherwise also writes the target image for easy visualization')

    parser.add_argument('--do_not_use_symlinks', action='store_true', help='For source and target images, by default symbolic links are created, otherwise files are copied')

    parser.add_argument('--retarget_data_directory', required=False, default=None,help='Looks for the datafiles in this directory')

    parser.add_argument('--do_not_visualize', action='store_true', help='visualizes the output otherwise')
    parser.add_argument('--do_not_print_images', action='store_true', help='prints the results otherwise')
    parser.add_argument('--do_not_compute_det_jac', action='store_true', help='computes the determinant of the Jacobian otherwise')
    parser.add_argument('--do_not_write_out_images', action='store_true', help='writes out the map and the warped image otherwise')

    args = parser.parse_args()


    slice_proportion_3d_was_specified = False
    if args.slice_proportion_3d is None:
        slice_proportion_3d = [0.5,0.5,0.5]
    else:
        slice_proportion_3d = [float(item) for item in args.slice_proportion_3d.split(',')]
        slice_proportion_3d_was_specified = True

    slice_mode_3d_was_specified = False
    if args.slice_mode_3d is None:
        slice_mode_3d = [0,1,2]
    else:
        slice_mode_3d = [int(item) for item in args.slice_mode_3d.split(',')]
        slice_mode_3d_was_specified = True

    if slice_mode_3d_was_specified and not slice_proportion_3d_was_specified:
        slice_proportion_3d = [0.5]*len(slice_mode_3d)

    if slice_proportion_3d_was_specified and not slice_mode_3d_was_specified:
        slice_mode_3d = list(range(len(slice_proportion_3d)))

    if len(slice_mode_3d)!=len(slice_proportion_3d):
        raise ValueError('There need to be the same number of proportions as there are 3D modes specified')

    json_file = args.config
    output_dir = args.output_directory

    used_pairs = torch.load(os.path.join(output_dir,'used_image_pairs.pt'))
    nr_of_computed_pairs = len(used_pairs['source_ids'])

    if args.compute_only_pair_nr is not None:
        pair_nrs = [args.compute_only_pair_nr]
    else:
        pair_nrs = list(range(nr_of_computed_pairs))

    for pair_nr in pair_nrs:
        print('Computing pair number: ' + str(pair_nr))
        compute_and_visualize_results(json_file=args.config,output_dir=output_dir,
                                      stage=args.stage_nr,
                                      compute_from_frozen=args.compute_from_frozen,
                                      pair_nr=pair_nr,
                                      slice_proportion_3d=slice_proportion_3d,
                                      slice_mode_3d=slice_mode_3d,
                                      visualize=not args.do_not_visualize,
                                      print_images=not args.do_not_print_images,
                                      write_out_images=not args.do_not_write_out_images,
                                      write_out_source_image=not args.do_not_write_source_image,
                                      write_out_target_image=not args.do_not_write_target_image,
                                      compute_det_of_jacobian=not args.do_not_compute_det_jac,
                                      retarget_data_directory=args.retarget_data_directory)

    if not args.do_not_print_images:
        if args.compute_from_frozen:
            print_output_dir = os.path.join(os.path.normpath(output_dir), 'pdf_frozen_stage_{:d}'.format(args.stage_nr))
        else:
            print_output_dir = os.path.join(os.path.normpath(output_dir), 'pdf_stage_{:d}'.format(args.stage_nr))

        # if we have pdfjam we create a summary pdf
        if os.system('which pdfjam') == 0:
            summary_pdf_name = os.path.join(print_output_dir, 'summary.pdf')

            if os.path.isfile(summary_pdf_name):
                os.remove(summary_pdf_name)

            print('Creating summary PDF: ')
            cmd = 'pdfjam {:} --nup 1x2 --outfile {:}'.format(os.path.join(print_output_dir, '*.pdf'), summary_pdf_name)
            os.system(cmd)
