import os
import torch
from . import model_factory as MF
from . import visualize_registration_results as vizReg
from . import image_sampling as IS
from .data_wrapper import AdaptVal
from .data_wrapper import USE_CUDA
from . import utils
from . import module_parameters as pars


def evaluate_model(ISource_in, ITarget_in, sz, spacing,
                   model_name=None,
                   use_map=None,
                   compute_inverse_map=False,
                   map_low_res_factor=None,
                   compute_similarity_measure_at_low_res=None,
                   spline_order=None,
                   individual_parameters=None,shared_parameters=None,params=None,extra_info=None,visualize=True,visual_param=None,given_weight=False, init_map=None,lowres_init_map=None, init_inverse_map=None,lowres_init_inverse_map=None,):

    """

    #todo: Support initial maps which are not identity

    :param ISource_in: source image (BxCxXxYxZ format)
    :param ITarget_in: target image (BxCxXxYxZ format)
    :param sz: size of the images (BxCxXxYxZ format)
    :param spacing: spacing for the images
    :param model_name: name of the desired model (string)
    :param use_map: if set to True then map-based mode is used
    :param compute_inverse_map: if set to True the inverse map will be computed
    :param map_low_res_factor: if set to None then computations will be at full resolution, otherwise at a fraction of the resolution
    :param compute_similarity_measure_at_low_res:
    :param spline_order: desired spline order for the sampler
    :param individual_parameters: individual registration parameters
    :param shared_parameters: shared registration parameters
    :param params: parameter dictionary (of model_dictionary type) which configures the model
    :param visualize: if set to True results will be visualized

    :return: returns a tuple (I_warped,phi,phi_inverse,model_dictionary), here I_warped = I_source\circ\phi, and phi_inverse is the inverse of phi; model_dictionary contains various intermediate results
    """

    ISource = AdaptVal(ISource_in)
    ITarget = AdaptVal(ITarget_in)

    if params is None:
        print('INFO: WARNING: no params specified, creating empty parameter dictionary')
        params = pars.ParameterDict()

    if model_name is None:
        model_name = params['model']['registration_model']['type']

    if use_map is None:
        use_map = params['model']['deformation']['use_map']

    if map_low_res_factor is None:
        map_low_res_factor = params['model']['deformation'][('map_low_res_factor', None, 'low_res_factor')]

    if compute_similarity_measure_at_low_res is None:
        compute_similarity_measure_at_low_res = params['model']['deformation'][('compute_similarity_measure_at_low_res', False, 'to compute Sim at lower resolution')]

    if spline_order is None:
        spline_order = params['model']['registration_model'][('spline_order', 1, 'Spline interpolation order; 1 is linear interpolation (default); 3 is cubic spline')]


    # set some initial default values
    lowResSize = None
    lowResSpacing = None

    lowResISource = None
    lowResIdentityMap = None

    # compute low-res versions if needed

    if map_low_res_factor is not None:
        lowResSize = utils._get_low_res_size_from_size(sz, map_low_res_factor)
        lowResSpacing = utils._get_low_res_spacing_from_spacing(spacing, sz, lowResSize)

        lowResISource = utils._compute_low_res_image(ISource, spacing, lowResSize,spline_order)
        # todo: can be removed to save memory; is more experimental at this point
        lowResITarget = utils._compute_low_res_image(ITarget, spacing, lowResSize,spline_order)

        # computes model at a lower resolution than the image similarity
        if compute_similarity_measure_at_low_res:
            mf = MF.ModelFactory(lowResSize, lowResSpacing, lowResSize, lowResSpacing)
        else:
            mf = MF.ModelFactory(sz, spacing, lowResSize, lowResSpacing)
    else:
        # computes model and similarity at the same resolution
        mf = MF.ModelFactory(sz, spacing, sz, spacing)

    model, criterion = mf.create_registration_model(model_name, params['model'], compute_inverse_map=compute_inverse_map)
    # set it to evaluation mode
    model.eval()

    print(model)

    if use_map:
        # create the identity map [-1,1]^d, since we will use a map-based implementation
        id = utils.identity_map_multiN(sz, spacing)
        identityMap = AdaptVal(torch.from_numpy(id))
        if map_low_res_factor is not None:
            # create a lower resolution map for the computations
            lowres_id = utils.identity_map_multiN(lowResSize, lowResSpacing)
            lowResIdentityMap = AdaptVal(torch.from_numpy(lowres_id))
            sampler = IS.ResampleImage()

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

    if shared_parameters is not None:
        model.load_shared_state_dict(shared_parameters)
    model_pars = utils.individual_parameters_to_model_parameters(individual_parameters)
    model.set_individual_registration_parameters(model_pars)
    if 'm'in individual_parameters and individual_parameters['m'] is not None:
        model.m.data = AdaptVal(individual_parameters['m'])
    if 'local_weights'in individual_parameters and individual_parameters['local_weights'] is not None:
        model.local_weights.data= AdaptVal(individual_parameters['local_weights'])

    opt_variables = {'iter': 0, 'epoch': 0,'extra_info':extra_info,'over_scale_iter_count':0}

    # now let's run the model

    rec_IWarped, rec_phiWarped, rec_phiInverseWarped = evaluate_model_low_level_interface(
        model=model,
        I_source=ISource,
        opt_variables=opt_variables,
        use_map=use_map,
        initial_map=identityMap if init_map is None else init_map,
        compute_inverse_map=compute_inverse_map,
        initial_inverse_map=identityMap if init_inverse_map is None else init_inverse_map,
        map_low_res_factor=map_low_res_factor,
        sampler=sampler,
        low_res_spacing=lowResSpacing,
        spline_order=spline_order,
        low_res_I_source=lowResISource,
        low_res_initial_map=lowResIdentityMap if lowres_init_map is None else lowres_init_map,
        low_res_initial_inverse_map=lowResIdentityMap if lowres_init_inverse_map is None else lowres_init_inverse_map,
        compute_similarity_measure_at_low_res=compute_similarity_measure_at_low_res)

    if use_map:
        rec_IWarped = utils.compute_warped_image_multiNC(ISource, rec_phiWarped, spacing, spline_order, zero_boundary=True)

    if use_map and map_low_res_factor is not None:
        vizImage, vizName = model.get_parameter_image_and_name_to_visualize(lowResISource)
    else:
        vizImage, vizName = model.get_parameter_image_and_name_to_visualize(ISource)

    if use_map:
        phi_or_warped_image = rec_phiWarped
    else:
        phi_or_warped_image = rec_IWarped
    if visual_param is None:
        visual_param = {}
        visual_param['visualize'] = visualize
        visual_param['save_fig'] = False
        visual_param['save_fig_num'] = -1
    else:
        visual_param['visualize'] = visualize
        save_fig_path = visual_param['save_fig_path']
        visual_param['save_fig_path_byname'] = os.path.join(save_fig_path, 'byname')
        visual_param['save_fig_path_byiter'] = os.path.join(save_fig_path, 'byiter')
        visual_param['iter'] = 'scale_' + str(0) + '_iter_' + str(0)

    if use_map:
        if compute_similarity_measure_at_low_res:
            I1Warped = utils.compute_warped_image_multiNC(lowResISource, phi_or_warped_image, lowResSpacing, spline_order)
            if visualize or visual_param['save_fig']:
                vizReg.show_current_images(iter=iter, iS=lowResISource, iT=lowResITarget, iW=I1Warped, vizImages=vizImage, vizName=vizName,
                                       phiWarped=phi_or_warped_image, visual_param=visual_param)
        else:
            I1Warped = utils.compute_warped_image_multiNC(ISource, phi_or_warped_image, spacing, spline_order)
            if visualize or visual_param['save_fig']:
                vizReg.show_current_images(iter=iter, iS=ISource, iT=ITarget, iW=I1Warped, vizImages=vizImage, vizName=vizName,
                                       phiWarped=phi_or_warped_image, visual_param=visual_param)
    else:
        if visualize or visual_param['save_fig']:
            vizReg.show_current_images(iter=iter, iS=ISource, iT=ITarget, iW=phi_or_warped_image, vizImages=vizImage, vizName=vizName, phiWarped=None, visual_param=visual_param)

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
    try:
        smoother.set_debug_retain_computed_local_weights(True)
    except:
        pass
    #model_pars = model.get_registration_parameters()
    if 'lam' in model_pars:
        m = utils.compute_vector_momentum_from_scalar_momentum_multiNC(model_pars['lam'], lowResISource, lowResSize,lowResSpacing)
    elif 'm' in model_pars:
        m = model_pars['m']
    else:
        raise ValueError('Expected a scalar or a vector momentum in model (use SVF for example)')

    # if not given_weight:
    #     v = smoother.smooth(m, None, dictionary_to_pass_to_smoother)
    #     local_weights = smoother.get_debug_computed_local_weights()
    #     local_pre_weights = smoother.get_debug_computed_local_pre_weights()
    # else:
    v = None
    local_weights=None
    local_pre_weights=None

    try:
        default_multi_gaussian_weights = smoother.get_default_multi_gaussian_weights()
        gaussian_stds=smoother.get_gaussian_stds()
    except:
        default_multi_gaussian_weights=None
        gaussian_stds=None

    model_dict = dict()
    model_dict['use_map'] = use_map
    model_dict['lowResISource'] = lowResISource
    model_dict['lowResITarget'] = lowResITarget
    model_dict['lowResSpacing'] = lowResSpacing
    model_dict['lowResSize'] = lowResSize
    model_dict['local_weights'] = local_weights
    model_dict['local_pre_weights'] = local_pre_weights
    model_dict['default_multi_gaussian_weights'] = default_multi_gaussian_weights
    model_dict['stds'] =gaussian_stds
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

    return rec_IWarped,rec_phiWarped, rec_phiInverseWarped, model_dict



def evaluate_model_low_level_interface(model,I_source,opt_variables=None,use_map=False,initial_map=None,compute_inverse_map=False,initial_inverse_map=None,
                                       map_low_res_factor=None,
                                       sampler=None,low_res_spacing=None,spline_order=1,
                                       low_res_I_source=None,low_res_initial_map=None,low_res_initial_inverse_map=None,compute_similarity_measure_at_low_res=False):
    """
    Evaluates a registration model. Core functionality for optimizer. Use evaluate_model for a convenience implementation which recomputes settings on the fly

    :param model: registration model
    :param I_source: source image (may not be used for map-based approaches)
    :param opt_variables: dictionary to be passed to an optimizer or here the evaluation routine (e.g., {'iter': self.iter_count,'epoch': self.current_epoch})
    :param use_map: if set to True then map-based mode is used
    :param initial_map: initial full-resolution map (will in most cases be the identity)
    :param compute_inverse_map: if set to True the inverse map will be computed
    :param initial_inverse_map: initial inverse map (will in most cases be the identity)
    :param map_low_res_factor: if set to None then computations will be at full resolution, otherwise at a fraction of the resolution
    :param sampler: sampler which takes care of upsampling maps from their low-resolution variants (when map_low_res_factor<1.)
    :param low_res_spacing: spacing of the low res map/image
    :param spline_order: desired spline order for the sampler
    :param low_res_I_source: low resolution source image
    :param low_res_initial_map: low resolution version of the initial map
    :param low_res_initial_inverse_map: low resolution version of the initial inverse map
    :param compute_similarity_measure_at_low_res: if set to True the similarity measure is also evaluated at low resolution (otherwise at full resolution)

    :return: returns a tuple (I_warped,phi,phi_inverse), here I_warped = I_source\circ\phi, and phi_inverse is the inverse of phi

    IMPORTANT: note that \phi is the map that maps from source to target image; I_warped is None for map_based solution, phi is None for image-based solution; phi_inverse is None if inverse is not computed
    """

    # do some checks to make sure everything is properly defined
    if use_map:
        if initial_map is None:
            raise ValueError('initial_map needs to be defined when using map mode')
        if compute_inverse_map:
            if initial_inverse_map is None:
                raise ValueError('inital_inverse_map needs to be defined when using map mode and requesting inverse map')
        if map_low_res_factor is not None:
            if sampler is None:
                raise ValueError('sampler needs to be specified when using map mode and computing at lower resolution')
            if low_res_spacing is None:
                raise ValueError('low_res_spacing needs to be speficied when using map mode and computing at lower resolution')
            if low_res_I_source is None:
                raise ValueError('low_res_I_source needs to be defined when using map mode and computing at lower resolution')
            if low_res_initial_map is None:
                raise ValueError('low_res_initial_map needs to be defined when using map mode and computing at lower resolution')
            if compute_inverse_map:
                if low_res_initial_inverse_map is None:
                    raise ValueError('low_res_initial_inverse_map needs to be defined when using map mode at lower resolution and requesting inverse map')

    else: # not using map
        if compute_inverse_map:
            raise ValueError('Cannot compute inverse map in non-map mode')
        if compute_similarity_measure_at_low_res:
            raise ValueError('Low res similarity measure computations are only supported in map mode')


    # actual evaluation code starts here
    rec_phiWarped = None
    rec_phiInverseWarped = None
    rec_IWarped = None

    if use_map:
        if map_low_res_factor is not None:
            if compute_similarity_measure_at_low_res:
                ret = model(low_res_initial_map, low_res_I_source,
                                 phi_inv=low_res_initial_inverse_map, variables_from_optimizer=opt_variables)
                if compute_inverse_map:
                    if type(ret) == tuple:  # if it is a tuple it is returning the inverse
                        rec_phiWarped = ret[0]
                        rec_phiInverseWarped = ret[1]
                    else:
                        rec_phiWarped = ret
                else:
                    rec_phiWarped = ret
            else:
                ret = model(low_res_initial_map, low_res_I_source,
                                 phi_inv=low_res_initial_inverse_map, variables_from_optimizer=opt_variables)
                if compute_inverse_map:
                    if type(ret) == tuple:
                        rec_tmp = ret[0]
                        rec_inv_tmp = ret[1]
                    else:
                        rec_tmp = ret
                        rec_inv_tmp = None
                else:
                    rec_tmp = ret
                # now upsample to correct resolution
                desiredSz = initial_map.size()[2::]
                rec_phiWarped, _ = sampler.upsample_image_to_size(rec_tmp, low_res_spacing, desiredSz,
                                                                            spline_order, zero_boundary=False)
                if compute_inverse_map and rec_inv_tmp is not None:
                    rec_phiInverseWarped, _ = sampler.upsample_image_to_size(rec_inv_tmp, low_res_spacing,
                                                                                       desiredSz, spline_order,
                                                                                       zero_boundary=False)
                else:
                    rec_phiInverseWarped = None
        else:
            # computing at full resolution
            ret = model(initial_map, I_source,
                             phi_inv=initial_inverse_map, variables_from_optimizer=opt_variables)
            if compute_inverse_map:
                if type(ret) == tuple:
                    rec_phiWarped = ret[0]
                    rec_phiInverseWarped = ret[1]
                else:
                    rec_phiWarped = ret
            else:
                rec_phiWarped = ret
    else:
        rec_IWarped = model(I_source, opt_variables)

    return rec_IWarped,rec_phiWarped,rec_phiInverseWarped
