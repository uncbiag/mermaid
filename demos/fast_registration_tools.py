# This is a collection of various tools for the fast registration

import set_pyreg_paths

# first do the imports
import torch
from torch.autograd import Variable
from pyreg.data_wrapper import AdaptVal, MyTensor
import pyreg.fileio as FIO
import pyreg.simple_interface as SI
import pyreg.smoother_factory as SF
import pyreg.image_sampling as IS
import pyreg.custom_optimizers as CO
import pyreg.model_factory as MF
import pyreg.visualize_registration_results as vizReg
from torch.nn.parameter import Parameter
import pyreg.module_parameters as pars
import pyreg.utils as utils

import numpy as np
import matplotlib.pyplot as plt
import os
import fnmatch
import h5py


# keep track of general parameters
params = pars.ParameterDict()

def find(pattern, path):
    """
    finds all files with given pattern under given path and returns sorted list of all files found
    :param pattern: pattern in files to find
    :param path: path to files to find
    :return: list of all files found with given pattern under given path
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return sorted(result)

def read_h5file(path):
    """
    reads the h5file under the given path and returns its dataset
    :param path: path to h5file
    :return: dataset of h5file in numpy array
    """

    data = h5py.File(path, 'r')
    data_set = data['/dataset'][()]
    data.close()
    return data_set

def write_h5file(path, dset):
    """
    writes dset in h5file under the given path
    :param path: path where to save h5file
    :param dset: dataset (numpy array) to be saved as h5file
    """
    print('writing h5file: '+path)
    data = h5py.File(path, "w")
    data.create_dataset("dataset", data=dset)
    data.close()

def find_central_segmentation(images, path, visualize):
    """
    computes the central segmentation as the image with the lowest SSD values
    when compared to all other images, saves central segmentation under given path
    :param images: list of images
    :param path: path where to save central segmentation
    :param visualize: if True - visualize central segmentation in plot
    """

    print('computing central segmentation')
    im_io = FIO.ImageIO()
    ssd_list = []
    for i ,im_name_i in enumerate(images):
        Ic_i, hdrc, spacing, _ = im_io.read_to_nc_format(filename=im_name_i, silent_mode=True)
        Ic_i_flat = Ic_i.flatten()
        for j, im_name_j in enumerate(images):
            Ic_j, hdrc, spacing, _ = im_io.read_to_nc_format(filename=im_name_j, silent_mode=True)
            Ic_j_flat = Ic_j.flatten()
            ssd = np.sum((Ic_i_flat-Ic_j_flat)*(Ic_i_flat-Ic_j_flat))
            if j==0:
                ssd_list.append(ssd)
            else:
                ssd_list[i]+= ssd
    ssd_min = np.argwhere(ssd_list == np.min(ssd_list))[0][0]
    print(ssd_min)

    Central, hdrc, spacing, _ = im_io.read_to_nc_format(filename=images[ssd_min], silent_mode=True)
    FIO.ImageIO().write(path, Central[0,0,:])

    if visualize:
        plt.imshow(Central[0, 0, ...], cmap='gray', vmin=0, vmax=1)
        plt.title('Central Segmentation')
        plt.colorbar()
        plt.show()

def compute_average_image(images,path,visualize):
    """
    computes the average image by adding all images and dividing by the number of images,
    saves average image under given path
    :param images: list of images
    :param path: path where to save average image
    :param visualize: if True - visualize average image in plot
    """
    # computes the average image by adding all images and dividing by the number of images,
    # saves average image under given path
    print('computing average image')
    im_io = FIO.ImageIO()
    Iavg = None
    for nr,im_name in enumerate(images):
        Ic,hdrc,spacing,_ = im_io.read_to_nc_format(filename=im_name, silent_mode=True)
        if nr==0:
            Iavg = Ic
        else:
            Iavg += Ic
    Iavg = Iavg / len(images)
    FIO.ImageIO().write(path, Iavg[0,0,:])

    if visualize:
        plt.imshow(Iavg[0, 0, ...], cmap='gray', vmin=0, vmax=1)
        plt.title('Average Image')
        plt.colorbar()
        plt.show()

def concatenate_momentums(first_momentum_filepath, second_momentum_filepath, concatenated_filepath):
    """
    Computes the concatenated momentums from 2 input momentum h5files
    :param first_momentum_filepath: path to first momentum h5file
    :param second_momentum_filepath: path to second momentum h5file
    :param concatenated_filepath: filepath under which the concatenated momentums will be saved as h5file
    :return: returns the concatenated momentums in nparray
    """
    # load first momentums
    mom_1_data = read_h5file(first_momentum_filepath)
    sz_1 = mom_1_data.shape

    # load second momentums
    mom_2_data = read_h5file(second_momentum_filepath)
    sz_2 = mom_2_data.shape

    # check if files are compatible
    assert len(sz_1)==len(sz_2)
    for i in range(len(sz_1)):
        assert sz_1[i]==sz_2[i]

    # concatenate momentums
    mom_concatenated = mom_1_data + mom_2_data

    # save concatenated momentums in h5 file
    write_h5file(concatenated_filepath, mom_concatenated)

    return mom_concatenated

def subtract_momentums(first_momentum_filepath, second_momentum_filepath, subtracted_filepath):
    """
    Computes the subtracted momentums from 2 input momentum h5files
    :param first_momentum_filepath: path to first momentum h5file
    :param second_momentum_filepath: path to second momentum h5file
    :param concatenated_filepath: filepath under which the subtracted momentums will be saved as h5file
    :return: returns the subtracted momentums in nparray
    """
    # load first momentums
    mom_1_data = read_h5file(first_momentum_filepath).squeeze()
    sz_1 = mom_1_data.shape

    # load second momentums
    mom_2_data = read_h5file(second_momentum_filepath).squeeze()
    sz_2 = mom_2_data.shape

    # check if files are compatible
    assert len(sz_1)==len(sz_2)
    for i in range(len(sz_1)):
        assert sz_1[i]==sz_2[i]

    # concatenate momentums
    mom_subtracted = mom_1_data - mom_2_data

    # save concatenated momentums in h5 file
    write_h5file(subtracted_filepath, mom_subtracted)

    return mom_subtracted

def single_registration(images, target_image_path, do_smoothing, registration_results_images_path,
                        registration_results_deformations_path, plot_and_save_def_grids):
    """
    registers multiple source images to 1 target image
    :param images: list of source images
    :param target_image_path: path to target image
    :param do_smoothing: if true - first register smoothed images, then continue on unsmoothed images
    :param registration_results_images_path: filepath under which the warped images will be saved as .nii.gz
    :param registration_results_deformations_path: filepath under which plots of the deformation field grids will be saved
    :param plot_and_save_def_grids: if true - plot and save deformation field grids
    :return: returns average image, list of deformation maps, list of momentums
    """
    im_io = FIO.ImageIO()

    # read in target image
    Iavg, hdrc, spacing, _ = im_io.read_to_nc_format(target_image_path)


    if do_smoothing:
        # create the target image as pyTorch variable
        Iavg_pt = AdaptVal(Variable(torch.from_numpy(Iavg), requires_grad=False))
        Iavg_beforeSmoothing = AdaptVal(Variable(torch.from_numpy(Iavg), requires_grad=False)).cpu().data.numpy()

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
        Iavg = Iavg_pt.cpu().data.numpy()


    # initialize lists to save maps and momentums of last cycle
    wp_list = []
    mom_list = []

    # register all images to the average image and while doing so compute a new average image

    for i, im_name in enumerate(images):
        print('Registering image ' + str(i+1) + '/' + str(len(images)))
        Ic, hdrc, spacing, _ = im_io.read_to_nc_format(filename=im_name)

        if do_smoothing:
            # create the source image as pyTorch variable
            Ic_pt = AdaptVal(Variable(torch.from_numpy(Ic), requires_grad=False))
            Ic_beforeSmoothing = AdaptVal(Variable(torch.from_numpy(Ic), requires_grad=False))

            # smoth a little bit
            Ic_pt = s.smooth(Ic_pt)
            Ic = Ic_pt.cpu().data.numpy()

            # register smoothed current image to smoothed target image
            si1 = SI.RegisterImagePair()
            si1.register_images(Ic, Iavg, spacing, model_name='svf_vector_momentum_map',
                               smoother_type='multiGaussian',
                               compute_similarity_measure_at_low_res=False,
                               map_low_res_factor=1.0,
                               visualize_step=None,
                               nr_of_iterations=300,
                               rel_ftol=1e-8,
                               similarity_measure_type="ncc",
                               similarity_measure_sigma=0.1,
                               params='fast_registration_params_1.json',
                               json_config_out_filename='fast_registration_params_1.json'
                               )

            # save momentum for second registration
            model_pars = si1.get_model_parameters()
            first_mom=(model_pars['m'].cpu().data.numpy().squeeze())

            if False:
                # visualize warped image and deformation field of first registration
                wp1 = si1.get_map()
                wi1 = utils.compute_warped_image_multiNC(Ic_beforeSmoothing, wp1, si1.spacing, 1)
                Iw = wi1.cpu().data.numpy()[0,0,125:225, :]
                deff_data = wp1.cpu().data.numpy()
                plt.clf()
                plt.imshow(Iw, cmap='gray')
                plt.contour(deff_data[0, 0, :][125:225, :], np.linspace(-1, 1, 300), colors='r', linestyles='solid')
                plt.contour(deff_data[0, 1, :][125:225, :], np.linspace(-1, 1, 300), colors='r', linestyles='solid')
                plt.title('deformation field %i' % (i+1))
                plt.show()
                plt.clf()

            # set all model parameters to zero
            si1.set_model_parameters_to_zero()

        # register current image to target image
        si = SI.RegisterImagePair()
        if do_smoothing:
            # initialize model with momentum from first registration
            si.set_model_parameters({'m': torch.from_numpy(first_mom)})
            si.register_images(Ic_beforeSmoothing.cpu().data.numpy(), Iavg_beforeSmoothing, spacing, model_name='svf_vector_momentum_map',
                               smoother_type='multiGaussian',
                               compute_similarity_measure_at_low_res=False,
                               map_low_res_factor=1.0,
                               visualize_step=None,
                               nr_of_iterations=100,
                               rel_ftol=1e-8,
                               similarity_measure_type="ncc",
                               similarity_measure_sigma=0.1,
                               params='fast_registration_params_2.json',
                               json_config_out_filename='fast_registration_params_2.json'
                               )
        else:
             si.register_images(Ic, Iavg, spacing, model_name='svf_vector_momentum_map',
                               smoother_type='multiGaussian',
                               compute_similarity_measure_at_low_res=False,
                               map_low_res_factor=1.0,
                               visualize_step=None,
                               nr_of_iterations=100,
                               rel_ftol=1e-8,
                               similarity_measure_type="ncc",
                               similarity_measure_sigma=0.1,
                               params='fast_registration_params_2.json',
                               json_config_out_filename='fast_registration_params_2.json'
                               )

        wi = si.get_warped_image()
        wp = si.get_map()
        model_pars = si.get_model_parameters()
        mom = (model_pars['m'].cpu().data.numpy().squeeze())

        mom_list.append(mom)
        wp_list.append(wp.cpu().data.numpy())


        current_filename = registration_results_images_path+'regImage' + str(i+1).zfill(4) + '.nii.gz'
        wi_data = wi.data
        im_io.write(current_filename, wi_data[0,0,:])


        if plot_and_save_def_grids:
            # visualize and save warped image and deformation field of the registration
            Iw = wi.cpu().data.numpy()[0,0,125:225, :]
            deff_data = wp.cpu().data.numpy()
            plt.clf()
            plt.imshow(Iw, cmap='gray')
            plt.contour(deff_data[0, 0, :][125:225, :], np.linspace(-1, 1, 300), colors='r', linestyles='solid')
            plt.contour(deff_data[0, 1, :][125:225, :], np.linspace(-1, 1, 300), colors='r', linestyles='solid')
            plt.title('deformation field %i' % (i+1))
            plt.savefig(registration_results_deformations_path+'reg_deformation_field_grid_300_' + str(i + 1).zfill(4) + '.png')
            plt.show()

        # update new average image
        if i == 0:
            newAvg = wi.data.cpu().numpy()
        else:
            newAvg += wi.data.cpu().numpy()

    # udate, save and visualize new average image
    Iavg = newAvg / len(images)
    Iavg = AdaptVal(Variable(torch.from_numpy(Iavg), requires_grad=False)).data.cpu().numpy()

    filename = registration_results_images_path+'Average_Image_firstReg.nii.gz'
    FIO.ImageIO().write(filename, Iavg[0,0, :])

    plt.imshow(Iavg[0, 0, ...], cmap='gray', vmin=0, vmax=1)
    plt.title('Average Image')
    plt.colorbar()
    plt.show()

    return Iavg, wp_list, mom_list

def multi_registration(images, target_images, do_smoothing, registration_results_path, mode, plot_and_save_def_grids):
    """
    registers multiple source images to multiple target images
    :param images: list of source images
    :param target_images: list of target images
    :param do_smoothing: if true - first register smoothed images, then continue on unsmoothed images
    :param registration_results_path: filepath under which the warped images will be saved as .nii.gz
    :param mode: choose train or test mode (for saving purposes)
    :param plot_and_save_def_grids: if true - plot and save deformation field grids
    :return: returns list of deformation maps, list of momentums
    """
    im_io = FIO.ImageIO()


    # initialize lists to save map and momentum of registration
    wp_list = []
    mom_list = []

    # register all images to the target images and while doing so compute a new average image
    for i, im_name in enumerate(images):
        print('Registering image ' + str(i+1) + '/' + str(len(images)))
        si = SI.RegisterImagePair()
        Ic, hdrc, spacing, _ = im_io.read_to_nc_format(filename=images[i])
        Iavg, _, _, _ = im_io.read_to_nc_format(filename=target_images[i])

        if do_smoothing:
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

            # create the source image as pyTorch variable
            Ic_pt = AdaptVal(Variable(torch.from_numpy(Ic), requires_grad=False))
            Ic_beforeSmoothing = AdaptVal(Variable(torch.from_numpy(Ic), requires_grad=False)).cpu().data.numpy()

            # smoth a little bit
            Ic_pt = s.smooth(Ic_pt)
            Ic = Ic_pt.cpu().data.numpy()

            # create the target image as pyTorch variable
            Iavg_pt = AdaptVal(Variable(torch.from_numpy(Iavg), requires_grad=False))
            Iavg_beforeSmoothing = AdaptVal(Variable(torch.from_numpy(Iavg), requires_grad=False)).cpu().data.numpy()

            # smoth a little bit
            Iavg_pt = s.smooth(Iavg_pt)
            Iavg = Iavg_pt.cpu().data.numpy()

            # register smoothed current image to smoothed target image
            si1 = SI.RegisterImagePair()
            si1.register_images(Ic, Iavg, spacing, model_name='svf_vector_momentum_map',
                                smoother_type='multiGaussian',
                                compute_similarity_measure_at_low_res=False,
                                map_low_res_factor=1.0,
                                visualize_step=None,
                                nr_of_iterations=300,
                                rel_ftol=1e-8,
                                similarity_measure_type="ncc",
                                similarity_measure_sigma=0.1,
                                params='fast_registration_params_1.json',
                                json_config_out_filename='fast_registration_params_1.json'
                                )

            # save momentum for second registration
            model_pars = si1.get_model_parameters()
            first_mom = (model_pars['m'].cpu().data.numpy().squeeze())

            if False:
                # visualize warped image and deformation field of first registration
                wp1 = si1.get_map()
                wi1 = utils.compute_warped_image_multiNC(Ic_beforeSmoothing, wp1, si1.spacing, 1)
                Iw = wi1.cpu().data.numpy()[0, 0, 125:225, :]
                deff_data = wp1.cpu().data.numpy()
                plt.clf()
                plt.imshow(Iw, cmap='gray')
                plt.contour(deff_data[0, 0, :][125:225, :], np.linspace(-1, 1, 300), colors='r', linestyles='solid')
                plt.contour(deff_data[0, 1, :][125:225, :], np.linspace(-1, 1, 300), colors='r', linestyles='solid')
                plt.title('deformation field %i' % (i + 1))
                plt.show()
                plt.clf()

            # set all model parameters to zero
            si1.set_model_parameters_to_zero()

            # register current image to target image
        si = SI.RegisterImagePair()
        if do_smoothing:
            # initialize model with momentum from first registration
            si.set_model_parameters({'m': torch.from_numpy(first_mom)})
            si.register_images(Ic_beforeSmoothing, Iavg_beforeSmoothing, spacing, model_name='svf_vector_momentum_map',
                               smoother_type='multiGaussian',
                               compute_similarity_measure_at_low_res=False,
                               map_low_res_factor=1.0,
                               visualize_step=None,
                               nr_of_iterations=100,
                               rel_ftol=1e-8,
                               similarity_measure_type="ncc",
                               similarity_measure_sigma=0.1,
                               params='findTheBug_GaussianWeights_2.json',
                               json_config_out_filename='findTheBug_GaussianWeights_2.json'
                               )
        else:
            si.register_images(Ic, Iavg, spacing, model_name='svf_vector_momentum_map',
                               smoother_type='multiGaussian',
                               compute_similarity_measure_at_low_res=False,
                               map_low_res_factor=1.0,
                               visualize_step=None,
                               nr_of_iterations=100,
                               rel_ftol=1e-8,
                               similarity_measure_type="ncc",
                               similarity_measure_sigma=0.1,
                               params='findTheBug_GaussianWeights_2.json',
                               json_config_out_filename='findTheBug_GaussianWeights_2.json'
                               )

        wi = si.get_warped_image()
        wp = si.get_map()
        model_pars = si.get_model_parameters()
        mom = (model_pars['m'].cpu().data.numpy().squeeze())

        mom_list.append(mom)
        wp_list.append(wp.cpu().data.numpy())

        current_filename = registration_results_path + mode+ '_regImage' + str(i + 1).zfill(4) + '.nii.gz'
        wi_data = wi.data
        im_io.write(current_filename, wi_data[0, 0, :])

        if plot_and_save_def_grids:
            # visualize and save warped image and deformation field of the registration
            Iw = wi.cpu().data.numpy()[0, 0, 125:225, :]
            deff_data = wp.cpu().data.numpy()
            plt.clf()
            plt.imshow(Iw, cmap='gray')
            plt.contour(deff_data[0, 0, :][125:225, :], np.linspace(-1, 1, 300), colors='r', linestyles='solid')
            plt.contour(deff_data[0, 1, :][125:225, :], np.linspace(-1, 1, 300), colors='r', linestyles='solid')
            plt.title('deformation field %i' % (i + 1))
            plt.savefig(registration_results_path + mode + '_reg_deformation_field_grid_300_' + str(i + 1).zfill(4) + '.png')
            plt.show()

    return wp_list, mom_list

def read_image_and_map_and_apply_map(image_filename,map_filename):
    """
    Reads an image and a map and applies the map to an image
    :param image_filename: input image filename
    :param map_filename: input map filename
    :return: the warped image and its image header as a tuple (im,hdr)
    """

    im_warped = None
    map,map_hdr,_,_ = FIO.MapIO().read_to_nc_format(map_filename)
    #im,hdr,_,_ = FIO.ImageIO().read_to_map_compatible_format(image_filename,map)
    im, hdr, _, _ = FIO.ImageIO().read_to_nc_format(image_filename)

    spacing = hdr['spacing']
    #TODO: check that the spacing is compatible with the map

    if (im is not None) and (map is not None):
        # make pytorch arrays for subsequent processing
        im_t = AdaptVal(Variable(torch.from_numpy(im), requires_grad=False))
        map_t = AdaptVal(Variable(torch.from_numpy(map[0,:]), requires_grad=False))
        im_warped = utils.t2np( utils.compute_warped_image_multiNC(im_t,map_t,spacing[-2:],1) )

        return im_warped,hdr
    else:
        print('Could not read map or image')
        return None,None

def _compute_low_res_image(I,spacing,low_res_size):
    sampler = IS.ResampleImage()
    low_res_image, _ = sampler.downsample_image_to_size(I, spacing, low_res_size[2::],1)
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

def evaluate_model(ISource_in,ITarget_in,sz,spacing,individual_parameters,shared_parameters,params,visualize=True,
                   compute_inverse_map=False):
    """
    evaluates model for given source and target image, size and spacing
    model specified by individual and shared parameters, as well as params from loaded jsonfile
    :param ISource_in: source image as torch variable
    :param ITarget_in: target image as torch variable
    :param sz: size of images
    :param spacing: spacing of images
    :param individual_parameters: dictionary containing the momentum
    :param shared_parameters: empty dictionary
    :param params: model parameter from loaded jsonfile
    :param visualize: if True - plots IS,IT,IW,chessboard,grid,momentum
    :param compute_inverse_map: if true - gives out inverse deformation map [inverse map = None if False]
    :return: returns IWarped, map, inverse map as torch variables
    """

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

    model, criterion = mf.create_registration_model(model_name, params['model'], compute_inverse_map)
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

    if False:
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
    rec_phiWarped_inverse = None

    if use_map:
        if map_low_res_factor is not None:
            if compute_similarity_measure_at_low_res:
                maps = model(lowResIdentityMap, lowResISource)
                if compute_inverse_map:
                    rec_phiWarped = maps[0]
                    rec_phiWarped_inverse=maps[1]
                else:
                    rec_phiWarped = maps
            else:
                maps = model(lowResIdentityMap, lowResISource)
                if compute_inverse_map:
                    rec_tmp = maps[0]
                    rec_tmp_inverse=maps[1]
                else:
                    rec_tmp = maps
                # now upsample to correct resolution
                desiredSz = identityMap.size()[2::]
                sampler = IS.ResampleImage()
                rec_phiWarped, _ = sampler.upsample_image_to_size(rec_tmp, spacing, desiredSz)
                if compute_inverse_map:
                    rec_phiWarped_inverse, _ = sampler.upsample_image_to_size(rec_tmp_inverse, spacing, desiredSz)
        else:
            maps = model(identityMap, ISource)
            if compute_inverse_map:
                rec_phiWarped = maps[0]
                rec_phiWarped_inverse = maps[1]
            else:
                rec_phiWarped = maps

    else:
        rec_IWarped = model(ISource)

    if use_map:
        rec_IWarped = utils.compute_warped_image_multiNC(ISource, rec_phiWarped, spacing,1)

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
            I1Warped = utils.compute_warped_image_multiNC(lowResISource, phi_or_warped_image, lowResSpacing,1)
            vizReg.show_current_images(iter, lowResISource, lowResITarget, I1Warped, vizImage, vizName,
                                       phi_or_warped_image, visual_param)
        else:
            I1Warped = utils.compute_warped_image_multiNC(ISource, phi_or_warped_image, spacing,1)
            vizReg.show_current_images(iter, ISource, ITarget, I1Warped, vizImage, vizName,
                                       phi_or_warped_image, visual_param)
    else:
        vizReg.show_current_images(iter, ISource, ITarget, phi_or_warped_image, vizImage, vizName, None, visual_param)

    return rec_IWarped,rec_phiWarped, rec_phiWarped_inverse