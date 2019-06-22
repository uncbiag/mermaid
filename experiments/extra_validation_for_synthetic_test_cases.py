from __future__ import print_function
from builtins import str
from builtins import range

# needs to be imported before matplotlib to assure proper plotting
import mermaid.visualize_registration_results as vizReg

import torch

import mermaid.fileio as FIO
import mermaid.image_sampling as IS
import mermaid.module_parameters as pars
from mermaid.data_wrapper import USE_CUDA, MyTensor, AdaptVal

import experiment_utils as eu

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import os


def _compute_stat_measures(im):
    d = dict()
    d['mean'] = im.mean()
    d['median'] = np.percentile(im, 50)
    d['std'] = im.std()
    return d

def compute_image_stats(im,label_image=None):
    d = dict()
    d['global'] = _compute_stat_measures(im)
    d['local'] = dict()

    if label_image is not None:
        lvals = np.unique(label_image)
        for l in lvals:
            d['local'][l] = _compute_stat_measures(im[label_image==l])

    return d

def compare_det_of_jac_from_map(map,gt_map,label_image,visualize=False,print_output_directory=None,clean_publication_directory=None,pair_nr=None):

    sz = np.array(map.shape[2:])
    # synthetic spacing
    spacing = np.array(1./(sz-1))

    map_torch =    AdaptVal(torch.from_numpy(map).float())
    gt_map_torch = AdaptVal(torch.from_numpy(gt_map).float())

    det_est = eu.compute_determinant_of_jacobian(map_torch, spacing)
    det_gt = eu.compute_determinant_of_jacobian(gt_map_torch, spacing)

    n = det_est-det_gt

    if visualize:

        if clean_publication_directory is None:
            plt.clf()

            plt.subplot(131)
            plt.imshow(det_gt)
            plt.colorbar()
            plt.title('det_gt')

            plt.subplot(132)
            plt.imshow(det_est)
            plt.colorbar()
            plt.title('det_est')

            plt.subplot(133)
            plt.imshow(n)
            plt.colorbar()
            plt.title('det_est - det_gt')

            if print_output_directory is None:
                plt.show()
            else:
                plt.savefig(os.path.join(print_output_directory, '{:0>3d}'.format(pair_nr) + '_det_jac_validation.pdf'))

        if clean_publication_directory is not None:
            plt.clf()
            plt.imshow(det_gt)
            plt.colorbar()
            plt.axis('image')
            plt.axis('off')
            plt.savefig(os.path.join(clean_publication_directory, 'det_gt_{:0>3d}'.format(pair_nr) + '_det_jac_validation.pdf'),bbox_inches='tight',pad_inches=0)

            plt.clf()
            plt.imshow(det_est)
            plt.colorbar()
            plt.axis('image')
            plt.axis('off')
            plt.savefig(os.path.join(clean_publication_directory, 'det_est_{:0>3d}'.format(pair_nr) + '_det_jac_validation.pdf'),bbox_inches='tight',pad_inches=0)

            plt.clf()
            plt.imshow(n)
            plt.colorbar()
            plt.axis('image')
            plt.axis('off')
            plt.savefig(os.path.join(clean_publication_directory, 'det_est_m_det_gt_{:0>3d}'.format(pair_nr) + '_det_jac_validation.pdf'),bbox_inches='tight',pad_inches=0)

    ds = compute_image_stats(n,label_image)
    return ds

def compare_map(map,gt_map,label_image,visualize=False,print_output_directory=None,clean_publication_directory=None,pair_nr=None):

    # compute vector norm difference
    n = ((map[0, 0, ...] - gt_map[0, 0, ...]) ** 2 + (map[0, 1, ...] - gt_map[0, 1, ...]) ** 2) ** 0.5

    if visualize:

        if clean_publication_directory is None:
            plt.clf()

            plt.subplot(221)
            plt.imshow(map[0,0,...]-gt_map[0,0,...])
            plt.colorbar()
            plt.title('phix-gt_phix')

            plt.subplot(222)
            plt.imshow(map[0,1,...]-gt_map[0,1,...])
            plt.colorbar()
            plt.title('phiy-gt_phiy')

            plt.subplot(223)
            plt.imshow(n)
            plt.colorbar()
            plt.title('2-norm error')

            plt.subplot(224)
            plt.imshow(label_image)
            plt.colorbar()
            plt.title('gt labels')

            if print_output_directory is None:
                plt.show()
            else:
                plt.savefig(os.path.join(print_output_directory, '{:0>3d}'.format(pair_nr) + '_map_validation.pdf'))

        if clean_publication_directory is not None:
            plt.clf()
            plt.imshow(map[0, 0, ...] - gt_map[0, 0, ...])
            plt.colorbar()
            plt.axis('image')
            plt.axis('off')
            plt.savefig(os.path.join(clean_publication_directory, 'phix_m_gt_phix_{:0>3d}'.format(pair_nr) + '_map_validation.pdf'),bbox_inches='tight',pad_inches=0)

            plt.clf()
            plt.imshow(map[0, 1, ...] - gt_map[0, 1, ...])
            plt.colorbar()
            plt.axis('image')
            plt.axis('off')
            plt.savefig(os.path.join(clean_publication_directory, 'phiy_m_gt_phiy_{:0>3d}'.format(pair_nr) + '_map_validation.pdf'),bbox_inches='tight',pad_inches=0)

            plt.clf()
            plt.imshow(n)
            plt.colorbar()
            plt.axis('image')
            plt.axis('off')
            plt.savefig(os.path.join(clean_publication_directory, 'phi_two_norm_error_{:0>3d}'.format(pair_nr) + '_map_validation.pdf'),bbox_inches='tight',pad_inches=0)

            plt.clf()
            plt.imshow(label_image)
            plt.colorbar()
            plt.axis('image')
            plt.axis('off')
            plt.savefig(os.path.join(clean_publication_directory, 'gt_labels_{:0>3d}'.format(pair_nr) + '_map_validation.pdf'),bbox_inches='tight',pad_inches=0)


    ds = compute_image_stats(n,label_image)
    return ds

def downsample_to_compatible_size_single_image(gt_weight,weight,interpolation_order=3):
    # downsample the ground truth weights if needed
    if gt_weight.shape==weight.shape:
        return gt_weight
    else:
        sampler = IS.ResampleImage()

        gt_weight_sz = gt_weight.shape
        gt_weight_reshaped = AdaptVal(torch.from_numpy(gt_weight.view().reshape([1, 1] + list(gt_weight_sz))).float())
        spacing = np.array([1., 1.])
        desired_size = weight.shape

        gt_weight_downsampled_t,_ = sampler.downsample_image_to_size(gt_weight_reshaped, spacing, desired_size, interpolation_order)
        gt_weight_downsampled = gt_weight_downsampled_t.detach().cpu().numpy()

        return gt_weight_downsampled

def downsample_to_compatible_size(gt_weights_orig,weights):

    gt_weights = np.zeros_like(weights)
    nr_of_weights = weights.shape[1]

    for n in range(nr_of_weights):
        gt_weights[0,n,...] = downsample_to_compatible_size_single_image(gt_weights_orig[0,n,...],weights[0,n,...])

    return gt_weights

def upsample_to_compatible_size_single_image(gt_weight,weight,interpolation_order=1):
    # upsample the weights if needed
    if gt_weight.shape==weight.shape:
        return weight
    else:
        sampler = IS.ResampleImage()

        weight_sz = weight.shape
        weight_reshaped = AdaptVal(torch.from_numpy(weight.view().reshape([1, 1] + list(weight_sz))).float())
        spacing = np.array([1., 1.])
        desired_size = gt_weight.shape

        weight_upsampled_t,_ = sampler.upsample_image_to_size(weight_reshaped, spacing, desired_size, interpolation_order)
        weight_upsampled = weight_upsampled_t.detach().cpu().numpy()

        return weight_upsampled

def upsample_to_compatible_size(gt_weights,weights_orig):

    weights = np.zeros_like(gt_weights)
    nr_of_weights = gt_weights.shape[1]

    for n in range(nr_of_weights):
        weights[0,n,...] = upsample_to_compatible_size_single_image(gt_weights[0,n,...],weights_orig[0,n,...])

    return weights

def _compute_overall_stds(weights,stds):

    nr_of_weights = weights.shape[1]
    nr_of_stds = len(stds)

    if nr_of_weights!=nr_of_stds:
        raise ValueError('Number of stds needs to be the same as the number of weights')

    overall_stds = np.zeros_like(weights[0,0,...])
    for i in range(nr_of_stds):
        overall_stds += weights[0,i,...]*stds[i]**2

    overall_stds = overall_stds**0.5

    return overall_stds

def compare_weights(weights_orig,gt_weights_orig,multi_gaussian_stds_synth,multi_gaussian_stds,label_image,visualize=False,print_output_directory=None,clean_publication_directory=None,pair_nr=None,upsample_weights=True):

    nr_of_weights = weights_orig.shape[1]
    nr_of_weights_gt = gt_weights_orig.shape[1]

    if nr_of_weights!=nr_of_weights_gt:
        raise ValueError('The number of weights for the ground truth and the estimate is not the same; make sure the same # and the same standard deviations were used')

    if upsample_weights:
        gt_weights = gt_weights_orig
        weights = upsample_to_compatible_size(gt_weights_orig,weights_orig)
    else:
        gt_weights = downsample_to_compatible_size(gt_weights_orig,weights_orig)
        weights = weights_orig

    max_std = max( max(multi_gaussian_stds), max(multi_gaussian_stds_synth))

    stds_synth = _compute_overall_stds(gt_weights,multi_gaussian_stds_synth)
    stds_computed = _compute_overall_stds(weights,multi_gaussian_stds)

    compatible_stds = (multi_gaussian_stds==multi_gaussian_stds_synth)
    if not compatible_stds:
        print('WARNING: standard deviations are not compatible; only comparing effective standard deviations and not the weights')

    if visualize:

        if compatible_stds:

            if clean_publication_directory is None:

                plt.clf()

                for n in range(nr_of_weights):
                    plt.subplot(3,nr_of_weights,1+n)
                    plt.imshow(gt_weights[0,n,...])
                    plt.colorbar()
                    plt.title('gt_w')

                for n in range(nr_of_weights):
                    plt.subplot(3, nr_of_weights, 1 + n + nr_of_weights)
                    plt.imshow(weights[0, n, ...])
                    plt.colorbar()
                    plt.title('w')

                for n in range(nr_of_weights):
                    plt.subplot(3, nr_of_weights, 1 + n + 2*nr_of_weights)
                    plt.imshow(weights[0, n, ...]-gt_weights[0,n,...])
                    plt.colorbar()
                    plt.title('w-gt_w')

                if print_output_directory is None:
                    plt.show()
                else:
                    plt.savefig(os.path.join(print_output_directory, '{:0>3d}'.format(pair_nr) + '_weights_validation.pdf'))

            if clean_publication_directory is not None:
                for n in range(nr_of_weights):
                    plt.clf()
                    plt.imshow(gt_weights[0, n, ...],clim=(0.0,1.0))
                    plt.colorbar()
                    plt.axis('image')
                    plt.axis('off')
                    plt.savefig(os.path.join(clean_publication_directory, 'gt_weight_{:d}_{:0>3d}'.format(n,pair_nr) + '_weights_validation.pdf'),bbox_inches='tight',pad_inches=0)

                for n in range(nr_of_weights):
                    plt.clf()
                    plt.imshow(weights[0, n, ...],clim=(0.0,1.0))
                    plt.colorbar()
                    plt.axis('image')
                    plt.axis('off')
                    plt.savefig(os.path.join(clean_publication_directory, 'estimated_weight_{:d}_{:0>3d}'.format(n,pair_nr) + '_weights_validation.pdf'),bbox_inches='tight',pad_inches=0)

                for n in range(nr_of_weights):
                    plt.clf()
                    plt.imshow(weights[0, n, ...] - gt_weights[0, n, ...],clim=(-1.0,1.0))
                    plt.colorbar()
                    plt.axis('image')
                    plt.axis('off')
                    plt.savefig(os.path.join(clean_publication_directory, 'estimated_m_gt_weight_{:d}_{:0>3d}'.format(n,pair_nr) + '_weights_validation.pdf'),bbox_inches='tight',pad_inches=0)


        if clean_publication_directory is None:

            plt.clf()

            plt.subplot(131)
            plt.imshow(stds_synth)
            plt.colorbar()
            plt.title('std(synth)')

            plt.subplot(132)
            plt.imshow(stds_computed)
            plt.colorbar()
            plt.title('std')

            plt.subplot(133)
            plt.imshow(stds_computed-stds_synth)
            plt.colorbar()
            plt.title('std-std(synth)')

            if print_output_directory is None:
                plt.show()
            else:
                plt.savefig(os.path.join(print_output_directory, '{:0>3d}'.format(pair_nr) + '_stds_validation.pdf'))

        if clean_publication_directory is not None:
            plt.clf()
            plt.imshow(stds_synth,clim=(0.0,max_std))
            plt.colorbar()
            plt.axis('image')
            plt.axis('off')
            plt.savefig(os.path.join(clean_publication_directory, 'std_synth_{:0>3d}'.format(pair_nr) + '_stds_validation.pdf'),bbox_inches='tight',pad_inches=0)

            plt.clf()
            plt.imshow(stds_computed,clim=(0.0,max_std))
            plt.colorbar()
            plt.axis('image')
            plt.axis('off')
            plt.savefig(os.path.join(clean_publication_directory, 'std_estimated_{:0>3d}'.format(pair_nr) + '_stds_validation.pdf'),bbox_inches='tight',pad_inches=0)

            plt.clf()
            plt.imshow(stds_computed - stds_synth,clim=(-max_std,max_std))
            plt.colorbar()
            plt.axis('image')
            plt.axis('off')
            plt.savefig(os.path.join(clean_publication_directory, 'std_estimated_m_synth_{:0>3d}'.format(pair_nr) + '_stds_validation.pdf'),bbox_inches='tight',pad_inches=0)

    weights_stats = dict()

    if compatible_stds:
        weights_stats['weights'] = dict()
        for n in range(nr_of_weights):
            weights_stats['weights'][n] = compute_image_stats(weights[0, n, ...]-gt_weights[0,n,...],label_image)

    weights_stats['overall_stds'] = compute_image_stats(stds_computed-stds_synth,label_image)

    return weights_stats

def compare_momentum(momentum,gt_momentum,label_image,visualize=False,print_output_directory=None,clean_publication_directory=None,pair_nr=None):

    if momentum.shape!=gt_momentum.shape:
        raise ValueError('Momentum comparisons are only supported for the same size')

    # compute vector norm difference
    n = ((momentum[0, 0, ...] - gt_momentum[0, 0, ...]) ** 2 +
         (momentum[0, 1, ...] - gt_momentum[0, 1, ...]) ** 2) ** 0.5

    if visualize:

        if clean_publication_directory is None:
            plt.clf()

            plt.subplot(131)
            plt.imshow(momentum[0, 0, ...] - gt_momentum[0, 0, ...])
            plt.colorbar()
            plt.title('mx-gt_mx')

            plt.subplot(132)
            plt.imshow(momentum[0, 1, ...] - gt_momentum[0, 1, ...])
            plt.colorbar()
            plt.title('my-gt_my')

            plt.subplot(133)
            plt.imshow(n)
            plt.colorbar()
            plt.title('2-norm error')

            if print_output_directory is None:
                plt.show()
            else:
                plt.savefig(os.path.join(print_output_directory, '{:0>3d}'.format(pair_nr) + '_momentum_validation.pdf'))

        if clean_publication_directory is not None:
            plt.clf()
            plt.imshow(momentum[0, 0, ...] - gt_momentum[0, 0, ...])
            plt.colorbar()
            plt.axis('image')
            plt.axis('off')
            plt.savefig(os.path.join(clean_publication_directory, 'mx_m_gt_mx_{:0>3d}'.format(pair_nr) + '_momentum_validation.pdf'),bbox_inches='tight',pad_inches=0)

            plt.clf()
            plt.imshow(momentum[0, 1, ...] - gt_momentum[0, 1, ...])
            plt.colorbar()
            plt.axis('image')
            plt.axis('off')
            plt.savefig(os.path.join(clean_publication_directory, 'my_m_gt_my_{:0>3d}'.format(pair_nr) + '_momentum_validation.pdf'),bbox_inches='tight',pad_inches=0)

            plt.clf()
            plt.imshow(n)
            plt.colorbar()
            plt.axis('image')
            plt.axis('off')
            plt.savefig(os.path.join(clean_publication_directory, 'm_two_norm_error_{:0>3d}'.format(pair_nr) + '_momentum_validation.pdf'),bbox_inches='tight',pad_inches=0)

    ds = compute_image_stats(n,label_image)
    return ds

def compute_and_visualize_validation_result(multi_gaussian_stds_synth,
                                            multi_gaussian_stds,
                                            compare_global_weights,
                                            image_and_map_output_directory,
                                            misc_output_directory,
                                            label_output_directory,
                                            print_output_directory,
                                            pair_nr,current_source_id,
                                            visualize=False,
                                            print_images=False,
                                            clean_publication_print=False,
                                            printing_single_pair=False):

    # load the computed results for map, weights, and momentum
    map_output_filename_pt = os.path.join(image_and_map_output_directory, 'map_{:05d}.pt'.format(pair_nr))
    weights_output_filename_pt = os.path.join(image_and_map_output_directory, 'weights_{:05d}.pt'.format(pair_nr))
    momentum_output_filename_pt = os.path.join(image_and_map_output_directory, 'momentum_{:05d}.pt'.format(pair_nr))

    map = torch.load(map_output_filename_pt).detach().cpu().numpy()
    momentum = torch.load(momentum_output_filename_pt).detach().cpu().numpy()

    if print_images:
        visualize = True

    if clean_publication_print:
        clean_publication_dir = os.path.join(print_output_directory, 'clean_publication_prints')
        if not os.path.exists(clean_publication_dir):
            print('INFO: creating directory {:s}'.format(clean_publication_dir))
            os.mkdir(clean_publication_dir)
    else:
        clean_publication_dir = None

    weights_dict = torch.load(weights_output_filename_pt)
    if not compare_global_weights:
        if 'local_weights' in weights_dict:
            weights = weights_dict['local_weights'].detach().cpu().numpy()
        else:
            raise ValueError('requested comparison of local weights, but local weights are not available')
    else:
        # there are only global weights
        # let's make them "local" so that we can use the same code for comparison everywhere
        global_weights = weights_dict['default_multi_gaussian_weights'].detach().cpu().numpy()
        nr_of_weights = len(global_weights)

        sz_m = list(momentum.shape)
        desired_sz = [sz_m[0]] + [nr_of_weights] + sz_m[2:]
        weights = np.zeros(desired_sz, dtype='float32')
        for n in range(nr_of_weights):
            weights[:, n, ...] = global_weights[n]

    # now get the corresponding ground truth values
    # (these are based on the source image and hence based on the source id)
    # load the ground truth results for map, weights, and momentum
    gt_map_filename = os.path.join(misc_output_directory, 'gt_map_{:05d}.pt'.format(current_source_id))
    gt_weights_filename = os.path.join(misc_output_directory, 'gt_weights_{:05d}.pt'.format(current_source_id))
    gt_momentum_filename = os.path.join(misc_output_directory, 'gt_momentum_{:05d}.pt'.format(current_source_id))
    label_output_filename = os.path.join(label_output_directory,'m{:d}.nii'.format(current_source_id))

    gt_map = torch.load(gt_map_filename)
    gt_weights = torch.load(gt_weights_filename)
    gt_momentum = torch.load(gt_momentum_filename)

    im_io = FIO.ImageIO()
    label_image,_,_,_ = im_io.read_to_nc_format(label_output_filename,silent_mode=True)
    label_image = label_image[0,0,:,:,0]

    if print_images:
        print_output_directory_eff = print_output_directory
    else:
        print_output_directory_eff = None

    if printing_single_pair or pair_nr==0:
        clean_publication_dir_eff = clean_publication_dir
        visualize = True
    else:
        # we don't want to print them for all
        clean_publication_dir_eff = None

    # now we can compare them
    d = dict()
    d['map_stats'] = compare_map(map,gt_map,label_image,visualize=visualize,print_output_directory=print_output_directory_eff,clean_publication_directory=clean_publication_dir_eff,pair_nr=pair_nr)

    d['det_jac_stats'] = compare_det_of_jac_from_map(map,gt_map,label_image,visualize=visualize,print_output_directory=print_output_directory_eff,clean_publication_directory=clean_publication_dir_eff,pair_nr=pair_nr)

    d['weight_stats'] = compare_weights(weights,gt_weights,multi_gaussian_stds_synth,multi_gaussian_stds,
                                        label_image=label_image,visualize=visualize,print_output_directory=print_output_directory_eff,clean_publication_directory=clean_publication_dir_eff,pair_nr=pair_nr)

    if momentum.shape==gt_momentum.shape:
        d['momentum_stats'] = compare_momentum(momentum,gt_momentum,label_image,visualize=visualize,print_output_directory=print_output_directory_eff,clean_publication_directory=clean_publication_dir_eff,pair_nr=pair_nr)

    return d

def _display_array_stats(a,name,file_stream=None):
    mean_str = 'mean({:s})={:f}'.format(name,np.mean(a))
    median_str = 'median({:s})={:f}'.format(name,np.percentile(a,50))
    std_str = 'std({:s})={:f}'.format(name,np.std(a))

    print(mean_str)
    print(median_str)
    print(std_str)

    if file_stream is not None:
        f.write(mean_str+'\n')
        f.write(median_str+'\n')
        f.write(std_str+'\n')

def display_stats(all_stats,name=None,file_stream=None):
    if all_stats is not None:
        if name is None:
            name = ''
        # recurse down the dictionary structure
        if np.isscalar(all_stats):
            print(name + ' = ' + all_stats)
        elif type(all_stats) == np.ndarray:
            _display_array_stats(all_stats,name,file_stream=file_stream)
        elif type(all_stats) == dict:
            for k in all_stats:
                current_name = name + '_' + str(k)
                display_stats(all_stats[k],current_name,file_stream=file_stream)
        else:
            raise ValueError('Type' + str(type(all_stats)) + ' is not supported')

def append_stats(all_stats,current_stats):
    if all_stats is None:
        return current_stats
    else:
        # recurse down the dictionary structure
        if np.isscalar(all_stats):
            return np.append(all_stats,current_stats)
        elif type(all_stats)==np.ndarray:
            return np.append(all_stats,current_stats)
        elif type(all_stats)==dict:
            for k in all_stats:
                all_stats[k] = append_stats(all_stats[k],current_stats[k])
        else:
            raise ValueError('Type' + str(type(all_stats)) + ' is not supported')

    return all_stats


def _show_global_local_boxplot_summary(current_stats,title,desired_stat='median'):

    # desired stat should be 'median' or 'mean'

    plt.clf()

    compound_results = []
    compound_names = []

    for k in current_stats['local']:
        compound_results.append(current_stats['local'][k][desired_stat])
        compound_names.append(str(k))

    compound_results.append(current_stats['global'][desired_stat])
    compound_names.append('global')

    eu.plot_boxplot(compound_results, compound_names)
    plt.title(title)

def show_boxplot_summary(all_stats, print_output_directory=None, visualize=False, print_images=False, clean_publication_print=False):
    # there is a lot of stats info in 'all_stats'
    # let's show the boxplots over the medians for the different regions and overall

    ws = all_stats['weight_stats']
    ms = all_stats['map_stats']
    dj = all_stats['det_jac_stats']

    if print_images:
        visualize = True

    if clean_publication_print:
        clean_publication_dir = os.path.join(print_output_directory, 'clean_publication_prints')
    else:
        clean_publication_dir = None

    desired_stats = ['mean','median']

    for current_stat in desired_stats:

        _show_global_local_boxplot_summary(ms,'map validation results (estimated-truth)',current_stat)
        if visualize:
            if print_output_directory is None:
                plt.show()
            else:
                plt.savefig(os.path.join(print_output_directory, 'stat_summary_map_validation_{:s}.pdf'.format(current_stat)))

        _show_global_local_boxplot_summary(dj, 'det of jac validation results (estimated-truth)', current_stat)
        if visualize:
            if print_output_directory is None:
                plt.show()
            else:
                plt.savefig(
                    os.path.join(print_output_directory, 'stat_summary_det_jac_validation_{:s}.pdf'.format(current_stat)))

        _show_global_local_boxplot_summary(ws['overall_stds'],'overall stds results (estimated-truth)',current_stat)
        if visualize:
            if print_output_directory is None:
                plt.show()
            else:
                plt.savefig(os.path.join(print_output_directory, 'stat_summary_overall_stds_validation_{:s}.pdf'.format(current_stat)))

        for k in ws['weights']:
            _show_global_local_boxplot_summary(ws['weights'][k], 'weight {:d} results (estimated-truth)'.format(k),current_stat)
            if visualize:
                if print_output_directory is None:
                    plt.show()
                else:
                    plt.savefig(os.path.join(print_output_directory, 'stat_summary_weight_{:d}_validation_{:s}.pdf'.format(k,current_stat)))

    if visualize and clean_publication_dir is not None:
        for current_stat in desired_stats:

            _show_global_local_boxplot_summary(ms, '', current_stat)
            plt.savefig(os.path.join(clean_publication_dir, 'clean_stat_summary_map_validation_{:s}.pdf'.format(current_stat)),bbox_inches='tight',pad_inches=0)

            _show_global_local_boxplot_summary(dj, '', current_stat)
            plt.savefig(os.path.join(clean_publication_dir, 'clean_stat_summary_det_jac_validation_{:s}.pdf'.format(current_stat)),bbox_inches='tight',pad_inches=0)

            _show_global_local_boxplot_summary(ws['overall_stds'],'',current_stat)
            plt.savefig(os.path.join(clean_publication_dir, 'clean_stat_summary_overall_stds_validation_{:s}.pdf'.format(current_stat)),bbox_inches='tight',pad_inches=0)

            for k in ws['weights']:
                _show_global_local_boxplot_summary(ws['weights'][k], '',current_stat)
                plt.savefig(os.path.join(clean_publication_dir,'clean_stat_summary_weight_{:d}_validation_{:s}.pdf'.format(k,current_stat)),bbox_inches='tight',pad_inches=0)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Computes additional validation measures for the synthetically generated data')

    parser.add_argument('--config', required=True, default=None, help='Main configuration file used for the experimental run (will automatically be stage adapted)')

    parser.add_argument('--output_directory', required=True, help='Where the output was stored (now this will be the input directory)')
    parser.add_argument('--input_image_directory', required=True, help='Directory where all the images are')
    parser.add_argument('--stage_nr', required=True, type=int, help='stage number for which the computations should be performed {0,1,2}')

    parser.add_argument('--compute_only_pair_nr', required=False, type=int, default=None, help='When specified only this pair is computed; otherwise all of them')
    parser.add_argument('--compute_from_frozen', action='store_true', help='computes the results from optimization results with frozen parameters')

    parser.add_argument('--do_not_visualize', action='store_true', help='visualizes the output otherwise')
    parser.add_argument('--do_not_print_images', action='store_true', help='prints the results otherwise')
    parser.add_argument('--clean_publication_print', action='store_true', help='Modifies the printing behavior so also clean images for publications are created; if combined with --compute_only_pair_nr then ONLY the clean images are created and no other output is generated')

    args = parser.parse_args()

    output_dir = args.output_directory
    input_image_directory = args.input_image_directory
    misc_output_directory = os.path.join(os.path.split(input_image_directory)[0],'misc')
    label_output_directory = os.path.join(os.path.split(input_image_directory)[0],'label_affine_icbm')

    synthetic_json = os.path.join(os.path.split(input_image_directory)[0],'config.json')
    # load the synthetic json to determine what stds were used to generate this data
    if not os.path.isfile(synthetic_json):
        raise ValueError('Could not find {:s}, the configuration file for the synthetically generated data'.format(synthetic_json))
    else:
        params_synth = pars.ParameterDict()
        params_synth.load_JSON(synthetic_json)
        # and now get the standard deviations
        multi_gaussian_stds_synth = params_synth[('multi_gaussian_stds',None,'standard deviations used to create the synthetic data')]
        if multi_gaussian_stds_synth is None:
            raise ValueError('Could not find key, multi_gaussian_stds, in {:s}'.format(synthetic_json))

    print_output_directory = os.path.join(os.path.normpath(output_dir), 'pdf_extra_validation_{:d}'.format(args.stage_nr))
    if not os.path.isdir(print_output_directory):
        os.mkdir(print_output_directory)

    stage = args.stage_nr
    if stage<2:
        compare_global_weights=True
    else:
        compare_global_weights=False

    compute_from_frozen = args.compute_from_frozen

    if compute_from_frozen:
        json_config = os.path.join(args.output_directory, 'frozen_out_stage_{:d}_'.format(stage) + os.path.split(args.config)[1])
    else:
        json_config = os.path.join(args.output_directory, 'out_stage_{:d}_'.format(stage) + os.path.split(args.config)[1])

    # load the json used for the stage run to determine what stds were used to generate this data
    if not os.path.isfile(json_config):
        raise ValueError('Could not find {:s}, the configuration file for the generated data'.format(json_config))
    else:
        params = pars.ParameterDict()
        params.load_JSON(json_config)
        # and now get the standard deviations
        multi_gaussian_stds = params['model']['registration_model']['forward_model']['smoother'][('multi_gaussian_stds', None,'multi gaussian stds')]
        multi_gaussian_weights = params['model']['registration_model']['forward_model']['smoother'][('multi_gaussian_weights', None,'multi gaussian weights')]

        if multi_gaussian_stds is None:
            raise ValueError('Could not find key, multi_gaussian_stds, in {:s}'.format(json_config))

        if multi_gaussian_weights is None:
            raise ValueError('Could not find key, multi_gaussian_weights, in {:s}'.format(json_config))

    used_pairs = torch.load(os.path.join(output_dir, 'used_image_pairs.pt'))
    nr_of_computed_pairs = len(used_pairs['source_ids'])

    source_ids = used_pairs['source_ids']

    if args.compute_only_pair_nr is not None:
        pair_nrs = [args.compute_only_pair_nr]
    else:
        pair_nrs = list(range(nr_of_computed_pairs))

    if compute_from_frozen:
        image_and_map_output_dir = os.path.join(os.path.normpath(output_dir),'model_results_frozen_stage_{:d}'.format(stage))
    else:
        image_and_map_output_dir = os.path.join(os.path.normpath(output_dir), 'model_results_stage_{:d}'.format(stage))

    extra_stats_filename_txt = os.path.join(image_and_map_output_dir,'extra_statistics.txt')
    extra_stats_filename_pt = os.path.join(image_and_map_output_dir,'extra_statistics.pt')

    all_stats = None

    #DEBUG
    #all_stats = torch.load(extra_stats_filename_pt)
    #show_boxplot_summary(all_stats, print_output_directory=print_output_directory,
    #                     visualize=not args.do_not_visualize)
    #DEBUG

    nr_of_pairs = len(pair_nrs)
    printing_single_pair = (nr_of_pairs==1)

    for pair_nr in pair_nrs:

        print('Computing pair {:d}/{:d}'.format(pair_nr+1,nr_of_pairs))

        # compute the validation results pair-by-pair
        current_source_id = source_ids[pair_nr]
        current_stats = compute_and_visualize_validation_result(multi_gaussian_stds_synth=multi_gaussian_stds_synth,
                                                                multi_gaussian_stds=multi_gaussian_stds,
                                                                compare_global_weights=compare_global_weights,
                                                                image_and_map_output_directory=image_and_map_output_dir,
                                                                misc_output_directory=misc_output_directory,
                                                                label_output_directory=label_output_directory,
                                                                print_output_directory=print_output_directory,
                                                                pair_nr=pair_nr,
                                                                current_source_id=current_source_id,
                                                                visualize=not args.do_not_visualize,
                                                                print_images=not args.do_not_print_images,
                                                                clean_publication_print=args.clean_publication_print,
                                                                printing_single_pair=printing_single_pair)

        all_stats = append_stats(all_stats,current_stats)

    if args.compute_only_pair_nr is None:
        show_boxplot_summary(all_stats,
                             print_output_directory=print_output_directory,
                             visualize=not args.do_not_visualize,
                             print_images=not args.do_not_print_images,
                             clean_publication_print=args.clean_publication_print)

        # now save and output the means and the standard deviations for all the values
        print('Writing text statistics output to {:s}'.format(extra_stats_filename_txt))
        f = open(extra_stats_filename_txt, 'w')
        display_stats(all_stats,file_stream=f)
        f.close()

        print('Writing pt statistics output to {:s}'.format(extra_stats_filename_pt))
        torch.save(all_stats,extra_stats_filename_pt)

        if not args.do_not_print_images:
            # if we have pdfjam we create a summary pdf
            if os.system('which pdfjam') == 0:
                summary_pdf_name = os.path.join(print_output_directory, 'extra_summary.pdf')

                if os.path.isfile(summary_pdf_name):
                    os.remove(summary_pdf_name)

                print('Creating summary PDF: ')
                cmd = 'pdfjam {:} --nup 1x2 --outfile {:}'.format(os.path.join(print_output_directory, '*.pdf'), summary_pdf_name)
                os.system(cmd)
