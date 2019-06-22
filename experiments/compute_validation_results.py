from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range

# needs to be imported before matplotlib to assure proper plotting
import mermaid.visualize_registration_results as vizReg

import os

import torch

import mermaid.utils as utils
import mermaid.fileio as fio

import scipy.io as sio
import numpy as np
import itk
# needs to be imported after itk to overwrite itk's incorrect error handling
import mermaid.fixwarnings

import nrrd
import torch

import matplotlib.pyplot as plt
import matplotlib

def new_warp_image_nn(label_map, phi, spacing):

    lm_t = torch.from_numpy(label_map)
    lm_t = lm_t.view([1,1]+list(lm_t.size()))

    return utils.get_warped_label_map(lm_t, phi, spacing, sched='nn')


def _warp_image_nn_1d(moving, phi):
    # get image dimensions
    dim1 = moving.shape[0]

    # round the deformation map to integer
    phi_round = np.round(phi).astype('int')
    idx_x = np.reshape(phi_round[0, :], (dim1, 1))

    # deal with extreme cases
    idx_x[idx_x < 0] = 0
    idx_x[idx_x > dim1 - 1] = dim1 - 1

    # get the wrapped results
    ind = np.ravel_multi_index([idx_x], [dim1])
    result = moving.flatten()[ind]
    result = np.reshape(result, (dim1))
    return result


def _warp_image_nn_2d(moving, phi):

    # get image dimensions
    dim1 = moving.shape[0]
    dim2 = moving.shape[1]

    # round the deformation map to integer
    phi_round = np.round(phi).astype('int')
    idx_x = np.reshape(phi_round[0, :, :], (dim1 * dim2, 1))
    idx_y = np.reshape(phi_round[1, :, :], (dim1 * dim2, 1))

    # deal with extreme cases
    idx_x[idx_x < 0] = 0
    idx_x[idx_x > dim1 - 1] = dim1 - 1
    idx_y[idx_y < 0] = 0
    idx_y[idx_y > dim2 - 1] = dim2 - 1

    # get the wrapped results
    ind = np.ravel_multi_index([idx_x, idx_y], [dim1, dim2])
    result = moving.flatten()[ind]
    result = np.reshape(result, (dim1, dim2))
    return result

def _warp_image_nn_3d(moving, phi):

    # get image dimensions
    dim1 = moving.shape[0]
    dim2 = moving.shape[1]
    dim3 = moving.shape[2]

    # round the deformation map to integer
    phi_round = np.round(phi).astype('int')
    idx_x = np.reshape(phi_round[0, :, :, :], (dim1 * dim2 * dim3, 1))
    idx_y = np.reshape(phi_round[1, :, :, :], (dim1 * dim2 * dim3, 1))
    idx_z = np.reshape(phi_round[2, :, :, :], (dim1 * dim2 * dim3, 1))

    # deal with extreme cases
    idx_x[idx_x < 0] = 0
    idx_x[idx_x > dim1 - 1] = dim1 - 1
    idx_y[idx_y < 0] = 0
    idx_y[idx_y > dim2 - 1] = dim2 - 1
    idx_z[idx_z < 0] = 0
    idx_z[idx_z > dim3 - 1] = dim3 - 1

    # get the wrapped results
    ind = np.ravel_multi_index([idx_x, idx_y, idx_z], [dim1, dim2, dim3])
    result = moving.flatten()[ind]
    result = np.reshape(result, (dim1, dim2, dim3))
    return result

def warp_image_nn(moving, phi):
    """
    Warp labels according to deformation field
    :param moving: Moving image in numpy array format
    :param phi: Deformation field in numpy array format
    :return: Warped labels according to deformation field
    """

    dim = len(moving.shape)

    if dim==1:
        return _warp_image_nn_1d(moving,phi)
    elif dim==2:
        return _warp_image_nn_2d(moving,phi)
    elif dim==3:
        return _warp_image_nn_3d(moving, phi)
    else:
        raise ValueError('Only dimensions 1, 2, and 3 supported')


def calculate_image_overlap(dataset_info, phi_path, source_labelmap_path, target_labelmap_path,
                            warped_labelmap_path,moving_id, target_id, use_sym_links=True):
    """
    Calculate the overlapping rate of a specified case
    :param dataset_info: dictionary containing all the validation dataset information
    :param dataset_dir: path to the label datasets
    :param phi_path: deformation field path
    :param moving_id: moving image id
    :param target_id: target image id
    :return:
    """

    Labels = None
    nr_of_labels = -1
    if dataset_info['label_name'] is not None:
        Labels = sio.loadmat(dataset_info['label_name'])
        nr_of_labels = (len(Labels['Labels']))
        result = np.zeros(nr_of_labels)
    else:
        if 'nr_of_labels' in dataset_info:
            nr_of_labels = dataset_info['nr_of_labels']
            result = np.zeros(nr_of_labels)
        else:
            raise ValueError('If matlab label file not given, nr_of_labels needs to be specified')

        # todo: not sure why these are floats, but okay for now

    im_io = fio.ImageIO()
    label_from_id = moving_id-dataset_info['start_id'] # typicall starts at 1
    label_to_id = target_id-dataset_info['start_id']

    label_from_filename = dataset_info['label_files_dir'] + dataset_info['label_prefix'] + '{:d}.nii'.format(label_from_id + dataset_info['start_id'])
    label_from, hdr, _, _ = im_io.read(label_from_filename, silent_mode=True, squeeze_image=True)

    label_to_filename = dataset_info['label_files_dir'] + dataset_info['label_prefix'] + '{:d}.nii'.format(label_to_id + dataset_info['start_id'])
    label_to, hdr, _, _ = im_io.read(label_to_filename, silent_mode=True, squeeze_image=True)

    map_io = fio.MapIO()

    phi,_,_,_ = map_io.read_from_validation_map_format(phi_path)
    warp_result = warp_image_nn(label_from, phi)

    im_io.write(warped_labelmap_path,warp_result,hdr)

    if source_labelmap_path is not None:
        if use_sym_links:
            utils.create_symlink_with_correct_ext(label_from_filename,source_labelmap_path)
        else:
            im_io.write(source_labelmap_path,label_from,hdr)

    if target_labelmap_path is not None:
        if use_sym_links:
            utils.create_symlink_with_correct_ext(label_to_filename,target_labelmap_path)
        else:
            im_io.write(target_labelmap_path,label_to,hdr)


    for label_idx in range(nr_of_labels):

        if Labels is not None:
            current_id = Labels['Labels'][label_idx][0]
        else:
            current_id = label_idx

        target_vol = float( (label_to==current_id).sum() )

        if target_vol==0 or current_id==0:
            result[label_idx] = np.nan
        else:
            intersection = float((np.logical_and(label_to == current_id, warp_result == current_id)).sum())
            result[label_idx] = intersection/target_vol

    single_result = result
    single_result = single_result[~np.isnan(single_result)]
    if len(single_result)==0:
        result_mean = np.nan
    else:
        result_mean = np.mean(single_result)

    return result_mean,single_result

    #print('overlapping rate of without unNaN value')
    #print(single_result)
    #print('Averaged overlapping rate')
    #print(result_mean)


def overlapping_plot(old_results_filename, new_results, boxplot_filename, visualize=True):
    """
    Plot the overlaping results of 14 old appraoch and the proposed appraoch
    :param old_results_filename: Old results stored in .mat format file
    :param new_results: Dictionary that contains the new results
    :return:
    """

    if old_results_filename is not None:
        old_results = sio.loadmat(old_results_filename)
    else:
        old_results = None

    # combine old names with proposed method
    compound_names = []

    if old_results is not None:
        for item in old_results['direc_name']:
            compound_names.append(str(item[0])[3:-2])

    compound_names.append('Proposed')

    # new results may only have a subset of the results
    if old_results is not None:
        old_results_selected = old_results['results'][new_results['ind'],:] # select the desired rows

        # combine data
        compound_results = np.concatenate((old_results_selected, np.array(new_results['mean_target_overlap']).reshape(-1, 1)), axis=1)
    else:
        compound_results = np.array(new_results['mean_target_overlap']).reshape(-1, 1)

    # create a figure instance
    fig = plt.figure(1, figsize=(8, 6))

    # create an axes instance
    ax = fig.add_subplot(111)

    # set axis tick
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.yaxis.set_tick_params(left=True, direction='in', width=1)
    ax.yaxis.set_tick_params(right=True, direction='in', width=1)
    ax.xaxis.set_tick_params(top=False, direction='in', width=1)
    ax.xaxis.set_tick_params(bottom=False, direction='in', width=1)

    # create the boxplot
    bp = plt.boxplot(compound_results, vert=True, whis=1.5, meanline=True, widths=0.16, showfliers=True,
                     showcaps=False, patch_artist=True, labels=compound_names)

    # rotate x labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    # set properties of boxes, medians, whiskers, fliers
    plt.setp(bp['medians'], color='orange')
    plt.setp(bp['boxes'], color='blue')
    # plt.setp(bp['caps'], color='b')
    plt.setp(bp['whiskers'], linestyle='-', color='blue')
    plt.setp(bp['fliers'], marker='o', markersize=5, markeredgecolor='blue')

    # matplotlib.rcParams['ytick.direction'] = 'in'
    # matplotlib.rcParams['xtick.direction'] = 'inout'

    # setup font
    font = {'family': 'normal', 'weight': 'semibold', 'size': 10}
    matplotlib.rc('font', **font)

    # set the line width of the figure
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    # set the range of the overlapping rate
    plt.ylim([0, 1.0])

    # add two lines to represent the lower quartile and upper quartile
    lower_quartile = np.percentile(compound_results[:, -1], 25)
    upper_quartile = np.percentile(compound_results[:, -1], 75)
    ax.axhline(lower_quartile, ls='-', color='r', linewidth=1)
    ax.axhline(upper_quartile, ls='-', color='r', linewidth=1)

    # set the target box to red color
    bp['boxes'][-1].set(color='red')
    bp['boxes'][-1].set(facecolor='red')
    bp['whiskers'][-1].set(color='red')
    bp['whiskers'][-2].set(color='red')
    bp['fliers'][-1].set(color='red', markeredgecolor='red')

    # save figure
    if boxplot_filename is not None:
        print('Saving boxplot to : ' + boxplot_filename )
        plt.savefig(boxplot_filename, dpi=1000, bbox_inches='tight')

    # show figure
    if visualize:
        plt.show()


def extract_id_from_cumc_filename(filename):
    r = os.path.split(filename)
    # these files have the format m1.nii, m12.nii, ...
    nr = int(r[1][1:-4])
    return nr

def create_stage_output_dir(output_dir,stage,compute_from_frozen=False):
    if not stage in [0,1,2]:
        raise ValueError('stages need to be {0,1,2}')

    if compute_from_frozen:
        stage_output_dir = os.path.join(os.path.normpath(output_dir), 'model_results_frozen_stage_{:d}'.format(stage))
    else:
        stage_output_dir = os.path.join(os.path.normpath(output_dir), 'model_results_stage_{:d}'.format(stage))

    return stage_output_dir

def create_filenames(id,output_dir,stage,compute_from_frozen=False):

    if not stage in [0,1,2]:
        raise ValueError('stages need to be {0,1,2}')

    stage_output_dir = create_stage_output_dir(output_dir,stage,compute_from_frozen)
    
    map_filename = os.path.join(stage_output_dir,'map_validation_format_{:05d}.nrrd'.format(id))
    warped_labelmap_filename = os.path.join(stage_output_dir,'warped_labelmap_{:05d}.nrrd'.format(id))
    source_labelmap_filename = os.path.join(stage_output_dir,'source_labelmap_{:05d}.nrrd'.format(id))
    target_labelmap_filename = os.path.join(stage_output_dir,'target_labelmap_{:05d}.nrrd'.format(id))

    return map_filename,source_labelmap_filename,target_labelmap_filename,warped_labelmap_filename,stage_output_dir


def get_result_indices(r,start_id,nr_of_images_in_dataset):

   D = {}
   cnt = 0
   for src_idx in np.arange(nr_of_images_in_dataset):
       for dst_idx in np.arange(nr_of_images_in_dataset):
           if src_idx == dst_idx:
               continue
           if (src_idx+1) in D:
               D[src_idx+1][dst_idx+1]=cnt
           else:
               D[src_idx+1]={}
               D[src_idx+1][dst_idx+1]=cnt
           cnt += 1

   indices = []
   for sid,tid in zip(r['source_id'],r['target_id']):
       indices.append(D[sid][tid])
   return indices


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Computes target overlap for registration results')

    parser.add_argument('--output_directory', required=True,
                        help='Main directory where the output was stored (now this will be the input directory)')
    parser.add_argument('--stage_nr', required=True, type=int,
                        help='stage number for which the computations should be performed {0,1,2}; shifted by one')
    parser.add_argument('--dataset_directory', required=True,
                        help='Main directory where dataset is stored; this directory should contain the subdirectory label_affine_icbm')

    parser.add_argument('--dataset', required=False, default=None, help='Which validation dataset is being used. Specify as [CUMC|LPBA|IBSR|MGH|SYNTH]; CUMC is the default')

    parser.add_argument('--compute_from_frozen', action='store_true', help='computes the results from optimization results with frozen parameters')

    parser.add_argument('--do_not_write_target_labelmap', action='store_true', help='otherwise also writes the target labelmap again for easy visualization')
    parser.add_argument('--do_not_write_source_labelmap', action='store_true', help='otherwise also writes the source labelmap again for easy visualization')

    parser.add_argument('--do_not_use_symlinks', action='store_true', help='For source and target labelmaps, by default symbolic links are created, otherwise files are copied')

    parser.add_argument('--do_not_visualize', action='store_true', help='visualizes the output otherwise')
    parser.add_argument('--do_not_print_images', action='store_true', help='prints the results otherwise')

    parser.add_argument('--save_overlap_filename', required=False, default=None,help='If specified write the result in this output file')

    args = parser.parse_args()

    # output_directory = './sample_3d_res/test_out'
    # stage = 2
    # dataset_directory = '/Users/mn/data/testdata/CUMC12'

    output_directory = args.output_directory
    dataset_directory = args.dataset_directory
    stage = args.stage_nr

    used_pairs = torch.load(os.path.join(output_directory, 'used_image_pairs.pt'))
    nr_of_computed_pairs = len(used_pairs['source_ids'])

    # these are the supported validation datasets
    validation_datasets = dict()
    validation_datasets['CUMC'] = {'nr_of_images_in_dataset': 12,
                                   'start_id': 1,
                                   'label_prefix': 'm',
                                   'label_name': './validation_mat/m_Labels.mat',
                                   'nr_of_labels': None,
                                   'label_files_dir': os.path.join(dataset_directory, 'label_affine_icbm/'),
                                   'old_klein_results_filename': './validation_mat/quicksilver_results/CUMC_results.mat'}

    validation_datasets['LPBA'] = {'nr_of_images_in_dataset': 40,
                                   'start_id': 1,
                                   'label_prefix': 's',
                                   'label_name': './validation_mat/l_Labels.mat',
                                   'nr_of_labels': None,
                                   'label_files_dir': os.path.join(dataset_directory, 'label_affine_icbm/'),
                                   'old_klein_results_filename': './validation_mat/quicksilver_results/LPBA_results.mat'}

    validation_datasets['IBSR'] = {'nr_of_images_in_dataset': 18,
                                   'start_id': 1,
                                   'label_prefix': 'c',
                                   'label_name': './validation_mat/c_Labels.mat',
                                   'nr_of_labels': None,
                                   'label_files_dir': os.path.join(dataset_directory, 'label_affine_icbm/'),
                                   'old_klein_results_filename': './validation_mat/quicksilver_results/IBSR_results.mat'}

    validation_datasets['MGH'] = {'nr_of_images_in_dataset': 10,
                                  'start_id': 1,
                                  'label_prefix': 'g',
                                  'label_name': './validation_mat/g_Labels.mat',
                                  'nr_of_labels': None,
                                  'label_files_dir': os.path.join(dataset_directory, 'label_affine_icbm/'),
                                  'old_klein_results_filename': './validation_mat/quicksilver_results/MGH_results.mat'}

    validation_datasets['SYNTH'] = {'nr_of_images_in_dataset': -1,
                                  'start_id': 1,
                                  'label_prefix': 'm',
                                  'label_name': None,
                                  'nr_of_labels': 4,
                                  'label_files_dir': os.path.join(dataset_directory, 'label_affine_icbm/'),
                                  'old_klein_results_filename': None}

    if args.dataset is None:
        print('INFO: Assuming that dataset is CUMC; if this is not the case, use the --dataset option')
        validation_dataset_name = 'CUMC'
    else:
        if args.dataset in list(validation_datasets.keys()):
            validation_dataset_name = args.dataset
        else:
            raise ValueError('Dataset needs to be [CUMC|MGH|LPBA|IBSR|SYNTH], but got ' + args.dataset)


    if args.save_overlap_filename is not None:
        save_results = True

        stage_output_dir = create_stage_output_dir(output_directory,stage,args.compute_from_frozen)
        base_overlap_filename = os.path.split(args.save_overlap_filename)[1]

        effective_overlap_filename = os.path.join(stage_output_dir,base_overlap_filename)
        print('Writing overlap results to: ' + effective_overlap_filename )
        
        res_file = open(effective_overlap_filename, 'w')
        res_file.write('source_id,target_id,mean overlap ratio\n')
    else:
        save_results = False

    validation_results = dict()
    validation_results['source_id'] = []
    validation_results['target_id'] = []
    validation_results['mean_target_overlap'] = []
    validation_results['single_results'] = []

    for n in range(nr_of_computed_pairs):
        current_source_image = used_pairs['source_images'][n]
        current_target_image = used_pairs['target_images'][n]

        source_id = extract_id_from_cumc_filename(current_source_image)
        target_id = extract_id_from_cumc_filename(current_target_image)

        current_map_filename,current_source_labelmap_filename,current_target_labelmap_filename,current_warped_labelmap_filename,stage_output_dir = \
            create_filenames(n,output_directory,stage,args.compute_from_frozen)

        if args.do_not_write_target_labelmap:
            current_target_labelmap_filename = None

        if args.do_not_write_source_labelmap:
            current_source_labelmap_filename = None

        print('current_map_filename: ' + current_map_filename)
            
        mean_result,single_results = calculate_image_overlap(validation_datasets[validation_dataset_name], current_map_filename,
                                                             current_source_labelmap_filename,
                                                             current_target_labelmap_filename,
                                                             current_warped_labelmap_filename,
                                                             source_id, target_id,
                                                             use_sym_links=not args.do_not_use_symlinks)

        validation_results['source_id'].append(source_id)
        validation_results['target_id'].append(target_id)
        validation_results['mean_target_overlap'].append(mean_result)
        validation_results['single_results'].append(single_results)

        print('mean label overlap for ' + str(source_id) + ' -> ' + str(target_id) + ': ' + str(mean_result))
        if save_results:
            res_file.write(str(source_id) + ', ' + str(target_id) + ', ' + str(mean_result) + '\n')


    mean_target_overlap_results = np.array(validation_results['mean_target_overlap'])
    print('\nOverall results:')
    print('min = ' + str(mean_target_overlap_results.min()))
    print('max = ' + str(mean_target_overlap_results.max()))
    print('mean = ' + str(mean_target_overlap_results.mean()))
    print('median = ' + str(np.percentile(mean_target_overlap_results,50)))
    print('\n')

    if save_results:

        res_file.write('\nOverall results:')
        res_file.write('\nmin = ' + str(mean_target_overlap_results.min()))
        res_file.write('\nmax = ' + str(mean_target_overlap_results.max()))
        res_file.write('\nmean = ' + str(mean_target_overlap_results.mean()))
        res_file.write('\nmedian = ' + str(np.percentile(mean_target_overlap_results, 50)))
        res_file.write('\n')

        res_file.close()

    if not args.dataset=='SYNTH':
        # synth has direct pairwise pairing, not all with respect to all others
        validation_results['ind'] = get_result_indices(validation_results,
                                                       validation_datasets[validation_dataset_name]['start_id'],
                                                       validation_datasets[validation_dataset_name]['nr_of_images_in_dataset'])

    validation_results_filename = os.path.join(stage_output_dir,'validation_results.pt')
    print('Saving the validation results to: ' + validation_results_filename)
    torch.save(validation_results,validation_results_filename)

    # now do the boxplot
    old_klein_results_filename = validation_datasets[validation_dataset_name]['old_klein_results_filename']

    boxplot_filename = None
    if not args.do_not_print_images:
        boxplot_filename = os.path.join(stage_output_dir,'boxplot_results.pdf')

    overlapping_plot(old_klein_results_filename, validation_results,boxplot_filename, not args.do_not_visualize)
