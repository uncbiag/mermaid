import set_pyreg_paths

import os

import torch
from torch.autograd import Variable

import pyreg.utils as utils
import pyreg.fileio as fio

import scipy.io as sio
import numpy as np
import itk

import nrrd
import torch

import matplotlib.pyplot as plt
import matplotlib

def new_warp_image_nn(label_map, phi, spacing):

    lm_t = Variable( torch.from_numpy(label_map),requires_grad=False)
    lm_t = lm_t.view([1,1]+list(lm_t.size()))

    return utils.get_warped_label_map(lm_t, phi, spacing, sched='nn')

def warp_image_nn(moving, phi):
    """
    Warp labels according to deformation field
    :param moving: Moving image in numpy array format
    :param phi: Deformation field in numpy array format
    :return: Warped labels according to deformation field
    """

    # get image demensions
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

def calculate_image_overlap(dataset_name, dataset_dir, phi_path, source_labelmap_path, target_labelmap_path,
                            warped_labelmap_path,moving_id, target_id):
    """
    Calculate the overlapping rate of a specified case
    :param dataset_name: 'LPBA', 'IBSR', 'CUMC' or 'MGH'
    :param dataset_dir: path to the label datasets
    :param phi_path: deformation field path
    :param moving_id: moving image id
    :param target_id: target image id
    :return:
    """

    if dataset_name == 'LPBA':
        label_name = './validation_mat/l_Labels.mat'
        label_files_dir = os.path.join(dataset_dir, 'LPBA_label_affine/')
        dataset_size = 40
        label_prefix = 's'
    elif dataset_name == 'IBSR':
        label_name = './validation_mat/c_Labels.mat'
        label_files_dir = os.path.join(dataset_dir, 'IBSR_label_affine/')
        dataset_size = 18
        label_prefix = 'c'
    elif dataset_name == 'CUMC':
        label_name = './validation_mat/m_Labels.mat'
        label_files_dir = os.path.join(dataset_dir, 'label_affine_icbm/')
        dataset_size = 12
        label_prefix = 'm'
    elif dataset_name == 'MGH':
        label_name = './validation_mat/g_Labels.mat'
        label_files_dir = os.path.join(dataset_dir, 'MGH_label_affine/')
        dataset_size = 10
        label_prefix = 'g'
    else:
        raise TypeError("Unknown Dataset Name: Dataset name must be 'LPBA', 'IBSR', 'CUMC' or 'MGH'")

    Labels = sio.loadmat(label_name)
    result = np.zeros((len(Labels['Labels'])))

    label_images = [None] * dataset_size

    # todo: not sure why these are floats, but okay for now

    im_io = fio.ImageIO()
    label_from_id = moving_id-1
    label_to_id = target_id-1

    label_from_filename = label_files_dir + label_prefix + str(label_from_id + 1) + '.nii'
    label_from, hdr, _, _ = im_io.read(label_from_filename, silent_mode=True)

    label_to_filename = label_files_dir + label_prefix + str(label_to_id + 1) + '.nii'
    label_to, hdr, _, _ = im_io.read(label_to_filename, silent_mode=True)

    map_io = fio.MapIO()

    phi,_,_,_ = map_io.read_from_validation_map_format(phi_path)
    phi = phi.squeeze()
    warp_result = warp_image_nn(label_from, phi)

    im_io.write(warped_labelmap_path,warp_result,hdr)

    if source_labelmap_path is not None:
        im_io.write(source_labelmap_path,label_from,hdr)

    if target_labelmap_path is not None:
        im_io.write(target_labelmap_path,label_to,hdr)


    for label_idx in range(len(Labels['Labels'])):

        current_id = Labels['Labels'][label_idx][0]

        target_vol = float( (label_to==current_id).sum() )

        if target_vol==0 or current_id==0:
            result[label_idx] = np.nan
        else:
            intersection = float((np.logical_and(label_to == current_id, warp_result == current_id)).sum())
            result[label_idx] = intersection/target_vol

    single_result = result
    single_result = single_result[~np.isnan(single_result)]
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
    old_results = sio.loadmat(old_results_filename)

    # combine old names with proposed method
    compound_names = []
    for item in old_results['direc_name']:
        compound_names.append(str(item[0])[3:-2])

    compound_names.append('Proposed')

    # new results may only have a subset of the results

    old_results_selected = old_results['results'][new_results['ind'],:] # select the desired rows

    # combine data
    compound_results = np.concatenate((old_results_selected, np.array(new_results['mean_target_overlap']).reshape(-1, 1)), axis=1)

    # create a figure instance
    fig = plt.figure(1, figsize=(8, 6))

    # create an axes instance
    ax = fig.add_subplot(111)

    # set axis tick
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.yaxis.set_tick_params(left='on', direction='in', width=1)
    ax.yaxis.set_tick_params(right='on', direction='in', width=1)
    ax.xaxis.set_tick_params(top='off', direction='in', width=1)
    ax.xaxis.set_tick_params(bottom='off', direction='in', width=1)

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
    plt.ylim([0, 0.8])

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
        plt.savefig(boxplot_filename, dpi=1000, bbox_inches='tight')

    # show figure
    if visualize:
        plt.show()


def extract_id_from_cumc_filename(filename):
    r = os.path.split(filename)
    # these files have the format m1.nii, m12.nii, ...
    nr = int(r[1][1:-4])
    return nr

def create_filenames(id,output_dir,stage,compute_from_frozen=False):

    if not stage in [0,1,2]:
        raise ValueError('stages need to be {0,1,2}')

    if compute_from_frozen:
        stage_output_dir = os.path.join(os.path.normpath(output_dir), 'model_results_frozen_stage_' + str(stage))
    else:
        stage_output_dir = os.path.join(os.path.normpath(output_dir), 'model_results_stage_' + str(stage))

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

    #todo: THis is currently specific for the CUMC data; make this more generic later

    import argparse

    parser = argparse.ArgumentParser(description='Computes target overlap for registration results')

    parser.add_argument('--output_directory', required=True,
                        help='Main directory where the output was stored (now this will be the input directory)')
    parser.add_argument('--stage_nr', required=True, type=int,
                        help='stage number for which the computations should be performed {0,1,2}; shifted by one')
    parser.add_argument('--dataset_directory', required=True,
                        help='Main directory where dataset is stored; this directory should contain the subdirectory label_affine_icbm')

    parser.add_argument('--compute_from_frozen', action='store_true', help='computes the results from optimization results with frozen parameters')

    parser.add_argument('--do_not_write_target_labelmap', action='store_true', help='otherwise also writes the target labelmap again for easy visualization')
    parser.add_argument('--do_not_write_source_labelmap', action='store_true', help='otherwise also writes the source labelmap again for easy visualization')


    parser.add_argument('--do_not_visualize', action='store_true', help='visualizes the output otherwise')
    parser.add_argument('--do_not_print_images', action='store_true', help='prints the results otherwise')

    parser.add_argument('--save_overlap_filename', required=False, default=None,help='If specified write the result in this output file')

    args = parser.parse_args()

    #output_directory = './sample_3d_res/test_out'
    #stage = 2
    #dataset_directory = '/Users/mn/data/testdata/CUMC12'

    output_directory = args.output_directory
    dataset_directory = args.dataset_directory
    stage = args.stage_nr

    used_pairs = torch.load(os.path.join(output_directory, 'used_image_pairs.pt'))
    nr_of_computed_pairs = len(used_pairs['source_ids'])

    if args.save_overlap_filename is not None:
        save_results = True
        res_file = open(args.save_overlap_filename, 'w')
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

        mean_result,single_results = calculate_image_overlap('CUMC', dataset_directory, current_map_filename,
                                                             current_source_labelmap_filename, current_target_labelmap_filename, current_warped_labelmap_filename, source_id, target_id)

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
        res_file.close()

    # todo: specific settings for CUMC (has 12 images; starts at image 1); make this more generic later
    validation_results['ind'] = get_result_indices(validation_results,start_id=1,nr_of_images_in_dataset=12)

    validation_results_filename = os.path.join(stage_output_dir,'validation_results.pt')
    print('Saving the validation results to: ' + validation_results_filename)
    torch.save(validation_results,validation_results_filename)

    # now do the boxplot
    if not args.do_not_visualize:
        old_results_filename = './validation_mat/quicksilver_results/CUMC_results.mat'

        boxplot_filename = None
        if not args.do_not_print_images:
            boxplot_filename = os.path.join(stage_output_dir,'boxplot_results.pdf')

        overlapping_plot(old_results_filename, validation_results,boxplot_filename, not args.do_not_visualize)