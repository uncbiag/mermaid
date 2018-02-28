import set_pyreg_paths

import os

import scipy.io as sio
import numpy as np
import itk

import nrrd
import torch

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

def calculate_image_overlap(dataset_name, dataset_dir, phi_path, moving_id, target_id):
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

    # todo: there should be no need to load all the label files here
    for i in range(dataset_size):
        label_images[i] = itk.GetArrayFromImage(
            itk.imread(label_files_dir + label_prefix + str(i + 1) + '.nii')).squeeze()

    label_from = label_images[moving_id - 1]
    label_to = label_images[target_id - 1]
    phi,hdr = nrrd.read(phi_path)
    phi = phi.squeeze()
    warp_result = warp_image_nn(label_from, phi)
    for label_idx in range(len(Labels['Labels'])):

        current_id = Labels['Labels'][label_idx]

        intersection = float( (np.logical_and(label_to==current_id, warp_result==current_id)).sum() )
        target_vol = float( (label_to==current_id).sum() )

        if target_vol==0:
            result[label_idx] = np.nan
        else:
            result[label_idx] = intersection/target_vol

    single_result = result
    single_result = single_result[~np.isnan(single_result)]
    result_mean = np.mean(single_result)

    return result_mean,single_result

    #print('overlapping rate of without unNaN value')
    #print(single_result)
    #print('Averaged overlapping rate')
    #print(result_mean)

def extract_id_from_cumc_filename(filename):
    r = os.path.split(filename)
    # these files have the format m1.nii, m12.nii, ...
    nr = int(r[1][1:-4])
    return nr

def create_map_filename(id,output_dir,stage):

    if not stage in [0,1,2]:
        raise ValueError('stages need to be {0,1,2} mapping to stage 1, 2, and 3 respectively')

    stage_output_dir = os.path.normpath(output_dir) + '_model_results_stage_' + str(stage)

    map_filename = os.path.join(stage_output_dir,'map_validation_format_{:05d}.nrrd'.format(id))
    return map_filename

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Computes target overlap for registration results')

    parser.add_argument('--output_directory', required=True,
                        help='Main directory where the output was stored (now this will be the input directory)')
    parser.add_argument('--stage_nr', required=True, type=int,
                        help='stage number for which the computations should be performed {0,1,2}; shifted by one')
    parser.add_argument('--dataset_directory', required=True,
                        help='Main directory where dataset is stored; this directory should contain the subdirectory label_affine_icbm')

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

    for n in range(nr_of_computed_pairs):
        current_source_image = used_pairs['source_images'][n]
        current_target_image = used_pairs['target_images'][n]

        source_id = extract_id_from_cumc_filename(current_source_image)
        target_id = extract_id_from_cumc_filename(current_target_image)

        current_map_filename = create_map_filename(n,output_directory,stage)

        mean_result,single_results = calculate_image_overlap('CUMC', dataset_directory, current_map_filename, source_id, target_id)

        print('mean label overlap for ' + str(source_id) + ' -> ' + str(target_id) + ': ' + str(mean_result))
        if save_results:
            res_file.write(str(source_id) + ', ' + str(target_id) + ', ' + str(mean_result) + '\n')

    if save_results:
        res_file.close()