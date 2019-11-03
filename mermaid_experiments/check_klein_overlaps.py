import numpy as np
import scipy.io as sio
import os

dataset_directory = '.'

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

old_klein_results_filename = validation_datasets[validation_dataset_name]['old_klein_results_filename']

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
    old_results_selected = old_results['results'][new_results['ind'], :]  # select the desired rows

    # combine data
    compound_results = np.concatenate(
        (old_results_selected, np.array(new_results['mean_target_overlap']).reshape(-1, 1)), axis=1)
else:
    compound_results = np.array(new_results['mean_target_overlap']).reshape(-1, 1)
