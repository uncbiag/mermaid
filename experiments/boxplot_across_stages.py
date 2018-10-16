import torch
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib

def overlapping_plot(old_results_filename, new_results, new_results_names, boxplot_filename, visualize=True):
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

    # remove the last two
    nr_of_methods_to_remove = 3

    compound_names = compound_names[0:-nr_of_methods_to_remove]

    for n in new_results_names:
        compound_names.append(n)

    # new results may only have a subset of the results
    if old_results is not None:
        old_results_selected = old_results['results'][new_results[0]['ind'],0:(-nr_of_methods_to_remove)] # select the desired rows

        # combine data
        compound_results = old_results_selected
        for current_new_result in new_results:
            compound_results = np.concatenate((compound_results, np.array(current_new_result['mean_target_overlap']).reshape(-1, 1)), axis=1)
    else:
        compound_results = new_results[0]
        for current_new_result in new_results[1:]:
            compound_results = np.concatenate((compound_results, np.array(current_new_result['mean_target_overlap']).reshape(-1, 1)), axis=1)

    # print out the results

    print('Results:\n')
    print('mean, std, perc1, perc5, median, perc95, perc99\n')
    for n in range(len(compound_names)):
        c_median = np.median(compound_results[:,n])
        c_mean = np.mean(compound_results[:,n])
        c_std = np.std(compound_results[:,n])
        c_perc_1 = np.percentile(compound_results[:,n],1)
        c_perc_5 = np.percentile(compound_results[:,n],5)
        c_perc_95 = np.percentile(compound_results[:,n],95)
        c_perc_99 = np.percentile(compound_results[:,n],99)

        print('{:s}: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(compound_names[n],c_mean,c_std,c_perc_1,c_perc_5,c_median,c_perc_95,c_perc_99))

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
    plt.ylim([0.2, 0.6])

    # add two lines to represent the lower quartile and upper quartile
    lower_quartile = np.percentile(compound_results[:, -1], 25)
    upper_quartile = np.percentile(compound_results[:, -1], 75)
    ax.axhline(lower_quartile, ls='-', color='r', linewidth=1)
    ax.axhline(upper_quartile, ls='-', color='r', linewidth=1)

    # add a dashed line representing the median
    med_quartile = np.percentile(compound_results[:, -1], 50)
    ax.axhline(med_quartile, ls=':', color='r', linewidth=1)

    # set the target box to red color

    nr_of_new_boxes = len(new_results_names)

    for n in range(nr_of_new_boxes):
        bp['boxes'][-(1+n)].set(color='red')
        bp['boxes'][-(1+n)].set(facecolor='red')
        bp['whiskers'][-(1+2*n)].set(color='red')
        bp['whiskers'][-(2+2*n)].set(color='red')
        bp['fliers'][-(1+n)].set(color='red', markeredgecolor='red')

    # save figure
    if boxplot_filename is not None:
        print('Saving boxplot to : ' + boxplot_filename )
        plt.savefig(boxplot_filename, dpi=1000, bbox_inches='tight')

    # show figure
    if visualize:
        plt.show()

#validation_data_dir = '/Users/mn/cumc3d_short_boxplots'
validation_data_dir = '/Users/mn/sim_results/cumc3d_val'

#dataset_directory = args.dataset_directory
#dataset_directory = '/Users/mn/PycharmProjects/mermaid/experiments/CUMC12_2d'
dataset_directory = ''
validation_dataset_name = 'CUMC'
boxplot_filename = 'cumc3d_short_boxplot.pdf'

validation_results_filenames = ['validation_results_stage_0.pt',
                                'validation_results_stage_1.pt',
                                'validation_results_stage_2.pt']

validation_results_names = ['stage 0', 'stage 1', 'stage 2']
validation_results = []
validation_results.append(torch.load(os.path.join(validation_data_dir,validation_results_filenames[0])))
validation_results.append(torch.load(os.path.join(validation_data_dir,validation_results_filenames[1])))
validation_results.append(torch.load(os.path.join(validation_data_dir,validation_results_filenames[2])))

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

# now do the boxplot
old_klein_results_filename = validation_datasets[validation_dataset_name]['old_klein_results_filename']

overlapping_plot(old_klein_results_filename, validation_results, validation_results_names, boxplot_filename, visualize=True)
