import torch
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy import stats
import matplotlib

def _get_p_as_text(p,max_text_frac=3,min_text_frac=10):

    p_levels = np.arange(min_text_frac,max_text_frac-1,-1)

    txt = None
    for cand_p in p_levels:
        if p<10.**(-cand_p):
            txt = '<1e-{}'.format(cand_p)
            break

    if txt is None:
        txt = '{:.3f}'.format(p)

    return txt


def print_results_as_text(results,p_significance):

    print('Results:\n')
    print('Significance level alpha={:.5f}'.format(p_significance))
    print('mean, std, perc1, perc5, median, perc95, perc99, p, type, significant\n')

    for k in results:
        c_mean = results[k]['mean']
        c_std = results[k]['std']
        c_perc_1 = results[k]['perc_1']
        c_perc_5 = results[k]['perc_5']
        c_perc_50 = results[k]['median']
        c_perc_95 = results[k]['perc_95']
        c_perc_99 = results[k]['perc_99']

        ttest = results[k]['ttest_wrt_stage2']
        mann_whitney = results[k]['mw_wrt_stage2']
        anderson_darling = results[k]['anderson_darling_wrt_stage2']

        if ttest is None or mann_whitney is None or anderson_darling is None:

            print('{:s}: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {}({}), {}' \
                  .format(k, c_mean, c_std, c_perc_1, c_perc_5, c_perc_50, c_perc_95, c_perc_99,
                          '-', '-', '-'))

        else:

            assert(anderson_darling.significance_level[2]==5.)
            if anderson_darling.statistic>anderson_darling.critical_values[2]:
                stat_res_p = ttest.pvalue/2 # because of one-sided test
                stat_is_greater = ttest.statistic>0 # one-sided test
                stat_res_type = 'T'
            else:
                stat_res_p = mann_whitney.pvalue # is already one-sided test no division by two necessary
                stat_res_type = 'MW'
                stat_is_greater = True

            is_significant = (stat_res_p < p_significance) and stat_is_greater
            if is_significant:
                is_significant_text = 'Y'
            else:
                is_significant_text = 'N'

            stat_res_p_as_text = _get_p_as_text(p=stat_res_p)

            print('{:s}: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {}({}), {}' \
                      .format(k, c_mean, c_std, c_perc_1, c_perc_5, c_perc_50, c_perc_95, c_perc_99,
                              stat_res_p_as_text, stat_res_type, is_significant_text))

def _make_bold_latex_text(val,make_bold=False):

    if make_bold:
        ret = '{{\\bf {:.3f}}}'.format(val)
    else:
        ret = '{:.3f}'.format(val)

    return ret

def _find_largest(results,name):

    current_largest = None
    for k in results:
        if current_largest is None:
            current_largest = results[k][name]
        else:
            if current_largest<results[k][name]:
                current_largest = results[k][name]

    return current_largest


def _find_smallest(results, name):
    current_smallest = None
    for k in results:
        if current_smallest is None:
            current_smallest = results[k][name]
        else:
            if current_smallest > results[k][name]:
                current_smallest = results[k][name]

    return current_smallest

def print_results_as_latex(results,p_significance, training_data=None, testing_data=None):

    if testing_data is None:
        testing_data = ''
    if training_data is None:
        training_data = ''

    print('Results:\n')
    print('Significance level alpha={:.5f}\n'.format(p_significance))

#     \renewcommand{\tabcolsep}{3pt}
#     \begin{table}
#         \begin{center}
#             \begin{tabular}{ | l | c | c | c | c | c | c | c | c |}
#                 \hline
#                 ~\textbf{Method}~ & ~\textbf{mean}~ & ~\textbf{std}~ & ~\textbf{1\ %}~ & ~\textbf{5\ %}~ & ~\textbf{50\ %}~ & ~\textbf{95\ %}~ & ~\textbf
#         {99\ %}~ & ~Better mean?~ \\
#                 \hline
#                 FLIRT & 0.394 & 0.031 & 0.334 & 0.345 & 0.396 & 0.442 & 0.463 & \cmark \\
#                 AIR & 0.423 & 0.030 & 0.362 & 0.377 & 0.421 & 0.483 & 0.492 & \cmark \\
#                 ANIMAL & 0.426 & 0.037 & 0.328 & 0.367 & 0.425 & 0.483 & 0.498 & \cmark \\
#                 ART & 0.503 & 0.031 & 0.446 & 0.452 & 0.506 & 0.556 & 0.563 & \cmark \\
#                 Demons & 0.462 & 0.029 & 0.407 & 0.421 & 0.461 & 0.510 & 0.531 & \cmark \\
#                 FNIRT & 0.463 & 0.036 & 0.381 & 0.410 & 0.463 & 0.519 & 0.537 & \cmark \\
#                 Fluid & 0.462 & 0.031 & 0.401 & 0.410 & 0.462 & 0.516 & 0.532 & \cmark \\
#                 SICLE & 0.419 & 0.044 & 0.300 & 0.330 & 0.424 & 0.475 & 0.504 & \cmark \\
#                 SyN & 0.514 & 0.033 & 0.454 & 0.460 & 0.515 & 0.565 & 0.578 & \cmark \\
#                 SPM5N8 & 0.365 & 0.045 & 0.257 & 0.293 & 0.370 & 0.426 & 0.455 & \cmark \\
#                 SPM5N & 0.420 & 0.031 & 0.361 & 0.376 & 0.418 & 0.471 & 0.494 & \cmark \\
#                 SPM5U & 0.438 & 0.029 & 0.373 & 0.394 & 0.437 & 0.489 & 0.502 & \cmark \\
#                 SPM5D & 0.512 & 0.056 & 0.262 & 0.445 & 0.523 & 0.570 & 0.579 & \cmark \\
#                 \hline\hline
#                 Stage 0 & 0.414 & 0.029 & 0.356 & 0.373 & 0.413 & 0.460 & 0.484 & \cmark \\
#                 Stage 1 & \textbf{0.516} & 0.032 & 0.456 & 0.460 & 0.521 & 0.565 & 0.572 & \cmark \\
#                 Stage 2 & \textbf{0.520} & 0.032 & 0.459 & 0.464 & 0.524 & 0.568 & 0.577 & \xmark \\
#                 \hline
#             \end{tabular}
#         \end{center}
#     \caption{Statistics for mean(over all labeled brain structures while disregarding the background) target overlap measures across the CUMC12 registration pairs for different methods from ~\cite{klein2009}.Our approach results in the highest target overlap ratios.There is a significant improvement between stages 0 and 1 and a small improvement between stages 1 and 2 (\ie, for the localized multi-Gaussian weights).Fig.~\ref{fig:boxplot_overlap_cumc3d_short} shows this
# data presented in the form of a boxplot.}
#     \label{tab:3d_cumc_results}
#     \end{table}

    print('\\renewcommand{\\tabcolsep}{3pt}')
    print('\\begin{table}')

    print('\t\\begin{center}')

    print('\t\t\\begin{tabular}{| l | c | c | c | c | c | c | c | c | c | c |}')
    print('\t\t\t\\hline')
    print('\t\t\t~\\textbf{Method}~ & ~\\textbf{mean}~ & ~\\textbf{std}~ & ~\\textbf{1\%}~ & ~\\textbf{5\%}~ & ~\\textbf{50\%}~ & ~\\textbf{95\%}~ & ~\\textbf{99\%}~ & ~p~ & ~type~ & sig?~ \\\\')
    print('\t\t\t\\hline')

    # here comes the actual data

    c_mean_largest = _find_largest(results,'mean')
    c_std_smallest = _find_smallest(results,'std')
    c_perc_1_largest = _find_largest(results,'perc_1')
    c_perc_5_largest = _find_largest(results,'perc_5')
    c_perc_50_largest = _find_largest(results,'median')
    c_perc_95_largest = _find_largest(results,'perc_95')
    c_perc_99_largest = _find_largest(results,'perc_99')

    for k in results:

        if k=='stage 0':
            print('\t\t\t\\hline')
            print('\t\t\t\\hline')

        c_mean = results[k]['mean']
        c_std = results[k]['std']
        c_perc_1 = results[k]['perc_1']
        c_perc_5 = results[k]['perc_5']
        c_perc_50 = results[k]['median']
        c_perc_95 = results[k]['perc_95']
        c_perc_99 = results[k]['perc_99']

        ttest = results[k]['ttest_wrt_stage2']
        mann_whitney = results[k]['mw_wrt_stage2']
        anderson_darling = results[k]['anderson_darling_wrt_stage2']

        c_mean_txt = _make_bold_latex_text(c_mean, c_mean >= c_mean_largest)
        c_std_txt = _make_bold_latex_text(c_std, c_std <= c_std_smallest)
        c_perc_1_txt = _make_bold_latex_text(c_perc_1, c_perc_1 >= c_perc_1_largest)
        c_perc_5_txt = _make_bold_latex_text(c_perc_5, c_perc_5 >= c_perc_5_largest)
        c_perc_50_txt = _make_bold_latex_text(c_perc_50, c_perc_50 >= c_perc_50_largest)
        c_perc_95_txt = _make_bold_latex_text(c_perc_95, c_perc_95 >= c_perc_95_largest)
        c_perc_99_txt = _make_bold_latex_text(c_perc_99, c_perc_99 >= c_perc_99_largest)

        if ttest is None or mann_whitney is None or anderson_darling is None:

            print('\t\t\t {:s} & {:s} & {:s} & {:s} & {:s} & {:s} & {:s} & {:s} & {} & ({}) & {} \\\\' \
                  .format(k, c_mean_txt, c_std_txt, c_perc_1_txt, c_perc_5_txt, c_perc_50_txt, c_perc_95_txt, c_perc_99_txt,
                          '-', '-', '-'))

        else:

            assert (anderson_darling.significance_level[2] == 5.)
            if anderson_darling.statistic > anderson_darling.critical_values[2]:
                stat_res_p = ttest.pvalue / 2  # because of one-sided test
                stat_is_greater = ttest.statistic > 0  # one-sided test
                stat_res_type = 'T'
            else:
                stat_res_p = mann_whitney.pvalue  # is already one-sided test no division by two necessary
                stat_res_type = 'MW'
                stat_is_greater = True

            is_significant = (stat_res_p < p_significance) and stat_is_greater
            if is_significant:
                is_significant_text = '\\cmark'
            else:
                is_significant_text = '\\xmark'

            stat_res_p_as_text = '$' + _get_p_as_text(p=stat_res_p) + '$'

            print('\t\t\t {:s} & {:s} & {:s} & {:s} & {:s} & {:s} & {:s} & {:s} & {} & ({}) & {} \\\\' \
                  .format(k, c_mean_txt, c_std_txt, c_perc_1_txt, c_perc_5_txt, c_perc_50_txt, c_perc_95_txt, c_perc_99_txt,
                          stat_res_p_as_text, stat_res_type, is_significant_text))

    print('\t\t\t\\hline')
    print('\t\t\\end{tabular}')

    print('\t\\end{center}')
    print('\t\caption{{Statistics for mean(over all labeled brain structures while disregarding the background) target overlap measures across the {} registration pairs for different methods from ~\cite{{klein2009}}. The method was trained on the {} dataset. Fig.~\\ref{{fig:boxplot_overlap_train_{}_test_{}}} shows this data presented in the form of a boxplot.}}'.format(testing_data,training_data,training_data,testing_data))

    print('\end{table}')
    print('\n')


def print_results_and_compute_statistics(compound_names,compound_results, training_data=None, testing_data=None):

    # do statistical comparisons with respect to the stage 2 results
    if compound_names[-1]!='stage 2':
        raise ValueError('Assuming that the last stage is stage 2 for comparisons')

    results = dict()

    stage2_vals = compound_results[:,-1]

    nr_of_comparisons = len(compound_names)-1
    last_n = nr_of_comparisons
    p_significance_bonferroni = 0.05/nr_of_comparisons

    for n in range(len(compound_names)):
        current_vals = compound_results[:,n]
        current_results = dict()
        current_results['median'] = np.median(compound_results[:, n])
        current_results['mean'] = np.mean(compound_results[:, n])
        current_results['std'] = np.std(compound_results[:, n])
        current_results['perc_1'] = np.percentile(compound_results[:, n], 1)
        current_results['perc_5'] = np.percentile(compound_results[:, n], 5)
        current_results['perc_95'] = np.percentile(compound_results[:, n], 95)
        current_results['perc_99'] = np.percentile(compound_results[:, n], 99)

        if n!=last_n:
            current_results['ttest_wrt_stage2'] = stats.ttest_rel(a=stage2_vals,b=current_vals)
            current_results['mw_wrt_stage2'] = stats.mannwhitneyu(x=stage2_vals, y=current_vals, use_continuity=True, alternative='greater')
            current_results['anderson_darling_wrt_stage2'] = stats.anderson(stage2_vals-current_vals)
        else: # is stage 2
            current_results['ttest_wrt_stage2'] = None
            current_results['mw_wrt_stage2'] = None
            current_results['anderson_darling_wrt_stage2'] = None

        results[compound_names[n]]=current_results

    print_results_as_text(results=results,p_significance=p_significance_bonferroni)
    print_results_as_latex(results=results,p_significance=p_significance_bonferroni, training_data=training_data, testing_data=testing_data)

def overlapping_plot(old_results_filename, new_results, new_results_names, boxplot_filename, visualize=True, training_data=None, testing_data=None):
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
            current_name = str(item[0])[2:-2]
            if current_name=='SPM5_Normalize_direct_8mm':
                current_name='SPM5N8'
            elif current_name=='SPM5_Normalize':
                current_name='SPM5N'
            elif current_name=='SPM5_UnifiedSegment':
                current_name='SPM5U'
            elif current_name=='SPM5_DARTEL_pairs':
                current_name='SPM5D'
            compound_names.append(current_name)

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
    print_results_and_compute_statistics(compound_names=compound_names,compound_results=compound_results, training_data=training_data, testing_data=testing_data)

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

def get_validation_results_filenames_from_base_directory(base_directory,desired_stages=[0,1,2]):
    validation_results = []
    for s in desired_stages:
        validation_results.append(os.path.join(base_directory,'model_results_stage_{}/validation_results.pt'.format(s)))

    return validation_results

def get_validation_datasets():

    dataset_directory = ''
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

    return validation_datasets

#validation_data_dir = '/Users/mn/cumc3d_short_boxplots'
#validation_data_dir = '/Users/mn/sim_results/cumc3d_val'


validation_datasets = get_validation_datasets()

datasets_to_test = ['mgh10','cumc12','ibsr18']
corresponding_validation_datasets = ['MGH','CUMC','IBSR18']

tv_penalty = 0.1
omt_penalty = 50.0

for train_dataset in datasets_to_test:
    for test_dataset,validation_dataset_name in zip(datasets_to_test,corresponding_validation_datasets):
        validation_data_dir = '/Users/mn/sim_results/pf-out_testing_train_{}_test_{}_3d_sqrt_w_K_sqrt'.format(train_dataset,test_dataset)
        if os.path.exists(validation_data_dir):
            print('\nPlotting for train={}/test={}'.format(train_dataset,test_dataset))
        else:
            print('\n\nData for train={}/test={} not found. IGNORING\n\n'.format(train_dataset,test_dataset))
            continue

        boxplot_filename = 'train_{}_test_{}_3d_boxplot.pdf'.format(train_dataset,test_dataset)

        base_directory = os.path.join(validation_data_dir,'out_testing_total_variation_weight_penalty_{:.6f}_omt_weight_penalty_{:.6f}'.format(tv_penalty,omt_penalty))
        validation_results_filenames = get_validation_results_filenames_from_base_directory(base_directory=base_directory)

        # validation_results_filenames = ['validation_results_stage_0.pt',
        #                                 'validation_results_stage_1.pt',
        #                                 'validation_results_stage_2.pt']

        validation_results_names = ['stage 0', 'stage 1', 'stage 2']
        validation_results = []
        validation_results.append(torch.load(os.path.join(validation_data_dir,validation_results_filenames[0])))
        validation_results.append(torch.load(os.path.join(validation_data_dir,validation_results_filenames[1])))
        validation_results.append(torch.load(os.path.join(validation_data_dir,validation_results_filenames[2])))

        # now do the boxplot
        old_klein_results_filename = validation_datasets[validation_dataset_name]['old_klein_results_filename']

        overlapping_plot(old_klein_results_filename, validation_results, validation_results_names, boxplot_filename,
                         visualize=True,testing_data=test_dataset,training_data=train_dataset)
