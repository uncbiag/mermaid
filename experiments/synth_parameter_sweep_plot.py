import experiment_utils as eu
import os
import glob
import torch
import copy
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import numpy as np

def get_values_and_names(current_stats,name_prefix='',desired_stat='mean'):
    compound_results = []
    compound_names = []

    for k in current_stats['local']:
        compound_results.append(current_stats['local'][k][desired_stat])
        compound_names.append('{:s}_{:s}'.format(name_prefix,str(k)))

    compound_results.append(current_stats['global'][desired_stat])
    compound_names.append('{:s}_global'.format(name_prefix))

    return compound_results,compound_names

def reorder_values(cr,cn,nr_of_measures):
    cr_s = []
    cn_s = []

    for n in range(nr_of_measures):
        cr_s += cr[n::nr_of_measures]
        cn_s += cn[n::nr_of_measures]

    return cr_s,cn_s

def spatially_normalize_stats(ms,spacing):
    if type(ms)==dict:
        for k in ms:
            spatially_normalize_stats(ms[k],spacing)
    else:
        ms/=spacing

def spatially_normalize_dj_stats(djs,spacing):
    if type(djs)==dict:
        for k in djs:
            spatially_normalize_stats(djs[k],spacing)
    else:
        djs/=(spacing**2) # we assume that there is isotropic spacing for this synthetic result

def merge_dicts(all_stats,all_names,current_stats,desired_stat,name_prefix):

    for k in current_stats['local']:
        if k in all_stats:
            all_stats[k].append(current_stats['local'][k][desired_stat])
        else:
            all_stats[k] = [current_stats['local'][k][desired_stat]]

        if k in all_names:
            all_names[k].append(name_prefix)
        else:
            all_names[k] = [name_prefix]

    if 'global' in all_stats:
        all_stats['global'].append(current_stats['global'][desired_stat])
    else:
        all_stats['global'] = [current_stats['global'][desired_stat]]

    if 'global' in all_names:
        all_names['global'].append(name_prefix)
    else:
        all_names['global'] = [name_prefix]

    return all_stats,all_names

def outliers_suppressed(text,showfliers=True):
    if showfliers==False:
        return text + ' (outliers suppressed)'
    else:
        return text

def plot_results(all_stats,all_names,nr_of_measures,showfliers,normalize_by_spacing,ylabel,output_prefix,title_prefix,
                 suppress_pattern=None,suppress_pattern_keep_first_as=None,
                 replace_pattern_from=None,
                 replace_pattern_to=None,
                 custom_ranges_raw=None,
                 custom_ranges_norm=None,
                 print_title=True,
                 show_labels = True,
                 fix_aspect = None,
                 print_output_directory=None):

    print('Results for prefix {:s}'.format(output_prefix))

    # direct visualization
    for k in all_stats:
        plt.clf()
        rs, rn = reorder_values(all_stats[k], all_names[k], nr_of_measures=nr_of_measures)
        eu.plot_boxplot(rs, rn, semilogy=False, showfliers=showfliers,
                        suppress_pattern=suppress_pattern,suppress_pattern_keep_first_as=suppress_pattern_keep_first_as,
                        replace_pattern_from=replace_pattern_from,
                        replace_pattern_to=replace_pattern_to,
                        show_labels=show_labels,fix_aspect=fix_aspect)
        if print_title:
            plt.title(outliers_suppressed('Raw: ' + title_prefix + ' ' + str(k), showfliers=showfliers))
        if normalize_by_spacing:
            plt.ylabel(ylabel + ' [pixel]')
        else:
            plt.ylabel(ylabel)

        if custom_ranges_raw is not None:
            plt.ylim(custom_ranges_raw[k])

        if print_output_directory is not None:
            plt.savefig(os.path.join(print_output_directory, 'raw_stat_{:s}_{:s}.pdf'.format(output_prefix,str(k))))
        else:
            plt.show()

    # median normalized (with respect to stage 0 -- first entry, check this; this should be more or less constant as it does not depend on the OMT or TV penalty
    all_stats_mn = copy.deepcopy(all_stats)
    for k in all_stats_mn:
        c_stats = all_stats_mn[k]
        print('Normalizing based on {:s}'.format(all_names[k][0]))
        median_for_normalization = np.percentile(c_stats[0], 50)
        for s in c_stats:
            s /= median_for_normalization

    # now plot it
    for k in all_stats:
        plt.clf()
        rs, rn = reorder_values(all_stats_mn[k], all_names[k], nr_of_measures=nr_of_measures)
        eu.plot_boxplot(rs, rn, semilogy=False, showfliers=showfliers,
                        suppress_pattern=suppress_pattern,
                        suppress_pattern_keep_first_as=suppress_pattern_keep_first_as,
                        replace_pattern_from=replace_pattern_from,
                        replace_pattern_to=replace_pattern_to,
                        show_labels = show_labels,
                        fix_aspect = fix_aspect)

        if print_title:
            plt.title(outliers_suppressed('Median normalized: ' + title_prefix + ' ' + str(k), showfliers=showfliers))

        plt.ylabel(ylabel + ' [unitless; normalized]')

        if custom_ranges_norm is not None:
            plt.ylim(custom_ranges_norm[k])

        if print_output_directory is not None:
            plt.savefig(os.path.join(print_output_directory, 'median_normalized_stat_{:s}_{:s}.pdf'.format(output_prefix,str(k))))
        else:
            plt.show()

    for k in all_stats:
        print('Results for ' + str(k) + ':')
        for s, n in zip(all_stats[k], all_names[k]):
            current_median_value = np.percentile(s, 50)
            print('{:s}: median={:f}'.format(n, current_median_value))

#datapath = '/Users/mn/sim_results/pf_out_testing_paper_experiment_wo_momentum_sqrt_w_K_sqrt_w_200_wo_noise_sc'
#stages = [0,1,2]

#datapath = '/Users/mn/sim_results/pf_out_testing_paper_experiment_wo_momentum_sqrt_w_K_sqrt_w_200_wo_noise_sc-skip-stage-1'
#datapath = '/Users/mn/sim_results/pf_out_paper_experiment_wo_momentum_sqrt_w_K_sqrt_w_300_wo_noise_sc'
#datapath = '/Users/mn/sim_results/pf_out_testing_paper_experiment_wo_momentum_sqrt_w_K_sqrt_w_200_wo_noise_sc-skip-stage-1'
#datapath = '/Users/mn/sim_results/pf_out_testing_paper_experiment_wo_momentum_sqrt_w_K_sqrt_w_200_wo_noise_sc_only_0.1_and_0.25'
#datapath = '/Users/mn/sim_results/pf_out_testing_paper_experiment_wo_momentum_sqrt_w_K_sqrt_w_200_wo_noise_sc'
datapath = '/Users/mn/sim_results/pf-out_testing_paper_experiment_wo_momentum_sqrt_w_K_sqrt_w_200_wo_noise_sc-skip-stage-1'

#stages = [0,1,2]
stages = [0,2]

squeezed_aspect_ratio = 0.35
#stages = [0,2]
nr_of_measures = len(stages)

if 1 in stages:
    suppress_pattern_keep_first_as = 's0'
    replace_pattern_from = None
    replace_pattern_to = None
else:
    suppress_pattern_keep_first_as = 'global'
    replace_pattern_from = '_s2'
    replace_pattern_to = '_l'

prefix = 'out_testing'
#prefix = 'out_training'

use_all_directories = False

if not use_all_directories:
    desired_tv = [0.01,0.1,0.25]
    desired_omt = [5,15,25,50,75,100]
else:
    desired_tv = None
    desired_omt = None

#datapath = './experimental_results_synth_2d'
#prefix = 'out_test'
#use_all_directories = True

#desired_stat = 'mean'
desired_stat = 'median'
use_custom_boxplot_ranges = False
showfliers = False
normalize_by_spacing = True
spacing = 1./(128.-1.)
print_output_directory = 'pdf_sweep-' + os.path.split(datapath)[1]
print_output_directory_no_title = 'pdf_sweep-no-title-' + os.path.split(datapath)[1]
print_output_directory_no_title_squeezed = 'pdf_sweep-squeezed-no-title-' + os.path.split(datapath)[1]
print_output_directory_no_title_no_label = 'pdf_sweep-no-title-no-label-' + os.path.split(datapath)[1]
print_output_directory_no_title_no_label_squeezed = 'pdf_sweep-squeezed-no-title-no-label-' + os.path.split(datapath)[1]

if print_output_directory is not None:
    print('Saving figure output in directory: {:s}'.format(print_output_directory))
    if not os.path.exists(print_output_directory):
        os.mkdir(print_output_directory)

if print_output_directory_no_title is not None:
    print('Saving figure output WITHOUT titles in directory: {:s}'.format(print_output_directory_no_title))
    if not os.path.exists(print_output_directory_no_title):
        os.mkdir(print_output_directory_no_title)

if print_output_directory_no_title_squeezed is not None:
    print('Saving squeezed figure output WITHOUT titles in directory: {:s}'.format(print_output_directory_no_title_squeezed))
    if not os.path.exists(print_output_directory_no_title_squeezed):
        os.mkdir(print_output_directory_no_title_squeezed)

if print_output_directory_no_title_no_label is not None:
    print('Saving figure output WITHOUT titles and labels in directory: {:s}'.format(print_output_directory_no_title_no_label))
    if not os.path.exists(print_output_directory_no_title_no_label):
        os.mkdir(print_output_directory_no_title_no_label)

if print_output_directory_no_title_no_label_squeezed is not None:
    print('Saving squeezed figure output WITHOUT titles and labels in directory: {:s}'.format(print_output_directory_no_title_no_label_squeezed))
    if not os.path.exists(print_output_directory_no_title_no_label_squeezed):
        os.mkdir(print_output_directory_no_title_no_label_squeezed)

if use_all_directories:
    desired_directories = glob.glob(os.path.join(datapath,'{:s}*'.format(prefix)))
else:
    desired_directories = []
    for c_tv in desired_tv:
        for c_omt in desired_omt:
            current_dir_name = os.path.join(datapath,'{}_total_variation_weight_penalty_{:f}_omt_weight_penalty_{:f}'.format(prefix,c_tv,c_omt))
            print('Adding directory: {}'.format(current_dir_name))
            desired_directories.append(current_dir_name)

    # desired_directories = ['/Users/mn/data/stat_results/out_test_total_variation_weight_penalty_0.010000_omt_weight_penalty_1.000000',
    #                        '/Users/mn/data/stat_results/out_test_total_variation_weight_penalty_0.010000_omt_weight_penalty_10.000000',
    #                        #'/Users/mn/data/stat_results/out_test_total_variation_weight_penalty_0.010000_omt_weight_penalty_100.000000',
    #                        '/Users/mn/data/stat_results/out_test_total_variation_weight_penalty_0.100000_omt_weight_penalty_1.000000',
    #                        '/Users/mn/data/stat_results/out_test_total_variation_weight_penalty_0.100000_omt_weight_penalty_10.000000',
    #                        #'/Users/mn/data/stat_results/out_test_total_variation_weight_penalty_0.100000_omt_weight_penalty_100.000000'
    #                        ]

split_keys = ['total_variation_weight_penalty','omt_weight_penalty']
abbrv_keys = ['t','o']

cr_all = []
cn_all = []

all_stats_map = dict()
all_names_map = dict()

all_stats_dj = dict()
all_names_dj = dict()

all_overlaps = []
all_overlap_names = []

for d in desired_directories:

    cr_all_stage = []
    cn_all_stage = []

    for s in stages:

        current_es_file_name = os.path.join(d,'model_results_stage_{:d}'.format(s),'extra_statistics.pt')
        es = torch.load(current_es_file_name)

        current_val_file_name = os.path.join(d,'model_results_stage_{:d}'.format(s),'validation_results.pt')
        val_res = torch.load(current_val_file_name)

        # get the current name
        dir_name = os.path.split(d)[1] # path name that contains the keys

        name_prefix_abbr,_ = eu.get_abbrv_case_descriptor(dir_name=dir_name,split_keys=split_keys,abbrv_keys=abbrv_keys)
        name_prefix = '{:s}_s{:d}'.format(name_prefix_abbr,s)

        # first deal with the extra statistics; the values for the map
        ms = es['map_stats']
        if normalize_by_spacing:
            spatially_normalize_stats(ms,spacing)

        all_stats_map,all_names_map = merge_dicts(all_stats_map,all_names_map,ms,desired_stat,name_prefix)

        # now deal with the determinant of Jacobian
        djs = es['det_jac_stats']
        #if normalize_by_spacing:
        #    spatially_normalize_dj_stats(djs,spacing)

        all_stats_dj,all_names_dj = merge_dicts(all_stats_dj,all_names_dj,djs,desired_stat,name_prefix)


        # now deal with the overlap measures
        om = val_res['mean_target_overlap']
        all_overlaps.append(om)
        all_overlap_names.append(name_prefix)


# visualization of overlaps
plt.clf()
rs,rn = reorder_values(all_overlaps,all_overlap_names,nr_of_measures=nr_of_measures)
eu.plot_boxplot(rs,rn,semilogy=False,showfliers=showfliers,suppress_pattern='_s0',suppress_pattern_keep_first_as=suppress_pattern_keep_first_as,
                replace_pattern_from=replace_pattern_from,replace_pattern_to=replace_pattern_to)
plt.ylabel('Dice')
plt.title('Dice overlap')

if print_output_directory is not None:
    plt.savefig(os.path.join(print_output_directory, 'dice.pdf'))

    plt.clf()
    eu.plot_boxplot(rs, rn, semilogy=False, showfliers=showfliers, suppress_pattern='_s0',suppress_pattern_keep_first_as=suppress_pattern_keep_first_as,
                    replace_pattern_from=replace_pattern_from, replace_pattern_to=replace_pattern_to)
    plt.ylabel('Dice')
    plt.savefig(os.path.join(print_output_directory_no_title, 'dice.pdf'))

else:
    plt.show()

# first the visualization for the map results

# custom ranges for the boxplots
if use_custom_boxplot_ranges:
    # only stage 0,2
    # custom_ranges_map_raw = {0.0:[0,6.5], 1.0:[0,6.5], 2.0:[0,6.5], 'global':[0,6.5]}
    # custom_ranges_map_norm = {0.0:[0,2.5], 1.0:[0,2.5], 2.0:[0,2.5], 'global':[0,2.5]}
    #
    # custom_ranges_det_jac_raw = {0.0:[-0.15,0.05], 1.0:[-0.4,0.4], 2.0:[-0.05,0.7], 'global':[-0.1,0.1]}
    # custom_ranges_det_jac_norm = {0.0:[-1.5,6.0], 1.0:[-10.0,10.0], 2.0:[-0.25,3.0], 'global':[-10.0,15.0]}

    # stages 0,1,2
    custom_ranges_map_raw = {0.0: [0, 6.0], 1.0: [0, 6.0], 2.0: [0, 6.0], 'global': [0, 6.0]}
    custom_ranges_map_norm = {0.0: [0, 2.5], 1.0: [0, 2.5], 2.0: [0, 2.5], 'global': [0, 2.5]}

    custom_ranges_det_jac_raw = {0.0: [-0.125, 0.05], 1.0: [-0.6, 0.4], 2.0: [-0.08, 0.65], 'global': [-0.08, 0.055]}
    custom_ranges_det_jac_norm = {0.0: [-2.5, 6.5], 1.0: [-17.5, 11.0], 2.0: [-0.25, 2.75], 'global': [-20.0, 27.5]}
else:
    custom_ranges_map_raw = None
    custom_ranges_map_norm = None
    custom_ranges_det_jac_raw = None
    custom_ranges_det_jac_norm = None


plot_results(all_stats=all_stats_map, all_names=all_names_map, nr_of_measures=nr_of_measures, showfliers=showfliers,
             normalize_by_spacing=normalize_by_spacing, ylabel='disp error (est-GT)', output_prefix='map',
             title_prefix = 'map (est-GT)',
             suppress_pattern = '_s0', suppress_pattern_keep_first_as = suppress_pattern_keep_first_as,
             replace_pattern_from = replace_pattern_from,
             replace_pattern_to = replace_pattern_to,
             custom_ranges_raw=custom_ranges_map_raw,
             custom_ranges_norm=custom_ranges_map_norm,
             print_title=True,
             print_output_directory=print_output_directory)

plot_results(all_stats=all_stats_dj, all_names=all_names_dj, nr_of_measures=nr_of_measures, showfliers=showfliers,
             normalize_by_spacing=False, ylabel='det(jac) error (est-GT)', output_prefix='det_jac',
             title_prefix='det_jac (est-GT)',
             suppress_pattern = '_s0', suppress_pattern_keep_first_as = suppress_pattern_keep_first_as,
             replace_pattern_from = replace_pattern_from,
             replace_pattern_to = replace_pattern_to,
             custom_ranges_raw=custom_ranges_det_jac_raw,
             custom_ranges_norm=custom_ranges_det_jac_norm,
             print_title=True,
             print_output_directory=print_output_directory)

# now print it without title
if print_output_directory_no_title is not None:

    plot_results(all_stats=all_stats_map, all_names=all_names_map, nr_of_measures=nr_of_measures, showfliers=showfliers,
                 normalize_by_spacing=normalize_by_spacing, ylabel='disp error (est-GT)', output_prefix='map',
                 title_prefix = 'map (est-GT)',
                 suppress_pattern = '_s0', suppress_pattern_keep_first_as = suppress_pattern_keep_first_as,
                 replace_pattern_from=replace_pattern_from,
                 replace_pattern_to=replace_pattern_to,
                 custom_ranges_raw=custom_ranges_map_raw,
                 custom_ranges_norm=custom_ranges_map_norm,
                 print_title=False,
                 print_output_directory=print_output_directory_no_title)

    plot_results(all_stats=all_stats_dj, all_names=all_names_dj, nr_of_measures=nr_of_measures, showfliers=showfliers,
                 normalize_by_spacing=False, ylabel='det(jac) error (est-GT)', output_prefix='det_jac',
                 title_prefix='det_jac (est-GT)',
                 suppress_pattern = '_s0', suppress_pattern_keep_first_as = suppress_pattern_keep_first_as,
                 replace_pattern_from=replace_pattern_from,
                 replace_pattern_to=replace_pattern_to,
                 custom_ranges_raw=custom_ranges_det_jac_raw,
                 custom_ranges_norm=custom_ranges_det_jac_norm,
                 print_title=False,
                 print_output_directory=print_output_directory_no_title)

# now print it without title and label
if print_output_directory_no_title_no_label is not None:
    plot_results(all_stats=all_stats_map, all_names=all_names_map, nr_of_measures=nr_of_measures,
                 showfliers=showfliers,
                 normalize_by_spacing=normalize_by_spacing, ylabel='disp error (est-GT)', output_prefix='map',
                 title_prefix='map (est-GT)',
                 suppress_pattern='_s0', suppress_pattern_keep_first_as= suppress_pattern_keep_first_as,
                 replace_pattern_from=replace_pattern_from,
                 replace_pattern_to=replace_pattern_to,
                 custom_ranges_raw=custom_ranges_map_raw,
                 custom_ranges_norm=custom_ranges_map_norm,
                 print_title=False,
                 show_labels=False,
                 print_output_directory=print_output_directory_no_title_no_label)

    plot_results(all_stats=all_stats_dj, all_names=all_names_dj, nr_of_measures=nr_of_measures,
                 showfliers=showfliers,
                 normalize_by_spacing=False, ylabel='det(jac) error (est-GT)', output_prefix='det_jac',
                 title_prefix='det_jac (est-GT)',
                 suppress_pattern='_s0', suppress_pattern_keep_first_as= suppress_pattern_keep_first_as,
                 replace_pattern_from=replace_pattern_from,
                 replace_pattern_to=replace_pattern_to,
                 custom_ranges_raw=custom_ranges_det_jac_raw,
                 custom_ranges_norm=custom_ranges_det_jac_norm,
                 print_title=False,
                 show_labels=False,
                 print_output_directory=print_output_directory_no_title_no_label)

# now do the squeezed plots

# now print it without title
if print_output_directory_no_title_squeezed is not None:

    plot_results(all_stats=all_stats_map, all_names=all_names_map, nr_of_measures=nr_of_measures, showfliers=showfliers,
                 normalize_by_spacing=normalize_by_spacing, ylabel='disp error (est-GT)', output_prefix='map',
                 title_prefix = 'map (est-GT)',
                 suppress_pattern = '_s0', suppress_pattern_keep_first_as = suppress_pattern_keep_first_as,
                 replace_pattern_from=replace_pattern_from,
                 replace_pattern_to=replace_pattern_to,
                 custom_ranges_raw=custom_ranges_map_raw,
                 custom_ranges_norm=custom_ranges_map_norm,
                 print_title=False,
                 fix_aspect=squeezed_aspect_ratio,
                 print_output_directory=print_output_directory_no_title_squeezed)

    plot_results(all_stats=all_stats_dj, all_names=all_names_dj, nr_of_measures=nr_of_measures, showfliers=showfliers,
                 normalize_by_spacing=False, ylabel='det(jac) error (est-GT)', output_prefix='det_jac',
                 title_prefix='det_jac (est-GT)',
                 suppress_pattern = '_s0', suppress_pattern_keep_first_as = suppress_pattern_keep_first_as,
                 replace_pattern_from=replace_pattern_from,
                 replace_pattern_to=replace_pattern_to,
                 custom_ranges_raw=custom_ranges_det_jac_raw,
                 custom_ranges_norm=custom_ranges_det_jac_norm,
                 print_title=False,
                 fix_aspect=squeezed_aspect_ratio,
                 print_output_directory=print_output_directory_no_title_squeezed)

# now print it without title and label
if print_output_directory_no_title_no_label_squeezed is not None:
    plot_results(all_stats=all_stats_map, all_names=all_names_map, nr_of_measures=nr_of_measures,
                 showfliers=showfliers,
                 normalize_by_spacing=normalize_by_spacing, ylabel='disp error (est-GT)', output_prefix='map',
                 title_prefix='map (est-GT)',
                 suppress_pattern='_s0', suppress_pattern_keep_first_as=suppress_pattern_keep_first_as,
                 replace_pattern_from=replace_pattern_from,
                 replace_pattern_to=replace_pattern_to,
                 custom_ranges_raw=custom_ranges_map_raw,
                 custom_ranges_norm=custom_ranges_map_norm,
                 print_title=False,
                 show_labels=False,
                 fix_aspect=squeezed_aspect_ratio,
                 print_output_directory=print_output_directory_no_title_no_label_squeezed)

    plot_results(all_stats=all_stats_dj, all_names=all_names_dj, nr_of_measures=nr_of_measures,
                 showfliers=showfliers,
                 normalize_by_spacing=False, ylabel='det(jac) error (est-GT)', output_prefix='det_jac',
                 title_prefix='det_jac (est-GT)',
                 suppress_pattern='_s0', suppress_pattern_keep_first_as=suppress_pattern_keep_first_as,
                 replace_pattern_from=replace_pattern_from,
                 replace_pattern_to=replace_pattern_to,
                 custom_ranges_raw=custom_ranges_det_jac_raw,
                 custom_ranges_norm=custom_ranges_det_jac_norm,
                 print_title=False,
                 show_labels=False,
                 fix_aspect=squeezed_aspect_ratio,
                 print_output_directory=print_output_directory_no_title_no_label_squeezed)


if print_output_directory is not None:
    # if we have pdfjam we create a summary pdf
    if os.system('which pdfjam') == 0:
        summary_pdf_name = os.path.join(print_output_directory, 'synth_validation_summary.pdf')

        if os.path.isfile(summary_pdf_name):
            os.remove(summary_pdf_name)

        print('Creating summary PDF: ')
        cmd = 'pdfjam {:} --nup 1x2 --outfile {:}'.format(os.path.join(print_output_directory, '*.pdf'),summary_pdf_name)
        #cmd = 'pdfjam {:} --outfile {:}'.format(os.path.join(print_output_directory, '*.pdf'),summary_pdf_name)
        os.system(cmd)
