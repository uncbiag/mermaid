import experiment_utils as eu
import os
import glob
import torch
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

def get_abbrv_case_descriptor(dir_name,split_keys,abbrv_keys):

    val_strs = []
    cur_dir_name = dir_name
    for k in split_keys:
        cur_dir_name = cur_dir_name.split(k+'_')[1]
        if cur_dir_name.find('_')!=-1:
            val_str = cur_dir_name.split('_')[0]
        else:
            val_str = cur_dir_name
        val_strs.append(val_str)

    abbrv = ''
    for va in zip(val_strs,abbrv_keys):
        abbrv += '{:s}={:.2f};'.format(va[1],float(va[0]))
    abbrv = abbrv[0:-1]
    return abbrv

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

datapath = '/Users/mn/data/stat_results'
prefix = 'out_test'

stages = [0,1,2]
#desired_stat = 'mean'
desired_stat = 'median'
use_all_directories = False
showfliers = False
normalize_by_spacing = True
spacing = 1./(128.-1.)
print_output_directory = 'pdf_sweep'

if print_output_directory is not None:
    print('Saving figure output in directory: {:s}'.format(print_output_directory))
    if not os.path.exists(print_output_directory):
        os.mkdir(print_output_directory)

if use_all_directories:
    desired_directories = glob.glob(os.path.join(datapath,'{:s}*'.format(prefix)))
else:
    desired_directories = ['/Users/mn/data/stat_results/out_test_total_variation_weight_penalty_0.010000_omt_weight_penalty_1.000000',
                           '/Users/mn/data/stat_results/out_test_total_variation_weight_penalty_0.010000_omt_weight_penalty_10.000000',
                           #'/Users/mn/data/stat_results/out_test_total_variation_weight_penalty_0.010000_omt_weight_penalty_100.000000',
                           '/Users/mn/data/stat_results/out_test_total_variation_weight_penalty_0.100000_omt_weight_penalty_1.000000',
                           '/Users/mn/data/stat_results/out_test_total_variation_weight_penalty_0.100000_omt_weight_penalty_10.000000',
                           #'/Users/mn/data/stat_results/out_test_total_variation_weight_penalty_0.100000_omt_weight_penalty_100.000000'
                           ]

split_keys = ['total_variation_weight_penalty','omt_weight_penalty']
abbrv_keys = ['t','o']

cr_all = []
cn_all = []

all_stats = dict()
all_names = dict()
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

        # first deal with the extra statistics; the values for the map
        ms = es['map_stats']
        if normalize_by_spacing:
            spatially_normalize_stats(ms,spacing)

        dir_name = os.path.split(d)[1] # path name that contains the keys

        name_prefix_abbr = get_abbrv_case_descriptor(dir_name=dir_name,split_keys=split_keys,abbrv_keys=abbrv_keys)
        name_prefix = '{:s}_s{:d}'.format(name_prefix_abbr,s)

        all_stats,all_names = merge_dicts(all_stats,all_names,ms,desired_stat,name_prefix)

        # now deal with the overlap measures
        om = val_res['mean_target_overlap']
        all_overlaps.append(om)
        all_overlap_names.append(name_prefix)

# visualization of overlaps
plt.clf()
rs,rn = reorder_values(all_overlaps,all_overlap_names,3)
eu.plot_boxplot(rs,rn,semilogy=False,showfliers=showfliers)
plt.title('Dice overlap')

if print_output_directory is not None:
    plt.savefig(os.path.join(print_output_directory, 'dice.pdf'))
else:
    plt.show()

# direct visualization
for k in all_stats:
    plt.clf()
    rs,rn = reorder_values(all_stats[k],all_names[k],3)
    eu.plot_boxplot(rs,rn,semilogy=False,showfliers=showfliers)
    plt.title(outliers_suppressed('Raw: ' + str(k),showfliers=showfliers))
    if normalize_by_spacing:
        plt.ylabel('disp error [pixel]')
    else:
        plt.ylabel('disp error')

    if print_output_directory is not None:
        plt.savefig(os.path.join(print_output_directory, 'raw_stat_{:s}.pdf'.format(str(k))))
    else:
        plt.show()

# median normalized (with respect to stage 0 -- first entry, check this; this should be more or less constant as it does not depend on the OMT or TV penalty
all_stats_mn = all_stats.copy()
for k in all_stats_mn:
    c_stats = all_stats_mn[k]
    print('Normalizing based on {:s}'.format(all_names[k][0]))
    median_for_normalization = np.percentile(c_stats[0],50)
    for s in c_stats:
        s/=median_for_normalization

# now plot it
for k in all_stats:
    plt.clf()
    rs,rn = reorder_values(all_stats_mn[k],all_names[k],3)
    eu.plot_boxplot(rs,rn,semilogy=False,showfliers=showfliers)
    plt.title(outliers_suppressed('Median normalized: ' + str(k),showfliers=showfliers))

    plt.ylabel('disp error [unitless; normalized]')

    if print_output_directory is not None:
        plt.savefig(os.path.join(print_output_directory, 'median_normalized_stat_{:s}.pdf'.format(str(k))))
    else:
        plt.show()

for k in all_stats:
    print('Results for ' + str(k) + ':')
    for s,n in zip(all_stats[k],all_names[k]):
        current_median_value = np.percentile(s,50)
        print('{:s}: median={:f}'.format(n,current_median_value))

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
