import os
import glob
import torch
import numpy as np
import experiment_utils as eu

#import tabulate

def create_latex_table(table,table_stds=None,row_names=None,column_names=None,additional_heading=None,print_beginning=True,print_end=True):

    nr_of_rows = len(table)
    nr_of_cols = len(table[0])
    indent = 5

    """
    \begin{tabular}{|l|c||c|c||c||c|c|}
    \hline
      & \textbf{mean} & \textbf{1\%} & \textbf{5\%} & \textbf{50\%} & \textbf{95\%} & \textbf{99\%} \\
      \hline
      \textbf{Stage 0} & 0.999(0.010) & 0.959(0.017) & 0.966(0.016) & 0.998(0.011) & 1.035(0.018) & 1.043(0.020) \\
      \textbf{Stage 1} & 1.003(0.022) & 0.062(0.043) & 0.229(0.035) & 0.837(0.035) & 2.313(0.083) & 4.087(0.301) \\
      \textbf{Stage 2} & 0.987(0.020) & -0.064(0.071) & 0.143(0.031) & 0.774(0.034) & 2.500(0.084) & 4.797(0.334) \\
      \hline
    \end{tabular}
    """

    str = ''

    if print_beginning:

        str += '\\begin{tabular}{|l|'
        for n in range(nr_of_cols):
            str += 'c|'
        str += '}\n'
        str += '\hline\n'

        if column_names is not None:
            str += ' '*indent
            for c in column_names:
                str += '& ' + c + ' '
            str += '\\\\\n'
            str += ' '*indent + '\\hline\n'

    if additional_heading:
        str += '\\hline'
        str += ' '*indent + '\multicolumn{{{}}}{{c}}{{{}}}\\\\\n\\hline\n'.format(nr_of_cols,additional_heading)

    for n in range(nr_of_rows):
        str += ' ' * indent
        if row_names is not None:
            str += '\\textbf{{{}}}'.format(row_names[n])
        for c in range(nr_of_cols):
            str += '& ' + '{:.2f}'.format(table[n][c])
            if table_stds is not None:
                str += '({:.2f})'.format(table_stds[n][c])
            str += ' '
        str += '\\\\\n'

    if print_end:
        str += '\\hline\n'
        str += '\end{tabular}\n'

    return str

# conda install -c conda-forge tabulate

#datapath = '/Users/mn/sim_results/pf-out_testing_train_cumc12_test_cumc12_3d_sqrt_w_K_sqrt'
datapath = '/Users/mn/sim_results/pf-out_testing_train_cumc12_test_cumc12_3d_sqrt_w_K_sqrt-skip-stage-1'
#datapath = '/Users/mn/sim_results/pf_out_paper_experiment_lpba40_2d_sqrt_w_K_sqrt'
#stages = [0,1,2]
stages = [0,2]
nr_of_measures = len(stages)
#data_from_generic_sweep_run = False
data_from_generic_sweep_run = True
use_all_directories = True
prefix = 'out_testing'
#prefix = 'out_training'

if data_from_generic_sweep_run:

    if not use_all_directories:
        desired_tv = [0.1] # [0.01,0.1,0.25]
        desired_omt = [5,7.5,10.0,12.5,15,25,50,75,100] #[5,15,25,50,75,100]
    else:
        desired_tv = None
        desired_omt = None

    if use_all_directories:
        desired_directories = glob.glob(os.path.join(datapath,'{:s}*'.format(prefix)))
    else:
        desired_directories = []
        for c_tv in desired_tv:
            for c_omt in desired_omt:
                current_dir_name = os.path.join(datapath,'{}_total_variation_weight_penalty_{:f}_omt_weight_penalty_{:f}'.format(prefix,c_tv,c_omt))
                print('Adding directory: {}'.format(current_dir_name))
                desired_directories.append(current_dir_name)

else:  # data not from sweep, directory is directly specified

    desired_directories = [datapath]



split_keys = ['total_variation_weight_penalty','omt_weight_penalty']
abbrv_keys = ['t','o']

for idx,d in enumerate(desired_directories):

    if data_from_generic_sweep_run:
        # get the current name
        dir_name = os.path.split(d)[1] # path name that contains the keys

        name_prefix_abbr,current_vals = eu.get_abbrv_case_descriptor(dir_name=dir_name,split_keys=split_keys,abbrv_keys=abbrv_keys)

        current_latex_header = '$\lambda_{{TV}}={:.1f}$, $\lambda_{{OMT}}={:.1f}$'.format(current_vals[0],current_vals[1])

        #print('Name = {}'.format(name_prefix_abbr))
    else:
        current_latex_header = None

    current_table = []
    current_stds_table = []

    keys_for_headers = ['mean', '1_perc', '5_perc', 'median', '95_perc', '99_perc']
    headers = ['\\textbf{mean}', '\\textbf{1\%}', '\\textbf{5\%}', '\\textbf{median}', '\\textbf{95\%}', '\\textbf{99\%}']

    row_names = []

    for s in stages:

        current_det_jac_name = os.path.join(d,'model_results_stage_{:d}'.format(s),'all_stat_det_of_jacobian.pt')
        dj = torch.load(current_det_jac_name)


        current_row = []
        current_std_row = []
        row_names.append('Stage {:d}'.format(s))

        for k in keys_for_headers:
            current_mean = np.mean(dj['nz_det_jac'][k])
            current_std = np.std(dj['nz_det_jac'][k])

            current_row.append(current_mean)
            current_std_row.append(current_std)

        current_table.append(current_row)
        current_stds_table.append(current_std_row)

    if data_from_generic_sweep_run:
        if idx==0:
            print_beginning=True
            print_end=False
        elif idx==len(desired_directories)-1:
            print_beginning=False
            print_end=True
        else:
            print_beginning=False
            print_end=False
    else:
        print_beginning=True
        print_end=True

    lt = create_latex_table(table=current_table,
                            table_stds=current_stds_table,
                            row_names=row_names,
                            column_names=headers,
                            additional_heading=current_latex_header,
                            print_beginning=print_beginning,
                            print_end=print_end)
    print(lt)





