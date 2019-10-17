from __future__ import print_function
from __future__ import absolute_import

from builtins import str
from builtins import range

import command_line_execution_tools as ce

import mermaid.module_parameters as pars

import os
import shutil

def _make_arg_list(args):
    arg_list = []
    for k in args:
        arg_list.append('--' + str(k))
        arg_list.append( str(args[k]) )

    return arg_list

def get_bash_precommand(cuda_visible_devices,pre_command=None):

    if (cuda_visible_devices is not None) and (pre_command is not None):
        ret = '{:s} && CUDA_VISIBLE_DEVICES={:d}'.format(pre_command,cuda_visible_devices)
    elif (cuda_visible_devices is not None) and (pre_command is None):
        ret = 'CUDA_VISIBLE_DEVICES={:d}'.format(cuda_visible_devices)
    elif (cuda_visible_devices is None) and (pre_command is not None):
        ret = '{:s} && '.format(pre_command)
    else:
        ret = ''
    return ret

def get_stage_log_filename(output_directory,stage,process_name=None):
    if process_name is not None:
        ret = os.path.join(output_directory,'log_stage_{:d}_{:s}.txt'.format(stage,process_name))
    else:
        ret = os.path.join(output_directory,'log_stage_{:d}.txt'.format(stage))
    return ret

def create_kv_string(kvs):
    ret = ''
    is_first = True
    for k in kvs:
        if is_first:
            ret += str(k)+'='+str(kvs[k])
            is_first = False
        else:
            ret += '\\;' + str(k) + '=' + str(kvs[k])

    return ret

def _escape_semicolons(s):
    s_split = s.split(';')
    if len(s_split)<2:
        return s
    else:
        ret = s_split[0]
        for c in s_split[1:]:
            ret += '\\;' + c
        return ret

def add_to_config_string(cs,cs_to_add):
    if (cs is None) or (cs==''):
        # we need to check if there are semicolons in the string to add, if so, these need to be escaped
        ret = _escape_semicolons(cs_to_add)
    elif cs_to_add is None:
        ret = cs
    else:
        cs_to_add_escaped = _escape_semicolons(cs_to_add)
        ret = cs + '\\;' + cs_to_add_escaped

    return ret

def run_optimization(stage,nr_of_epochs,image_pair_config_pt,nr_of_image_pairs,
                     only_run_stage0_with_unchanged_config,
                     skip_stage_1_and_start_stage_2_from_stage_0,
                     input_image_directory,output_directory,
                     main_json,
                     load_shared_parameters_for_stages_from_directory,
                     only_optimize_over_registration_parameters_for_stage_nr,
                     seed,
                     key_value_overwrites=dict(),string_key_value_overwrites=dict(),
                     cuda_visible_devices=None,pre_command=None):

    if stage==0:
        all_nr_of_epochs = '{:d},1,1'.format(nr_of_epochs)
    elif stage==1:
        all_nr_of_epochs = '1,{:d},1'.format(nr_of_epochs)
    elif stage==2:
        all_nr_of_epochs = '1,1,{:d}'.format(nr_of_epochs)
    else:
        raise ValueError('Unknown stage; stage needs to be 0,1, or 2')

    # create key-value string
    config_kv_string = create_kv_string(key_value_overwrites)

    if 'default' in string_key_value_overwrites:
        config_kv_string = add_to_config_string(config_kv_string,string_key_value_overwrites['default'])
    if stage in string_key_value_overwrites:
        config_kv_string = add_to_config_string(config_kv_string,string_key_value_overwrites[stage])

    args = {'input_image_directory': input_image_directory,
            'output_directory': output_directory,
            'nr_of_epochs': all_nr_of_epochs,
            'config': main_json,
            'stage_nr': stage,
            'seed': seed,
            }

    if image_pair_config_pt is not None:
        args['image_pair_config_pt'] = image_pair_config_pt
    elif nr_of_image_pairs is not None:
        if nr_of_image_pairs>0:
            args['nr_of_image_pairs'] = nr_of_image_pairs

    if not config_kv_string=='':
        args['config_kvs'] = config_kv_string

    if only_run_stage0_with_unchanged_config:
        args['only_run_stage0_with_unchanged_config']=''

    if skip_stage_1_and_start_stage_2_from_stage_0:
        args['skip_stage_1_and_start_stage_2_from_stage_0']=''

    if load_shared_parameters_for_stages_from_directory is not None:
        args['load_shared_parameters_for_stages_from_directory '] = load_shared_parameters_for_stages_from_directory

    if only_optimize_over_registration_parameters_for_stage_nr is not None:
        args['only_optimize_over_registration_parameters_for_stage_nr'] = only_optimize_over_registration_parameters_for_stage_nr

    # now run the command
    cmd_arg_list = _make_arg_list(args)
    current_python_script = 'multi_stage_smoother_learning.py'

    entire_pre_command = get_bash_precommand(cuda_visible_devices=cuda_visible_devices,pre_command=pre_command)
    log_file = get_stage_log_filename(output_directory=output_directory,stage=stage,process_name='opt')
    ce.execute_python_script_via_bash(current_python_script, cmd_arg_list, pre_command=entire_pre_command, log_file=log_file)

def run_visualization(stage,output_directory,main_json,cuda_visible_devices=None,pre_command=None,compute_only_pair_nr=None,only_recompute_validation_measures=False):

    args = {'config': main_json,
            'output_directory': output_directory,
            'stage_nr': stage,
            'do_not_visualize': '',
            'clean_publication_print': ''}

    if compute_only_pair_nr is not None:
        print('INFO: Only computing pair nr: ' + str(compute_only_pair_nr))
        args['compute_only_pair_nr'] = compute_only_pair_nr

    if only_recompute_validation_measures:
        print('INFO: Only recomputing the Jacobians')
        args['only_recompute_validation_measures'] = ''

    # now run the command
    cmd_arg_list = _make_arg_list(args)
    current_python_script = 'visualize_multi_stage.py'

    entire_pre_command = get_bash_precommand(cuda_visible_devices=cuda_visible_devices,pre_command=pre_command)
    log_file = get_stage_log_filename(output_directory=output_directory,stage=stage,process_name='viz')
    ce.execute_python_script_via_bash(current_python_script, cmd_arg_list, pre_command=entire_pre_command, log_file=log_file)

def run_validation(stage,output_directory,validation_dataset_directory,validation_dataset_name,cuda_visible_devices=None,pre_command=None):

    args = {'stage_nr':stage,
            'output_directory': output_directory,
            'dataset_directory': validation_dataset_directory,
            'do_not_visualize': '',
            'save_overlap_filename': 'overlaps.txt'}

    args['dataset'] = validation_dataset_name

    # now run the command
    cmd_arg_list = _make_arg_list(args)
    current_python_script = 'compute_validation_results.py'

    entire_pre_command = get_bash_precommand(cuda_visible_devices=cuda_visible_devices,pre_command=pre_command)
    log_file = get_stage_log_filename(output_directory=output_directory,stage=stage,process_name='val')
    ce.execute_python_script_via_bash(current_python_script, cmd_arg_list, pre_command=entire_pre_command, log_file=log_file)

def run_extra_validation(stage,input_image_directory,output_directory,main_json,cuda_visible_devices=None,pre_command=None,compute_only_pair_nr=None):

    args = {'stage_nr': stage,
            'input_image_directory': input_image_directory,
            'output_directory': output_directory,
            'config': main_json,
            'do_not_visualize': '',
            'clean_publication_print': ''}

    if compute_only_pair_nr is not None:
        print('INFO: Only computing pair nr: ' + str(compute_only_pair_nr))
        args['compute_only_pair_nr'] = compute_only_pair_nr

    # now run the command
    cmd_arg_list = _make_arg_list(args)
    current_python_script = 'extra_validation_for_synthetic_test_cases.py'

    entire_pre_command = get_bash_precommand(cuda_visible_devices=cuda_visible_devices,pre_command=pre_command)
    log_file = get_stage_log_filename(output_directory=output_directory,stage=stage,process_name='eval')
    ce.execute_python_script_via_bash(current_python_script, cmd_arg_list, pre_command=entire_pre_command, log_file=log_file)

def run(stage,nr_of_epochs,main_json,
        image_pair_config_pt,
        nr_of_image_pairs,
        input_image_directory,
        output_base_directory,
        postfix,
        validation_dataset_directory,
        validation_dataset_name,
        previous_output_base_directory,
        previous_postfix,
        only_optimize_over_registration_parameters_for_stage_nr,
        load_shared_parameters_from_previous_stage_nr,
        move_to_directory,
        key_value_overwrites=dict(),
        string_key_value_overwrites=dict(),
        seed=1234,
        cuda_visible_devices=None,
        pre_command=None,
        parts_to_run=None,
        compute_only_pair_nr=None,
        only_run_stage0_with_unchanged_config=False,
        skip_stage_1_and_start_stage_2_from_stage_0=False,
        only_recompute_validation_measures=False
        ):

    if parts_to_run is None:
        parts_to_run = dict()
        parts_to_run['run_optimization'] = True
        parts_to_run['run_visualization'] = True
        parts_to_run['run_validation'] = True
        parts_to_run['run_extra_validation'] = True

    output_directory = os.path.join(output_base_directory, 'out_' + postfix )

    if not os.path.exists(output_base_directory):
        print('Creating output base directory: {:s}'.format(output_base_directory))
        os.mkdir(output_base_directory)

    if not os.path.exists(output_directory):
        print('Creating output directory: {:s}'.format(output_directory))
        os.mkdir(output_directory)

    if parts_to_run['run_optimization']:

        if load_shared_parameters_from_previous_stage_nr:
            # we need to load the previous shared parameters, so construct the filename from where to load first

            load_shared_parameters_for_stages_from_directory = os.path.join(previous_output_base_directory, 'out_' + previous_postfix )

            # load_shared_parameters_from_file = os.path.join(previous_output_base_directory, 'out_' + previous_postfix,
            #                                                 'results_after_stage_{}'.format(
            #                                                     load_shared_parameters_from_previous_stage_nr),
            #                                                 'shared', 'shared_parameters.pt')

            # if move_to_directory is None:
            #     load_shared_parameters_from_file = os.path.join(previous_output_base_directory, 'out_' + previous_postfix,
            #                                                     'results_after_stage_{}'.format(
            #                                                         load_shared_parameters_from_previous_stage_nr),
            #                                                     'shared', 'shared_parameters.pt')
            # else:
            #     load_shared_parameters_from_file = os.path.join(move_to_directory, 'out_' + previous_postfix,
            #                                                     'results_after_stage_{}'.format(
            #                                                         load_shared_parameters_from_previous_stage_nr),
            #                                                     'shared', 'shared_parameters.pt')

            #if not os.path.exists(load_shared_parameters_from_file):
            #    raise ValueError('Could not find file: {}'.format(load_shared_parameters_from_file))
        else:
            #load_shared_parameters_from_file = None
            load_shared_parameters_for_stages_from_directory = None

        run_optimization(stage=stage,
                         nr_of_epochs=nr_of_epochs,
                         image_pair_config_pt=image_pair_config_pt,
                         nr_of_image_pairs=nr_of_image_pairs,
                         only_run_stage0_with_unchanged_config=only_run_stage0_with_unchanged_config,
                         skip_stage_1_and_start_stage_2_from_stage_0=skip_stage_1_and_start_stage_2_from_stage_0,
                         input_image_directory=input_image_directory,
                         output_directory=output_directory,
                         main_json=main_json,
                         load_shared_parameters_for_stages_from_directory=load_shared_parameters_for_stages_from_directory,
                         only_optimize_over_registration_parameters_for_stage_nr=only_optimize_over_registration_parameters_for_stage_nr,
                         key_value_overwrites=key_value_overwrites,
                         string_key_value_overwrites=string_key_value_overwrites,
                         seed=seed,
                         cuda_visible_devices=cuda_visible_devices,
                         pre_command=pre_command)
    else:
        print('INFO: Optimization will not be run')

    if not only_run_stage0_with_unchanged_config:

        if not (stage==1 and skip_stage_1_and_start_stage_2_from_stage_0):

            if parts_to_run['run_visualization']:
                run_visualization(stage=stage,
                                  output_directory=output_directory,
                                  main_json=main_json,
                                  cuda_visible_devices=cuda_visible_devices,
                                  pre_command=pre_command,
                                  compute_only_pair_nr=compute_only_pair_nr,
                                  only_recompute_validation_measures=only_recompute_validation_measures)
            else:
                print('INFO: Visualization will not be run')

            if (validation_dataset_directory is not None) and (validation_dataset_name is not None):

                if parts_to_run['run_validation']:
                    run_validation(stage=stage,
                                   output_directory=output_directory,
                                   validation_dataset_directory=validation_dataset_directory,
                                   validation_dataset_name=validation_dataset_name,
                                   cuda_visible_devices=cuda_visible_devices,
                                   pre_command=pre_command)
                else:
                    print('INFO: Validation will not be run')

                if validation_dataset_name=='SYNTH':
                    if parts_to_run['run_extra_validation']:
                        run_extra_validation(stage=stage,
                                             input_image_directory=input_image_directory,
                                             output_directory=output_directory,
                                             main_json=main_json,
                                             cuda_visible_devices=cuda_visible_devices,
                                             pre_command=pre_command,
                                             compute_only_pair_nr=compute_only_pair_nr)
                    else:
                        print('INFO: Extra validation will not be run')

            else:
                print('Validation not computed, because validation dataset is not specified')

        else:
            print('INFO: stage 1 is skipped, because stage 2 would be computed from stage 0')



def move_output_directory(move_to_directory,output_base_directory,postfix):
    # This optionally allows to move the output directory to a different directory after computation
    # This is for example useful when computing using an SSD, but then moving the result to slower storage afterwards

    output_directory = os.path.join(output_base_directory, 'out_' + postfix)

    if move_to_directory is not None:
        if not os.path.exists(move_to_directory):
            print('Creating output directory: {:s}'.format(move_to_directory))
            os.mkdir(move_to_directory)

        relocate_dir = os.path.join(move_to_directory, 'out_' + postfix)
        if os.path.exists(relocate_dir):
            print('Removing directory {:s}'.format(relocate_dir))
            shutil.rmtree(relocate_dir)

        print('Moving directory {:s} to {:s}'.format(output_directory, relocate_dir))
        shutil.copytree(output_directory, relocate_dir, symlinks=True)

        # now delete the original directory
        print('Removing directory {:s}'.format(output_directory))
        shutil.rmtree(output_directory)

        # and create a link
        moved_directory_name = os.path.abspath(relocate_dir)
        print('Creating symbolic link {} -> {}'.format(output_directory,moved_directory_name))
        os.symlink(moved_directory_name, output_directory)

def _get_kv_info(kv_name,vals_str):

    # values
    if vals_str is None:
        vals = None
    else:
        vals = [float(item) for item in vals_str.split(',')]

    if kv_name is None:
        kv_label = None
    else:
        # last part after period (to create name to save)
        kv_label = kv_name.split('.')[-1]

    return kv_label,vals

def get_sweep_values(sweep_value_name_a, sweep_values_a,
                     sweep_value_name_b, sweep_values_b,
                     sweep_value_name_c, sweep_values_c):


    kv_label_a,kv_values_a = _get_kv_info(sweep_value_name_a,sweep_values_a)
    kv_label_b,kv_values_b = _get_kv_info(sweep_value_name_b,sweep_values_b)
    kv_label_c,kv_values_c = _get_kv_info(sweep_value_name_c,sweep_values_c)

    # create tensor product results (stored in an array with dictionary elements for key-value pairs)

    if (kv_values_a is not None) and (kv_values_b is None) and (kv_values_c is None):
        # only sweep over a
        ret = []
        for va in kv_values_a:
            ret.append({'label':kv_label_a + '_{:f}'.format(va), 'kvs': {sweep_value_name_a:va}})

    elif (kv_values_a is not None) and (kv_values_b is not None) and (kv_values_c is None):
        # sweep over a and b
        ret = []
        for va in kv_values_a:
            for vb in kv_values_b:
                ret.append({'label': kv_label_a + '_{:f}_'.format(va) + kv_label_b + '_{:f}'.format(vb),
                            'kvs': {sweep_value_name_a:va,sweep_value_name_b:vb}})

    elif (kv_values_a is not None) and (kv_values_b is not None) and (kv_values_c is not None):
        # sweep over a, b, and c
        ret = []
        for va in kv_values_a:
            for vb in kv_values_b:
                for vc in kv_values_c:
                    ret.append({'label': kv_label_a + '_{:f}_'.format(va) + kv_label_b + '_{:f}_'.format(vb) + kv_label_c + '_{:f}'.format(vc),
                                'kvs': {sweep_value_name_a:va, sweep_value_name_b:vb, sweep_value_name_c:vc}})

    else:
        # No tensor product to sweep over specified
        return None

    return ret

def merge_kvs(kva,kvb):
    ret = kva.copy()
    ret.update(kvb)

    return ret

def _none_if_empty(par):
    if type(par)==type(pars.ParameterDict()):
        if par.isempty():
            return None
        else:
            return par
    else:
        return par

if __name__ == "__main__":

    # These can be set once and for all (will likely never need to be specified on the command line
    #input_image_directory = 'synthetic_example_out/brain_affine_icbm'
    #image_pair_config_pt = 'synthetic_example_out/used_image_pairs.pt'
    #output_base_directory = 'experimental_results'
    #validation_dataset_directory = 'synthetic_example_out'

    # Example for how to call it:
    # --dataset_config dataset.json --config test2d_025.json --sweep_value_name_a model.registration_model.forward_model.smoother.deep_smoother.total_variation_weight_penalty
    # --sweep_values_a 0.01,0.1,1.0,10 --sweep_value_name_b model.registration_model.forward_model.smoother.omt_weight_penalty
    # --sweep_values_b 1.,10.,100. --move_to_directory test_sweep --multi_gaussian_weights_stage_0 [0.,0.,0.,1.0]
    # --config_kvs optimizer.sgd.individual.lr=0.01\;model.optimizer.sgd.shared.lr=0.0 --precommand "source activate p27"
    #
    # directories can also be specified with other settings. If a json file is specified, but does not exist
    # it will generate a default one

    import argparse

    parser = argparse.ArgumentParser(description='Main driver to perform experiments')

    parser.add_argument('--config', required=True, help='The main json configuration file to generate the results')

    parser.add_argument('--dataset_config', required=False, default=None, help='Allows to specify all the needed directories and the dataset type in a JSON file.')
    """dataset_config contains settings for: image_pair_config_pt, input_image_directory, output_base_directory, validation_dataset, and validation_dataset_directory"""

    parser.add_argument('--cuda_device', required=False, default=0, type=int, help='The CUDA device the code is run on if CUDA is enabled')
    parser.add_argument('--precommand', required=False, default=None, type=str, help='Command to execute before; for example to activate a virtual environment; "source activate p27"')

    parser.add_argument('--dataset', required=False, default=None, help='Which validation dataset is being used: [CUMC|LPBA|IBSR|MGH|SYNTH]; if none is specified validation results are not computed')
    parser.add_argument('--input_image_directory', required=False, default=None, help='Directory where all the images are')
    parser.add_argument('--output_base_directory', required=False, default='experimental_results', help='Base directory where the output is stored')
    parser.add_argument('--validation_dataset_directory', required=False, default=None, help='Directory where the validation data can be found')
    parser.add_argument('--image_pair_config_pt', required=False, default=None, help='If specified then the image-pairs are determined based on the information in this file; can simply point to a previous configuration.pt file')

    parser.add_argument('--previous_dataset_config', required=False, default=None,
                        help='Point to the directory config file of a previous run; this allows reusing these results for example to compute evaluations on a separate test set. Use this together with --load_shared_parameters_from_previous_stage_nr and --only_optimize_over_registration_parameters_for_stage_nr')

    parser.add_argument('--load_shared_parameters_from_previous_stage_nr', action='store_true',
                        help='If specified will load the shared parameters for a given stage from the corresponding previous stage')

    parser.add_argument('--only_optimize_over_registration_parameters_for_stage_nr', required=False, type=str,default=None,
                        help='If set, only the registration parameters (e.g., momentum) are optimized for the given stages (as a comma separated list; e.g., 0,1,2), but not the smoother. Default is None (i.e., optimization happens in all stages). Best used in conjunction with --load_shared_parameters_for_previous_stage_nr; similar in effect to freezing iterations, but will be stored in default directories; so good for validation ')

    parser.add_argument('--previous_postfix', required=False, default=None, help='Output subdirectory of previous run, out_postfix')

    parser.add_argument('--nr_of_image_pairs', required=False, type=int, default=0, help='number of image pairs that will be used; if not set all pairs will be used')

    parser.add_argument('--compute_only_pair_nr', required=False, type=int, default=None, help='When specified only this pair is computed; otherwise all of them')
    parser.add_argument('--create_clean_publication_print_for_pair_nr', required=False, type=int, default=None, help="When specified creates the publication prints only for the specified pairs number")
    parser.add_argument('--only_recompute_validation_measures', action='store_true', help='When set only the valiation measures are recomputed (nothing else; no images are written and no PDFs except for the validation boxplot are created)')

    parser.add_argument('--only_run_stage0_with_unchanged_config', action='store_true', help='This is a convenience setting which allows using the script to run any json config file unchanged (as if it were stage 0); i.e., it will optimize over the smoother if set as such; should only be used for debugging; will terminate after stage 0.')
    parser.add_argument('--skip_stage_1_and_start_stage_2_from_stage_0', action='store_true', help='If set, stage 2 is initialized from stage 0, without computing stage 1')

    parser.add_argument('--move_to_directory', required=False, default=None, help='If specified results will be move to this directory after computation (for example to move from SSD to slower storage)')

    parser.add_argument('--postfix', required=False, default=None, help='Output subdirectory will be out_postfix')

    parser.add_argument('--main_synth_directory', required=False, default=None, help='Main directory for synthetic data, e.g., synthetic_example_out. If set takes care of all directories expect for the output directory')

    parser.add_argument('--stage_nr', required=False, type=str, default='0,1,2', help='Which stages should be run {0,1,2} as a comma separated list; default is all: 0,1,2, but can also just be 1,2')
    parser.add_argument('--nr_of_epochs', required=False,type=str, default=None, help='number of epochs for *all* the three stages as a comma separated list')
    parser.add_argument('--seed', required=False, type=int, default=1234, help='Sets the random seed which affects data shuffling')

    parser.add_argument('--config_kvs', required=False, default=None, help='Allows specifying key value pairs (for all stages) that will override json settings; in format k1.k2.k3=val1;k1.k2=val2')
    parser.add_argument('--config_kvs_0', required=False, default=None, help='Allows specifying key value pairs (for stage 0) that will override json settings; in format k1.k2.k3=val1;k1.k2=val2')
    parser.add_argument('--config_kvs_1', required=False, default=None, help='Allows specifying key value pairs (for stage 1) that will override json settings; in format k1.k2.k3=val1;k1.k2=val2')
    parser.add_argument('--config_kvs_2', required=False, default=None, help='Allows specifying key value pairs (for stage 2) that will override json settings; in format k1.k2.k3=val1;k1.k2=val2')

    parser.add_argument('--sweep_value_name_a', required=False, default=None, type=str, help='Allows specifying a configuration value in k1.k2.k3 format for which we sweep over values (specified for all levels)')
    parser.add_argument('--sweep_values_a', required=False, default=None, type=str, help='Values for sweeping as a comma separated list (sorry, currently only for scalars)')

    parser.add_argument('--sweep_value_name_b', required=False, default=None, type=str, help='Allows specifying a configuration value in k1.k2.k3 format for which we sweep over values (specified for all levels; tensor product w/a)')
    parser.add_argument('--sweep_values_b', required=False, default=None, type=str, help='Values for sweeping as a comma separated list (sorry, currently only for scalars)')

    parser.add_argument('--sweep_value_name_c', required=False, default=None, type=str, help='Allows specifying a configuration value in k1.k2.k3 format for which we sweep over values (spefified for all levels; tensor product w/ a and b)')
    parser.add_argument('--sweep_values_c', required=False, default=None, type=str, help='Values for sweeping as a comma separated list (sorry, currently only for scalars)')

    parser.add_argument('--multi_gaussian_weights_stage_0', required=False, default=None, type=str, help='Convenience function to specify a multi-Gaussian weight as a string for stage 0: e.g., [0.0,0.0,0.0,1.0]')

    # run only certain parts of the pipline
    parser.add_argument('--run_optimization', action='store_true', help='If specified then only specified parts of the pipline are run')
    parser.add_argument('--run_visualization', action='store_true', help='If specified then only specified parts of the pipline are run')
    parser.add_argument('--run_validation', action='store_true', help='If specified then only specified parts of the pipline are run')
    parser.add_argument('--run_extra_validation', action='store_true', help='If specified then only specified parts of the pipline are run')

    parser.add_argument('--run_vizval_only_for_stage_2', action='store_true', help='If set to True the costly visualization and evaluation is only run for stage 2')

    args = parser.parse_args()

    if args.load_shared_parameters_from_previous_stage_nr and args.previous_dataset_config is None:
        raise ValueError('To specify a previous stage for the shared parameters requires a previous dataset configuration file')

    if args.nr_of_image_pairs==0:
        nr_of_image_pairs = None
    else:
        nr_of_image_pairs = args.nr_of_image_pairs

    parts_to_run = dict()
    parts_to_run['run_optimization'] = False
    parts_to_run['run_visualization'] = False
    parts_to_run['run_validation'] = False
    parts_to_run['run_extra_validation'] = False

    if args.run_optimization or args.run_visualization or args.run_validation or args.run_extra_validation:
        parts_to_run['run_optimization'] = args.run_optimization
        parts_to_run['run_visualization'] = args.run_visualization
        parts_to_run['run_validation'] = args.run_validation
        parts_to_run['run_extra_validation'] = args.run_extra_validation
    else:
        # by default everything runs
        parts_to_run['run_optimization'] = True
        parts_to_run['run_visualization'] = True
        parts_to_run['run_validation'] = True
        parts_to_run['run_extra_validation'] = True

    input_image_directory = None
    image_pair_config_pt = None
    output_base_directory = None
    validation_dataset_directory = None
    validation_dataset_name = None

    if args.main_synth_directory is not None:
        print('Specifying directories based on SYNTH main directory: {:s}'.format(args.main_synth_directory))

        validation_dataset_name = 'SYNTH'

        input_image_directory = os.path.join(args.main_synth_directory,'brain_affine_icbm')
        image_pair_config_pt = os.path.join(args.main_synth_directory,'used_image_pairs.pt')
        validation_dataset_directory = args.main_synth_directory

    # if this is not synth data, we first check if a JSON file was used to specify the directories
    elif args.dataset_config is not None:
        print('Reading dataset configuration from: {:s}'.format(args.dataset_config))
        data_params = pars.ParameterDict()
        save_dataset_config = False

        if not os.path.exists(args.dataset_config):
            print('INFO: config file {:s} does not exist. Creating a generic one.'.format(args.dataset_config))
            input_image_directory_default = 'synthetic_example_out/brain_affine_icbm'
            image_pair_config_pt_default = 'synthetic_example_out/used_image_pairs.pt'
            output_base_directory_default = 'experimental_results_synth_2d'
            validation_dataset_directory_default = 'synthetic_example_out'
            validation_dataset_name_default = 'SYNTH'
            save_dataset_config = True
        else:
            input_image_directory_default = None
            image_pair_config_pt_default = None
            output_base_directory_default = None
            validation_dataset_directory_default = None
            validation_dataset_name_default = None

            data_params.load_JSON(args.dataset_config)

        input_image_directory = _none_if_empty(data_params[('input_image_directory',input_image_directory_default,'directory where the input images live')])
        image_pair_config_pt = _none_if_empty(data_params[('image_pair_config_pt',image_pair_config_pt_default,'file which specifies the image pairs if desired')])
        output_base_directory = data_params[('output_base_directory',output_base_directory_default,'where the output should be stored')]
        validation_dataset_directory = _none_if_empty(data_params[('validation_dataset_directory',validation_dataset_directory_default,'the main directory containing the validation data')])
        validation_dataset_name = _none_if_empty(data_params[('validation_dataset',validation_dataset_name_default,'CUMC|MGH|LPBA|IBSR|SYNTH')])

        if save_dataset_config:
            data_params.write_JSON(args.dataset_config)
            print('Edit this configuration file and then run again')
            exit()

    # now read from the command line what is still missing
    if input_image_directory is None:
        if args.input_image_directory is None:
            raise ValueError('--input_image_directory needs to be specified')
        else:
            input_image_directory = args.input_image_directory

    if output_base_directory is None:
        output_base_directory = args.output_base_directory

    if image_pair_config_pt is None:
        image_pair_config_pt = args.image_pair_config_pt

    if validation_dataset_directory is None:
        validation_dataset_directory = args.validation_dataset_directory

    if validation_dataset_name is None:

        validation_datasets = ['CUMC','MGH','LPBA','IBSR','SYNTH']

        if args.dataset is None:
            print('INFO: No dataset was specied. No validation is performed. Use --dataset option if validation is desired or speficy via --dataset_config')
            validation_dataset_name = None
        else:
            if args.dataset in validation_datasets:
                validation_dataset_name = args.dataset
            else:
                raise ValueError('Dataset needs to be [CUMC|MGH|LPBA|IBSR|SYNTH], but got ' + args.dataset)

    main_json = args.config # e.g., 'test2d_025.json'

    # directory to move results to after computation
    move_to_directory = args.move_to_directory

    # which GPU to run on
    cuda_visible_devices = args.cuda_device

    # random seed
    seed = args.seed

    # epochs
    if args.nr_of_epochs is None:
        nr_of_epochs = [1,1,1]
    else:
        nr_of_epochs = [int(item) for item in args.nr_of_epochs.split(',')]

    if len(nr_of_epochs)!=3:
        raise ValueError('Number of epochs needs to be defined for the three different stages')

    if args.stage_nr is None:
        stage_nr = [0, 1, 2]
    else:
        stage_nr = [int(item) for item in args.stage_nr.split(',')]

    # sweep values setup
    sweep_values = get_sweep_values(args.sweep_value_name_a,args.sweep_values_a,
                                    args.sweep_value_name_b, args.sweep_values_b,
                                    args.sweep_value_name_c, args.sweep_values_c)


    # check if there is a previous dataset config file specified

    #load_shared_parameters_from_previous_stage_nr = False
    previous_output_base_directory = None
    if args.previous_dataset_config is not None:
        print('Reading previous dataset configuration from: {:s}'.format(args.previous_dataset_config))
        previous_data_params = pars.ParameterDict()

        if not os.path.exists(args.previous_dataset_config):
            raise ValueError('Could not find previous dataset config file: {}'.format(args.previous_dataset_config))
        else:
            print('Reading previous dataset config file: {}'.format(args.previous_dataset_config))

            previous_data_params.load_JSON(args.previous_dataset_config)
            previous_output_base_directory = previous_data_params[('output_base_directory','','previous output directory')]
            if previous_output_base_directory=='':
                raise ValueError('Could not find key output_base_directory in file {}'.format(args.previous_dataset_config))

            # todo: probably remove this part
            # if args.load_shared_parameters_from_previous_stage_nr is None:
            #     load_shared_parameters_from_previous_stage_nr = None
            # else:
            #     specified_previous_stage_nr = [int(item) for item in args.load_shared_parameters_from_previous_stage_nr.split(',')]
            #     if len(specified_previous_stage_nr)==1:
            #         if specified_previous_stage_nr[0] in [0,1,2]:
            #             load_shared_parameters_from_previous_stage_nr = specified_previous_stage_nr[0]
            #         else:
            #             raise ValueError('The previous stage number needs to be in {0,1,2}. Instead it was: {}'.format(specified_previous_stage_nr[0]))
            #     else:
            #         raise ValueError('Exactly one previous stage nr needs to be specified. Instead the following was specified: {}'.format(args.load_shared_parameters_from_previous_stage_nr))

    else:
        if args.load_shared_parameters_from_previous_stage_nr:
            raise ValueError('Previous dataset config was not specified use --previous_dataset_config')


    # what stages we want to optimize parameters for (typically these are all; but for evaluation on
    # a separate test set one might want to restrict this)
    only_optimize_over_registration_parameters_for_stage_nr = args.only_optimize_over_registration_parameters_for_stage_nr

    if args.load_shared_parameters_from_previous_stage_nr:
        if args.previous_postfix is None:
            previous_postfix = 'training'
        else:
            previous_postfix = args.previous_postfix
    else:
        previous_postfix = None

    if args.postfix is None:
        if previous_postfix is not None:
            postfix = 'init_from_' + previous_postfix
        else:
            postfix = 'training'
    else:
        postfix = args.postfix

    if previous_postfix is not None:
        if previous_postfix==postfix:
            raise ValueError('previous_postfix={} and postfix={} are not allowed to be the same to avoid overwriting results; use --postfix and --previous_postfix to specify differt values'.format(previous_postfix,postfix))

    # key value pairs
    kvos_str = dict()
    kvos_str['default'] = args.config_kvs
    kvos_str[0] = args.config_kvs_0
    kvos_str[1] = args.config_kvs_1
    kvos_str[2] = args.config_kvs_2

    if args.multi_gaussian_weights_stage_0 is not None:
        # multi-Gaussian weight specified for stage 0
        if kvos_str[0] is not None:
            kvos_str[0] += '\;model.registration_model.forward_model.smoother.multi_gaussian_weights={:s}'.format(args.multi_gaussian_weights_stage_0)
        else:
            kvos_str[0] = 'model.registration_model.forward_model.smoother.multi_gaussian_weights={:s}'.format(args.multi_gaussian_weights_stage_0)

    #todo: make this more generic
    # These are our default configurations (can be adapted as desired)
    kvo = dict()
    #kvo['optimizer.sgd.individual.lr'] = 0.01
    #kvo['optimizer.sgd.shared.lr'] = 0.01

    # check if we only want to create some prints for publications
    compute_only_pair_nr = args.compute_only_pair_nr

    if args.create_clean_publication_print_for_pair_nr is not None:
        print('INFO: Only publication output will be generated as --create_clean_publication_print_for_pair_nr was specified')
        compute_only_pair_nr = args.create_clean_publication_print_for_pair_nr
        parts_to_run['run_optimization'] = False
        parts_to_run['run_visualization'] = True
        parts_to_run['run_validation'] = False
        parts_to_run['run_extra_validation'] = True

    if args.only_recompute_validation_measures:
        # so far only the visualization part is supported here, as we use it to recompute the boxplot and the Jacobian measures
        parts_to_run['run_optimization'] = False
        parts_to_run['run_visualization'] = True
        parts_to_run['run_validation'] = False
        parts_to_run['run_extra_validation'] = False

    # determine of all parts will be run
    all_parts_will_be_run = True
    if (parts_to_run['run_optimization'] == False) \
            or (parts_to_run['run_visualization'] == False) \
            or (parts_to_run['run_validation'] == False) \
            or (parts_to_run['run_extra_validation'] == False):
        all_parts_will_be_run = False

    if sweep_values is None:
        # no sweeping, just do a regular call

        for stage in stage_nr:

            parts_to_run_for_current_stage = parts_to_run.copy()
            if args.run_vizval_only_for_stage_2 and stage!=2:
                parts_to_run_for_current_stage['run_visualization'] = False
                parts_to_run_for_current_stage['run_validation'] = False
                parts_to_run_for_current_stage['run_extra_validation'] = False

            run(stage=stage,
                nr_of_epochs=nr_of_epochs[stage],
                main_json=main_json,
                image_pair_config_pt=image_pair_config_pt,
                nr_of_image_pairs=nr_of_image_pairs,
                input_image_directory=input_image_directory,
                output_base_directory=output_base_directory,
                postfix=postfix,
                previous_postfix=previous_postfix,
                validation_dataset_directory=validation_dataset_directory,
                validation_dataset_name=validation_dataset_name,
                previous_output_base_directory=previous_output_base_directory,
                only_optimize_over_registration_parameters_for_stage_nr=
                only_optimize_over_registration_parameters_for_stage_nr,
                load_shared_parameters_from_previous_stage_nr=args.load_shared_parameters_from_previous_stage_nr,
                move_to_directory=move_to_directory,
                key_value_overwrites=kvo,
                string_key_value_overwrites=kvos_str,
                seed=seed,
                cuda_visible_devices=cuda_visible_devices,
                parts_to_run=parts_to_run_for_current_stage,
                pre_command=args.precommand,
                compute_only_pair_nr=compute_only_pair_nr,
                only_run_stage0_with_unchanged_config=args.only_run_stage0_with_unchanged_config,
                skip_stage_1_and_start_stage_2_from_stage_0=args.skip_stage_1_and_start_stage_2_from_stage_0,
                only_recompute_validation_measures=args.only_recompute_validation_measures
                )

        if all_parts_will_be_run:
            move_output_directory(move_to_directory=move_to_directory,
                                  output_base_directory=output_base_directory,
                                  postfix=postfix)
        else:
            if move_to_directory is not None:
                print('INFO: Ignored move request to directory {:s} as not all components were run'.format(
                    move_to_directory))

    else:
        # we sweep over all the entries

        nr_of_sweep_values = len(sweep_values)
        for counter,s in enumerate(sweep_values):
            current_kvs = s['kvs']
            current_label = s['label']
            current_postfix = postfix + '_' + current_label

            if previous_postfix is not None:
                current_previous_postfix = previous_postfix + '_' + current_label
            else:
                current_previous_postfix = None

            combined_kvs = merge_kvs(kvo,current_kvs)

            print('Computing sweep {:d}/{:d} for {:s}'.format(counter+1,nr_of_sweep_values,current_label))

            for stage in stage_nr:

                parts_to_run_for_current_stage = parts_to_run.copy()
                if args.run_vizval_only_for_stage_2 and stage != 2:
                    parts_to_run_for_current_stage['run_visualization'] = False
                    parts_to_run_for_current_stage['run_validation'] = False
                    parts_to_run_for_current_stage['run_extra_validation'] = False

                run(stage=stage,
                    nr_of_epochs=nr_of_epochs[stage],
                    main_json=main_json,
                    image_pair_config_pt=image_pair_config_pt,
                    nr_of_image_pairs=nr_of_image_pairs,
                    input_image_directory=input_image_directory,
                    output_base_directory=output_base_directory,
                    postfix=current_postfix,
                    previous_postfix=current_previous_postfix,
                    validation_dataset_directory=validation_dataset_directory,
                    validation_dataset_name=validation_dataset_name,
                    previous_output_base_directory=previous_output_base_directory,
                    only_optimize_over_registration_parameters_for_stage_nr=
                    only_optimize_over_registration_parameters_for_stage_nr,
                    load_shared_parameters_from_previous_stage_nr=args.load_shared_parameters_from_previous_stage_nr,
                    move_to_directory=move_to_directory,
                    key_value_overwrites=combined_kvs,
                    string_key_value_overwrites=kvos_str,
                    seed=seed,
                    cuda_visible_devices=cuda_visible_devices,
                    pre_command=args.precommand,
                    parts_to_run=parts_to_run_for_current_stage,
                    compute_only_pair_nr=compute_only_pair_nr,
                    only_run_stage0_with_unchanged_config=args.only_run_stage0_with_unchanged_config,
                    skip_stage_1_and_start_stage_2_from_stage_0=args.skip_stage_1_and_start_stage_2_from_stage_0,
                    only_recompute_validation_measures=args.only_recompute_validation_measures
                    )

            if all_parts_will_be_run:
                move_output_directory(move_to_directory=move_to_directory,
                                      output_base_directory=output_base_directory,
                                      postfix=current_postfix)
            else:
                if move_to_directory is not None:
                    print('INFO: Ignored move request to directory {:s} as not all components were run'.format(move_to_directory))












