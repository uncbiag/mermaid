from __future__ import print_function
from builtins import str
from builtins import range

from __future__ import print_function
from __future__ import absolute_import

from . import command_line_execution_tools as ce

import set_pyreg_paths
import pyreg.module_parameters as pars

import os

def _make_arg_list(args):
    arg_list = ''
    for k in args:
        arg_list.append('--' + str(k))
        arg_list.appedn( str(args[k]) )

    return arg_list

def run_optimization(stage,nr_of_epochs,image_pair_config_pt,input_image_directory,output_directory,
                     main_json,seed,cuda_visible_devices=None):

    config_kvs = 'model.optimizer.sgd.individual.lr=0.01;model.optimizer.sgd.shared.lr=0.01'
    if stage==0:
        stage0_weights = [0.0,0.0,0.3,0.7]
        config_kvs = 'model.registration_model.forward_model.smoother.multi_gaussian_weights={:s}'.format(stage0_weights)
        all_nr_of_epochs = '{:s},1,1'.format(nr_of_epochs)
    elif stage==1:
        all_nr_of_epochs = '1,{:s},1'.format(nr_of_epochs)
    elif stage==2:
        all_nr_of_epochs = '1,1,{:s}'.format(nr_of_epochs)
    else:
        raise ValueError('Unknown stage; stage needs to be 0,1, or 2')

    args = {'image_pair_config_pt': image_pair_config_pt,
            'input_image_directory': input_image_directory,
            'output_directory': output_directory,
            'nr_of_epochs': all_nr_of_epochs,
            'config': main_json,
            'config_kvs': config_kvs,
            'stage': stage,
            'seed': seed,
            }

    # now run the command
    cmd_arg_list = _make_arg_list(args)
    current_command = 'python multi_stage_smoother_learning.py'

    ce.executeCommand(current_command, cmd_arg_list)

def run_visualization(stage,output_directory,main_json,cuda_visible_devices=None):

    args = {'config': main_json,
            'output_directory': output_directory,
            'stage_nr': stage,
            'do_not_visualize': ''}

    # now run the command
    cmd_arg_list = _make_arg_list(args)
    current_command = 'python visualize_multi_stage.py'

    ce.executeCommand(current_command, cmd_arg_list)

def run_validation(stage,output_directory,validation_dataset_directory,cuda_visible_devices=None):

    args = {'stage_nr':stage,
            'output_directory': output_directory,
            'dataset_directory': validation_dataset_directory,
            'dataset': 'SYNTH',
            'do_not_visualize': '',
            'save_overlap_filename': 'overlaps.txt'}

    # now run the command
    cmd_arg_list = _make_arg_list(args)
    current_command = 'python compute_validation_results.py'
    ce.executeCommand(current_command, cmd_arg_list)

def run_extra_validation(stage,input_image_directory,output_directory,main_json,cuda_visible_devices=None):

    args = {'stage_nr': stage,
            'input_image_directory': input_image_directory,
            'output_directory': output_directory,
            'config': main_json,
            'do_not_visualize': ''}

    # now run the command
    cmd_arg_list = _make_arg_list(args)
    current_command = 'python extra_validation_for_synthetic_test_cases.py'
    ce.executeCommand(current_command, cmd_arg_list)

def run(stage,nr_of_epochs,main_json,
        image_pair_config_pt,
        input_image_dir,
        output_base_dir,postfix,
        validation_dataset_dir,
        cuda_visible_devices=None):

    seed = 1234
    output_dir = os.path.join(output_base_dir, 'out_' + postfix )

    run_optimization(stage=stage,
                     nr_of_epochs=nr_of_epochs,
                     image_pair_config_pt=image_pair_config_pt,
                     input_image_directory=input_image_dir,
                     output_directory=output_dir,
                     main_json=main_json,
                     seed=seed,
                     cuda_visible_devices=cuda_visible_devices)

    run_visualization(stage=stage,
                      output_directory=output_dir,
                      main_json=main_json,
                      cuda_visible_devices=cuda_visible_devices)

    run_validation(stage=stage,
                   output_directory=output_dir,
                   validation_dataset_directory=validation_dataset_dir,
                   cuda_visible_devices=cuda_visible_devices)

    run_extra_validation(stage=stage,
                         input_image_directory=input_image_dir,
                         output_directory=output_dir,
                         main_json=main_json,
                         cuda_visible_devices=cuda_visible_devices)




    #if __name__ == "__main__":
#
#    import argparse
#
#    parser = argparse.ArgumentParser(description='Registers batches of two images based on OASIS data (for testing)')
#
#    parser.add_argument('--model_name', required=False, default='svf_vector_momentum_map',help='model that should be used for the registration')
#
#    parser.add_argument('--input_image_directory', required=True, help='Directory where all the images are')


#MAIN_JSON=$1
#CUDA_VISIBLE_DEVICES=$2
#POSTFIX=$3
#STAGE=$4
#NREPOCHS=$5

#SEED=1234

#PTFILE="/ssd_data_mn/complex_synthetic_example_out/used_image_pairs.pt"
#INPUT_DIR="/ssd_data_mn/complex_synthetic_example_out/brain_affine_icbm"
#OUTPUT_BASE_DIR="/ssd_data_mn/experimental_results_synth_2d"
#VALIDATION_DATASET_DIR="/ssd_data_mn/complex_synthetic_example_out"

#export CUDA_VISIBLE_DEVICES










