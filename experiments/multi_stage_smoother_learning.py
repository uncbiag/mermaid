import set_pyreg_paths
import pyreg.config_parser as cp

# first do the torch imports
import torch
import multiprocessing as mp

import pyreg.simple_interface as si
import pyreg.module_parameters as pars

import glob
import os

import shutil

def do_registration(source_images,target_images,model_name,output_directory,
                    nr_of_epochs,nr_of_iterations,visualize_step,json_in,json_out,
                    optimize_over_deep_network=False,
                    optimize_over_weights=False,
                    start_from_previously_saved_parameters=True):

    reg = si.RegisterImagePair()

    # load the json file and make necessary modifications
    params_in = pars.ParameterDict()
    print('Loading settings from file: ' + json_in)
    params_in.load_JSON(json_in)

    params_in['optimizer']['batch_settings']['nr_of_epochs'] = nr_of_epochs
    params_in['optimizer']['batch_settings']['parameter_output_dir'] = output_directory
    params_in['optimizer']['batch_settings']['start_from_previously_saved_parameters'] = start_from_previously_saved_parameters

    params_in['model']['registration_model']['forward_model']['smoother']['type'] = 'learned_multiGaussianCombination'
    params_in['model']['registration_model']['forward_model']['smoother']['optimize_over_deep_network'] = optimize_over_deep_network
    params_in['model']['registration_model']['forward_model']['smoother']['optimize_over_smoother_stds'] = False
    params_in['model']['registration_model']['forward_model']['smoother']['optimize_over_smoother_weights'] = optimize_over_weights
    params_in['model']['registration_model']['forward_model']['smoother']['start_optimize_over_smoother_parameters_at_iteration'] = 0

    spacing = None
    reg.register_images(source_images, target_images, spacing,
                        model_name=model_name,
                        nr_of_iterations=nr_of_iterations,
                        visualize_step=visualize_step,
                        json_config_out_filename=json_out,
                        use_batch_optimization=True,
                        params=params_in)

def get_n_pairwise_image_combinations(input_directory,n=10):
    all_files = glob.glob(os.path.join(input_directory,'*.*'))
    nr_of_files = len(all_files)

    source_files = []
    target_files = []

    current_n = 0
    for i in range(nr_of_files):
        c_source = all_files[i]
        for j in range(nr_of_files):
            if i!=j:
                c_target = all_files[j]
                print(str(current_n) + ': Source: ' + c_source + ' -> target: ' + c_target)
                source_files.append(c_source)
                target_files.append(c_target)
                current_n += 1
                if n is not None:
                    if current_n>=n:
                        return source_files,target_files

    return source_files,target_files


if __name__ == "__main__":

    torch.set_num_threads(mp.cpu_count())

    import argparse

    parser = argparse.ArgumentParser(description='Registers batches of two images based on OASIS data (for testing)')

    parser.add_argument('--model_name', required=False, default='svf_scalar_momentum_map',help='model that should be used for the registration')

    parser.add_argument('--input_image_directory', required=True, help='Directory where all the images are')
    parser.add_argument('--output_directory', required=True, help='Where the output is stored')
    parser.add_argument('--nr_of_image_pairs', required=False, type=int, default=20, help='number of image pairs that will be used; if not set all pairs will be used')

    parser.add_argument('--nr_of_epochs', required=False,type=str, default=None, help='number of epochs for the three stages as a comma separated list')
    parser.add_argument('--nr_of_iterations_per_batch', required=False,type=int, default=5, help='number of iterations per mini-batch')

    parser.add_argument('--retain_intermediate_stage_results',required=False, default=True, help='If set to true, will backup results between stages to not overwrite them')

    parser.add_argument('--visualize', action='store_true', help='visualizes the output')
    parser.add_argument('--visualize_step', required=False, type=int, default=20, help='Number of iterations between visualization output')

    parser.add_argument('--config', required=True, default=None, help='Configuration file to read in')

    args = parser.parse_args()

    if args.nr_of_epochs is None:
        nr_of_epochs = [1,1,1]
    else:
        nr_of_epochs = [int(item) for item in args.nr_of_epochs.split(',')]

    if len(nr_of_epochs)!=3:
        raise ValueError('Number of epochs needs to be defined for the three different stages')

    if args.visualize:
        visualize_step = args.visualize_step
    else:
        visualize_step = None

    if args.nr_of_image_pairs==0:
        nr_of_image_pairs = None
    else:
        nr_of_image_pairs = args.nr_of_image_pairs

    source_images,target_images = get_n_pairwise_image_combinations(args.input_image_directory,nr_of_image_pairs)

    if not os.path.exists(args.output_directory):
        print('Creating output directory: ' + args.output_directory)
        os.makedirs(args.output_directory)

    print('Running stage 1: optimize only using given weights')
    in_json = args.config
    out_json_stage_1 = 'out_stage_1_' + args.config

    do_registration(
        source_images=source_images,
        target_images=target_images,
        model_name=args.model_name,
        output_directory=args.output_directory,
        nr_of_epochs=nr_of_epochs[0],
        nr_of_iterations=args.nr_of_iterations_per_batch,
        visualize_step=visualize_step,
        json_in=in_json,
        json_out=out_json_stage_1,
        optimize_over_deep_network=False,
        optimize_over_weights=False,
        start_from_previously_saved_parameters=False
    )

    if args.retain_intermediate_stage_results:
        print('Backing up the stage 1 results')
        backup_dir = os.path.realpath(args.output_directory)+'_after_stage_1'
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(args.output_directory, backup_dir)

    print('Running stage 2: now continue optimizing, but optimizing over the global weights')
    in_json = out_json_stage_1
    out_json_stage_2 = 'out_stage_2_' + args.config

    do_registration(
        source_images=source_images,
        target_images=target_images,
        model_name=args.model_name,
        output_directory=args.output_directory,
        nr_of_epochs=nr_of_epochs[1],
        nr_of_iterations=args.nr_of_iterations_per_batch,
        visualize_step=visualize_step,
        json_in=in_json,
        json_out=out_json_stage_2,
        optimize_over_deep_network=False,
        optimize_over_weights=True,
        start_from_previously_saved_parameters=True
    )

    if args.retain_intermediate_stage_results:
        print('Backing up the stage 2 results')
        backup_dir = os.path.realpath(args.output_directory) + '_after_stage_2'
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(args.output_directory, backup_dir)

    print('Running stage 3: now optimize over the network (keeping everything else fixed)')
    in_json = out_json_stage_2
    out_json_stage_3 = 'out_stage_3_' + args.config

    do_registration(
        source_images=source_images,
        target_images=target_images,
        model_name=args.model_name,
        output_directory=args.output_directory,
        nr_of_epochs=nr_of_epochs[2],
        nr_of_iterations=args.nr_of_iterations_per_batch,
        visualize_step=visualize_step,
        json_in=in_json,
        json_out=out_json_stage_3,
        optimize_over_deep_network=True,
        optimize_over_weights=False,
        start_from_previously_saved_parameters=True
    )

