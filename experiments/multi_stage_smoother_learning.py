import set_pyreg_paths
import pyreg.config_parser as cp

# first do the torch imports
import torch
import multiprocessing as mp

import pyreg.simple_interface as si
import pyreg.module_parameters as pars

import glob
import os

import random

import shutil

def do_registration(source_images,target_images,model_name,output_directory,
                    nr_of_epochs,nr_of_iterations,map_low_res_factor,
                    visualize_step,json_in,json_out,
                    optimize_over_deep_network=False,
                    optimize_over_weights=False,
                    start_from_previously_saved_parameters=True):

    reg = si.RegisterImagePair()

    # load the json file and make necessary modifications
    params_in = pars.ParameterDict()
    print('Loading settings from file: ' + json_in)
    params_in.load_JSON(json_in)

    if map_low_res_factor is None:
        map_low_res_factor = params_in['model']['deformation'][('map_low_res_factor',1.0,'low res factor for the map')]

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
                        map_low_res_factor=map_low_res_factor,
                        visualize_step=visualize_step,
                        json_config_out_filename=json_out,
                        use_batch_optimization=True,
                        params=params_in)

def get_n_pairwise_image_combinations(input_directory,n=10,no_random_shuffle=False,suffix=None):

    if suffix is None:
        all_files = glob.glob(os.path.join(input_directory,'*.*'))
    else:
        all_files = glob.glob(os.path.join(input_directory,'*.'+suffix))

    nr_of_files = len(all_files)

    source_files = []
    target_files = []
    source_ids = []
    target_ids = []

    current_n = 0
    for i in range(nr_of_files):
        c_source = all_files[i]
        for j in range(nr_of_files):
            if i!=j:
                c_target = all_files[j]
                if no_random_shuffle:
                    print(str(current_n) + ': Source: ' + c_source + ' -> target: ' + c_target)
                source_files.append(c_source)
                target_files.append(c_target)
                source_ids.append(i)
                target_ids.append(j)
                current_n += 1
                if n is not None:
                    if current_n>=n and no_random_shuffle:
                        return source_files,target_files,source_ids,target_ids

    if n<nr_of_files and not no_random_shuffle:
        # we now do a random selection
        ind = range(len(source_ids))
        random.shuffle(ind)

        source_files_ret = []
        target_files_ret = []
        source_ids_ret = []
        target_ids_ret = []

        for i in range(n):
            current_ind = ind[i]
            source_files_ret.append(source_files[current_ind])
            target_files_ret.append(target_files[current_ind])
            source_ids_ret.append(source_ids[current_ind])
            target_ids_ret.append(target_ids[current_ind])
            print(str(i) + ': Source: ' + source_files[current_ind] + ' -> target: ' + target_files[current_ind])

        return source_files_ret,target_files_ret,source_ids_ret,target_ids_ret

    else:
        return source_files,target_files,source_ids,target_ids


if __name__ == "__main__":

    torch.set_num_threads(mp.cpu_count())

    import argparse

    parser = argparse.ArgumentParser(description='Registers batches of two images based on OASIS data (for testing)')

    parser.add_argument('--model_name', required=False, default='svf_scalar_momentum_map',help='model that should be used for the registration')

    parser.add_argument('--input_image_directory', required=True, help='Directory where all the images are')
    parser.add_argument('--output_directory', required=True, help='Where the output is stored')
    parser.add_argument('--nr_of_image_pairs', required=False, type=int, default=20, help='number of image pairs that will be used; if not set all pairs will be used')
    parser.add_argument('--map_low_res_factor', required=False, type=float, default=None, help='map low res factor')

    parser.add_argument('--nr_of_epochs', required=False,type=str, default=None, help='number of epochs for the three stages as a comma separated list')
    parser.add_argument('--nr_of_iterations_per_batch', required=False,type=int, default=5, help='number of iterations per mini-batch')

    parser.add_argument('--retain_intermediate_stage_results',required=False, default=True, help='If set to true, will backup results between stages to not overwrite them')

    parser.add_argument('--visualize', action='store_true', help='visualizes the output')
    parser.add_argument('--visualize_step', required=False, type=int, default=20, help='Number of iterations between visualization output')

    parser.add_argument('--noshuffle', action='store_true', help='Does not use dataset shuffling if using a subset of the images via --nr_of_image_pairs')

    parser.add_argument('--suffix', required=False, default=None, help='Allows setting a suffix for the files to read; e.g., specifiy --suffix nii ant only nii files will be considered')

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

    source_images,target_images,source_ids,target_ids = get_n_pairwise_image_combinations(args.input_image_directory,
                                                                    nr_of_image_pairs,
                                                                    no_random_shuffle=args.noshuffle,
                                                                    suffix=args.suffix)

    if not os.path.exists(args.output_directory):
        print('Creating output directory: ' + args.output_directory)
        os.makedirs(args.output_directory)

    # first save how the data was created
    d = dict()
    d['source_images'] = source_images
    d['target_images'] = target_images
    d['source_ids'] = source_ids
    d['target_ids'] = target_ids

    used_image_pairs_filename_pt = os.path.join(args.output_directory,'used_image_pairs.pt')
    used_image_pairs_filename_txt = os.path.join(args.output_directory,'used_image_pairs.txt')

    torch.save(d,used_image_pairs_filename_pt)

    # also save it as a text file for easier readability
    f = open(used_image_pairs_filename_txt, 'w')
    f.write('Image pair id, source id, target id, source file name, target file name\n')
    for i in range(len(source_images)):
        out_str = str(i) + ', '
        out_str += str(source_ids[i]) + ', '
        out_str += str(target_ids[i]) + ', '
        out_str += str(source_images[i]) + ', '
        out_str += str(target_images[i])
        out_str += '\n'

        f.write(out_str)
    f.close()

    print('Running stage 1: optimize only using given weights')
    in_json = args.config
    out_json_stage_1 = os.path.join(os.path.split(args.config)[0],'out_stage_1_' + os.path.split(args.config)[1])

    do_registration(
        source_images=source_images,
        target_images=target_images,
        model_name=args.model_name,
        output_directory=args.output_directory,
        nr_of_epochs=nr_of_epochs[0],
        nr_of_iterations=args.nr_of_iterations_per_batch,
        map_low_res_factor=args.map_low_res_factor,
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
    out_json_stage_2 = os.path.join(os.path.split(args.config)[0],'out_stage_2_' + os.path.split(args.config)[1])

    do_registration(
        source_images=source_images,
        target_images=target_images,
        model_name=args.model_name,
        output_directory=args.output_directory,
        nr_of_epochs=nr_of_epochs[1],
        nr_of_iterations=args.nr_of_iterations_per_batch,
        map_low_res_factor=args.map_low_res_factor,
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
    out_json_stage_3 = os.path.join(os.path.split(args.config)[0],'out_stage_3_' + os.path.split(args.config)[1])

    do_registration(
        source_images=source_images,
        target_images=target_images,
        model_name=args.model_name,
        output_directory=args.output_directory,
        nr_of_epochs=nr_of_epochs[2],
        nr_of_iterations=args.nr_of_iterations_per_batch,
        visualize_step=visualize_step,
        map_low_res_factor=args.map_low_res_factor,
        json_in=in_json,
        json_out=out_json_stage_3,
        optimize_over_deep_network=True,
        optimize_over_weights=False,
        start_from_previously_saved_parameters=True
    )

