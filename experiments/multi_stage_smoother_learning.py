import set_pyreg_paths
import pyreg.config_parser as cp

# first do the torch imports
import torch
import multiprocessing as mp

import pyreg.simple_interface as si
import pyreg.module_parameters as pars
from pyreg.data_wrapper import USE_CUDA

import glob
import os
import ast

import random

import shutil

def do_registration(source_images,target_images,model_name,output_directory,
                    nr_of_epochs,nr_of_iterations,map_low_res_factor,
                    visualize_step,json_in,json_out,
                    optimize_over_deep_network=False,
                    optimize_over_weights=False,
                    freeze_parameters=False,
                    start_from_previously_saved_parameters=True,
                    args_kvs=None):

    reg = si.RegisterImagePair()

    # load the json file if it is a file and make necessary modifications
    if type(json_in)==pars.ParameterDict:
        params_in=json_in
    else:
        params_in = pars.ParameterDict()
        print('Loading settings from file: ' + json_in)
        params_in.load_JSON(json_in)

    add_key_value_pairs_to_params(params_in, args_kvs)

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
    params_in['model']['registration_model']['forward_model']['smoother']['freeze_parameters'] = freeze_parameters

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


def add_key_value_pairs_to_params(params,kvs):
    """
    Allows overwriting values in the parameter structure
    :param params: ParameterDict object
    :param kvs: key value pairs string
    :return: key value pairs will now be part of params
    """

    if kvs is not None:

        # first break it up into the different key-value arguments
        d = dict()
        for item in kvs.split(','):
            kv = item.split('=')
            if len(kv)!=2:
                print('Cannot parse key value pair ' + item + ' ; expected form key:value, but got more than one ":"; ignoring pair')
            else:
                d[kv[0]]=kv[1]

        # now go through the dictionary and create the proper entries for the parameter dict

        for key in d:
            val = d[key]
            sub_keys = key.split('.')
            nr_of_sub_keys = len(sub_keys)
            current_pars = params
            for i in range(nr_of_sub_keys-1):
                current_pars = current_pars[sub_keys[i]]
            # and now the last one
            current_pars[sub_keys[-1]] = ast.literal_eval(val)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Registers batches of two images based on OASIS data (for testing)')

    parser.add_argument('--model_name', required=False, default='svf_vector_momentum_map',help='model that should be used for the registration')

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

    parser.add_argument('--stage_nr', required=False, type=str, default=None, help='Which stages should be run {0,1,2} as a comma separated list')

    parser.add_argument('--seed', required=False, type=int, default=None, help='Sets the random seed which affects data shuffling')

    parser.add_argument('--do_not_read_used_images_from_file', action='store_true', help='If set the image pairs are recomputed. Use with EXTREME care as stage/frozen results may become inconsistent')

    parser.add_argument('--frozen_nr_of_epochs', required=False,type=str, default=None, help='number of epochs to run the three stages with frozen parameters (for refinement)')
    parser.add_argument('--only_compute_frozen_epochs', action='store_true', help='if specified will not redo the optimization, but will only compute the frozen results based on previous optimizatin results')

    parser.add_argument('--config_kvs', required=False, default=None, help='Allows specifying key value pairs that will override json settings; in format k1.k2.k3=val1, k1.k2=val2')

    args = parser.parse_args()

    if args.seed is not None:
        print('Setting the random seed to {:}'.format(args.seed))
        random.seed(args.seed)

    print('Loading settings from file: ' + args.config)
    params = pars.ParameterDict()
    params.load_JSON(args.config)

    torch.set_num_threads(mp.cpu_count())

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

    compute_frozen_epochs = False
    if args.frozen_nr_of_epochs is None:
        frozen_nr_of_epochs = [0,0,0]
    else:
        compute_frozen_epochs = True
        frozen_nr_of_epochs = [int(item) for item in args.frozen_nr_of_epochs.split(',')]


    if args.visualize:
        visualize_step = args.visualize_step
    else:
        visualize_step = None

    if args.nr_of_image_pairs==0:
        nr_of_image_pairs = None
    else:
        nr_of_image_pairs = args.nr_of_image_pairs

    if not os.path.exists(args.output_directory):
        print('Creating output directory: ' + args.output_directory)
        os.makedirs(args.output_directory)

    results_output_directory = os.path.join(args.output_directory, 'results')
    if os.path.exists(results_output_directory):
        print('Removing temporary results in directory ' + results_output_directory)
        shutil.rmtree(results_output_directory)

    print('Creating results output directory: ' + results_output_directory)
    os.makedirs(results_output_directory)

    used_image_pairs_filename_pt = os.path.join(args.output_directory, 'used_image_pairs.pt')
    used_image_pairs_filename_txt = os.path.join(args.output_directory, 'used_image_pairs.txt')

    create_new_image_pairs = False

    if args.do_not_read_used_images_from_file:
        if os.path.isfile(used_image_pairs_filename_pt):
            print('WARNING! WARNING! image pair file ' + used_image_pairs_filename_pt + ' exists, but will not be used')

        create_new_image_pairs = True

    else:
        if os.path.isfile(used_image_pairs_filename_pt):

            print('INFO: Loading image pair configuration from: ' + used_image_pairs_filename_pt)

            up = torch.load(used_image_pairs_filename_pt)
            source_images = up['source_images']
            target_images = up['target_images']
            source_ids = up['source_ids']
            target_ids = up['target_ids']
        else:
            create_new_image_pairs = True

    if create_new_image_pairs:

        print('INFO: creating new image pairs')

        source_images, target_images, source_ids, target_ids = get_n_pairwise_image_combinations(
            args.input_image_directory,
            nr_of_image_pairs,
            no_random_shuffle=args.noshuffle,
            suffix=args.suffix)

        # now save how the data was created
        d = dict()
        d['source_images'] = source_images
        d['target_images'] = target_images
        d['source_ids'] = source_ids
        d['target_ids'] = target_ids

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

    #
    # Stage 0 (fixed weights) starts here
    #

    out_json_stage_0 = os.path.join(args.output_directory, 'out_stage_0_' + os.path.split(args.config)[1])

    if 0 in stage_nr:

        if not args.only_compute_frozen_epochs:

            print('Running stage 0: optimize only using given weights')
            in_json = args.config

            do_registration(
                source_images=source_images,
                target_images=target_images,
                model_name=args.model_name,
                output_directory=results_output_directory,
                nr_of_epochs=nr_of_epochs[0],
                nr_of_iterations=args.nr_of_iterations_per_batch,
                map_low_res_factor=args.map_low_res_factor,
                visualize_step=visualize_step,
                json_in=in_json,
                json_out=out_json_stage_0,
                optimize_over_deep_network=False,
                optimize_over_weights=False,
                freeze_parameters=False,
                start_from_previously_saved_parameters=False,
                args_kvs=args.config_kvs
            )

            if args.retain_intermediate_stage_results:
                print('Backing up the stage 0 results')
                backup_dir = os.path.realpath(results_output_directory)+'_after_stage_0'
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                shutil.copytree(results_output_directory, backup_dir)

        if compute_frozen_epochs:

            print('Computing {:} frozen epochs for stage 0'.format(frozen_nr_of_epochs[0]))
            in_json = out_json_stage_0
            frozen_out_json_stage_0 = os.path.join(args.output_directory, 'frozen_out_stage_0_' + os.path.split(args.config)[1])

            if args.only_compute_frozen_epochs:
                # copy the earlier results of stage 0 into the current results directory
                # (otherwise they are already in this directory and we can continue from there)
                print('Restoring stage 0 results for continuation')
                if os.path.exists(results_output_directory):
                    shutil.rmtree(results_output_directory)
                backup_dir = os.path.realpath(results_output_directory) + '_after_stage_0'
                if os.path.exists(backup_dir):
                    shutil.copytree(backup_dir,results_output_directory)
                else:
                    raise ValueError('could not restore stage 0 results')

            do_registration(
                source_images=source_images,
                target_images=target_images,
                model_name=args.model_name,
                output_directory=results_output_directory,
                nr_of_epochs=frozen_nr_of_epochs[0],
                nr_of_iterations=args.nr_of_iterations_per_batch,
                map_low_res_factor=args.map_low_res_factor,
                visualize_step=visualize_step,
                json_in=in_json,
                json_out=frozen_out_json_stage_0,
                optimize_over_deep_network=False,
                optimize_over_weights=False,
                freeze_parameters=True,
                start_from_previously_saved_parameters=True,
                args_kvs=args.config_kvs
            )

            if args.retain_intermediate_stage_results:
                print('Backing up the frozen stage 0 results')
                backup_dir = os.path.realpath(results_output_directory) + '_frozen_after_stage_0'
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                shutil.copytree(results_output_directory, backup_dir)

    #
    # Stage 1 (optimizing over global weights) starts here
    #

    out_json_stage_1 = os.path.join(args.output_directory,'out_stage_1_' + os.path.split(args.config)[1])

    if 1 in stage_nr:

        if not args.only_compute_frozen_epochs:

            print('Running stage 1: now continue optimizing, but optimizing over the global weights')
            in_json = out_json_stage_0

            backup_dir_stage_0 = os.path.realpath(results_output_directory)+'_after_stage_0'
            if os.path.exists(backup_dir_stage_0):
                print('Copying ' + backup_dir_stage_0 + ' to ' + results_output_directory)
                shutil.rmtree(results_output_directory)
                shutil.copytree(backup_dir_stage_0,results_output_directory)

            do_registration(
                source_images=source_images,
                target_images=target_images,
                model_name=args.model_name,
                output_directory=results_output_directory,
                nr_of_epochs=nr_of_epochs[1],
                nr_of_iterations=args.nr_of_iterations_per_batch,
                map_low_res_factor=args.map_low_res_factor,
                visualize_step=visualize_step,
                json_in=in_json,
                json_out=out_json_stage_1,
                optimize_over_deep_network=False,
                optimize_over_weights=True,
                freeze_parameters=False,
                start_from_previously_saved_parameters=True,
                args_kvs=args.config_kvs
            )

            if args.retain_intermediate_stage_results:
                print('Backing up the stage 1 results')
                backup_dir = os.path.realpath(results_output_directory) + '_after_stage_1'
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                shutil.copytree(results_output_directory, backup_dir)

        if compute_frozen_epochs:

            print('Computing {:} frozen epochs for stage 1'.format(frozen_nr_of_epochs[1]))
            in_json = out_json_stage_1
            frozen_out_json_stage_1 = os.path.join(args.output_directory,
                                                   'frozen_out_stage_1_' + os.path.split(args.config)[1])

            if args.only_compute_frozen_epochs:
                # copy the earlier results of stage 0 into the current results directory
                # (otherwise they are already in this directory and we can continue from there)
                print('Restoring stage 1 results for continuation')
                if os.path.exists(results_output_directory):
                    shutil.rmtree(results_output_directory)
                backup_dir = os.path.realpath(results_output_directory) + '_after_stage_1'
                if os.path.exists(backup_dir):
                    shutil.copytree(backup_dir, results_output_directory)
                else:
                    raise ValueError('could not restore stage 1 results')

            do_registration(
                source_images=source_images,
                target_images=target_images,
                model_name=args.model_name,
                output_directory=results_output_directory,
                nr_of_epochs=frozen_nr_of_epochs[1],
                nr_of_iterations=args.nr_of_iterations_per_batch,
                map_low_res_factor=args.map_low_res_factor,
                visualize_step=visualize_step,
                json_in=in_json,
                json_out=frozen_out_json_stage_1,
                optimize_over_deep_network=False,
                optimize_over_weights=True,
                freeze_parameters=True,
                start_from_previously_saved_parameters=True,
                args_kvs=args.config_kvs
            )

            if args.retain_intermediate_stage_results:
                print('Backing up the frozen stage 1 results')
                backup_dir = os.path.realpath(results_output_directory) + '_frozen_after_stage_1'
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                shutil.copytree(results_output_directory, backup_dir)

    #
    # Stage 2 (optimizing over NN weights) starts here
    #

    out_json_stage_2 = os.path.join(args.output_directory, 'out_stage_2_' + os.path.split(args.config)[1])

    if 2 in stage_nr:

        if not args.only_compute_frozen_epochs:

            print('Running stage 2: now optimize over the network (keeping everything else fixed)')
            in_json = out_json_stage_1

            backup_dir_stage_1 = os.path.realpath(results_output_directory) + '_after_stage_1'
            if os.path.exists(backup_dir_stage_1):
                print('Copying ' + backup_dir_stage_1 + ' to ' + results_output_directory)
                shutil.rmtree(results_output_directory)
                shutil.copytree(backup_dir_stage_1, results_output_directory)

            do_registration(
                source_images=source_images,
                target_images=target_images,
                model_name=args.model_name,
                output_directory=results_output_directory,
                nr_of_epochs=nr_of_epochs[2],
                nr_of_iterations=args.nr_of_iterations_per_batch,
                visualize_step=visualize_step,
                map_low_res_factor=args.map_low_res_factor,
                json_in=in_json,
                json_out=out_json_stage_2,
                optimize_over_deep_network=True,
                optimize_over_weights=False,
                freeze_parameters=False,
                start_from_previously_saved_parameters=True,
                args_kvs=args.config_kvs
            )

            if args.retain_intermediate_stage_results:
                print('Backing up the stage 2 results')
                backup_dir = os.path.realpath(results_output_directory) + '_after_stage_2'
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                shutil.copytree(results_output_directory, backup_dir)

        if compute_frozen_epochs:

            print('Computing {:} frozen epochs for stage 2'.format(frozen_nr_of_epochs[2]))
            in_json = out_json_stage_2
            frozen_out_json_stage_2 = os.path.join(args.output_directory,
                                                   'frozen_out_stage_2_' + os.path.split(args.config)[1])

            if args.only_compute_frozen_epochs:
                # copy the earlier results of stage 0 into the current results directory
                # (otherwise they are already in this directory and we can continue from there)
                print('Restoring stage 2 results for continuation')
                if os.path.exists(results_output_directory):
                    shutil.rmtree(results_output_directory)
                backup_dir = os.path.realpath(results_output_directory) + '_after_stage_2'
                if os.path.exists(backup_dir):
                    shutil.copytree(backup_dir, results_output_directory)
                else:
                    raise ValueError('could not restore stage 2 results')

            do_registration(
                source_images=source_images,
                target_images=target_images,
                model_name=args.model_name,
                output_directory=results_output_directory,
                nr_of_epochs=frozen_nr_of_epochs[2],
                nr_of_iterations=args.nr_of_iterations_per_batch,
                map_low_res_factor=args.map_low_res_factor,
                visualize_step=visualize_step,
                json_in=in_json,
                json_out=frozen_out_json_stage_2,
                optimize_over_deep_network=True,
                optimize_over_weights=False,
                freeze_parameters=True,
                start_from_previously_saved_parameters=True,
                args_kvs=args.config_kvs
            )

            if args.retain_intermediate_stage_results:
                print('Backing up the frozen stage 2 results')
                backup_dir = os.path.realpath(results_output_directory) + '_frozen_after_stage_2'
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                shutil.copytree(results_output_directory, backup_dir)


    # Do cleanup

    # now remove temporary results directory
    if os.path.exists(results_output_directory):
        print('Removing temporary results in directory ' + results_output_directory)
        shutil.rmtree(results_output_directory)