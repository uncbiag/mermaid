from __future__ import print_function
from builtins import str
from builtins import range

import set_pyreg_paths

import torch
from torch.autograd import Variable

import pyreg.module_parameters as pars
from pyreg.data_wrapper import USE_CUDA, AdaptVal, MyTensor

import pyreg.fileio as FIO
import pyreg.utils as utils

import numpy as np

import matplotlib.pyplot as plt

import os

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Computes additional validation measures for the synthetically generated data')
    parser.add_argument('--output_directory', required=True, help='Where the output was stored (now this will be the input directory)')
    parser.add_argument('--input_image_directory', required=True, help='Directory where all the images are')
    parser.add_argument('--stage_nr', required=True, type=int, help='stage number for which the computations should be performed {0,1,2}')

    parser.add_argument('--compute_only_pair_nr', required=False, type=int, default=None, help='When specified only this pair is computed; otherwise all of them')
    parser.add_argument('--compute_from_frozen', action='store_true', help='computes the results from optimization results with frozen parameters')

    args = parser.parse_args()

    output_dir = args.output_directory
    input_image_directory = args.input_image_directory
    misc_output_directory = os.path.join(os.path.split(input_image_directory)[0],'misc')

    stage = args.stage_nr
    compute_from_frozen = args.compute_from_frozen

    used_pairs = torch.load(os.path.join(output_dir, 'used_image_pairs.pt'))
    nr_of_computed_pairs = len(used_pairs['source_ids'])

    source_ids = used_pairs['source_ids']

    if args.compute_only_pair_nr is not None:
        pair_nrs = [args.compute_only_pair_nr]
    else:
        pair_nrs = list(range(nr_of_computed_pairs))

    if compute_from_frozen:
        image_and_map_output_dir = os.path.join(os.path.normpath(output_dir),'model_results_frozen_stage_{:d}'.format(stage))
        print_output_dir = os.path.join(os.path.normpath(output_dir), 'pdf_frozen_stage_{:d}'.format(stage))
    else:
        image_and_map_output_dir = os.path.join(os.path.normpath(output_dir), 'model_results_stage_{:d}'.format(stage))

    for pair_nr in pair_nrs:

        # load the computed results for map, weights, and momentum
        map_output_filename_pt = os.path.join(image_and_map_output_dir,'map_{:05d}.pt'.format(pair_nr))
        weights_output_filename_pt = os.path.join(image_and_map_output_dir,'weights_{:05d}.pt'.format(pair_nr))
        momentum_output_filename_pt = os.path.join(image_and_map_output_dir,'momentum_{:05d}.pt'.format(pair_nr))

        map = torch.load(map_output_filename_pt).data.cpu.numpy()
        weights = torch.load(weights_output_filename_pt)
        momentum = torch.load(momentum_output_filename_pt).data.cpu().numpy()

        # now get the corresponding ground truth values
        # (these are based on the source image and hence based on the source id)
        current_source_id = source_ids[pair_nr]
        # load the ground truth results for map, weights, and momentum
        gt_map_filename = os.path.join(misc_output_directory, 'gt_map_{:05d}.pt'.format(current_source_id))
        gt_weights_filename = os.path.join(misc_output_directory, 'gt_weights_{:05d}.pt'.format(current_source_id))
        gt_momentum_filename = os.path.join(misc_output_directory, 'gt_momentum_{:05d}.pt'.format(current_source_id))

        gt_map = torch.load(gt_map_filename)
        gt_weights = torch.load(gt_weights_filename)
        gt_momentum = torch.load(gt_momentum_filename)

        print('Hello')
