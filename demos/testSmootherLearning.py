from builtins import str
from builtins import range
import set_pyreg_paths
import pyreg.config_parser as cp

# first do the torch imports
import torch
import numpy as np
import multiprocessing as mp

import pyreg.simple_interface as si
import pyreg.fileio as FIO


def get_image_range(im_from,im_to):
    f = []
    for i in range(im_from,im_to):
        current_filename = '../test_data/oasis_2d/oasis2d_' + str(i).zfill(4) + '.nrrd'
        f.append( current_filename )
    return f


def do_registration(model_name,nr_of_iterations,nr_of_image_pairs,
                    visualize_step,
                    map_low_res_factor,rel_ftol,json_in,json_out,
                    checkpoint_dir,resume_from_last_checkpoint,
                    optimizer_name,res_pt):

    # first load the images
    im_io = FIO.ImageIO()

    # load a bunch of images as source
    I0, hdr, spacing0, _ = im_io.read_batch_to_nc_format(get_image_range(0, nr_of_image_pairs))
    sz = np.array(I0.shape)
    # and a bunch of images as target images
    I1, hdr, spacing1, _ = im_io.read_batch_to_nc_format(get_image_range(nr_of_image_pairs, 2*nr_of_image_pairs))

    assert (np.all(spacing0 == spacing1))

    reg = si.RegisterImagePair()

    reg.register_images(I0,I1,spacing0,
                model_name=model_name,
                nr_of_iterations=nr_of_iterations,
                visualize_step=visualize_step,
                map_low_res_factor=map_low_res_factor,
                rel_ftol=rel_ftol,
                json_config_out_filename=json_out,
                use_consensus_optimization=True,
                checkpoint_dir=checkpoint_dir,
                resume_from_last_checkpoint=resume_from_last_checkpoint,
                optimizer_name=optimizer_name,
                params=json_in)

    h = reg.get_history()

    pars = reg.get_model_parameters()

    Iw = reg.get_warped_image()
    phi = reg.get_map()

    vars_to_save = dict()
    vars_to_save['registration_pars'] = pars
    vars_to_save['I0'] = I0
    vars_to_save['I1'] = I1
    vars_to_save['sz'] = sz
    vars_to_save['Iw'] = Iw
    vars_to_save['phi'] = phi
    vars_to_save['spacing'] = spacing0
    vars_to_save['params'] = reg.get_params()
    vars_to_save['history'] = h

    torch.save(vars_to_save,res_pt)


if __name__ == "__main__":

    torch.set_num_threads(mp.cpu_count())

    import argparse

    parser = argparse.ArgumentParser(description='Registers batches of two images based on OASIS data (for testing)')

    parser.add_argument('--model_name', required=False, default='svf_scalar_momentum_map', help='model that should be used for the registration')

    parser.add_argument('--nr_of_image_pairs', required=False, type=int, default=10, help='number of image pairs that will be used')
    parser.add_argument('--nr_of_iterations', required=False, type=int, default=10, help='number of iterations')
    parser.add_argument('--map_low_res_factor', required=False, type=float, default=1.0, help='map low res factor')
    parser.add_argument('--rel_ftol', required=False, type=float, default=1e-6, help='relative tolerance to stop optimization')

    parser.add_argument('--visualize', required=False, default=False, help='visualizes the output')
    parser.add_argument('--visualize_step', required=False, type=int, default=10, help='Number of iterations between visualization output')

    parser.add_argument('--config', required=False, default=None, help='Configuration file to read in')
    parser.add_argument('--used_config', required=False, default=None, help='Name to write out the used configuration')

    parser.add_argument('--res_pt', required=False, default='out.pt', help='File to write all the registration results to')
    parser.add_argument('--checkpointing_dir', required=False, default='checkpoints', help='directory where the checkpoints are saved')
    parser.add_argument('--resume_from_last_checkpoint', required=False, type=bool, default=False, help='when set to True starts from last checkpoint')
    parser.add_argument('--optimizer_name', required=False, default='lbfgs_ls', help='Optimizer type: lbfgs_ls|adam|sgd')

    args = parser.parse_args()

    if args.visualize:
        visualize_step = args.visualize_step
    else:
        visualize_step = None

    do_registration(model_name=args.model_name,
                    nr_of_iterations=args.nr_of_iterations,
                    nr_of_image_pairs=args.nr_of_image_pairs,
                    visualize_step=visualize_step,
                    map_low_res_factor=0.5,
                    rel_ftol=args.rel_ftol,
                    json_in=args.config,
                    json_out=args.used_config,
                    checkpoint_dir=args.checkpointing_dir,
                    resume_from_last_checkpoint=args.resume_from_last_checkpoint,
                    optimizer_name=args.optimizer_name,
                    res_pt=args.res_pt)
