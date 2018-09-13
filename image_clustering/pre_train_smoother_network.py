from __future__ import print_function
from __future__ import absolute_import
from builtins import object

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import re

import set_pyreg_paths
import pyreg.fileio as FIO
import pyreg.module_parameters as pars
import pyreg.deep_smoothers as deep_smoothers
import pyreg.deep_networks as dn
import pyreg.image_sampling as IS

from pyreg.data_wrapper import MyTensor, AdaptVal, USE_CUDA

device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

import matplotlib.pyplot as plt
import numpy as np
import random


#todo: the following three functions are duplicate from visualize_multi_stage: make this consistent and put it in one place

def _compute_low_res_image(I,spacing,low_res_size,spline_order):
    sampler = IS.ResampleImage()
    low_res_image, _ = sampler.downsample_image_to_size(I, spacing, low_res_size[2::],spline_order)
    return low_res_image

def _get_low_res_size_from_size(sz, factor):
    """
    Returns the corresponding low-res size from a (high-res) sz
    :param sz: size (high-res)
    :param factor: low-res factor (needs to be <1)
    :return: low res size
    """
    if (factor is None) or (factor>=1):
        print('WARNING: Could not compute low_res_size as factor was ' + str( factor ))
        return sz
    else:
        lowResSize = np.array(sz)
        lowResSize[2::] = (np.ceil((np.array(sz[2::]) * factor))).astype('int16')

        if lowResSize[-1]%2!=0:
            lowResSize[-1]-=1
            print('\n\nWARNING: forcing last dimension to be even: fix properly in the Fourier transform later!\n\n')

        return lowResSize

def _get_low_res_spacing_from_spacing(spacing, sz, lowResSize):
    """
    Computes spacing for the low-res parameterization from image spacing
    :param spacing: image spacing
    :param sz: size of image
    :param lowResSize: size of low re parameterization
    :return: returns spacing of low res parameterization
    """
    #todo: check that this is the correct way of doing it
    return spacing * (np.array(sz[2::])-1) / (np.array(lowResSize[2::])-1)


# create a dataloader

class ImageAndWeightDataset(Dataset):
    """keeps track of pairwise image as well as checkpoints for their parameters"""

    def __init__(self, image_filenames, rel_path, params):

        self.params = params

        self.image_filenames = image_filenames
        self.rel_path = rel_path

        self.params[('image_and_weight_data_loader', {}, 'data loader settings')]
        cparams = self.params['image_and_weight_data_loader']
        self.intensity_normalize = cparams[('intensity_normalize',False,'intensity normalize images when reading')]
        self.squeeze_image = cparams[('squeeze_image',True,'squeezes image first (e.g, from 1x128x128 to 128x128)')]
        self.normalize_spacing = cparams[('normalize_spacing',True,'normalizes the image spacing')]

    def __len__(self):
        return len(self.image_filenames)

    def _get_image_filename(self,idx):
        return os.path.join(rel_path,self.image_filenames[idx])

    def _get_file_idx_from_image_filename(self,image_filename):
        # this will be a number at the end. Filenames are expected to be of the format m20.nii
        file_idx = int(re.findall(r'\d+',image_filename)[0])
        return file_idx

    def _get_weight_filename(self,idx):
        current_image_filename = self._get_image_filename(idx)
        file_idx = self._get_file_idx_from_image_filename(current_image_filename)
        # this will be one directory up, in the subdirectory misc and by name gt_weights_00001.pt
        pn, _ = os.path.split(current_image_filename)
        pn, _ = os.path.split(pn) # because we need to get one directory up
        weight_filename = os.path.join(pn,'misc','gt_weights_{:05d}.pt'.format(file_idx))
        return weight_filename

    def __getitem__(self,idx):

        # load the actual images
        current_source_filename = self._get_image_filename(idx)

        im_io = FIO.ImageIO()
        ISource,hdr,spacing,normalized_spacing = im_io.read_batch_to_nc_format([current_source_filename],
                                                intensity_normalize=self.intensity_normalize,
                                                squeeze_image=self.squeeze_image,
                                                normalize_spacing=self.normalize_spacing,
                                                silent_mode=True)

        current_weight_filename = self._get_weight_filename(idx)
        weights = torch.load(current_weight_filename)

        sample = dict()
        sample['idx'] = idx
        sample['ISource'] = ISource[0,...] # as we only loaded a batch-of-one we remove the first dimension
        sample['gt_weights'] = weights[0,...]
        sample['spacing'] = normalized_spacing

        return sample

def compute_variances(weights,stds):
    nr_s = len(stds)
    nr_w = weights.size()[1]
    assert(nr_s==nr_w)

    ret = torch.zeros_like(weights[:,0,...])
    for n in range(nr_w):
        ret += weights[:,n,...]*stds[n]**2

    return ret

def _plot_current_images(I,gt_variances,pred_variances,gt_weights,weights,
                         display_colorbar,
                         selected_image,
                         write_prefix=None):

    clean_print_path = 'clean_figures'
    if not os.path.exists(clean_print_path):
        print('INFO: Creating directory {}'.format(clean_print_path))
        os.makedirs(clean_print_path)

    nr_of_weights = weights.size()[1]

    plt.clf()
    # plot the input (source) image
    if write_prefix is None:
        plt.subplot(4, nr_of_weights, 1)
    plt.imshow(I[currently_selected_image, 0, ...].detach().cpu().numpy())
    plt.axis('image')
    plt.axis('off')
    if display_colorbar:
        plt.colorbar()

    if write_prefix is not None:
        plt.savefig(os.path.join(clean_print_path,'input_image_{}.pdf'.format(write_prefix)),bbox_inches='tight',pad_inches=0)
        plt.clf()
    else:
        plt.subplot(4, nr_of_weights, 2)

    plt.imshow((gt_variances[currently_selected_image,...].detach().cpu().numpy())**0.5)
    plt.axis('image')
    plt.axis('off')
    if display_colorbar:
        plt.colorbar()

    if write_prefix is not None:
        plt.savefig(os.path.join(clean_print_path,'gt_stds_{}.pdf'.format(write_prefix)),bbox_inches='tight',pad_inches=0)
        plt.clf()
    else:
        plt.subplot(4, nr_of_weights, 3)

    plt.imshow((pred_variances[currently_selected_image, ...].detach().cpu().numpy())**0.5)
    plt.axis('image')
    plt.axis('off')
    if display_colorbar:
        plt.colorbar()

    if write_prefix is not None:
        plt.savefig(os.path.join(clean_print_path,'pred_stds_{}.pdf'.format(write_prefix)),bbox_inches='tight',pad_inches=0)
        plt.clf()
    else:
        plt.subplot(4, nr_of_weights, 4)

    plt.imshow((pred_variances[currently_selected_image, ...].detach().cpu().numpy())**0.5-(gt_variances[currently_selected_image,...].detach().cpu().numpy())**0.5)
    plt.axis('image')
    plt.axis('off')
    if display_colorbar:
        plt.colorbar()

    if write_prefix is not None:
        plt.savefig(os.path.join(clean_print_path,'diff_pred_m_gt_stds_{}.pdf'.format(write_prefix)),bbox_inches='tight',pad_inches=0)
        plt.clf()

    # plot the ground truth weights
    for nw in range(nr_of_weights):
        if write_prefix is None:
            plt.subplot(4,nr_of_weights,nr_of_weights+1+nw)
        else:
            plt.clf()
        plt.imshow(gt_weights[currently_selected_image, nw, ...].detach().cpu().numpy(),clim=(0.0,1.0))
        plt.axis('image')
        plt.axis('off')
        if display_colorbar:
            plt.colorbar()
        if write_prefix is not None:
            plt.savefig(os.path.join(clean_print_path, 'gt_weights_{}_{}.pdf'.format(nw,write_prefix)),
                        bbox_inches='tight', pad_inches=0)

    # plot the computed weights
    for nw in range(nr_of_weights):
        if write_prefix is None:
            plt.subplot(4, nr_of_weights, 2*nr_of_weights + 1 + nw)
        else:
            plt.clf()
        plt.imshow(weights[currently_selected_image, nw, ...].detach().cpu().numpy(),clim=(0.0,1.0))
        plt.axis('image')
        plt.axis('off')
        if display_colorbar:
            plt.colorbar()
        if write_prefix is not None:
            plt.savefig(os.path.join(clean_print_path, 'pred_weights_{}_{}.pdf'.format(nw,write_prefix)),
                        bbox_inches='tight', pad_inches=0)

    # plot the computed weights
    for nw in range(nr_of_weights):
        if write_prefix is None:
            plt.subplot(4, nr_of_weights, 3 * nr_of_weights + 1 + nw)
        else:
            plt.clf()
        plt.imshow(weights[currently_selected_image, nw, ...].detach().cpu().numpy()-gt_weights[currently_selected_image, nw, ...].detach().cpu().numpy(), clim=(-1.0, 1.0))
        plt.axis('image')
        plt.axis('off')
        if display_colorbar:
            plt.colorbar()
        if write_prefix is not None:
            plt.savefig(os.path.join(clean_print_path, 'diff_pred_m_gt_weights_{}_{}.pdf'.format(nw,write_prefix)),
                        bbox_inches='tight', pad_inches=0)

    if write_prefix is None:
        plt.show()

rel_path = '../experiments'
only_evaluate = True

if only_evaluate:
    input_directory = 'test_nn_synthetic_example_out_kernel_weighting_type_sqrt_w_K_sqrt_w'
else:
    input_directory = 'synthetic_example_out_kernel_weighting_type_sqrt_w_K_sqrt_w'

json_config = 'test_simple_consistent.json'

image_input_directory = os.path.join(rel_path,input_directory,'brain_affine_icbm')
json_config_full_path = os.path.join(rel_path, json_config)

params = pars.ParameterDict()
params.load_JSON(json_config_full_path)

map_low_res_factor = params['model']['deformation'][('map_low_res_factor',1.0,'Map low res factor')]
spline_order = params['model']['registration_model'][('spline_order', 1, 'Spline interpolation order; 1 is linear interpolation (default); 3 is cubic spline')]
params_smoother = params['model']['registration_model']['forward_model']['smoother']

all_omt_weight_penalties = [5.0,15.0,25.0,50.0,75.0,100.0] # 1.0
all_tv_penalties = [0.1]

for omt_weight_penalty in all_omt_weight_penalties:
    for tv_penalty in all_tv_penalties:
        params_smoother['omt_weight_penalty'] = omt_weight_penalty
        params_smoother['deep_smoother']['total_variation_penalty'] = tv_penalty
        #params_smoother['deep_smoother']['network_penalty'] = 0.0

        case_identifier_str = 'tv_{:f}_omt_{:f}'.format(tv_penalty,omt_weight_penalty)
        save_network_state_dict_file = 'network_conf_{}.pt'.format(case_identifier_str)

        gaussian_stds = torch.from_numpy(np.array(params_smoother['multi_gaussian_stds'],dtype='float32'))
        global_multi_gaussian_weights = torch.from_numpy(np.array(params_smoother['multi_gaussian_weights'],dtype='float32'))
        nr_of_weights = len(gaussian_stds)

        dim = 2
        batch_size = 40
        nr_of_epochs = 100
        visualize_intermediate_results = True
        only_display_epoch_results = True
        display_interval = 10

        if only_evaluate:
            if nr_of_epochs!=1:
                print('INFO: Setting number of epochs to 1 for evaluation-only mode')
                nr_of_epochs = 1
            if batch_size!=1:
                print('INFO: Setting batch size to 1 for evaluation-only mode')
                batch_size = 1
            if display_interval!=1:
                print('INFO: Setting display interval to 1 for evalutation-only mode')
                display_interval = 1

        reconstruct_variances = False
        reconstruct_stds = True

        display_colorbar = True

        reconstruction_weight = 100.0*batch_size
        only_use_reconstruction_loss = False
        disable_weight_range_penalty = False

        lr = 0.025/batch_size
        seed = 75

        if seed is not None:
            print('Setting the seed to: {}'.format(seed))
            random.seed(seed)
            torch.manual_seed(seed)

        used_image_pairs_file = os.path.join(rel_path,input_directory,'used_image_pairs.pt')
        used_image_pairs = torch.load(used_image_pairs_file)
        # only the source images have the weights
        input_images = used_image_pairs['source_images']

        dataset = ImageAndWeightDataset(image_filenames=input_images, rel_path=rel_path, params=params)
        if only_evaluate:
            print('INFO: disabled random shuffling')
            trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        else:
            trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        # create the two network and the loss function

        #reconstruction_criterion = nn.MSELoss().to(device)  # nn.L1Loss().to(device)
        reconstruction_criterion = nn.L1Loss().to(device)

        nr_of_datasets = len(dataset)
        nr_of_batches = len(trainloader)

        deep_smoother = None
        optimizer = None
        all_optimization_parameters = None

        for epoch in range(nr_of_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            running_reconstruction_loss = 0.0
            running_totalvariation_loss = 0.0
            running_omt_loss = 0.0
            running_weight_range_loss = 0.0
            running_l2_weight_loss = 0.0
            running_effective_weight_range_loss = 0.0

            for i, data in enumerate(trainloader, 0):

                # get the inputs
                input_dict = data
                gt_weights_orig = input_dict['gt_weights'].to(device)
                spacing_orig = input_dict['spacing'][0].detach().cpu().numpy()  # all of the spacings are the same, so just use one
                I_orig = input_dict['ISource'].to(device)
                sz_orig = np.array(list(I_orig.size()))

                downsample_data = False
                if map_low_res_factor is not None:
                    if not np.isclose(map_low_res_factor,1.0):
                        downsample_data = True

                if downsample_data:
                    sz_low_res = _get_low_res_size_from_size(sz_orig, map_low_res_factor)
                    spacing_low_res = _get_low_res_spacing_from_spacing(spacing_orig, sz_orig, sz_low_res)
                    I = _compute_low_res_image(I_orig, spacing_orig, sz_low_res, spline_order)
                    gt_weights = _compute_low_res_image(gt_weights_orig, spacing_orig, sz_low_res, spline_order)

                    im_sz = sz_low_res[2:]
                    spacing = spacing_low_res
                else:
                    I = I_orig
                    gt_weights = gt_weights_orig
                    im_sz = sz_orig[2:]
                    spacing = spacing_orig


                additional_inputs = dict()
                additional_inputs['I0'] = None
                additional_inputs['I1'] = None
                additional_inputs['m'] = None

                if deep_smoother is None:
                    deep_smoother = deep_smoothers.DeepSmootherFactory(nr_of_gaussians=nr_of_weights,
                                                                       gaussian_stds=gaussian_stds,
                                                                       dim=dim,
                                                                       spacing=spacing,
                                                                       im_sz=im_sz).create_deep_smoother(params_smoother).to(device)

                    if only_evaluate:
                        print('INFO: Loading the network state from {}'.format(save_network_state_dict_file))
                        deep_smoother.network.load_state_dict(torch.load(save_network_state_dict_file))

                weights, pre_weights, pre_weights_input = deep_smoother._compute_weights_and_preweights(I=I,
                                                                                     additional_inputs=additional_inputs,
                                                                                     global_multi_gaussian_weights=global_multi_gaussian_weights,
                                                                                     iter=epoch)

                current_penalty,current_omt_penalty, current_tv_penalty, current_diffusion_penalty, current_weight_range_penalty = \
                    deep_smoother._compute_penalty_from_weights_preweights_and_input_to_preweights(I=I,
                                                                                                   weights=weights,
                                                                                                   pre_weights=pre_weights,
                                                                                                   input_to_preweights=pre_weights_input,
                                                                                                   global_multi_gaussian_weights=global_multi_gaussian_weights)

                if optimizer is None:

                    all_optimization_parameters = deep_smoother.parameters()
                    print(deep_smoother)

                    # create the optimizer
                    optimizer = optim.SGD(all_optimization_parameters, lr=lr, momentum=0.9, nesterov=True)
                    # optimizer = optim.SGD(all_optimization_parameters, lr=lr, momentum=0.9, nesterov=True, weight_decay=0.001)

                    # create scheduler
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                                           verbose=True,
                                                                           factor=0.5,
                                                                           patience=5)

                # zero the parameter gradients
                optimizer.zero_grad()

                # compute resulting variances
                gt_variances = compute_variances(gt_weights, gaussian_stds)
                pred_variances = compute_variances(weights, gaussian_stds)

                # compute losses
                if reconstruct_variances:
                    reconstruction_loss = reconstruction_weight * reconstruction_criterion(pred_variances, gt_variances)
                elif reconstruct_stds:
                    reconstruction_loss = reconstruction_weight * reconstruction_criterion(torch.sqrt(pred_variances), torch.sqrt(gt_variances))
                else: # directly penalizing weights differences
                    reconstruction_loss = reconstruction_weight * reconstruction_criterion(weights, gt_weights)

                if disable_weight_range_penalty:
                    used_weight_range_penalty = 0*current_weight_range_penalty
                    effective_weight_range_penalty = current_weight_range_penalty
                else:
                    used_weight_range_penalty = current_weight_range_penalty
                    effective_weight_range_penalty = current_weight_range_penalty

                current_l2_weight_penalty = deep_smoother.compute_l2_parameter_weight_penalty()

                # compute the overall loss
                if only_use_reconstruction_loss:
                    loss = reconstruction_loss + used_weight_range_penalty + current_l2_weight_penalty
                else:
                    loss = reconstruction_loss + used_weight_range_penalty + current_l2_weight_penalty + current_tv_penalty + current_omt_penalty + current_diffusion_penalty

                if not only_evaluate:
                    # compute the gradient
                    loss.backward()

                    optimizer.step()
                    scheduler.step(loss.item())

                if only_display_epoch_results:
                    running_reconstruction_loss += reconstruction_loss.item() / nr_of_datasets
                    running_loss += loss.item() / nr_of_datasets
                    running_totalvariation_loss += current_tv_penalty.item() / nr_of_datasets
                    running_omt_loss += current_omt_penalty.item() / nr_of_datasets

                    running_effective_weight_range_loss += effective_weight_range_penalty.item() / nr_of_datasets
                    running_weight_range_loss += effective_weight_range_penalty.item() / nr_of_datasets

                    running_l2_weight_loss += current_l2_weight_penalty.item() / nr_of_datasets

                    if i == nr_of_batches - 1:
                        # print statistics
                        print(
                            'Epoch: {}; loss={:.6f}, r_loss={:.6f}, tv_loss={:.6f}, omt_loss={:.6f}, wr_loss={:.6f}, weight_loss={:.6f}'
                            .format(epoch + 1, running_loss,
                                    running_reconstruction_loss,
                                    running_totalvariation_loss,
                                    running_omt_loss,
                                    running_weight_range_loss,
                                    running_l2_weight_loss))
                else:
                    # print statistics

                    print(
                        'Epoch: {}; batch: {}; loss={:.6f}, r_loss={:.6f}, tv_loss={:.6f}, omt_loss={:.6f}, wr_loss={:.6f}, weight_loss={:.6f}'
                        .format(epoch + 1, i + 1, loss.item(),
                                reconstruction_loss.item(),
                                current_tv_penalty.item(),
                                current_omt_penalty.item(),
                                effective_weight_range_penalty.item(),
                                current_l2_weight_penalty.item()))

                if (i == 0 and (epoch % display_interval == 0)) or only_evaluate:

                    nr_of_current_images = I.size()[0]

                    if only_evaluate:
                        currently_selected_image = 0
                        _plot_current_images(I, gt_variances, pred_variances, gt_weights, weights,
                                             display_colorbar=True,
                                             selected_image=currently_selected_image,
                                             write_prefix=case_identifier_str)
                        _plot_current_images(I, gt_variances, pred_variances, gt_weights, weights,
                                             display_colorbar=False,
                                             selected_image=currently_selected_image,
                                             write_prefix=case_identifier_str+'_wo_colorbar')
                        break
                    else:
                        currently_selected_image = np.random.randint(low=0, high=nr_of_current_images)
                        _plot_current_images(I,gt_variances,pred_variances,gt_weights,weights,
                                             display_colorbar=display_colorbar,
                                             selected_image=currently_selected_image)


        if not only_evaluate:
            print('Finished Training')
            print('Saving network state dict to {}'.format(save_network_state_dict_file))
            torch.save(deep_smoother.network.state_dict(),save_network_state_dict_file)
        else:
            print('Finished Evaluation')

