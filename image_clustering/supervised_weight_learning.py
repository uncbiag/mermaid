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

from pyreg.data_wrapper import MyTensor, AdaptVal, USE_CUDA

device = torch.device("cuda:0" if (USE_CUDA and torch.cuda.is_available()) else "cpu")

import matplotlib.pyplot as plt
import numpy as np
import random

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

only_evaluate = False

rel_path = '../experiments'

save_network_state_dict_file = 'network_conf.pt'

if only_evaluate:
    input_directory = 'test_nn_synthetic_example_out_kernel_weighting_type_sqrt_w_K_sqrt_w'
else:
    input_directory = 'synthetic_example_out_kernel_weighting_type_sqrt_w_K_sqrt_w'

image_input_directory = os.path.join(rel_path,input_directory,'brain_affine_icbm')

dim = 2
batch_size = 20
nr_of_epochs = 100
visualize_intermediate_results = True
only_display_epoch_results = True
image_offset = 1.0

if only_evaluate:
    if nr_of_epochs!=1:
        print('INFO: Setting number of epochs to 1 for evaluation-only mode')
        nr_of_epochs = 1
    if batch_size!=1:
        print('INFO: Setting batch size to 1 for evaluation-only mode')
        batch_size = 1

# todo: read this values from the configuration file
nr_of_weights = 4
#global_multi_gaussian_weights = torch.from_numpy(np.array([0.25,0.25,0.25,0.25],dtype='float32'))
global_multi_gaussian_weights = torch.from_numpy(np.array([0.0,0.0,0.3,0.7],dtype='float32'))
gaussian_stds = torch.from_numpy(np.array([0.01,0.05,0.1,0.2],dtype='float32'))

normalization_type = 'group' # '['batch', 'instance', 'layer', 'group', 'none']
use_noisy_convolution = False
noisy_convolution_std = 0.05

use_noise_layers = False
noise_layer_std = 0.025
last_noise_layer_std = 0.025

use_color_tv = True
use_omt_tv_weighting = False
reconstruct_variances = False
reconstruct_stds = True

display_colorbar = True

im_sz = np.array([128,128])
reconstruction_weight = 10000.0
totalvariation_weight = 0.001
omt_weight = 0.0001
omt_power=1.0
omt_use_log_transformed_std=True
lr = 0.05
seed = 75

if seed is not None:
    print('Setting the seed to: {}'.format(seed))
    random.seed(seed)
    torch.manual_seed(seed)

used_image_pairs_file = os.path.join(rel_path,input_directory,'used_image_pairs.pt')
used_image_pairs = torch.load(used_image_pairs_file)
# only the source images have the weights
input_images = used_image_pairs['source_images']

# todo: check if smoothing of input image is beneficial for edge detection
# todo: if so, also implement it in the main code base

# todo: think about where to inject the noise in the noisy convolution, before or after the normalization (likely after the normalization)
# todo: weight the total variation term by the OMT cost. In this way it will be cheaper to introduce jumps for cheaper terms

#network_type = dn.Unet_no_skip
#network_type = dn.Unet
network_type = dn.Simple_consistent

params = pars.ParameterDict()
params['normalization_type'] = normalization_type
params['use_noisy_convolution'] = use_noisy_convolution
params['noisy_convolution_std'] = noisy_convolution_std

params['use_noise_layers'] = use_noise_layers
params['noise_layer_std'] = noise_layer_std
params['last_noise_layer_std'] = last_noise_layer_std

params['edge_penalty_gamma'] = 2. #10.

dataset = ImageAndWeightDataset(image_filenames=input_images, rel_path=rel_path, params=params)
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

# create the two network and the loss function

reconstruction_criterion = nn.MSELoss().to(device)  # nn.L1Loss().to(device)
totalvariation_criterion = None
omt_criterion = None

reconstruction_unet = network_type(dim=dim, n_in_channel=1, n_out_channel=nr_of_weights, im_sz=im_sz, params=params).to(device)
reconstruction_unet.initialize_network_weights()

if only_evaluate:
    print('INFO: Loading the network state from {}'.format(save_network_state_dict_file))
    reconstruction_unet.load_state_dict(torch.load(save_network_state_dict_file))

all_optimization_parameters = reconstruction_unet.parameters()

print(reconstruction_unet)

# create the optimizer
optimizer = optim.SGD(all_optimization_parameters, lr=lr, momentum=0.9, nesterov=True)
#optimizer = optim.SGD(all_optimization_parameters, lr=lr, momentum=0.9, nesterov=True, weight_decay=0.001)

nr_of_datasets = len(dataset)
nr_of_batches = len(trainloader)

# pre-weights versus weights
deep_network_local_weight_smoothing = 0.02
deep_network_weight_smoother = None


for epoch in range(nr_of_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    running_reconstruction_loss = 0.0
    running_totalvariation_loss = 0.0
    running_omt_loss = 0.0 # todo: add the OMT loss

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        input_dict = data
        unscaled_images = input_dict['ISource'].to(device)
        gt_weights = input_dict['gt_weights'].to(device)
        inputs = input_dict['ISource'].to(device) - image_offset
        spacing = input_dict['spacing'][0].detach().cpu().numpy()  # all of the spacings are the same, so just use one
        volumeElement = spacing.prod()

        if totalvariation_criterion is None:
            # now we have the spacing information and can instantiate it
            totalvariation_criterion = dn.TotalVariationLoss(dim=dim, im_sz=im_sz, spacing=spacing,
                                                             use_omt_weighting=use_omt_tv_weighting,
                                                             gaussian_stds=gaussian_stds,
                                                             omt_power=omt_power,
                                                             omt_use_log_transformed_std=omt_use_log_transformed_std,
                                                             params=params).to(device)
        if omt_criterion is None:
            # now we have the spacing information and can instantiate it
            omt_criterion = dn.OMTLoss(spacing=spacing, desired_power=omt_power,
                                       use_log_transform=omt_use_log_transformed_std, params=params).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        reconstruction_outputs = reconstruction_unet(inputs)
        #pre_weights = deep_smoothers.weighted_softmax(reconstruction_outputs, dim=1, weights=global_multi_gaussian_weights)
        pre_weights = deep_smoothers.weighted_linear_softmax(reconstruction_outputs, dim=1, weights=global_multi_gaussian_weights)
        #pre_weights = deep_smoothers.stable_softmax(reconstruction_outputs, dim=1)

        # enforce minimum weight for numerical reasons
        # todo: check if this is really needed, this was for the sqrt
        # todo, maybe make this conditional
        pre_weights = deep_smoothers._project_weights_to_min(pre_weights, 0.001, norm_type='sum')

        # instantiate the extra smoother if weight is larger than 0 and it has not been initialized yet
        if deep_network_local_weight_smoothing > 0 and deep_network_weight_smoother is None:
            import pyreg.smoother_factory as sf

            s_m_params = pars.ParameterDict()
            s_m_params['smoother']['type'] = 'gaussian'
            s_m_params['smoother']['gaussian_std'] = deep_network_local_weight_smoothing
            deep_network_weight_smoother = sf.SmootherFactory(inputs.size()[2::], spacing=spacing).create_smoother(s_m_params)

        if deep_network_local_weight_smoothing > 0:
            # now we smooth all the weights
            weights = deep_network_weight_smoother.smooth(pre_weights)
            # make sure they are all still positive (#todo: may not be necessary, since we set a minumum weight above now; but risky as we take the square root below)

            # todo: put back in?
            #weights = torch.clamp(weights, 0.0, 1.0)
        else:
            weights = pre_weights

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

        totalvariation_loss = totalvariation_weight * totalvariation_criterion(input_images=unscaled_images,label_probabilities=pre_weights,use_color_tv=use_color_tv)

        omt_loss = omt_weight * omt_criterion(weights, gaussian_stds)

        # compute the overall loss
        loss = reconstruction_loss + totalvariation_loss + omt_loss

        if not only_evaluate:
            # compute the gradient
            loss.backward()
            optimizer.step()

        if only_display_epoch_results:
            running_reconstruction_loss += reconstruction_loss.item() / nr_of_datasets
            running_loss += loss.item() / nr_of_datasets
            running_totalvariation_loss += totalvariation_loss.item() / nr_of_datasets
            running_omt_loss += omt_loss.item() / nr_of_datasets

            if i == nr_of_batches - 1:
                # print statistics
                print(
                    'Epoch: {}; loss={:.3f}, r_loss={:.3f}, tv_loss={:.3f}, omt_loss={:.3f}'
                    .format(epoch + 1, running_loss, running_reconstruction_loss, running_totalvariation_loss, running_omt_loss))
        else:
            # print statistics
            print(
                'Epoch: {}; batch: {}; loss={:.3f}, r_loss={:.3f}, tv_loss={:.3f}, omt_loss={:.3f}'
                .format(epoch + 1, i + 1, loss.item(), reconstruction_loss.item(), totalvariation_loss.item(), omt_loss.item()))

        if i == 0 and (epoch % 10 == 0):

            nr_of_current_images = inputs.size()[0]
            currently_selected_image = np.random.randint(low=0, high=nr_of_current_images)

            plt.clf()
            # plot the input (source) image
            plt.subplot(4, nr_of_weights, 1)
            plt.imshow(inputs[currently_selected_image, 0, ...].detach().cpu().numpy())
            plt.axis('image')
            plt.axis('off')
            if display_colorbar:
                plt.colorbar()

            plt.subplot(4, nr_of_weights, 2)
            plt.imshow((gt_variances[currently_selected_image,...].detach().cpu().numpy())**0.5)
            plt.axis('image')
            plt.axis('off')
            if display_colorbar:
                plt.colorbar()

            plt.subplot(4, nr_of_weights, 3)
            plt.imshow((pred_variances[currently_selected_image, ...].detach().cpu().numpy())**0.5)
            plt.axis('image')
            plt.axis('off')
            if display_colorbar:
                plt.colorbar()

            plt.subplot(4, nr_of_weights, 4)
            plt.imshow((pred_variances[currently_selected_image, ...].detach().cpu().numpy())**0.5-(gt_variances[currently_selected_image,...].detach().cpu().numpy())**0.5)
            plt.axis('image')
            plt.axis('off')
            if display_colorbar:
                plt.colorbar()

            # plot the ground truth weights
            for nw in range(nr_of_weights):
                plt.subplot(4,nr_of_weights,nr_of_weights+1+nw)
                plt.imshow(gt_weights[currently_selected_image, nw, ...].detach().cpu().numpy(),clim=(0.0,1.0))
                plt.axis('image')
                plt.axis('off')
                if display_colorbar:
                    plt.colorbar()

            # plot the computed weights
            for nw in range(nr_of_weights):
                plt.subplot(4, nr_of_weights, 2*nr_of_weights + 1 + nw)
                plt.imshow(weights[currently_selected_image, nw, ...].detach().cpu().numpy(),clim=(0.0,1.0))
                plt.axis('image')
                plt.axis('off')
                if display_colorbar:
                    plt.colorbar()

            # plot the computed weights
            for nw in range(nr_of_weights):
                plt.subplot(4, nr_of_weights, 3 * nr_of_weights + 1 + nw)
                plt.imshow(weights[currently_selected_image, nw, ...].detach().cpu().numpy()-gt_weights[currently_selected_image, nw, ...].detach().cpu().numpy(), clim=(-1.0, 1.0))
                plt.axis('image')
                plt.axis('off')
                if display_colorbar:
                    plt.colorbar()

            #plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
            plt.show()

print('Finished Training')

if not only_evaluate:
    print('Saving network state dict to {}'.format(save_network_state_dict_file))
    torch.save(reconstruction_unet.state_dict(),save_network_state_dict_file)
