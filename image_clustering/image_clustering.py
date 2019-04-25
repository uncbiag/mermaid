from __future__ import print_function
from __future__ import absolute_import
from builtins import object

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob

import set_pyreg_paths
import pyreg.deep_smoothers as deep_smoothers
import pyreg.fileio as FIO
import pyreg.module_parameters as pars

from pyreg.data_wrapper import MyTensor, AdaptVal, USE_CUDA

device = torch.device("cuda:0" if (USE_CUDA and torch.cuda.is_available()) else "cpu")

import matplotlib.pyplot as plt

import numpy as np

import deep_networks as dn

# create a dataloader

class ImageDataset(Dataset):
    """keeps track of pairwise image as well as checkpoints for their parameters"""

    def __init__(self, image_filenames, params):

        self.params = params

        self.image_filenames = image_filenames

        self.params[('data_loader', {}, 'data loader settings')]
        cparams = self.params['data_loader']
        self.intensity_normalize = cparams[('intensity_normalize',False,'intensity normalize images when reading')]
        self.squeeze_image = cparams[('squeeze_image',True,'squeezes image first (e.g, from 1x128x128 to 128x128)')]
        self.normalize_spacing = cparams[('normalize_spacing',True,'normalizes the image spacing')]

    def __len__(self):
        return len(self.image_filenames)

    def _get_image_filename(self,idx):
        return self.image_filenames[idx]

    def __getitem__(self,idx):

        # load the actual images
        current_source_filename = self._get_image_filename(idx)

        im_io = FIO.ImageIO()
        ISource,hdr,spacing,normalized_spacing = im_io.read_batch_to_nc_format([current_source_filename],
                                                intensity_normalize=self.intensity_normalize,
                                                squeeze_image=self.squeeze_image,
                                                normalize_spacing=self.normalize_spacing,
                                                silent_mode=True)

        sample = dict()
        sample['idx'] = idx
        sample['ISource'] = ISource[0,...] # as we only loaded a batch-of-one we remove the first dimension
        sample['spacing'] = normalized_spacing

        return sample

compute_unsupervised_segmentation = True # otherwise just does image reconstruction (as a sanity check)
    
dim = 2

nr_of_desired_clusters = 3
batch_size = 20
nr_of_epochs = 100
visualize_intermediate_results = True
only_display_epoch_results = True
image_offset = 1.0


clustering_weight = 0.0
reconstruction_weight = 50.0
totalvariation_weight = 25.0
entropy_weight = 1.0
global_entropy_weight = 0.01

#network_type = dn.encoder_decoder_small
network_type = dn.Unet_no_skip

image_input_directory = '../experiments/synthetic_example_out_kernel_weighting_type_w_K/brain_affine_icbm/'
input_images = glob.glob(os.path.join(image_input_directory,'*.nii'))
params = pars.ParameterDict()

dataset = ImageDataset(image_filenames=input_images,params=params)
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)


# create the two Unets and the loss functions

reconstruction_criterion = nn.MSELoss().to(device) # nn.L1Loss().to(device)

if compute_unsupervised_segmentation:
    cluster_unet = network_type(dim=dim,n_in_channel=1,n_out_channel=nr_of_desired_clusters).to(device)
    print(cluster_unet)

    reconstruction_unet = network_type(dim=dim,n_in_channel=nr_of_desired_clusters,n_out_channel=1).to(device)
    totalvariation_criterion = dn.TotalVariationLoss(dim=dim,params=params).to(device)
    entropy_criterion = dn.HLoss().to(device)
    global_entropy_criterion = dn.GlobalHLoss().to(device)
    clustering_criterion = dn.ClusteringLoss(dim=dim,params=params).to(device)
    all_optimization_parameters = list(cluster_unet.parameters()) + list(reconstruction_unet.parameters())

else:
    print('Only running as reconstruction')
    reconstruction_unet = network_type(dim=dim,n_in_channel=1,n_out_channel=1).to(device)
    all_optimization_parameters = reconstruction_unet.parameters()

print(reconstruction_unet)


# create the optimizer
optimizer = optim.SGD(all_optimization_parameters, lr=0.00001, momentum=0.9, nesterov=True)

nr_of_datasets = len(dataset)
nr_of_batches = len(trainloader)

for epoch in range(nr_of_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    running_reconstruction_loss = 0.0
    running_entropy_loss = 0.0
    running_global_entropy_loss = 0.0
    running_totalvariation_loss = 0.0
    running_clustering_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        input_dict = data
        unscaled_images = input_dict['ISource'].to(device)
        inputs = input_dict['ISource'].to(device)-image_offset
        spacing = input_dict['spacing'][0].detach().cpu().numpy() # all of the spacings are the same, so just use one
        # for now this is just like an auto-encoder
        desired_outputs = inputs

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        if compute_unsupervised_segmentation:
            cluster_outputs = cluster_unet(inputs)
            # now map this through a softmax to obtain the label probabilities
            label_probabilities  = deep_smoothers.stable_softmax(cluster_outputs,dim=1)
            # compute the clustering loss
            clustering_loss = clustering_weight*clustering_criterion(input_images=inputs,spacing=spacing,label_probabilities=label_probabilities)

            entropy_loss = entropy_weight*entropy_criterion(label_probabilities,spacing=spacing)
            # we want to maximize global entropy so let's minimize the negative one
            negative_global_entropy_loss = -global_entropy_weight*global_entropy_criterion(label_probabilities,spacing=spacing)
            totalvariation_loss = totalvariation_weight*totalvariation_criterion(input_images=unscaled_images,spacing=spacing,label_probabilities=label_probabilities)

            # try to reconstruct the image
            reconstruction_outputs = reconstruction_unet(cluster_outputs)

        else:
            reconstruction_outputs = reconstruction_unet(inputs)
            
        # compute the reconstruction loss
        reconstruction_loss = reconstruction_weight*reconstruction_criterion(reconstruction_outputs, desired_outputs)

        # compute the overall loss as a combination of reconstruction and clustering loss
        if compute_unsupervised_segmentation:
            loss = reconstruction_loss + clustering_loss + totalvariation_loss + entropy_loss + negative_global_entropy_loss
        else:
            loss = reconstruction_loss

        # compute the gradient
        loss.backward()

        optimizer.step()

        if only_display_epoch_results:
            running_reconstruction_loss += reconstruction_loss.item()/nr_of_datasets

            if compute_unsupervised_segmentation:
                running_loss += loss.item()/nr_of_datasets
                running_global_entropy_loss += negative_global_entropy_loss.item()/nr_of_datasets
                running_entropy_loss += entropy_loss.item()/nr_of_datasets
                running_totalvariation_loss += totalvariation_loss.item()/nr_of_datasets
                running_clustering_loss += clustering_loss.item()/nr_of_datasets

            if i==nr_of_batches-1:
                # print statistics
                print('Epoch: {}; loss={:.3f}, r_loss={:.3f}, tv_loss={:.3f}, h_loss={:.3f}, neg_global_h_loss={:.3f}, c_loss={:.3f}'
                      .format(epoch + 1, running_loss, running_reconstruction_loss, running_totalvariation_loss,
                              running_entropy_loss, running_global_entropy_loss, running_clustering_loss))

        else:
            # print statistics
            print('Epoch: {}; batch: {}; loss={:.3f}, r_loss={:.3f}, tv_loss={:.3f}, h_loss={:.3f}, neg_global_h_loss={:.3f}, c_loss={:.3f}'
                  .format(epoch+1, i+1,loss.item(),reconstruction_loss.item(),totalvariation_loss.item(),
                          entropy_loss.item(),negative_global_entropy_loss.item(),clustering_loss.item()))

        if i==0 and (epoch%10==0):

            nr_of_current_images = inputs.size()[0]
            currently_selected_image = np.random.randint(low=0, high=nr_of_current_images)

            plt.clf()
            plt.subplot(2,3,1)
            plt.imshow(inputs[currently_selected_image,0,...].detach().cpu().numpy())
            plt.colorbar()
            plt.subplot(2,3,2)
            plt.imshow(reconstruction_outputs[currently_selected_image,0,...].detach().cpu().numpy())
            plt.colorbar()
            if compute_unsupervised_segmentation:
                plt.subplot(2,3,4)
                plt.imshow(label_probabilities[currently_selected_image,0,...].detach().cpu().numpy())
                plt.colorbar()
                plt.subplot(2, 3, 5)
                plt.imshow(label_probabilities[currently_selected_image, 1, ...].detach().cpu().numpy())
                plt.colorbar()
                plt.subplot(2, 3, 6)
                plt.imshow(label_probabilities[currently_selected_image, 2, ...].detach().cpu().numpy())
                plt.colorbar()
                
            plt.show()

print('Finished Training')
