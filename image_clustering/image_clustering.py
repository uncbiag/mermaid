import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob

import set_pyreg_paths
import pyreg.deep_smoothers as deep_smoothers
import pyreg.fileio as FIO
import pyreg.module_parameters as pars
import pyreg.finite_differences as fd

from pyreg.data_wrapper import MyTensor, AdaptVal, USE_CUDA

device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

import matplotlib.pyplot as plt

import numpy as np

def DimConv(dim):
    if dim==1:
        return nn.Conv1d
    elif dim==2:
        return nn.Conv2d
    elif dim==3:
        return nn.Conv3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')

def DimBatchNorm(dim):
    if dim==1:
        return nn.BatchNorm1d
    elif dim==2:
        return nn.BatchNorm2d
    elif dim==3:
        return nn.BatchNorm3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')

def DimConvTranspose(dim):
    if dim==1:
        return nn.ConvTranspose1d
    elif dim==2:
        return nn.ConvTranspose2d
    elif dim==3:
        return nn.ConvTranspose3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')

def DimMaxPool(dim):
    if dim==1:
        return nn.MaxPool1d
    elif dim==2:
        return nn.MaxPool2d
    elif dim==3:
        return nn.MaxPool3d
    else:
        raise ValueError('Only supported for dimensions 1, 2, and 3')



class conv_bn_rel(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride=1, active_unit='relu', same_padding=False,
                 bn=False, reverse=False,group = 1,dilation = 1):
        super(conv_bn_rel, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        if not reverse:
            self.conv = DimConv(dim)(in_channels, out_channels, kernel_size, stride, padding=padding, groups=group,dilation=dilation)
        else:
            self.conv = DimConvTranspose(dim)(in_channels, out_channels, kernel_size, stride, padding=padding,groups=group,dilation=dilation)

        self.bn = DimBatchNorm(dim)(out_channels, eps=0.0001, momentum=0, affine=True) if bn else None
        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        elif active_unit =='leaky_relu':
            self.active_unit = nn.LeakyReLU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x


###########################################

# actual definitions of the UNets

class encoder_decoder_small(nn.Module):
    """
    unet include 2 down path (1/4)  and 2 up path (4)
    """
    def __init__(self, dim, n_in_channel, n_out_channel):
        # each dimension of the input should be 16x
        super(encoder_decoder_small,self).__init__()
        self.down_path_1 = conv_bn_rel(dim, n_in_channel, 8, 4, stride=2, active_unit='relu', same_padding=True, bn=False)
        self.down_path_2 = conv_bn_rel(dim, 8, 16, 4, stride=2, active_unit='relu', same_padding=True, bn=False)

        # output_size = strides * (input_size-1) + kernel_size - 2*padding

        self.up_path_2 = conv_bn_rel(dim, 16, 8, 4, stride=2, active_unit='relu', same_padding=True, bn=False,reverse=True)
        self.up_path_1 = conv_bn_rel(dim, 8, n_out_channel, 4, stride=2, active_unit='None', same_padding=True, bn=False, reverse=True)

    def forward(self, x):
        d1 = self.down_path_1(x)
        d2 = self.down_path_2(d1)

        u2 = self.up_path_2(d2)
        u1 = self.up_path_1(u2)

        output = u1

        return output

class Unet(nn.Module):
    """
    unet include 4 down path (1/16)  and 4 up path (16)
    """
    def __init__(self, dim, n_in_channel, n_out_channel):
        # each dimension of the input should be 16x
        super(Unet,self).__init__()
        self.down_path_1 = conv_bn_rel(dim, n_in_channel, 16, 3, stride=1, active_unit='relu', same_padding=True, bn=False)
        self.down_path_2_1 = conv_bn_rel(dim, 16, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        self.down_path_2_2 = conv_bn_rel(dim, 32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=False)
        self.down_path_4_1 = conv_bn_rel(dim, 32, 32, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        self.down_path_4_2 = conv_bn_rel(dim, 32, 32, 3, stride=1, active_unit='relu', same_padding=True, bn=False)
        self.down_path_8_1 = conv_bn_rel(dim, 32, 64, 3, stride=2, active_unit='relu', same_padding=True, bn=False)
        self.down_path_8_2 = conv_bn_rel(dim, 64, 64, 3, stride=1, active_unit='relu', same_padding=True, bn=False)
        self.down_path_16 = conv_bn_rel(dim, 64, 64, 3, stride=2, active_unit='relu', same_padding=True, bn=False)


        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.up_path_8_1 = conv_bn_rel(dim, 64, 64, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=False,reverse=True)
        self.up_path_8_2 = conv_bn_rel(dim, 128, 64, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=False)
        self.up_path_4_1 = conv_bn_rel(dim, 64, 64, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=False,reverse=True)
        self.up_path_4_2 = conv_bn_rel(dim, 96, 32, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=False)
        self.up_path_2_1 = conv_bn_rel(dim, 32, 32, 2, stride=2, active_unit='leaky_relu', same_padding=False, bn=False,reverse=True)
        self.up_path_2_2 = conv_bn_rel(dim, 64, 8, 3, stride=1, active_unit='leaky_relu', same_padding=True, bn=False)
        self.up_path_1_1 = conv_bn_rel(dim, 8, 8, 2, stride=2, active_unit='None', same_padding=False, bn=False, reverse=True)
        self.up_path_1_2 = conv_bn_rel(dim, 24, n_out_channel, 3, stride=1, active_unit='None', same_padding=True, bn=False)

    def forward(self, x):
        d1 = self.down_path_1(x)
        d2_1 = self.down_path_2_1(d1)
        d2_2 = self.down_path_2_2(d2_1)
        d4_1 = self.down_path_4_1(d2_2)
        d4_2 = self.down_path_4_2(d4_1)
        d8_1 = self.down_path_8_1(d4_2)
        d8_2 = self.down_path_8_2(d8_1)
        d16 = self.down_path_16(d8_2)


        u8_1 = self.up_path_8_1(d16)
        u8_2 = self.up_path_8_2(torch.cat((d8_2,u8_1),1))
        u4_1 = self.up_path_4_1(u8_2)
        u4_2 = self.up_path_4_2(torch.cat((d4_2,u4_1),1))
        u2_1 = self.up_path_2_1(u4_2)
        u2_2 = self.up_path_2_2(torch.cat((d2_2, u2_1), 1))
        u1_1 = self.up_path_1_1(u2_2)
        output = self.up_path_1_2(torch.cat((d1, u1_1), 1))

        return output

# custom loss function

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, spacing):
        volumeElement = spacing.prod()
        batch_size = x.size()[0]
        b = x*torch.log(x)
        #F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        #b = -1.0 * b.sum()*volumeElement/batch_size
        b = -1.0*b.sum()*volumeElement/batch_size
        return b

class GlobalHLoss(nn.Module):
    def __init__(self):
        super(GlobalHLoss, self).__init__()

    def forward(self, x, spacing):

        nr_of_labels = x.size()[1]
        P = MyTensor(nr_of_labels).zero_()
        sz = list(x.size())
        nr_of_elements = [sz[0]] + sz[2:]
        current_norm = float(np.array(nr_of_elements).prod().astype('float32')) # number of pixels
        for n in range(nr_of_labels):
            P[n] = (x[:,n,...].sum())/current_norm

        # now compute the entropy over this
        b = MyTensor(1).zero_()
        for n in range(nr_of_labels):
            b = b-P[n]*torch.log(P[n])

        return b

class TotalVariationLoss(nn.Module):
    """
    Loss function for image clustering (this is here a relaxation of normalized cuts)
    """

    def __init__(self, dim, params):
        """

        :param params: ParameterDict() object to hold and keep track of general parameters
        """
        super(TotalVariationLoss, self).__init__()
        self.params = params
        """ParameterDict() parameters"""
        self.dim = dim
        # dimension

    # todo: merge this with the code in deep_smoothers.py
    def compute_local_weighted_tv_norm(self, I, weights, spacing, nr_of_gaussians, pnorm=2):

        volumeElement = spacing.prod()
        sum_square_of_total_variation_penalty = MyTensor(nr_of_gaussians).zero_()
        # first compute the edge map
        g_I = deep_smoothers.compute_localized_edge_penalty(I[:, 0, ...], spacing, self.params)
        batch_size = I.size()[0]

        # now computed weighted TV norm channel-by-channel, square it and then take the square root (this is like in color TV)
        for g in range(nr_of_gaussians):
            c_local_norm_grad = deep_smoothers._compute_local_norm_of_gradient(weights[:, g, ...], spacing, pnorm)

            to_sum = g_I * c_local_norm_grad * volumeElement / batch_size
            current_tv = (to_sum).sum()
            sum_square_of_total_variation_penalty[g] = current_tv**2

        total_variation_penalty = torch.norm(sum_square_of_total_variation_penalty,p=2)
        return total_variation_penalty


    def forward(self, input_images, spacing, label_probabilities):
        # first compute the weighting functions

        nr_of_gaussians = label_probabilities.size()[1]
        current_penalty = self.compute_local_weighted_tv_norm(input_images, label_probabilities, spacing, nr_of_gaussians)
        return current_penalty


class ClusteringLoss(nn.Module):
    """
    Loss function for image clustering (this is here a relaxation of normalized cuts)
    """

    def __init__(self, dim, params):
        """

        :param params: ParameterDict() object to hold and keep track of general parameters
        """
        super(ClusteringLoss, self).__init__()
        self.params = params
        """ParameterDict() parameters"""
        self.dim = dim
        # dimension

    def _compute_cut_cost_for_label_k_1d(self,w_edge,p):
        raise ValueError('Not yet implemented')

    def _compute_cut_cost_for_label_k_3d(self, w_edge, p):
        raise ValueError('Not yet implemented')

    def _compute_cut_cost_for_label_k_2d(self,w_edge,p):
        # todo: maybe the current cost is not ideal and we should try the one from the paper
        # todo: but at least it would be consistent with our total variation implementation
        batch_size = p.size()[0]
        cut_cost = AdaptVal(torch.zeros(batch_size))

        # needs to be batch B x X x Y x Z format (as input)
        fdt = fd.FD_torch(spacing=np.array([1.0]*self.dim))
        p_xp = fdt.dXf(p)
        p_yp = fdt.dYf(p)

        for b in range(batch_size):

            nom = (p[b,...]*(p[b,...] + w_edge[b,...]*(p_xp[b,...]+p_yp[b,...]))).sum()
            denom = (p[b,...]*(1.0+2*w_edge[b,...])).sum()

            cut_cost[b] = nom/denom

        return cut_cost

    def _compute_cut_cost_for_label_k(self, w_edge, p):
        if self.dim==1:
            return self._compute_cut_cost_for_label_k_1d(w_edge, p)
        elif self.dim==2:
            return self._compute_cut_cost_for_label_k_2d(w_edge, p)
        elif self.dim==3:
            return self._compute_cut_cost_for_label_k_3d(w_edge, p)
        else:
            raise ValueError('Only defined for dimensions {1,2,3}')

    def forward(self, input_images, spacing, label_probabilities):
        # first compute the weighting functions
        localized_edge_penalty = deep_smoothers.compute_localized_edge_penalty(input_images[:,0,...],spacing,self.params)

        batch_size = label_probabilities.size()[0]
        nr_of_clusters = label_probabilities.size()[1]
        current_penalties = AdaptVal(torch.ones(batch_size)*nr_of_clusters)

        for k in range(nr_of_clusters):
            current_penalties -= self._compute_cut_cost_for_label_k(w_edge=localized_edge_penalty,p=label_probabilities[:,k,...])

        current_penalty = current_penalties.sum()
        return current_penalty


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

dim = 2
nr_of_desired_clusters = 3
batch_size = 20
nr_of_epochs = 100
visualize_intermediate_results = True
only_display_epoch_results = True
image_offset = 1.0

reconstruction_weight = 10.0
totalvariation_weight = 1.0
entropy_weight = 25.0
global_entropy_weight = 1.0


image_input_directory = '../experiments/synthetic_example_out_kernel_weighting_type_w_K/brain_affine_icbm/'
input_images = glob.glob(os.path.join(image_input_directory,'*.nii'))
params = pars.ParameterDict()

dataset = ImageDataset(image_filenames=input_images,params=params)
trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)


#network_type = encoder_decoder_small
network_type = Unet

# create the two Unets, the first one to create the clustering and the second one for reconstruction
cluster_unet = network_type(dim=dim,n_in_channel=1,n_out_channel=nr_of_desired_clusters).to(device)
print(cluster_unet)

reconstruction_unet = network_type(dim=dim,n_in_channel=nr_of_desired_clusters,n_out_channel=1).to(device)
print(reconstruction_unet)

# create the loss functions
reconstruction_criterion = nn.L1Loss().to(device)
totalvariation_criterion = TotalVariationLoss(dim=dim,params=params).to(device)
entropy_criterion = HLoss().to(device)
global_entropy_criterion = GlobalHLoss().to(device)
clustering_criterion = ClusteringLoss(dim=dim,params=params).to(device)

# create the optimizer
all_optimization_parameters = list(cluster_unet.parameters()) + list(reconstruction_unet.parameters())
optimizer = optim.SGD(all_optimization_parameters, lr=0.00025, momentum=0.9, nesterov=True)

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
        inputs = input_dict['ISource'].to(device)-image_offset
        spacing = input_dict['spacing'][0].detach().cpu().numpy() # all of the spacings are the same, so just use one
        # for now this is just like an auto-encoder
        desired_outputs = inputs

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        cluster_outputs = cluster_unet(inputs)
        # now map this through a softmax to obtain the label probabilities
        label_probabilities  = deep_smoothers.stable_softmax(cluster_outputs,dim=1)
        ## compute the clustering loss
        #clustering_loss = clustering_criterion(input_images=inputs,spacing=spacing,label_probabilities=label_probabilities)
        clustering_loss = AdaptVal(torch.zeros(1))

        entropy_loss = entropy_weight*entropy_criterion(label_probabilities,spacing=spacing)
        # we want to maximize global entropy so let's minimize the negative one
        negative_global_entropy_loss = -global_entropy_weight*global_entropy_criterion(label_probabilities,spacing=spacing)
        totalvariation_loss = totalvariation_weight*totalvariation_criterion(input_images=inputs,spacing=spacing,label_probabilities=label_probabilities)

        # try to reconstruct the image
        reconstruction_outputs = reconstruction_unet(cluster_outputs)
        # compute the reconstruction loss
        reconstruction_loss = reconstruction_weight*reconstruction_criterion(reconstruction_outputs, desired_outputs)

        # compute the overall loss as a combination of reconstruction and clustering loss
        loss = reconstruction_loss + clustering_loss + totalvariation_loss + entropy_loss + negative_global_entropy_loss
        # compute the gradient
        loss.backward()

        optimizer.step()

        if only_display_epoch_results:
            running_loss += loss.item()/nr_of_datasets
            running_reconstruction_loss += reconstruction_loss.item()/nr_of_datasets
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