from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from builtins import object
from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass

import torch
import torch.nn as nn
from . import finite_differences as fd
from . import deep_networks as dn
from .data_wrapper import USE_CUDA, MyTensor, AdaptVal

class AdaptiveWeightLoss(with_metaclass(ABCMeta,nn.Module)):
    def __init__(self, nr_of_gaussians, gaussian_stds, dim, spacing, im_sz=None, omt_power=1.0,
                 params=None):
        super(AdaptiveWeightLoss, self).__init__()

        self.dim = dim
        self.pnorm = 2
        self.print_count = 0.
        self.print_every_n_iter = 30
        self.epoch = None

        self.spacing = spacing
        self.fdt = fd.FD_torch(self.spacing)
        self.volumeElement = self.spacing.prod()
        cparams = params[('deep_smoother', {})]
        self.params = cparams

        # check that the largest standard deviation is the largest one
        if gaussian_stds.max() > gaussian_stds[-1]:
            raise ValueError('The last standard deviation needs to be the largest')
        self.use_weighted_linear_softmax = params[('use_weighted_linear_softmax', True,
                                                   'If set to ture use the use_weighted_linear_softmax to compute the pre-weights, otherwise use stable softmax')]  # 25
        if self.use_weighted_linear_softmax:
            print(" the weighted_linear_softmax is used")
        else:
            print(" the stable softmax is used")
        self.omt_weight_penalty = params[('omt_weight_penalty', 25, 'Penalty for the optimal mass transport')]  # 25
        self.omt_use_log_transformed_std = params[('omt_use_log_transformed_std', True,
                                                   'If set to true the standard deviations are log transformed for the computation of OMT')]
        """if set to true the standard deviations are log transformed for the OMT computation"""

        self.omt_power = params[
            ('omt_power', 1.0, 'Power for the optimal mass transport (i.e., to which power distances are penalized')]
        """optimal mass transport power"""

        self.gaussianWeight_min = params[('gaussian_weight_min', 0.001, 'minimal allowed weight for the Gaussians')]
        """minimal allowed weight during optimization"""

        self.preweight_input_range_weight_penalty = params[('preweight_input_range_weight_penalty', 1.0,
                                                            'Penalty for the input to the preweight computation; weights should be between 0 and 1. If they are not they get quadratically penalized; use this with weighted_linear_softmax only.')]

        self.weighting_type = self.params[
            ('weighting_type', 'sqrt_w_K_sqrt_w', 'Type of weighting: w_K|w_K_w|sqrt_w_K_sqrt_w')]
        admissible_weighting_types = ['w_K', 'w_K_w', 'sqrt_w_K_sqrt_w']
        if self.weighting_type not in admissible_weighting_types:
            raise ValueError('Unknown weighting_type: needs to be  w_K|w_K_w|sqrt_w_K_sqrt_w')

        self.diffusion_weight_penalty = self.params[
            ('diffusion_weight_penalty', 0.0, 'Penalized the squared gradient of the weights')]
        self.total_variation_weight_penalty = self.params[
            ('total_variation_weight_penalty', 0.1, 'Penalize the total variation of the weights if desired')]
        self.weight_range_init_weight_penalty = self.params[
            ('weight_range_init_weight_penalty', 0., 'Penalize to the range of the weights')]
        self.weight_range_epoch_factor = self.params[
            ('weight_range_factor', 6, 'the factor control the change of the penality ')]


        self.nr_of_gaussians = nr_of_gaussians
        self.gaussian_stds = gaussian_stds

        self.computed_weights = None
        """stores the computed weights if desired"""

        self.computed_pre_weights = None
        """stores the computed pre weights if desired"""

        self.current_penalty = 0.
        """to stores the current penalty (for example OMT) after running through the model"""

        self.deep_network_local_weight_smoothing = self.params[('deep_network_local_weight_smoothing', 0.02,
                                                                '0.02 prefered,How much to smooth the local weights (implemented by smoothing the resulting velocity field) to assure sufficient regularity')]
        """Smoothing of the local weight fields to assure sufficient regularity of the resulting velocity"""

        self.deep_network_weight_smoother = None
        """The smoother that does the smoothing of the weights; needs to be initialized in the forward model"""

        """These are parameters for the edge detector; put them here so that they are generated in the json file"""
        """This allows propagating the parameter between stages"""
        """There are not used for anything directly here"""
        self.params[('edge_penalty_gamma', 10.0, 'Constant for edge penalty: 1.0/(1.0+gamma*||\\nabla I||*min(spacing)')]
        self.params[('edge_penalty_write_to_file', False,
                     'If set to True the edge penalty is written into a file so it can be debugged')]
        self.params[('edge_penalty_filename', 'DEBUG_edge_penalty.nrrd', 'Edge penalty image')]
        self.params[('edge_penalty_terminate_after_writing', False,
                     'Terminates the program after the edge file has been written; otherwise file may be constantly overwritten')]

        self.estimate_around_global_weights = self.params[('estimate_around_global_weights', True,
                                                           'If true, a weighted softmax is used so the default output (for input zero) are the global weights')]

        self.network_penalty = self.params[
            ('network_penalty', 1e-5, 'factor by which the L2 norm of network weights is penalized')]
        """penalty factor for L2 norm of network weights"""

        # loss functions
        self.tv_loss = dn.TotalVariationLoss(dim=dim, im_sz=im_sz, spacing=spacing,
                                             use_omt_weighting=False,
                                             gaussian_stds=self.gaussian_stds,
                                             omt_power=self.omt_power,
                                             omt_use_log_transformed_std=self.omt_use_log_transformed_std,
                                             params=self.params)
        if USE_CUDA:
            self.tv_loss= self.tv_loss.cuda()

        self.omt_loss = dn.OMTLoss(spacing=spacing, desired_power=self.omt_power,
                                   use_log_transform=self.omt_use_log_transformed_std, params=params,img_sz=im_sz)
        if USE_CUDA:
            self.omt_loss= self.omt_loss.cuda()

        self.preweight_input_range_loss = dn.WeightInputRangeLoss()
        self.weight_range_loss = dn.WeightRangeLoss(self.dim, self.weight_range_epoch_factor,self.weighting_type)



    def _compute_penalty_from_weights_preweights_and_input_to_preweights(self, I, weights, pre_weights, input_to_preweights=None,
                                                                         global_multi_gaussian_weights=None):
        # compute the total variation penalty; compute this on the pre (non-smoothed) weights
        total_variation_penalty = MyTensor(1).zero_()
        if self.total_variation_weight_penalty > 0:
            # total_variation_penalty += self.compute_local_weighted_tv_norm(I=I,weights=pre_weights)
            total_variation_penalty += self.tv_loss(input_images=I, label_probabilities=pre_weights, use_color_tv=True)

        diffusion_penalty = MyTensor(1).zero_()
        if self.diffusion_weight_penalty > 0:
            for g in range(self.nr_of_gaussians):
                diffusion_penalty += self.compute_diffusion(pre_weights[:, g, ...])

        current_diffusion_penalty = self.diffusion_weight_penalty * diffusion_penalty
        omt_penalty = MyTensor(1).zero_()
        omt_epoch_factor = 1.
        if self.omt_weight_penalty > 0:
            if self.weighting_type == 'w_K_w':
                omt_penalty = self.omt_loss(weights=weights ** 2, gaussian_stds=self.gaussian_stds)
            else:
                omt_penalty = self.omt_loss(weights=weights, gaussian_stds=self.gaussian_stds)
                # omt_epoch_factor =self.omt_loss.cal_weights_for_omt(self.epoch)
        preweight_input_range_penalty = MyTensor(1).zero_()
        if self.preweight_input_range_weight_penalty > 0:
            if self.estimate_around_global_weights:
                preweight_input_range_penalty = self.preweight_input_range_loss(input_to_preweights,
                                                                                spacing=self.spacing,
                                                                                use_weighted_linear_softmax=True,
                                                                                weights=global_multi_gaussian_weights,
                                                                                min_weight=self.gaussianWeight_min,
                                                                                max_weight=1.0,
                                                                                dim=1)
            else:
                preweight_input_range_penalty = self.preweight_input_range_loss(input_to_preweights,
                                                                                spacing=self.spacing,
                                                                                use_weighted_linear_softmax=False,
                                                                                weights=None,
                                                                                min_weight=self.gaussianWeight_min,
                                                                                max_weight=1.0,
                                                                                dim=None)
        weight_range_penalty = MyTensor(1).zero_()
        weight_range_epoch_factor = 0.
        if self.weight_range_init_weight_penalty > 0:
            weight_range_penalty = self.weight_range_loss(weights, self.spacing, global_multi_gaussian_weights)
            assert self.epoch >= 0
            weight_range_epoch_factor = self.weight_range_loss.cal_weights_for_weightrange(self.epoch)
        balance_factor = (1. - weight_range_epoch_factor) if self.weight_range_init_weight_penalty > 0 else 1.
        current_omt_penalty = self.omt_weight_penalty * omt_epoch_factor * omt_penalty * balance_factor
        current_tv_penalty = self.total_variation_weight_penalty * total_variation_penalty* balance_factor
        current_preweight_input_range_penalty = self.preweight_input_range_weight_penalty * preweight_input_range_penalty
        current_range_weight_penalty = self.weight_range_init_weight_penalty * weight_range_epoch_factor * weight_range_penalty
        current_penalty = current_omt_penalty + current_tv_penalty + current_diffusion_penalty + current_preweight_input_range_penalty + current_range_weight_penalty

        current_batch_size = I.size()[0]

        if self.print_count % self.print_every_n_iter == 0:
            print("loss counting epoch:{} and  current balance factor for omt loss is {}".format(self.epoch,balance_factor))
            print('     TV/TV_penalty = ' + str(total_variation_penalty.item() / current_batch_size) + '/' \
                  + str(current_tv_penalty.item() / current_batch_size) + \
                  '; OMT/OMT_penalty = ' + str(omt_penalty.item() / current_batch_size) + '/' \
                  + str(current_omt_penalty.item() / current_batch_size) + \
                  '; WR/WR_penalty = ' + str(weight_range_penalty.item() / current_batch_size) + '/' \
                  + str(current_range_weight_penalty.item() / current_batch_size) + \
                  '; PWI/PWI_penalty = ' + str(preweight_input_range_penalty.item() / current_batch_size) + '/' \
                  + str(current_preweight_input_range_penalty.item() / current_batch_size) + \
                  '; diffusion_penalty = ' + str(current_diffusion_penalty.item() / current_batch_size))
            with torch.no_grad():
                self.displacy_weight_info(weights, global_multi_gaussian_weights)
                print("current gaussian std is {}".format(self.gaussian_stds))
        self.print_count += 1

        return current_penalty, current_omt_penalty, current_tv_penalty, current_diffusion_penalty, current_preweight_input_range_penalty


    def displacy_weight_info(self,weight,global_multi_gaussian_weights):
        def _display_stats(weights, weighted_std_map,global_multi_gaussian_weights,weighting_type):
            global_multi_gaussian_weights = global_multi_gaussian_weights.detach().cpu().numpy()
            weights_for_each_channel = [weights[:,i,...].mean().item() for i in range(weights.shape[1])]
            weights_for_each_channel = ["%.2f" % w for w in weights_for_each_channel]
            wstd_min = weighted_std_map.min().detach().cpu().numpy()
            wstd_max = weighted_std_map.max().detach().cpu().numpy()
            wstd_mean = weighted_std_map.mean().detach().cpu().numpy()
            wstd_std = weighted_std_map.std().detach().cpu().numpy()

            print('gbw: {},weight:{}; combined std: [{:.2f},{:.2f},{:.2f}]({:.2f})'.format(global_multi_gaussian_weights,weights_for_each_channel,wstd_min,wstd_mean,wstd_max,wstd_std))
        weight = weight**2 if self.weighting_type =='w_k_w' else weight
        weight = weight.detach()
        gaussian_stds = self.gaussian_stds
        gaussian_stds = gaussian_stds.detach()
        view_sz = [1] + [len(gaussian_stds)] + [1] * self.dim
        gaussian_stds = gaussian_stds.view(*view_sz)
        weighted_std_map = weight * (gaussian_stds ** 2)
        weighted_std_map = torch.sqrt(torch.sum(weighted_std_map, 1, keepdim=True))
        _display_stats(weight.float(),weighted_std_map.float(),global_multi_gaussian_weights,self.weighting_type)
