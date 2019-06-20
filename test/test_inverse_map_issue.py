#!/usr/bin/env python
"""
For debug the issue that the combination of forward map and inverse map is not an identity map
2D toy images are used for debugging
Created by zhenlinx on 9/4/18
"""

import sys
import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["CUDA_CACHE_PATH"] = "/playpen/zhenlinx/.cuda_cache"
sys.path.append(os.path.realpath(".."))
sys.path.append(os.path.realpath("../mermaid"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from mermaid.utils import apply_affine_transform_to_map_multiNC, get_inverse_affine_param, update_affine_param
import mermaid.simple_interface as SI
import mermaid.fileio as FIO
import mermaid.utils as pyreg_utils

from mermaid.data_wrapper import MyTensor, AdaptVal, USE_CUDA
device = torch.device("cuda:0" if (torch.cuda.is_available() and USE_CUDA) else "cpu")

import imageio

class Test2DRegister:
    def __init__(self, reg_models):
        """
        initialize with a sequence of registration mode
        :param reg_models: a list of tuples (model_name:string, model_setting: json_file_name(string) or an ParameterDict object)
        """
        self.si_ = SI.RegisterImagePair()
        self.model_0_name, self.model_0_setting = reg_models[0]
        self.model_1_name, self.model_1_setting = reg_models[1]
        self.im_io = FIO.ImageIO()

        self.target_image_np = None
        self.moving_image_np = None
        self.target_mask = None
        self.moving_mask = None

        self.Ab = None
        self.map = None
        self.inverse_map = None

    def set_target_image(self, target_image_path):
        """
        Read target image from given path and normalize their spacing such that the maximum size of all
        dimensions is 1.
        :param target_image_path:
        """
        if os.path.splitext(target_image_path)[-1] == "png":
            self.target_image_np = np.array(imageio.imread(target_image_path))
            self.target_spacing_normalized = 1/(self.target_image_np.shape)

        else:
            self.target_image_np, self.hdrc_target, self.target_spacing_normalized, _ = self.im_io.read_to_nc_format(
                filename=target_image_path, intensity_normalize=True,
                silent_mode=True)


    def set_moving_image(self, moving_path):
        """
        Read moving image from given path and normalize their spacing such that the maximum size of all
        dimensions is 1.
        :param moving_path:
        """
        if os.path.splitext(moving_path)[-1] == "png":
            self.moving_image_np = np.array(imageio.imread(moving_path))
            self.moving_spacing_normalized = 1/(self.moving_image_np.shape)
        else:
            self.moving_image_np, self.hdrc_moving, self.moving_spacing_normalized, _ = self.im_io.read_to_nc_format(
                filename=moving_path, intensity_normalize=True,
                silent_mode=True)

    def register_image_pair(self):
        self.register_image_pair_affine()
        self.register_image_pair_svf()

    def register_image_pair_affine(self, config_saving_path=None):

        print("\n######### Affine Registration Stage #########\n")

        # if there are init affine params
        if self.map is not None and self.inverse_map is not None:
            self.si_.set_initial_map(self.map, self.inverse_map)
            print("Initialize with existing Affine params")

        self.si_.register_images(self.moving_image_np, self.target_image_np, self.moving_spacing_normalized,
                                 model_name=self.model_0_name,
                                 use_multi_scale=True,
                                 rel_ftol=1e-7,
                                 visualize_step = 20,
                                 json_config_out_filename=config_saving_path,
                                 compute_inverse_map=True,
                                 params=self.model_0_setting)

        # if there are non-identity init affine params, combine them with affine registration result
        if self.Ab is not None:
            self.Ab = update_affine_param(self.Ab, self.si_.opt.optimizer.ssOpt.model.Ab.detach()) # forward affine parameters
        else:
            self.Ab = self.si_.opt.optimizer.ssOpt.model.Ab.detach()
        self.inv_Ab = get_inverse_affine_param(self.Ab)
        # affine_param = self.Ab.detach().cpu().numpy().reshape((4, 3))
        # affine_param = np.transpose(affine_param)
        # print(" the affine param is {}".format(affine_param))
        self.map = self.si_.opt.optimizer.ssOpt.get_map().detach()

        # self.map = apply_affine_transform_to_map_multiNC(
        #     self.Ab, torch.from_numpy(pyreg_utils.identity_map_multiN(self.moving_image_np.shape,
        #                                                                   self.moving_spacing_normalized)).cuda())

        self.inverse_map = apply_affine_transform_to_map_multiNC(
            self.inv_Ab, torch.from_numpy(pyreg_utils.identity_map_multiN(self.moving_image_np.shape,
                                                                          self.moving_spacing_normalized)).to(device))

    def register_image_pair_svf(self, config_saving_path=None):
        print("\n######### svf Registration Stage #########\n")
        self.si_.opt = None
        if self.map is not None and self.inverse_map is not None:
            self.si_.set_initial_map(self.map, self.inverse_map)

        self.si_.register_images(self.moving_image_np, self.target_image_np, self.moving_spacing_normalized,
                                 model_name=self.model_1_name,
                                 use_multi_scale=True,
                                 rel_ftol=1e-7,
                                 json_config_out_filename=config_saving_path,
                                 compute_inverse_map=True,
                                 params=self.model_1_setting)

        # WARNING: the inverse map from si_ is the combination of svf_inverse and initial FORWARD map, thus it have to
        # be combined with affine_inverse TWICE to get the correct final inverse map
        inversed_map_svf = self.si_.get_inverse_map().detach()
        # double_inv_Ab = update_affine_param(self.inv_Ab, self.inv_Ab)
        # self.inverse_map = apply_affine_transform_to_map_multiNC(self.inv_Ab, inversed_map_svf)
        self.map = self.si_.opt.optimizer.ssOpt.get_map().detach()
        self.inverse_map = inversed_map_svf

    def save_map(self, map_path, inverse_map_path=None):
        np.save(map_path, self.map.cpu().data)
        if inverse_map_path:
            np.save(inverse_map_path, self.inverse_map.cpu().numpy())

    def get_map(self):
        return self.map

    def get_inverse_map(self):
        return self.inverse_map

    def get_circular_map(self):

        return pyreg_utils.compute_warped_image_multiNC(self.map.to(device), self.inverse_map.to(device),
                                                    self.moving_spacing_normalized, 1, False).cpu().numpy().squeeze() / (self.moving_spacing_normalized.reshape((-1,) + (1,) * len(self.moving_spacing_normalized)))

    def show_maps(self, title=None):
        """Function to draw current map/inverse_map/circular_map"""
        circular_map = self.get_circular_map()
        forward_map = self.map.cpu().numpy().squeeze()/(self.target_spacing_normalized.reshape(2,1,1))
        inverse_map = self.inverse_map.cpu().numpy().squeeze()/(self.moving_spacing_normalized.reshape(2,1,1))

        map_shape = circular_map.shape
        x = np.arange(0, map_shape[1])
        y = np.arange(0, map_shape[2])
        X, Y = np.meshgrid(x, y)
        figs, axs = plt.subplots(2, 3)
        plt.title(title)
        CSX = axs[0,0].contour(X, Y, circular_map[0, :, :], list(range(0, map_shape[1], 40)), colors=['white'])
        axs[0,0].imshow(circular_map[0, :, :])
        axs[0,0].clabel(CSX, inline=1, fontsize=10)
        axs[0,0].set_title('Circular map\n X coutours')

        CSY = axs[1,0].contour(X, Y, circular_map[1, :, :], list(range(0, map_shape[2], 40)), colors=['white'])
        axs[1,0].imshow(circular_map[1, :, :])
        axs[1,0].clabel(CSY, inline=1, fontsize=10)
        axs[1,0].set_title('Y coutours')

        CSX = axs[0,1].contour(X, Y, forward_map[0, :, :], list(range(0, map_shape[1], 40)), colors=['white'])
        axs[0,1].imshow(forward_map[0, :, :])
        axs[0,1].clabel(CSX, inline=1, fontsize=10)
        axs[0,1].set_title('Forward map\n X coutours')

        CSY = axs[1,1].contour(X, Y, forward_map[1, :, :], list(range(0, map_shape[2], 40)), colors=['white'])
        axs[1,1].imshow(forward_map[1, :, :])
        axs[1,1].clabel(CSY, inline=1, fontsize=10)
        axs[1,1].set_title('Y coutours')

        CSX = axs[0,2].contour(X, Y, inverse_map[0, :, :], list(range(0, map_shape[1], 40)), colors=['white'])
        axs[0,2].imshow(inverse_map[0, :, :])
        axs[0,2].clabel(CSX, inline=1, fontsize=10)
        axs[0,2].set_title('Inverse map\n X coutours')

        CSY = axs[1,2].contour(X, Y, inverse_map[1, :, :], list(range(0, map_shape[2], 40)), colors=['white'])
        axs[1,2].imshow(inverse_map[1, :, :])
        axs[1,2].clabel(CSY, inline=1, fontsize=10)
        axs[1,2].set_title('Y coutours')
        plt.show()


def debug_inverse_map_issue():
    # moving_img_path = '../test_data/brain_slices/ws_slice.nrrd'
    # target_img_path = '../test_data/brain_slices/wt_slice.nrrd'
    moving_img_path = '../test_data/toy_2d/test1m.png'
    target_img_path = '../test_data/toy_2d/test1f.png'

    affine_setting_name = 'json/test_affine_settings.json'
    svf_setting_name = 'json/test_svf_settings.json'

    reg = Test2DRegister([('affine_map', affine_setting_name),
                           ('svf_vector_momentum_map', svf_setting_name)])
    reg.set_moving_image(moving_img_path)
    reg.set_target_image(target_img_path)

    reg.register_image_pair_affine()
    reg.show_maps('After Affine Registration')

    reg.register_image_pair_svf()
    reg.show_maps('After SVF Registration')


    pass




if __name__ == '__main__':
    debug_inverse_map_issue()