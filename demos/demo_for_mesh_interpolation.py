import os
import sys

sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../pyreg'))
sys.path.insert(0,os.path.abspath('../pyreg/libraries'))
import matplotlib as matplt
from pyreg.config_parser import MATPLOTLIB_AGG
if MATPLOTLIB_AGG:
    matplt.use('Agg')
import numpy as np
import torch
import torch.nn.functional as F
import random
from pyreg.utils import apply_affine_transform_to_map_multiNC, get_inverse_affine_param, compute_warped_image_multiNC, \
    update_affine_param, identity_map_multiN
import pyreg.simple_interface as SI
import pyreg.fileio as FIO
torch.cuda.set_device(1)
    
def setup_pair_register():
    """
    in this step, we do some stepup for mermaid
    :return:
    """
    register_param = {}
    si = SI.RegisterImagePair()
    register_param['si'] = si
    register_param['model0_name']= 'affine_map'
    register_param['model1_name']= 'svf_vector_momentum_map'

    return register_param


def read_source_and_moving_img(moving_path, target_path):
    """
    read img into numpy
    :return:
    """
    im_io = FIO.ImageIO()

    moving, hdrc0, spacing0, _ = im_io.read_to_nc_format(filename=moving_path, intensity_normalize=True)
    target, hdrc, spacing1, _ = im_io.read_to_nc_format(filename=target_path, intensity_normalize=True)

    return moving, target, spacing1, hdrc0

def register_pair_img(moving, target, spacing, register_param, hdrc0=None):
    si = register_param['si']
    model0_name = register_param['model0_name']
    model1_name = register_param['model1_name']

    #moving[np.where(moving==0)]=0.6
    si.set_initial_map(None)
    si.register_images(moving, target, spacing,
                            model_name=model0_name,
                            use_multi_scale=True,
                       visualize_step=None,
                            rel_ftol=1e-7,
                            json_config_out_filename='output_cur_settings.json',
                            compute_inverse_map=True,
                            params='cur_settings.json')

    Ab = si.opt.optimizer.ssOpt.model.Ab
    affine_param = Ab.detach().cpu().numpy().reshape((4, 3))
    affine_param = np.transpose(affine_param)
    print(" the affine param is {}".format(affine_param))

    inv_Ab = get_inverse_affine_param(Ab.detach())
    #id_Ab = update_affine_param(Ab,inv_Ab)
    identity_map = torch.Tensor(identity_map_multiN(moving.shape, spacing)).cuda()
    inversed_affine_map = apply_affine_transform_to_map_multiNC(inv_Ab, identity_map)




    print("let's come to step 2 ")

    affine_map = si.opt.optimizer.ssOpt.get_map()
    si.opt = None
    si.set_initial_map(affine_map.detach())

    si.register_images(moving, target, spacing,
                            model_name=model1_name,
                            use_multi_scale=True,
                       visualize_step=None,
                            rel_ftol=1e-7,
                            json_config_out_filename='output_settings_lbfgs.json',
                            compute_inverse_map=True,
                            params='cur_settings_lbfgs.json')

    warped_image = si.get_warped_image()
    inversed_map_svf = si.get_inverse_map().detach()
    inversed_map =  compute_warped_image_multiNC(inversed_affine_map,inversed_map_svf,spacing,1)

    #inv_Ab = update_affine_param(inv_Ab, inv_Ab)
    #inversed_map = apply_affine_transform_to_map_multiNC(inv_Ab, inversed_map_svf)
    #forward_map = si.get_map()
    forward_map=si.opt.optimizer.ssOpt.get_map()
    forward_inversed_map =  compute_warped_image_multiNC(forward_map, inversed_map,spacing,1)
    difference = forward_inversed_map.detach().cpu().numpy()-identity_map.detach().cpu().numpy()
    print(difference)
    print(np.sum(difference))

    return inversed_map, warped_image


def read_mesh_into_tensor():
    ##########  your code here ############33

    # example
    #  write a new function     read_mesh_into_tensor    B*3*N*1*1
    ##################    using randomized mesh for debugging   ###############################3
    mesh = torch.rand(1, 3, 200, 1, 1).cuda()

    return mesh

def do_points_interoplation(inversed_map, points, spacing):
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)

    #points_wraped = compute_warped_image_multiNC(inversed_map, poipointsnts, spacing=spacing, spline_order=1, zero_boundary=True)
    points_wraped = F.grid_sample(inversed_map, points.permute([0, 2, 3, 4, 1]), mode='bilinear', padding_mode='border')
    return points_wraped


def main():
    moving_img_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9085290_20040915_SAG_3D_DESS_LEFT_016610243503_image.nii.gz'
    target_img_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9085290_20051103_SAG_3D_DESS_LEFT_016610952703_image.nii.gz'

    register_param = setup_pair_register()
    moving,target, spacing, _ = read_source_and_moving_img(moving_img_path, target_img_path)
    inversed_map = register_pair_img(moving,target, spacing, register_param)
    mesh = read_mesh_into_tensor()
    interoplated_result = do_points_interoplation(inversed_map, mesh, spacing)


if __name__ == '__main__':
    main()