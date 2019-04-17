
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import os
import torch
import sys
import random
sys.path.insert(0,'/playpen/zyshen/reg_clean/mermaid')

from combine_shape import generate_shape_objects,get_image,randomize_pair
from moving_shape import MovingShape, MovingShapes
from utils_for_regularizer import get_single_gaussian_smoother
from utils_for_general import add_texture_on_img
from tools.visual_tools import plot_2d_img,save_smoother_map
from mermaid.pyreg.data_wrapper import MyTensor
from multiprocessing import *
import progressbar as pb
from functools import partial

import matplotlib.pyplot as plt
multi_gaussian_weight = np.array([0.2,0.5,0.2,0.1])
default_multi_gaussian_weight = np.array([0,0,0,1.])
multi_gaussian_stds = np.array([0.05,0.1,0.15,0.2])
def get_settings_for_shapes(img_sz):
    shape_setting_list = [
    {'name':'e1','type':'ellipse','img_sz':img_sz,'center_pos':[-0.0,-0.1],'radius':[0.5,0.7],'rotation':0,'use_weight':True},
    {'name':'c1','type':'circle','img_sz':img_sz,'center_pos':[-0.2,0.1],'radius':0.12,'use_weight':True},
    {'name':'poly_1','type':'poly','img_sz':img_sz,'vertices':[[0.0,0.2,0.3,0.1],[-0.3,-0.4,-0.3,-0.05]],'rotation':30,'use_weight':True},
    {'name':'e2','type':'ellipse','img_sz':img_sz,'center_pos':[-0.1,0.1],'radius':[0.6,0.8],'rotation':30,'use_weight':True},
    {'name':'tri_2','type':'tri','img_sz':img_sz,'center_pos':[-0.3,0.1],'height':0.3,'width':0.3,'rotation':30,'use_weight':True},
    {'name':'rec_2','type':'rect','img_sz':img_sz,'center_pos':[0.1,-0.4],'height':0.4,'width':0.2,'rotation':45,'use_weight':True},
    ]
    return shape_setting_list





def get_shapes_settings(img_sz, index= [0,1,2]):
    shape_setting_list = get_settings_for_shapes(img_sz)
    shape_setting_list = [shape_setting_list[i] for i in index]
    return shape_setting_list



def get_adaptive_weight_map(img_sz,shape_setting_list):
    spacing = 1./(np.array(img_sz)-1)
    shape_list = generate_shape_objects(shape_setting_list)
    moving_shape_list = []
    local_smoother = get_single_gaussian_smoother(gaussian_std=0.02,sz=img_sz,spacing=spacing)
    for i,shape in enumerate(shape_list):
         using_weight = shape_setting_list[i]['use_weight']
         moving_shape =  MovingShape(shape, multi_gaussian_weight, using_weight=using_weight, weight_type='w_K_w')
         moving_shape_list.append(moving_shape)

    moving_shapes =MovingShapes(img_sz, moving_shape_list, default_multi_gaussian_weight, multi_gaussian_stds, local_smoother = local_smoother)
    weight_map = moving_shapes.create_weight_map()
    return weight_map


def generate_random_pairs(fpath=None,fname=None,random_state=None):
    img_sz = [200,200]
    fg_setting={'type':['ellipse','rect'], 'scale':[0.7,0.9],'t_var':0.1}
    bg_setting={'type':['ellipse','rect','tri'], 'scale':[0.1,0.3],'t_var':0.1}
    overlay_num=3
    bg_num = 7
    add_texture=False
    complete_flg =False
    while not complete_flg:
        s_setting_list, t_setting_list, complete_flg = randomize_pair(img_sz=img_sz,fg_setting=fg_setting,bg_setting=bg_setting,overlay_num=overlay_num,bg_num=bg_num,random_state=random_state)
    get_and_save_img_and_weight(img_sz, s_setting_list, add_texture=add_texture, color=None, fpath=fpath, fname=fname + '_s')
    get_and_save_img_and_weight(img_sz, t_setting_list, add_texture=add_texture, color=None, fpath=fpath, fname=fname + '_t')



def get_and_save_img_and_weight(img_sz,shape_setting_list,add_texture=False, color=None,fpath=None,fname=None):

    img,color = get_image(img_sz,shape_setting_list,color)
    if add_texture:
        img =add_texture_on_img(img.numpy())
    weight = get_adaptive_weight_map(img_sz,shape_setting_list)
    if fpath and fname:
        os.makedirs(fpath,exist_ok=True)
        pth = os.path.join(fpath,fname)
        torch.save(img,pth+'_img.pt')
        torch.save(weight, pth+'_weight.pt')
        plot_2d_img(img,name='img',path= pth+'img.png')
        save_smoother_map(weight,gaussian_stds=MyTensor(multi_gaussian_stds),t=0,path=pth+'weight.png',weighting_type='w_K_w')


# def get_pair(fpath,fname):
#     img_sz =np.array([200,200])
#     index = [0,1,2]
#     shape_setting_list = get_shapes_settings(img_sz, index)
#     color = get_and_save_img_and_weight(img_sz, shape_setting_list, color=None,fpath=fpath,fname=fname+'_s')
#     index = [3,4,5]
#     shape_setting_list = get_shapes_settings(img_sz, index)
#     get_and_save_img_and_weight(img_sz,shape_setting_list,color=color,fpath=fpath,fname=fname+'_t')



def sub_process(index,fpath,fname):
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(index)).start()
    count = 0
    for i in index:
        generate_random_pairs(fpath, 'id_{:03d}_{}'.format(i, fname))
        count +=1
        pbar.update(count)
    pbar.finish()



if __name__ == '__main__':
    fpath = '/playpen/zyshen/debugs/syn_expr_0414'
    fname = 'debug'
    num_pairs_to_generate =40
    random_seed = 2018
    np.random.seed(random_seed)
    random.seed(random_seed)
    random_state = np.random.RandomState(random_seed)
    num_of_workers= 1
    sub_p = partial(sub_process,fpath=fpath,fname=fname)
    if num_of_workers>1:
        split_index = np.array_split(np.array(range(num_pairs_to_generate)), num_of_workers)
        procs = []
        for i in range(num_of_workers):
            p = Process(target=sub_p, args=(split_index[i],))
            p.start()
            print("pid:{} start:".format(p.pid))
            procs.append(p)
        for p in procs:
            p.join()

    else:
        for i in range(num_pairs_to_generate):
            if i % 5 == 0:
                print("generating the {} the pair".format(i))
            generate_random_pairs(fpath, 'id_{:03d}_{}'.format(i, fname), random_state)


















