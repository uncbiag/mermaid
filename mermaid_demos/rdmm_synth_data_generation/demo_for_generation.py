
from __future__ import print_function
from __future__ import absolute_import
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ''

sys.path.insert(0,os.path.abspath('../..'))
import numpy as np

import torch

import random
from glob import glob
import argparse

from mermaid_demos.rdmm_synth_data_generation.combine_shape import generate_shape_objects,get_image,randomize_pair
from mermaid_demos.rdmm_synth_data_generation.moving_shape import MovingShape, MovingShapes
from mermaid_demos.rdmm_synth_data_generation.utils_for_regularizer import get_single_gaussian_smoother
from mermaid_demos.rdmm_synth_data_generation.utils_for_general import add_texture_on_img,plot_2d_img, save_smoother_map,write_list_into_txt, get_file_name
from mermaid.data_wrapper import MyTensor
from multiprocessing import *
import progressbar as pb
from functools import partial
import matplotlib.pyplot as plt



multi_gaussian_weight = np.array([0.2,0.5,0.3,0.0])
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

    moving_shapes =MovingShapes(img_sz, moving_shape_list, default_multi_gaussian_weight, local_smoother = local_smoother)
    weight_map = moving_shapes.create_weight_map()
    return weight_map


def generate_random_pairs(fpath=None,fname=None,random_state=None):
    img_sz = [200,200]
    fg_setting={'type':['ellipse','rect'], 'scale':[0.8,0.9],'t_var':0.1}
    bg_setting={'type':['ellipse','rect','tri'], 'scale':[0.15,0.25],'t_var':0.1}
    overlay_num=2
    bg_num = 6
    add_texture=False
    complete_flg =False
    while not complete_flg:
        s_setting_list, t_setting_list, complete_flg = randomize_pair(img_sz=img_sz,fg_setting=fg_setting,bg_setting=bg_setting,overlay_num=overlay_num,bg_num=bg_num,random_state=random_state)
        print("failed to find an instance, now try again ...")
    get_and_save_img_and_weight(img_sz, s_setting_list, add_texture=add_texture, color=None, fpath=fpath, fname=fname + '_s')
    get_and_save_img_and_weight(img_sz, t_setting_list, add_texture=add_texture, color=None, fpath=fpath, fname=fname + '_t')

def visualize(img,visual):
    if visual:
        plt.style.use('bmh')
        fig, ax = plt.subplots()
        plt.imshow(img[0,0].cpu().numpy(),alpha =0.8)
        #fig.patch.set_visible(False)
        ax.axis('off')
        plt.show()

def smooth_img(img,img_sz,gaussian_std=0.02):
    spacing = 1./(np.array(img_sz)-1)
    local_smoother = get_single_gaussian_smoother(gaussian_std=gaussian_std,sz=img_sz,spacing=spacing)
    smoothed_img = local_smoother.smooth(img)
    return smoothed_img

def get_and_save_img_and_weight(img_sz,shape_setting_list,add_texture=False, color=None,fpath=None,fname=None):

    img,color = get_image(img_sz,shape_setting_list,color)

    if add_texture:
        img =add_texture_on_img(img.numpy())
    #img = smooth_img(img,img_sz,gaussian_std=0.01)
    #visualize(img,True)
    weight = get_adaptive_weight_map(img_sz,shape_setting_list)
    if fpath and fname:
        os.makedirs(fpath,exist_ok=True)
        pth = os.path.join(fpath,fname)
        torch.save(img,pth+'_img.pt')
        torch.save(weight, pth+'_weight.pt')
        plot_2d_img(img,name='img',path= pth+'_img.png')
        save_smoother_map(weight,gaussian_stds=MyTensor(multi_gaussian_stds),t=0,path=pth+'weight.png',weighting_type='w_K_w')


def get_txt_from_generated_images(folder_path,output_folder):
    """ to get the pair information from folder_path and then transfer into standard txt file"""
    output_folder = os.path.join(output_folder,'test')
    os.makedirs(output_folder,exist_ok=True)
    s_post = 's_img.pt'
    t_post = 't_img.pt'
    s_weight ='s_weight.pt'
    t_weight ='t_weight.pt'
    folder_path = os.path.abspath(folder_path)
    s_post_path = os.path.join(folder_path, '**','*'+s_post )
    s_path_list = glob(s_post_path, recursive=True)
    t_path_list = [s_path.replace(s_post,t_post) for s_path in s_path_list]
    sw_path_list = [s_path.replace(s_post,s_weight) for s_path in s_path_list]
    tw_path_list = [s_path.replace(s_post,t_weight) for s_path in s_path_list]
    num_pair = len(s_path_list)
    st_path_list = [[s_path_list[i],t_path_list[i]] for i in range(num_pair)]
    stw_path_list = [[sw_path_list[i],tw_path_list[i]] for i in range(num_pair)]
    st_name_list = [get_file_name(s_path_list[i])+'_'+ get_file_name(t_path_list[i]) for i in range(num_pair)]
    st_path_txt = os.path.join(output_folder,'pair_path_list.txt')
    stw_path_txt = os.path.join(output_folder,'pair_weight_path_list.txt')
    st_name_txt = os.path.join(output_folder,'pair_name_list.txt')
    write_list_into_txt(st_path_txt,st_path_list)
    write_list_into_txt(stw_path_txt,stw_path_list)
    write_list_into_txt(st_name_txt,st_name_list)


# def get_reg_pair(fpath,fname):
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


def get_data(folder_path=None):
    fpath = folder_path
    fname = 'debug'
    num_pairs_to_generate = 40
    random_seed = 2018
    np.random.seed(random_seed)
    random.seed(random_seed)
    random_state = np.random.RandomState(random_seed)
    num_of_workers = 20
    sub_p = partial(sub_process, fpath=fpath, fname=fname)
    if num_of_workers > 1:
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


def get_txt(folder_path=None,output_folder=None):
    get_txt_from_generated_images(folder_path, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a synthetic registration examples for RDMM related experiments')
    parser.add_argument('-dp','--data_saving_path', required=False, type=str, default='./data_output',
                        help='path of the folder saving synthesis data')
    parser.add_argument('-di','--data_task_path', required=False, type=str, default='./data_task',
                        help='path of the folder recording data info for registration tasks')
    args = parser.parse_args()
    data_saving_path =args.data_saving_path  #'/playpen/zyshen/debugs/syn_expr_0708' '/playpen/zyshen/debugs/syn_expr_0422_2'
    data_task_path = args.data_task_path  #'/playpen/zyshen/data/syn_data/test_0708'  '/playpen/zyshen/data/syn_data/test''
    get_data(data_saving_path)
    get_txt(data_saving_path,data_task_path)


















