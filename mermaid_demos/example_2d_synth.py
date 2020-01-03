""""
demo on RDMM on 2d synthetic image registration
"""
import matplotlib as matplt
matplt.use('Agg')
import sys,os
os.environ["CUDA_VISIBLE_DEVICES"] = ''

import torch
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../mermaid'))
import mermaid.module_parameters as pars
import mermaid.simple_interface as SI
from mermaid.model_evaluation import evaluate_model, evaluate_model_low_level_interface
from mermaid.data_wrapper import AdaptVal,MyTensor
from mermaid.metrics import get_multi_metric
from skimage.draw._random_shapes import _generate_random_colors
import mermaid.finite_differences as fdt
import mermaid.utils as utils
import numpy as np
from multiprocessing import * 
import progressbar as pb
from functools import partial


def get_pair_list(folder_path):
    pair_path = os.path.join(folder_path,'pair_path_list.txt')
    fname_path = os.path.join(folder_path,'pair_name_list.txt')
    pair_path_list = read_txt_into_list(pair_path)
    pair_name_list = read_txt_into_list(fname_path)
    return pair_path_list, pair_name_list

def get_init_weight_list(folder_path):
    weight_path = os.path.join(folder_path,'pair_weight_path_list.txt')
    init_weight_path = read_txt_into_list(weight_path)
    return init_weight_path

def read_txt_into_list(file_path):
    import re
    lists= []
    with open(file_path,'r') as f:
        content = f.read().splitlines()
        if len(content)>0:
            lists= [[x if x!='None'else None for x in re.compile('\s*[,|\s+]\s*').split(line)] for line in content]
            lists = [list(filter(lambda x: x is not None, items)) for items in lists]
        lists = [item[0] if len(item) == 1 else item for item in lists]
    return lists


def get_mermaid_setting(path,output_path):
    params = pars.ParameterDict()
    params.load_JSON(path)
    os.makedirs(output_path,exist_ok=True)
    output_path = os.path.join(output_path,'mermaid_setting.json')
    params.write_JSON(output_path,save_int=False)

def setting_visual_saving(expr_folder,pair_name,expr_name='',folder_name='intermid'):
    extra_info = pars.ParameterDict()
    extra_info['expr_name'] = expr_name
    extra_info['visualize']=False
    extra_info['save_fig']=True
    extra_info['save_fig_path']=os.path.join(expr_folder,folder_name)
    extra_info['save_fig_num'] = -1
    extra_info['save_excel'] =False
    extra_info['pair_name'] = [pair_name]
    return  extra_info

def affine_optimization(moving,target,spacing,fname_list,l_moving=None,l_target=None):
    si = SI.RegisterImagePair()
    extra_info={}
    extra_info['pair_name'] = fname_list
    si.opt = None
    si.set_initial_map(None)
    si.register_images(moving, target, spacing,extra_info=extra_info,LSource=l_moving,LTarget=l_target,
                            model_name='affine_map',
                            map_low_res_factor=1.0,
                            nr_of_iterations=100,
                            visualize_step=None,
                            optimizer_name='sgd',
                            use_multi_scale=True,
                            rel_ftol=0,
                            similarity_measure_type='lncc',
                            similarity_measure_sigma=1.,
                            params ='../mermaid/mermaid_demos/rdmm_synth_data_generation/cur_settings_affine.json')
    output = si.get_warped_image()
    phi = si.opt.optimizer.ssOpt.get_map()
    disp = si.opt.optimizer.ssOpt.model.Ab
    # phi = phi*2-1
    phi = phi.detach().clone()
    return output.detach_(), phi.detach_(), disp.detach_(), si


def nonp_optimization(si, moving,target,spacing,fname,l_moving=None,l_target=None, init_weight= None,expr_folder= None,mermaid_setting_path=None):
    affine_map = None
    if si is not None:
        affine_map = si.opt.optimizer.ssOpt.get_map()

    si =  SI.RegisterImagePair()
    extra_info = setting_visual_saving(expr_folder,fname)
    si.opt = None
    if affine_map is not None:
        si.set_initial_map(affine_map.detach())
    if init_weight is not None:
        si.set_weight_map(init_weight.detach(),freeze_weight=True)

    si.register_images(moving, target, spacing, extra_info=extra_info, LSource=l_moving,
                            LTarget=l_target,
                            map_low_res_factor=0.5,
                            visualize_step=30,
                            optimizer_name='lbfgs_ls',
                            use_multi_scale=True,
                            rel_ftol=0,
                            similarity_measure_type='lncc',
                            params=mermaid_setting_path)
    output = si.get_warped_image()
    phi = si.opt.optimizer.ssOpt.get_map()
    model_param = si.get_model_parameters()
    if len(model_param)==2:
        m, weight_map = model_param['m'], model_param['local_weights']
        return output.detach_(), phi.detach_(), m.detach(), weight_map.detach()
    else:
        m = model_param['m']
        return output.detach_(), phi.detach_(), m.detach(), None


def compute_jacobi(phi,spacing):
    spacing = spacing  # the disp coorindate is [-1,1]
    fd = fdt.FD_torch(spacing)
    dfx = fd.dXc(phi[:, 0, ...])
    dfy = fd.dYc(phi[:, 1, ...])
    jacobi_det = dfx * dfy
    jacobi_abs = - torch.sum(jacobi_det[jacobi_det < 0.])
    print("the current  sum of neg determinant of the  jacobi is {}".format(jacobi_abs))
    return jacobi_abs.item()






def get_input(img_pair,weight_pair=None):
    s_path, t_path = img_pair
    moving = torch.load(s_path)
    target = torch.load(t_path)
    moving_init_weight= None
    target_init_weight= None
    if weight_pair is not None:
        sw_path, tw_path = weight_pair
        moving_init_weight = torch.load(sw_path)
        target_init_weight = torch.load(tw_path)
    spacing = 1. / (np.array(moving.shape[2:]) - 1)
    return moving, target, spacing, moving_init_weight,target_init_weight



def get_analysis_input(img_pair,expr_folder,pair_name,color_image=False,model_name='rdmm'):
    s_path, t_path = img_pair
    moving = torch.load(s_path)
    target = torch.load(t_path)
    spacing = 1. / (np.array(moving.shape[2:]) - 1)
    ana_path = os.path.join(expr_folder, 'analysis')
    ana_path = os.path.join(ana_path, pair_name)
    phi = torch.load(os.path.join(ana_path, 'phi.pt'))
    m = torch.load(os.path.join(ana_path, 'm.pt'))
    if color_image:
        moving,color_info = generate_color_image(moving,random_seed=2016)
        target,color_info = generate_color_image(target, color_info=color_info)
        moving = MyTensor(moving)
        target = MyTensor(target)

    if model_name == 'rdmm':
        weight_map = torch.load(os.path.join(ana_path, 'weight_map.pt'))
    else:
        weight_map = None
    return moving, target, spacing, weight_map, phi, m





def generate_color_image(refer_image,num_channels=3,intensity_range=None,random_seed=None,color_info= None):

    if type(refer_image) is torch.Tensor:
        refer_image = refer_image.cpu().numpy()
        refer_image = np.squeeze(refer_image)
    image_shape = refer_image.shape
    if intensity_range is None:
        intensity_range = (0, 254) if num_channels == 1 else ((0, 254), )
    else:
        tmp = (intensity_range, ) if num_channels == 1 else intensity_range
        for intensity_pair in tmp:
            for intensity in intensity_pair:
                if not (0 <= intensity <= 255):
                    msg = 'Intensity range must lie within (0, 255) interval'
                    raise ValueError(msg)

    random = np.random.RandomState(random_seed)
    image_shape = (image_shape[0], image_shape[1], num_channels)
    image = np.ones(image_shape, dtype=np.uint8) * 0
    filled = np.zeros(image_shape, dtype=bool)
    refer_intensity = np.unique(refer_image)
    num_shapes = len(refer_intensity)
    if color_info is None:
        colors = _generate_random_colors(num_shapes, num_channels,
                                         intensity_range, random)
    else:
        colors = color_info
    for shape_idx in range(num_shapes):
        indices = np.where(refer_image==refer_intensity[shape_idx])

        if not filled[indices].any():
            image[indices] = colors[shape_idx]
            filled[indices] = True
            continue

    image = image.astype(np.float32)/intensity_range[0][1]
    image = np.transpose(image,[2,0,1])
    image_shape = [1]+ list(image.shape)
    image = image.reshape(*image_shape)
    return image, colors


def save_color_image(image,pair_name,target):
    pass



def wrap_data(data_list):
    return [AdaptVal(data) for data in data_list]



def do_single_pair_registration(pair,pair_name, weight_pair, do_affine=True,expr_folder=None,mermaid_setting_path=None):
    moving, target, spacing, moving_init_weight, _ =get_input(pair,weight_pair)
    moving, target, moving_init_weight = wrap_data([moving, target, moving_init_weight])
    si = None
    if do_affine:
        af_img, af_map, af_param, si =affine_optimization(moving,target,spacing,pair_name)
    return_val = nonp_optimization(si, moving, target, spacing, pair_name,init_weight=moving_init_weight,expr_folder=expr_folder,mermaid_setting_path=mermaid_setting_path)
    save_single_res(return_val, pair_name, expr_folder)

def do_pair_registration(pair_list, pair_name_list, weight_pair_list,do_affine=True,expr_folder=None,mermaid_setting_path=None):
    num_pair = len(pair_list)
    for i in range(num_pair):
        do_single_pair_registration(pair_list[i],pair_name_list[i],weight_pair_list[i] if weight_pair_list else None,do_affine=do_affine,expr_folder=expr_folder,mermaid_setting_path=mermaid_setting_path)




def visualize_res(res, saving_path=None):
    pass



def save_single_res(res,pair_name, expr_folder):
    ana_path = os.path.join(expr_folder,'analysis')
    ana_path = os.path.join(ana_path,pair_name)
    os.makedirs(ana_path,exist_ok=True)
    output, phi, m, weight_map = res
    torch.save(phi.cpu(),os.path.join(ana_path,'phi.pt'))
    torch.save(m.cpu(),os.path.join(ana_path,'m.pt'))
    if weight_map is not None:
        torch.save(weight_map.cpu(),os.path.join(ana_path,'weight_map.pt'))








def sub_process(index,pair_list, pair_name_list, weight_pair_list,do_affine,expr_folder, mermaid_setting_path):
    num_pair = len(index)
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=num_pair).start()
    count = 0
    for i in index:
        do_single_pair_registration(pair_list[i],pair_name_list[i],weight_pair_list[i] if weight_pair_list else None,do_affine=do_affine,expr_folder=expr_folder, mermaid_setting_path=mermaid_setting_path)
        count += 1
        pbar.update(count)
    pbar.finish()


def get_labeled_image(img_pair):
    s_path, t_path = img_pair
    moving = torch.load(s_path)
    target = torch.load(t_path)
    moving_np  = moving.cpu().numpy()
    target_np = target.cpu().numpy()
    ind_value_list = np.unique(moving_np)
    ind_value_list_target = np.unique(target_np)
    assert len(set(ind_value_list)-set(ind_value_list_target))==0
    lmoving = torch.zeros_like(moving)
    ltarget = torch.zeros_like(target)
    ind_value_list.sort()
    for i,value in enumerate(ind_value_list):
        lmoving[moving==value]= i
        ltarget[target==value]= i

    return AdaptVal(lmoving),AdaptVal(ltarget)




def analyze_on_pair_res(pair_list,pair_name_list,expr_folder=None,color_image=False,model_name='rdmm'):
    num_pair = len(pair_list)
    score_list = []
    jacobi_list = []
    for i in range(num_pair):
        score, avg_jacobi = analyze_on_single_res(pair_list[i],pair_name_list[i],expr_folder,color_image,model_name)
        print("the current score of {} is {}".format(pair_name_list[i],score))
        print("the current jacobi of {} is {}".format(pair_name_list[i],avg_jacobi))
        score_list.append(score)
        jacobi_list.append(avg_jacobi)

    average_score =np.array(score_list).mean()
    scores = np.array(score_list)
    jacobis = np.array(jacobi_list)
    print("the average score of the registration is {}".format(average_score))
    print("the average jacobi of the registration is {}".format(jacobis.mean()))
    saving_folder = os.path.join(expr_folder, 'score')
    os.makedirs(saving_folder, exist_ok=True)
    np.save(os.path.join(saving_folder,'score.npy'),scores)
    saving_folder = os.path.join(expr_folder, 'jacobi')
    os.makedirs(saving_folder, exist_ok=True)
    np.save(os.path.join(saving_folder,'jacobi.npy'),jacobis)
    with open(os.path.join(expr_folder,'res.txt'),'w') as f:
        f.write("the average score of the registration is {}".format(average_score))
        f.write("the average jacobi of the registration is {}".format((jacobis.mean())))





def analyze_on_single_res(pair,pair_name, expr_folder=None, color_image=False,model_name='rdmm'):
    moving, target, spacing, moving_init_weight, phi,m = get_analysis_input(pair,expr_folder,pair_name,color_image=color_image,model_name=model_name)
    lmoving, ltarget =get_labeled_image(pair)
    params = pars.ParameterDict()
    params.load_JSON(os.path.join(expr_folder,'mermaid_setting.json'))
    individual_parameters = dict(m=m,local_weights=moving_init_weight)
    sz = np.array(moving.shape)
    saving_folder = os.path.join(expr_folder, 'analysis')
    saving_folder = os.path.join(saving_folder, pair_name)
    saving_folder = os.path.join(saving_folder,'res_analysis')
    os.makedirs(saving_folder,exist_ok=True)
    extra_info = None
    visual_param = None

    extra_info = {'fname':[pair_name],'saving_folder':[saving_folder]}
    visual_param = setting_visual_saving(expr_folder, pair_name,folder_name='color')

    res= evaluate_model(moving, target, sz, spacing,
                   use_map=True,
                   compute_inverse_map=False,
                   map_low_res_factor=0.5,
                   compute_similarity_measure_at_low_res=False,
                   spline_order=1,
                   individual_parameters=individual_parameters,
                   shared_parameters=None, params=params, extra_info=extra_info,visualize=False,visual_param=visual_param, given_weight=True)
    phi = res[1]
    lres = utils.compute_warped_image_multiNC(lmoving, phi, spacing, 0, zero_boundary=True)
    scores = get_multi_metric(lres,ltarget,rm_bg=True)
    avg_jacobi = compute_jacobi(phi,spacing)
    return scores['label_batch_avg_res']['dice'], avg_jacobi





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Registeration demo for 2d synthetic data')
    parser.add_argument('--expr_name', required=False, default='rdmm_synth_demo',
                        help='the name of the experiment')
    parser.add_argument('--data_task_path', required=False, default='./rdmm_synth_data_generation/data_task',
                        help='the path of data task folder')
    parser.add_argument('--model_name', required=False, default='rdmm',
                        help='non-parametric method, vsvf/lddmm/rdmm are currently supported in this demo')
    parser.add_argument('--use_predefined_weight_in_rdmm',required=False,action='store_true',
                        help='this flag is only for RDMM model, if set true, the predefined regularizer mask will be loaded and only the momentum will be optimized; if set false, both weight and momenutm will be jointly optimized')
    parser.add_argument('--mermaid_setting_path', required=False, default=None,
                        help='path of mermaid setting json')
    num_of_pair_to_process = 4
    args = parser.parse_args()
    root_path = args.data_task_path
    model_name = args.model_name 
    use_init_weight = args.use_predefined_weight_in_rdmm
    mermaid_setting_path = args.mermaid_setting_path
    if mermaid_setting_path is None:
        print("the mermaid_setting_path is not provided, load the default settings instead")
        if model_name == 'rdmm':
                mermaid_setting_path = os.path.join('./2d_example_synth', 'rdmm_setting_predefined.json' if use_init_weight else 'rdmm_setting.json')
        elif model_name =='lddmm':
            mermaid_setting_path = './2d_example_synth/lddmm_setting.json'
        elif model_name == 'vsvf':
            mermaid_setting_path='./2d_example_synth/vsvf_setting.json'
        else:
            raise ValueError("the default setting of {} is not founded".format(model_name))

    expr_name =args.expr_name
    output_root_path = os.path.join(root_path,'test')
    expr_folder = os.path.join(root_path,expr_name)
    do_affine = False
    os.makedirs(expr_folder,exist_ok=True)
    pair_path_list, pair_name_list = get_pair_list(output_root_path)
    pair_path_list=pair_path_list[:num_of_pair_to_process]
    pair_name_list=pair_name_list[:num_of_pair_to_process]
    init_weight_path_list = None
    if use_init_weight:
        init_weight_path_list = get_init_weight_list(output_root_path)
    do_optimization = True   #todo make sure this is true in optimization
    do_evaluation = True
    color_image = True
    if do_optimization:
        get_mermaid_setting(mermaid_setting_path,expr_folder)


        num_of_workers = 1 #for unknown reason, multi-thread not work
        num_files = len(pair_name_list)
        if num_of_workers > 1:
            sub_p = partial(sub_process, pair_list=pair_path_list, pair_name_list=pair_name_list,
                            weight_pair_list=init_weight_path_list, do_affine=do_affine, expr_folder=expr_folder,mermaid_setting_path=mermaid_setting_path)
            split_index = np.array_split(np.array(range(num_files)), num_of_workers)
            procs = []
            for i in range(num_of_workers):
                p = Process(target=sub_p, args=(split_index[i],))
                p.start()
                print("pid:{} start:".format(p.pid))
                procs.append(p)
            for p in procs:
                p.join()
        else:
            do_pair_registration(pair_path_list, pair_name_list, init_weight_path_list,do_affine=do_affine,expr_folder=expr_folder,mermaid_setting_path=mermaid_setting_path)
    if do_evaluation:
        analyze_on_pair_res(pair_path_list, pair_name_list,expr_folder=expr_folder,color_image=color_image,model_name=model_name)


