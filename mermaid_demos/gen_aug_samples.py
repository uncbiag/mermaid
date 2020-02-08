"""
the first demo is to input a txt file
each line includes the moving image, target image
the output should be the warped image, the momentum
( we would remove the first demo into easyreg package)


the second demo is to input a txt file
each line include the moving image, label,  momentum1, momentum2,...
the output should be the the warped image ( by random momentum, by random time) and the corresponding warped label
"""
""""
demo on RDMM on 2d synthetic image registration
"""
import matplotlib as matplt
matplt.use('Agg')
import sys,os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../mermaid'))
from mermaid.model_evaluation import evaluate_model
import random
from mermaid.utils import *
import numpy as np
import SimpleITK as sitk
import nibabel as nib

def get_pair_list(txt_pth):
    moving_momentum_path_list = read_txt_into_list(txt_pth)
    return moving_momentum_path_list

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
    return params



def get_file_name(file_path,last_ocur=True):
    if not last_ocur:
        name= os.path.split(file_path)[1].split('.')[0]
    else:
        name = os.path.split(file_path)[1].rsplit('.',1)[0]
    name = name.replace('.nii','')
    name = name.replace('.','d')
    return name


def get_input(moving_momentum_path_list, init_weight_path_list):

    fr_sitk = lambda x: torch.Tensor(sitk.GetArrayFromImage(sitk.ReadImage(x)))
    spacing = np.flipud(sitk.ReadImage(moving_momentum_path_list[0]).GetSpacing())

    moving = fr_sitk(moving_momentum_path_list[0])[None][None]
    l_moving = fr_sitk(moving_momentum_path_list[1])[None][None]
    momentum_list =[np.transpose(fr_sitk(path))[None] for path in moving_momentum_path_list[2:]]

    if init_weight_path_list is not None:
        init_weight_list=[[fr_sitk(path) for path in init_weight_path_list]]
    else:
        init_weight_list=None
    fname_list =[get_file_name(path) for path in moving_momentum_path_list[2:]]
    fname_list = [fname.replace("_0000Momentum",'') for fname in fname_list]
    fname_list = [fname.replace("_0000_Momentum",'') for fname in fname_list]
    return moving, l_moving, momentum_list, init_weight_list, fname_list, spacing




def generate_list_res(moving_momentum_path_list,init_weight_path_list,output_path, mermaid_setting_path,compute_inversez):
    num_pair = len(moving_momentum_path_list)
    #num_aug = int(1500./num_pair)
    for i in range(num_pair):
        num_aug=20
        moving, l_moving, momentum_list, init_weight_list, fname_list, spacing = get_input(moving_momentum_path_list[i], init_weight_path_list[i] if init_weight_path_list else None)
        num_momentum  = len(momentum_list)
        for _ in range(num_aug):
            selected_index = random.sample(list(range(num_momentum)), 2)
            weight = random.random()
            rand_weight = [weight, 1.0 -weight]
            time_aug = random.random()*3-1#random.sample([-1,-0.75, -0.5, -0.25,0.25,0.5, 0.75, 1, 1.25,1.5,1.75, 2.0], 1)[0] # random.random()*3-1
            momentum  = rand_weight[0]*momentum_list[selected_index[0]] + rand_weight[1]*momentum_list[selected_index[1]]
            init_weight = None
            if init_weight_list is not None:
                init_weight = random.sample(init_weight_list,1)
            fname = fname_list[selected_index[0]] +'_'+fname_list[selected_index[1]]+'_{:.4f}_{:.4f}_t_{:.2f}'.format(rand_weight[0],rand_weight[1], time_aug)
            fname = fname.replace('.','d')
            generate_single_res(moving,l_moving, momentum,init_weight, fname, time_aug, output_path, mermaid_setting_path, spacing,moving_momentum_path_list[i][0],compute_inverse)
       



def generate_single_res(moving,l_moving, momentum, init_weight, fname, time_aug, output_path, mermaid_setting_path, spacing, moving_path, compute_inverse):
    params = pars.ParameterDict()
    params.load_JSON(mermaid_setting_path)
    params['model']['registration_model']['forward_model']['tTo'] = time_aug
    input_img_sz = [1,1]+ [int(sz*2) for sz in momentum.shape[2:]]
    org_spacing = 1.0/(np.array(moving.shape[2:])-1)
    input_spacing = 1.0/(np.array(input_img_sz[2:])-1)
    if not  input_img_sz == list(moving.shape):
        input_img,_ = resample_image(moving,spacing,input_img_sz)
    else:
        input_img = moving
    individual_parameters = dict(m=momentum,local_weights=init_weight)
    sz = np.array(input_img.shape)
    extra_info = None
    visual_param = None
    res= evaluate_model(input_img, input_img, sz, input_spacing,
                   use_map=True,
                   compute_inverse_map=compute_inverse,
                   map_low_res_factor=0.5,
                   compute_similarity_measure_at_low_res=False,
                   spline_order=1,
                   individual_parameters=individual_parameters,
                   shared_parameters=None, params=params, extra_info=extra_info,visualize=False,visual_param=visual_param, given_weight=False)
    phi = res[1]
    phi_new,_ = resample_image(phi,spacing,[1,3]+list(moving.shape[2:]))
    warped = compute_warped_image_multiNC(moving,phi_new,org_spacing,spline_order=1)
    l_warped = compute_warped_image_multiNC(l_moving,phi_new,org_spacing,spline_order=0)
    save_image_with_given_reference(warped,[moving_path],output_path,[fname+'_image'])
    save_image_with_given_reference(l_warped,[moving_path], output_path,[fname+'_label'])
    if compute_inverse:
        inv_phi_new,_ = resample_image(phi,spacing,[1,3]+list(moving.shape[2:]))
        save_deformation(inv_phi_new,output_path,[fname+'_inv_map'])

def save_deformation(phi,output_path,fname_list):
    phi_np = phi.detach().cpu().numpy()
    for i in range(phi_np.shape[0]):
        phi = nib.Nifti1Image(phi_np[i], np.eye(4))
        nib.save(phi, os.path.join(output_path,fname_list[i]+'.nii.gz'))


def resample_image(I,spacing,desiredSize, spline_order=1,zero_boundary=False,identity_map=None):
    """
    Resample an image to a given desired size

    :param I: Input image (expected to be of BxCxXxYxZ format)
    :param spacing: array describing the spatial spacing
    :param desiredSize: array for the desired size (excluding B and C, i.e, 1 entry for 1D, 2 for 2D, and 3 for 3D)
    :return: returns a tuple: the downsampled image, the new spacing after downsampling
    """
    desiredSize = desiredSize[2:]
    sz = np.array(list(I.size()))
    # check that the batch size and the number of channels is the same
    nrOfI = sz[0]
    nrOfC = sz[1]

    desiredSizeNC = np.array([nrOfI,nrOfC]+list(desiredSize))

    newspacing = spacing*((sz[2::].astype('float')-1.)/(desiredSizeNC[2::].astype('float')-1.)) ###########################################
    if identity_map is not None:
        idDes= identity_map
    else:
        idDes = AdaptVal(torch.from_numpy(identity_map_multiN(desiredSizeNC,newspacing)))
    # now use this map for resampling
    ID = compute_warped_image_multiNC(I, idDes, newspacing, spline_order,zero_boundary)

    return ID, newspacing


def save_image_with_given_reference(img=None,reference_list=None,path=None,fname=None):

    num_img = len(fname)
    os.makedirs(path,exist_ok=True)
    for i in range(num_img):
        img_ref = sitk.ReadImage(reference_list[i])
        if img is not None:
            if type(img) == torch.Tensor:
                img = img.detach().cpu().numpy()
            spacing_ref = img_ref.GetSpacing()
            direc_ref = img_ref.GetDirection()
            orig_ref = img_ref.GetOrigin()
            img_itk = sitk.GetImageFromArray(img[i,0])
            img_itk.SetSpacing(spacing_ref)
            img_itk.SetDirection(direc_ref)
            img_itk.SetOrigin(orig_ref)
        else:
            img_itk=img_ref
        fn =  '{}_batch_'.format(i)+fname if not type(fname)==list else fname[i]
        fpath = os.path.join(path,fn+'.nii.gz')
        sitk.WriteImage(img_itk,fpath)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Registeration demo for 2d synthetic data')
    parser.add_argument('--txt_folder_path', required=False, default='',
                        help='the file path of moving momentum txt')
    parser.add_argument('--output_path', required=False, default='./rdmm_synth_data_generation/data_task',
                        help='the path of task folder')
    parser.add_argument('--use_predefined_weight_in_rdmm',required=False,action='store_true',
                        help='this flag is only for RDMM model, if set true, the predefined regularizer mask will be loaded and only the momentum will be optimized; if set false, both weight and momenutm will be jointly optimized')
    parser.add_argument('--mermaid_setting_path', required=False, default=None,
                        help='path of mermaid setting json')
    parser.add_argument('--compute_inverse', required=False, action='store_true',
                        help='compute the inverse map')
    num_of_pair_to_process = 4
    args = parser.parse_args()
    txt_folder_path = args.txt_folder_path
    use_init_weight = False # for now, the init weight has yet supported
    mermaid_setting_path = args.mermaid_setting_path
    compute_inverse = args.compute_inverse

    output_path = args.output_path
    do_affine = False
    os.makedirs(output_path,exist_ok=True)
    moving_momentum_path_list = get_pair_list(txt_folder_path)
    init_weight_path_list = None
    if use_init_weight:
        init_weight_path_list = get_init_weight_list(txt_folder_path)
        
    generate_list_res(moving_momentum_path_list,init_weight_path_list,output_path, mermaid_setting_path,compute_inverse)


