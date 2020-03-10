"""
Input txt for atlas to image ,  i.e. train txt  atlas, image, atlas_label
folder for color image
folder for transformation
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
from mermaid.data_utils import *
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import copy
from glob import glob


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


def gen_aug_from_brainstorm(color_path_list,trans_path_list, atlas_label_path, color_to_trans_switcher,output_folder,img_for_reference,input_0_1=True):
    color_name_list = [get_file_name(path) for path in color_path_list]
    trans_name_list = [get_file_name(path) for path in trans_path_list]
    name_list = [name.replace(color_to_trans_switcher[1],'') for name in trans_name_list]
    fr_sitk = lambda x:sitk.GetArrayFromImage(sitk.ReadImage(x))
    color_list = [fr_sitk(color_path)[None] for color_path in color_path_list]
    trans_path_list = [np.transpose(fr_sitk(trans_path),[3,2,1,0]) for trans_path in trans_path_list]
    color = torch.Tensor(np.stack(color_list))
    color = (color+1.0)/2 if not input_0_1 else color
    trans = torch.Tensor(np.stack(trans_path_list))
    atlas_label = torch.Tensor(fr_sitk(atlas_label_path))[None][None]
    spacing = 1./(np.array(color.shape[2:])-1)
    num_aug = 1500
    batch =10
    num_iter = int(num_aug/batch)
    index_list = list(range(len(color_name_list)))
    atlas_label = atlas_label.repeat(batch,1,1,1,1)
    for i in range(num_iter):
        index_trans = random.sample(index_list,batch)
        index_color = random.sample(index_list,batch)
        color_cur = color[index_color]
        trans_cur = trans[index_trans]
        img_names = [name_list[index_color[i]]+'_color_'+ name_list[index_trans[i]]+"_phi_image" for i in range(batch)]
        label_names = [name_list[index_color[i]]+'_color_'+ name_list[index_trans[i]]+"_phi_label" for i in range(batch)]

        warped_img = compute_warped_image_multiNC(color_cur,trans_cur,spacing,spline_order=1,zero_boundary=True)
        label = compute_warped_image_multiNC(atlas_label,trans_cur,spacing,spline_order=0,zero_boundary=True)
        save_image_with_given_reference(warped_img,[img_for_reference]*batch,output_folder,img_names)
        save_image_with_given_reference(label,[img_for_reference]*batch,output_folder,label_names)



def gen_aug_from_brainstorm_not_read_in_memory(color_path_list,trans_path_list, atlas_label_path, color_to_trans_switcher,output_folder,img_for_reference,input_0_1=True):
    color_name_list = [get_file_name(path) for path in color_path_list]
    trans_name_list = [get_file_name(path) for path in trans_path_list]
    name_list = [name.replace(color_to_trans_switcher[1],'') for name in trans_name_list]
    fr_sitk = lambda x:sitk.GetArrayFromImage(sitk.ReadImage(x))
    color_list = [fr_sitk(color_path)[None] for color_path in color_path_list]
    color = torch.Tensor(np.stack(color_list))
    color = (color+1.0)/2 if not input_0_1 else color
    atlas_label = torch.Tensor(fr_sitk(atlas_label_path))[None][None]
    spacing = 1./(np.array(color.shape[2:])-1)
    num_aug = 300
    batch =10
    num_iter = int(num_aug/batch)
    index_color_list = list(range(len(color_name_list)))
    index_trans_list = list(range(len(trans_name_list)))
    atlas_label = atlas_label.repeat(batch,1,1,1,1)
    for i in range(num_iter):
        index_trans = random.sample(index_trans_list,batch)
        index_color = random.sample(index_color_list,batch)
        trans_list = [np.transpose(fr_sitk(trans_path_list[i]), [3, 2, 1, 0]) for i in index_trans]
        trans_cur = torch.Tensor(np.stack(trans_list))
        color_cur = color[index_color]
        img_names = [name_list[index_color[i]]+'_color_'+ trans_name_list[index_trans[i]]+"_phi_image" for i in range(batch)]
        label_names = [name_list[index_color[i]]+'_color_'+ trans_name_list[index_trans[i]]+"_phi_label" for i in range(batch)]

        warped_img = compute_warped_image_multiNC(color_cur,trans_cur,spacing,spline_order=1,zero_boundary=True)
        label = compute_warped_image_multiNC(atlas_label,trans_cur,spacing,spline_order=0,zero_boundary=True)
        save_image_with_given_reference(warped_img,[img_for_reference]*batch,output_folder,img_names)
        save_image_with_given_reference(label,[img_for_reference]*batch,output_folder,label_names)



# color_folder = "/playpen-raid/zyshen/data/oai_reg/brainstorm/color_lrfix_res/reg/res/records/3D"
# trans_folder = "/playpen-raid/zyshen/data/oai_reg/brainstorm/trans_lrfix_res/reg/res/records"
# color_type="_atlas_image_test_iter_0_warped_test_iter_0_warped.nii.gz"
# color_to_trans_switcher = ("atlas_image_test_iter_0_warped_test_iter_0_warped","phi")
# output_folder ="/playpen-raid1/zyshen/data/oai_reg/brain_storm/data_aug_fake_img_disp"
# atlas_label_path ="/playpen-raid/zyshen/data/oai_seg/atlas_label.nii.gz"
# img_for_reference ="/playpen-raid/zyshen/data/oai_seg/atlas_image.nii.gz"
# os.makedirs(output_folder,exist_ok=True)
# color_path_list = glob(os.path.join(color_folder,"*"+color_type))
# color_name_list = [get_file_name(path) for path in color_path_list]
# trans_path_list = [os.path.join(trans_folder,color_name.replace(*color_to_trans_switcher)+'.nii.gz') for color_name in color_name_list]
# gen_aug_from_brainstorm(color_path_list,trans_path_list, atlas_label_path, color_to_trans_switcher,output_folder,img_for_reference, input_0_1=False)


def inverse_phi_name(fname):
    fcomp = fname.split("_")
    f_inverse = fcomp[2]+'_'+fcomp[3]+'_'+fcomp[0]+'_'+fcomp[1]+"_phi"
    return f_inverse

# atlas_pair = "/playpen-raid1/zyshen/data/oai_seg/atlas/atlas/atlas_to.txt"
# color_folder = "/playpen-raid1/zyshen/data/oai_seg/atlas/atlas/test_sm/reg/res/records/3D"
# pair_list = read_txt_into_list(atlas_pair)
# img_name_list = [get_file_name(pair[1]) for pair in pair_list]
# color_path_list = [os.path.join(color_folder,fname + "_atlas_image_test_iter_0_warped.nii.gz") for fname in img_name_list]
# color_name_list = [get_file_name(path) for path in color_path_list]
# trans_folder = "/playpen-raid/zyshen/data/oai_reg/brainstorm/trans_lrfix_res/reg/res/records"
# color_to_trans_switcher = ("test_iter_0_warped","phi")
# trans_path_list = [os.path.join(trans_folder,inverse_phi_name(color_name.replace(*color_to_trans_switcher))+'.nii.gz') for color_name in color_name_list]
# output_folder ="/playpen-raid1/zyshen/data/oai_reg/brain_storm/data_aug_real_img_disp"
# atlas_label_path ="/playpen-raid/zyshen/data/oai_seg/atlas_label.nii.gz"
# img_for_reference ="/playpen-raid/zyshen/data/oai_seg/atlas_image.nii.gz"
# os.makedirs(output_folder,exist_ok=True)
# gen_aug_from_brainstorm(color_path_list,trans_path_list, atlas_label_path, color_to_trans_switcher,output_folder,img_for_reference)
#




# atlas_pair = "/playpen-raid1/zyshen/data/oai_seg/atlas/atlas/atlas_to.txt"
# color_folder = "/playpen-raid1/zyshen/data/oai_seg/atlas/atlas/test_sm/reg/res/records/3D"
# pair_list = read_txt_into_list(atlas_pair)
# img_name_list = [get_file_name(pair[1]) for pair in pair_list]
# color_path_list = [os.path.join(color_folder,fname + "_atlas_image_test_iter_0_warped.nii.gz") for fname in img_name_list]
# trans_folder = "/playpen-raid1/zyshen/data/oai_reg/brain_storm/data_aug_fluid_sr"
# color_to_trans_switcher = ("test_iter_0_warped","phi_map")
# trans_path_list =glob(os.path.join(trans_folder,'*'+'phi_map.nii.gz'))
# output_folder ="/playpen-raid1/zyshen/data/oai_reg/brain_storm/data_aug_real_img_fluid_sr"
# atlas_label_path ="/playpen-raid/zyshen/data/oai_seg/atlas_label.nii.gz"
# img_for_reference ="/playpen-raid/zyshen/data/oai_seg/atlas_image.nii.gz"
# os.makedirs(output_folder,exist_ok=True)
# gen_aug_from_brainstorm_not_read_in_memory(color_path_list,trans_path_list, atlas_label_path, color_to_trans_switcher,output_folder,img_for_reference)
#


color_folder = "/playpen-raid/zyshen/data/oai_reg/brainstorm/color_lrfix_res/reg/res/records/3D"
color_type="_atlas_image_test_iter_0_warped_test_iter_0_warped.nii.gz"
color_path_list = glob(os.path.join(color_folder,"*"+color_type))
trans_folder = "/playpen-raid1/zyshen/data/oai_reg/brain_storm/data_aug_fluid_sr"
color_to_trans_switcher = ("test_iter_0_warped","phi_map")
trans_path_list =glob(os.path.join(trans_folder,'*'+'phi_map.nii.gz'))
output_folder ="/playpen-raid1/zyshen/data/oai_reg/brain_storm/data_aug_fake_img_fluid_sr"
atlas_label_path ="/playpen-raid/zyshen/data/oai_seg/atlas_label.nii.gz"
img_for_reference ="/playpen-raid/zyshen/data/oai_seg/atlas_image.nii.gz"
os.makedirs(output_folder,exist_ok=True)
gen_aug_from_brainstorm_not_read_in_memory(color_path_list,trans_path_list, atlas_label_path, color_to_trans_switcher,output_folder,img_for_reference,input_0_1=False)




# color_folder = "/playpen-raid/zyshen/data/oai_reg/brainstorm/color_lrfix_res/reg/res/records/3D"
# trans_folder = "/playpen-raid/zyshen/data/oai_seg/atlas/phi_train"
# color_type="_atlas_image_test_iter_0_warped_test_iter_0_warped.nii.gz"
# color_to_trans_switcher = ("atlas_image_test_iter_0_warped_test_iter_0_warped","phi")
# output_folder ="/playpen-raid1/zyshen/data/oai_reg/brain_storm/data_aug_fake_img_fluidt1"
# atlas_label_path ="/playpen-raid/zyshen/data/oai_seg/atlas_label.nii.gz"
# img_for_reference ="/playpen-raid/zyshen/data/oai_seg/atlas_image.nii.gz"
# os.makedirs(output_folder,exist_ok=True)
# color_path_list = glob(os.path.join(color_folder,"*"+color_type))
# color_name_list = [get_file_name(path) for path in color_path_list]
# trans_path_list = [os.path.join(trans_folder,color_name.replace(*color_to_trans_switcher)+'.nii.gz') for color_name in color_name_list]
# gen_aug_from_brainstorm(color_path_list,trans_path_list, atlas_label_path, color_to_trans_switcher,output_folder,img_for_reference, input_0_1=False)
#
#
# atlas_pair = "/playpen-raid1/zyshen/data/oai_seg/atlas/atlas/atlas_to.txt"
# color_folder = "/playpen-raid1/zyshen/data/oai_seg/atlas/atlas/test_sm/reg/res/records/3D"
# pair_list = read_txt_into_list(atlas_pair)
# img_name_list = [get_file_name(pair[1]) for pair in pair_list]
# color_path_list = [os.path.join(color_folder,fname + "_atlas_image_test_iter_0_warped.nii.gz") for fname in img_name_list]
# color_name_list = [get_file_name(path) for path in color_path_list]
# trans_folder = "/playpen-raid/zyshen/data/oai_seg/atlas/phi_train"
# color_to_trans_switcher = ("test_iter_0_warped","phi")
# trans_path_list = [os.path.join(trans_folder,inverse_phi_name(color_name.replace(*color_to_trans_switcher))+'.nii.gz') for color_name in color_name_list]
# output_folder ="/playpen-raid1/zyshen/data/oai_reg/brain_storm/data_aug_real_img_fluidt1"
# atlas_label_path ="/playpen-raid/zyshen/data/oai_seg/atlas_label.nii.gz"
# img_for_reference ="/playpen-raid/zyshen/data/oai_seg/atlas_image.nii.gz"
# os.makedirs(output_folder,exist_ok=True)
# gen_aug_from_brainstorm(color_path_list,trans_path_list, atlas_label_path, color_to_trans_switcher,output_folder,img_for_reference)
#
