from __future__ import print_function
# from builtins import range
from os import listdir
from os.path import isfile, join
from glob import glob
import os
import numpy as np
#
import sys
import torch
import random
PYTHON_VERSION = 3
if sys.version_info[0] < 3:
    PYTHON_VERSION = 2
from . import fileio
from . import module_parameters as pars

def list_dic(path):
    """
    Returns all the dictionaries found in the given path.
    :param path: Input path
    :return: list of dictionaries in this path.
    """
    return [ dic for dic in listdir(path) if not isfile(join(path,dic))]


def list_pairwise(path, img_type, full_comb, sched):
    """
     return the list of  paths of the paired image  [N,2]
    :param path:  path of the folder
    :param img_type: filter and get the image of certain type
    :param full_comb: if full_comb, return all possible pairs, if not, return pairs in increasing order
    :param sched: sched can be inter personal or intra personal ('intra'|'inter')
    :return:
    """
    if sched == 'intra':
        dic_list = list_dic(path)
        pair_list = intra_pair(path, dic_list, img_type, full_comb)
    elif sched == 'inter':
        pair_list = inter_pair(path, img_type, full_comb)
    else:
        raise ValueError("schedule should be 'inter' or 'intra'")
    return pair_list

def check_full_comb_on(full_comb):
    if full_comb == False:
        print("only return the pair in order, to get more pairs, set the 'full_comb' True")
    else:
        print(" 'full_comb' is on, if you don't need all possible pair, set the 'full_com' False")


def inter_pair(path, type, full_comb=False, mirrored=False):
    """
    get the paired filename list
    :param path: dic path
    :param type: type filter, here should be [*1_a.bmp, *2_a.bmp]
    :param full_comb: if full_comb, return all possible pairs, if not, return pairs in increasing order
    :param mirrored: double the data,  generate pair2_pair1 from pair1_pair2
    :return: [N,2]
    """
    check_full_comb_on(full_comb)
    pair_list=[]
    for sub_type in type:
        f_path = join(path,'**', sub_type)
        if PYTHON_VERSION == 3: #python3
            f_filter = glob(f_path, recursive=True)
        else:
            f_filter = []
            import fnmatch
            for root, dirnames, filenames in os.walk(path):
                for filename in fnmatch.filter(filenames, sub_type):
                    f_filter.append(os.path.join(root, filename))

        f_num = len(f_filter)
        if not full_comb:
            pair = [[f_filter[idx], f_filter[idx + 1]] for idx in range(f_num - 1)]
        else:  # too many pairs , easy to out of memory
            raise ValueError("Warning, too many pairs, be sure the disk is big enough. Comment this line if you want to continue ")
            pair = []
            for i in range(f_num - 1):
                pair_tmp = [[f_filter[i], f_filter[idx + i]] for idx in range(1,f_num - i)]
                pair += pair_tmp
        pair_list += pair
    if mirrored:
        pair_list = mirror_pair(pair_list)
    return pair_list


def mirror_pair(pair_list):
    """
    double the data,  generate pair2_pair1 from pair1_pair2    :param pair_list:
    :return:
    """
    return pair_list + [[pair[1],pair[0]] for pair in pair_list]


def intra_pair(path, dic_list, type, full_comb, mirrored=False):
    """

    :param path: dic path
    :param dic_list: each elem in list contain the path of folder which contains the instance from the same person
    :param type: type filter, here should be [*1_a.bmp, *2_a.bmp]
    :param full_comb:  if full_comb, return all possible pairs, if not, return pairs in increasing order
    :return: [N,2]
    """
    check_full_comb_on(full_comb)
    pair_list = []
    for dic in dic_list:
        if PYTHON_VERSION == 3:
            f_path = join(path, dic, type[0])
            f_filter = glob(f_path)
        else:
            f_filter = []
            f_path = join(path, dic)
            import fnmatch
            for root, dirnames, filenames in os.walk(f_path):
                for filename in fnmatch.filter(filenames, type[0]):
                    f_filter.append(os.path.join(root, filename))
        f_num = len(f_filter)
        assert f_num != 0
        if not full_comb:
            pair = [[f_filter[idx], f_filter[idx + 1]] for idx in range(f_num - 1)]
        else:
            pair = []
            for i in range(f_num - 1):
                pair_tmp = [[f_filter[i], f_filter[idx + i]] for idx in range(1,f_num - i)]
                pair += pair_tmp
        pair_list += pair
    if mirrored:
        pair_list = mirror_pair(pair_list)
    return pair_list


def find_corr_map(pair_path_list, label_path):
    """
    get the label path from the image path, assume the file name is the same
    :param pair_path_list: the path list of the image
    :param label_path: the path of the label folder
    :return:
    """
    return [[os.path.join(label_path, os.path.split(pth)[1]) for pth in pair_path] for pair_path in pair_path_list]


def make_dir(path):
    """
    Creates a directory
    :param path: 
    :return: 
    """
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)
    return is_exist

def divide_data_set(root_path, pair_name_list, ratio):
    """
    divide the dataset into root_path/train root_path/val root_path/test
    :param root_path: the root path for saving the task_dataset
    :param pair_name_list: list of name of the saved pair  like img1_img2
    :param ratio: tuple of (train_ratio, val_ratio, test_ratio) from all the pairs
    :return:  full path of each file

    """
    train_ratio = ratio[0]
    val_ratio = ratio[1]
    pair_num = len(pair_name_list)
    sub_path = {x:os.path.join(root_path,x) for x in ['train', 'val', 'test']}
    nt = [make_dir(sub_path[key]) for key in sub_path]
    if sum(nt):
        raise ValueError("the data has already exist, due to randomly assignment schedule, the program block\n" \
                          "manually delete the folder to reprepare the data")
    train_num = int(train_ratio * pair_num)
    val_num = int(val_ratio*pair_num)
    pair_name_sub_list={}
    pair_name_sub_list['train'] = pair_name_list[:train_num]
    pair_name_sub_list['val'] = pair_name_list[train_num: train_num+val_num]
    pair_name_sub_list['test'] = pair_name_list[train_num+val_num:]
    saving_path_list = [os.path.join(sub_path[x],pair_name+'.h5py') for x in ['train', 'val', 'test'] for pair_name in pair_name_sub_list[x] ]
    return saving_path_list


def generate_pair_name(pair_path_list,sched='default'):
    """
    rename the filename for different dataset,
    :param pair_path_list: path of generated file
    :param sched: 'mixed' 'custom
    :return: return filename of the pair image
    """
    if sched == 'mixed':
        return mixed_pair_name(pair_path_list)
    elif sched == 'custom':
        return custom_pair_name(pair_path_list)
    elif sched =="default":
        return default_pair_name


def default_pair_name(pair_path):
    source_path, target_path = pair_path
    f = lambda x: os.path.split(x)
    assert source_path != target_path,"the source image should be different to the target image"
    while True:
        s = get_file_name(f(source_path)[-1])
        t = get_file_name(f(target_path)[-1])
        if s !=t:
            break
        else:
            source_path, target_path = f(source_path)[0],f(target_path)[0]
    pair_name = s+"_"+t
    return pair_name



def mixed_pair_name(pair_path_list):
    """
    the filename is orgnized as: patient1_id_slice1_id_patient2_id_slice2_id
    :param pair_path_list:
    :return:
    """
    f = lambda name: os.path.split(name)
    get_in = lambda x: os.path.splitext(f(x)[1])[0]
    get_fn = lambda x: f(f(x)[0])[1]
    get_img_name = lambda x: get_fn(x)+'_'+get_in(x)
    img_pair_name_list = [get_img_name(pair_path[0])+'_'+get_img_name(pair_path[1]) for pair_path in pair_path_list]
    return img_pair_name_list

def custom_pair_name(pair_path_list):
    """
    the filename is orgnized as: slice1_id_slice2_id
    :param pair_path_list:
    :return:
    """
    f = lambda name: os.path.split(name)
    get_img_name = lambda x: os.path.splitext(f(x)[1])[0]
    img_pair_name_list = [get_img_name(pair_path[0])+'_'+get_img_name(pair_path[1]) for pair_path in pair_path_list]
    return img_pair_name_list

def check_same_size(img, standard):
    """
    make sure all the image are of the same size
    :param img: img to compare
    :param standard: standarded image size
    """
    assert img.shape == standard, "img size must be the same"


def normalize_img(image, sched='tp'):
    """
    normalize image,
    warning, default [-1,1], which would be tricky when dealing with the bilinear,
    which default background is 0
    :param sched: 'ntp': percentile 0.95 then normalized to [-1,1] 'tp': percentile then [0,1], 'p' percentile
\   :return: normalized image
    """
    if sched == 'ntp':
        image[:] = image / np.percentile(image, 95) * 0.95
        image[:] = 2 * (image - np.min(image)) / (np.max(image) - np.min(image)) - 1
    elif sched == 'tp':
        image[:] = image / np.percentile(image, 95) * 0.95
        image[:] = (image - np.min(image)) / (np.max(image) - np.min(image))
    elif sched == 'p':
        image[:] = image / np.percentile(image, 95) * 0.95
    elif sched == 't':
        image[:] = (image - np.min(image)) / (np.max(image) - np.min(image))

#TODO: clean this up. There really should not be separate functions to read images here. This all needs to be in fileio

def file_io_read_img(path, is_label, normalize_spacing=True, normalize_intensities=True, squeeze_image=True, adaptive_padding=4):
    normalize_intensities = False if is_label else normalize_intensities
    im, hdr, spacing, squeezed_spacing = fileio.ImageIO().read(path, normalize_intensities, squeeze_image, normalize_spacing, adaptive_padding)
    if squeeze_image:
        spacing = squeezed_spacing
    else:
        spacing = spacing
    info = { 'spacing':spacing, 'img_size': im.shape}
    return im, info

def file_io_read_img_slice(path, slicing, axis, is_label, normalize_spacing=True, normalize_intensities=True, squeeze_image=True,adaptive_padding=4):
    """

    :param path: file path
    :param slicing: int, the nth slice of the img would be sliced
    :param axis: int, the nth axis of the img would be sliced
    :param is_label:  the img is label
    :param normalize_spacing: normalized the spacing
    :param normalize_intensities: normalized the img
    :param squeeze_image:
    :param adaptive_padding: padding the img to favored size, (divided by certain number, here is 4), here using default 4 , favored by cuda fft
    :return:
    """
    normalize_intensities = False if is_label else normalize_intensities
    im, hdr, spacing, squeezed_spacing = fileio.ImageIO().read(path, normalize_intensities, squeeze_image, normalize_spacing, adaptive_padding)
    if squeeze_image:
        spacing = squeezed_spacing
    else:
        spacing = spacing

    if axis == 1:
        slice = im[slicing]
        slicing_spacing = spacing[1:]
    elif axis == 2:
        slice = im[:,slicing,:]
        slicing_spacing = np.asarray([spacing[0], spacing[2]])
    elif axis == 3:
        slice = im[:,:,slicing]
        slicing_spacing = spacing[:2]
    else:
        raise ValueError("slicing axis exceed, should be 1-3")
    info = { 'spacing':slicing_spacing, 'img_size': slice.shape}
    return slice, info


def save_sz_sp_to_json(info, output_path):
    """
    save img size and img spacing info into json
    :param info:
    :param output_path:
    :return:
    """
    par = pars.ParameterDict()
    par[('info',{},'shared information of data')]
    par['info'][('img_sz',info['img_size'], 'size of image')]
    par['info'][('spacing',info['spacing'].tolist(), 'size of image')]
    par.write_JSON(os.path.join(output_path,'info.json'))


def read_file(path, type='h5py'):
    """
    return dictionary contain 'data' and   'info': img_h, img_w, pair_num, pair_path
    :param path:
    :param type:
    :return:
    """
    if type == 'h5py':
        f = h5py.File(path, 'r')
        data = f['data'][:]
        info = {}
        label= None
        if '/label' in f:
            label = f['label'][:]
        for key in f.attrs:
            info[key]= f.attrs[key]
        info['pair_path'] = f['pair_path'][:]
        f.close()
        return {'data':data, 'info': info, 'label':label}


def write_file(path, dic, type='h5py'):
    """

    :param path: file path
    :param dic:  which has three item : numpy 'data', numpy 'label'if exists,  dic 'info' , string list 'pair_path',
    :param type:
    :return:
    """
    if type == 'h5py':
        f = h5py.File(path, 'w')
        f.create_dataset('data',data=dic['data'])
        if dic['label'] is not None:
            f.create_dataset('label', data= dic['label'] )
        for key, value in list(dic['info'].items()):
            f.attrs[key] = value
        #asciiList = [[path.encode("ascii", "ignore") for path in pair] for pair in dic['pair_path']]
        asciiList = [path.encode("ascii", "ignore") for path in dic['pair_path']]
        string_dt = h5py.special_dtype(vlen=str)
        f.create_dataset('pair_path', data=asciiList,dtype=string_dt)
        f.close()
    else:
        raise ValueError('only h5py supported currently')


def save_to_h5py(path, img_pair_list, info, img_pair_path_list, label_pair_list=None,verbose=True):
    """

    :param path:  path for saving file
    :param img_pair_list: list of image pair
    :param info: additional info
    :param img_pair_path_list:  list of path/name of image pair
    :param img_pair_path_list:  list of path/name of corresponded label pair

    :return:
    """
    dic = {'data': img_pair_list, 'info': info, 'pair_path':img_pair_path_list, 'label': label_pair_list}
    write_file(path, dic, type='h5py')
    if verbose:
        print('data saved: {}'.format(path))
        print(dic['info'])
        print("the shape of pair{}".format(dic['data'][:].shape))
        print('the location of the first file pair\n{}'.format(img_pair_path_list[0]))



def read_txt_into_list(file_path):
    lists= []
    with open(file_path,'r') as f:
        content = f.read().splitlines()
        if len(content)>0:
            lists= [line.split('     ') for line in content]
        lists= [item[0] if len(item)==1 else item for item in lists]
    return lists



def read_h5py_file(path, type='h5py'):
    """
    return dictionary contain 'data' and   'info': start_coord, end_coord, file_path
    :param path:
    :param type:
    :return:
    """
    if type == 'h5py':
        f = h5py.File(path, 'r')
        data = f['data'][:]
        info = {}
        label = None
        if '/label' in f:
            label = f['label'][:]
        for key in f.attrs:
            info[key] = f.attrs[key]
        #info['file_id'] = f['file_id'][:]
        f.close()
        return {'data': data, 'info': info, 'label': label}


def get_file_name(file_path,last_ocur=True):
    if not last_ocur:
        name= os.path.split(file_path)[1].split('.')[0]
    else:
        name = os.path.split(file_path)[1].rsplit('.',1)[0]
    name = name.replace('.nii','')
    name = name.replace('.','d')
    return name


def sitk_read_img_to_std_tensor(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(path)
    img_np = sitk.GetArrayFromImage(img)
    img_np = np.expand_dims(np.expand_dims(img_np,0),0).astype(np.float32)
    return torch.from_numpy(img_np)


def sitk_read_img_to_std_numpy(path):
    import SimpleITK as sitk

    img = sitk.ReadImage(path)
    img_np = sitk.GetArrayFromImage(img)
    img_np = np.expand_dims(np.expand_dims(img_np,0),0).astype(np.float32)
    return img_np


def save_image_with_given_reference(img=None,reference_list=None,path=None,fname=None):
    import SimpleITK as sitk

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
