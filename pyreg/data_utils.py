from os import listdir
from os.path import isfile, join
from glob import glob
import os
from scipy import misc
import numpy as np
import h5py
import skimage
from data_utils import *
import SimpleITK as sitk
import sys
import random
PYTHON_VERSION = 3
if sys.version_info[0] < 3:
    PYTHON_VERSION = 2




def list_dic(path):
    return [ dic for dic in listdir(path) if not isfile(join(path,dic))]



def list_pairwise(path, img_type, skip, sched):
    """
     return the list of  paths of the paired image  [N,2]
    :param path:  path of the folder
    :param img_type: filter and get the image of certain type
    :param skip: if skip, return all possible pairs, if not, return pairs in increasing order
    :param sched: sched can be inter personal or intra personal
    :return:
    """
    if sched == 'intra':
        dic_list = list_dic(path)
        pair_list = intra_pair(path, dic_list, img_type, skip)
    elif sched == 'inter':
        pair_list = inter_pair(path, img_type, skip)
    else:
        raise ValueError("schedule should be 'inter' or 'intra'")
    return pair_list

def check_full_comb_on(full_comb):
    if full_comb == False:
        print("only return the pair in order, to get more pairs, set the 'full_comb' True")
    else:
        print(" 'full_comb' is on, if you don't need all possible pair, set the 'full_com' False")


def inter_pair(path, type, skip=False, mirrored=False):
    """
    get the paired filename list
    :param path: dic path
    :param type: type filter, here should be [*1_a.bmp, *2_a.bmp]
    :param skip: if skip, return all possible pairs, if not, return pairs in increasing order
    :return: [N,2]
    """
    check_full_comb_on(skip)
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
        if not skip:
            pair = [[f_filter[idx], f_filter[idx + 1]] for idx in range(f_num - 1)]
        else:  # too many pairs , easy to out of memory
            raise ValueError("Warnning, too many pairs, be sure the disk is big enough. Comment this line if you want to continue ")
            pair = []
            for i in range(f_num - 1):
                pair_tmp = [[f_filter[i], f_filter[idx + i]] for idx in range(1,f_num - i)]
                pair += pair_tmp
        pair_list += pair
    if mirrored:
        pair_list = mirror_pair(pair_list)
    return pair_list



def mirror_pair(pair_list):
    return pair_list + [[pair[1],pair[0]] for pair in pair_list]



def intra_pair(path, dic_list, type, skip, mirrored=False):
    """

    :param path: dic path
    :param dic_list: each elem in list contain the path of folder which contains the instance from the same person
    :param type: type filter, here should be [*1_a.bmp, *2_a.bmp]
    :param skip:  if skip, return all possible pairs, if not, return pairs in increasing order
    :return: [N,2]
    """
    check_full_comb_on(skip)
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
        if not skip:
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


def load_as_data(pair_list):
    """
    :param pair_list:  pair_list is the list of the path of the paired image
    :return: img_pair_list, type: numpy, size: (2N,2,img_h,img_w)    2N contains the N pair and N reversed pair
             info, type: dic, items: img_h, img_w, pair_num, pair_path
    """
    img_pair_list = []
    img_pair_path_list = []
    standard=()

    for i, pair in enumerate(pair_list):
        img1 =read_itk_img(pair[0])
        img2 =read_itk_img(pair[1])

        # check img size
        if i==0:
            standard = img1.shape
            check_same_size(img2, standard)
        else:
            check_same_size(img1,standard)
            check_same_size(img2,standard)
        normalize_img(img1)
        normalize_img(img2)
        img_pair_list += [(img1, img2)]
        img_pair_list += [(img2, img1)]
        img_pair_path_list += [[pair[0],pair[1]]]
        img_pair_path_list += [[pair[1], pair[0]]]

    assert len(img_pair_list) == 2*len(pair_list)
    info = {'img_h': standard[0], 'img_w': standard[1], 'pair_num': len(img_pair_list)}
    return np.asarray(img_pair_list), info, img_pair_path_list


def find_corr_map(pair_path_list, label_path):
    return [[os.path.join(label_path, os.path.split(pth)[1]) for pth in pair_path] for pair_path in pair_path_list]



def make_dir(path):
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
        raise ValueError, "the data has already exist, due to randomly assignment schedule, the program block\n" \
                          "manually delete the folder to reprepare the data"
    train_num = int(train_ratio * pair_num)
    val_num = int(val_ratio*pair_num)
    pair_name_sub_list={}
    pair_name_sub_list['train'] = pair_name_list[:train_num]
    pair_name_sub_list['val'] = pair_name_list[train_num: train_num+val_num]
    pair_name_sub_list['test'] = pair_name_list[train_num+val_num:]
    saving_path_list = [os.path.join(sub_path[x],pair_name+'.h5py') for x in ['train', 'val', 'test'] for pair_name in pair_name_sub_list[x] ]
    return saving_path_list






def generate_pair_name(pair_path_list,sched='mixed'):
    if sched == 'mixed':
        return mixed_pair_name(pair_path_list)
    elif sched == 'custom':
        return custom_pair_name(pair_path_list)


def mixed_pair_name(pair_path_list):
    f = lambda name: os.path.split(name)
    get_in = lambda x: os.path.splitext(f(x)[1])[0]
    get_fn = lambda x: f(f(x)[0])[1]
    get_img_name = lambda x: get_fn(x)+'_'+get_in(x)
    img_pair_name_list = [get_img_name(pair_path[0])+'_'+get_img_name(pair_path[1]) for pair_path in pair_path_list]
    return img_pair_name_list

def custom_pair_name(pair_path_list):
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

def normalize_img(image, sched='ntp'):
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


def read_itk_img(path):
    """
    :param path:
    :return: numpy image
    """
    itkimage = sitk.ReadImage(path)
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    return np.squeeze(ct_scan)


def read_itk_img_slice(path, slicing):
    """
    :param path:
    :return: numpy image
    """
    itkimage = sitk.ReadImage(path)
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = np.squeeze(sitk.GetArrayFromImage(itkimage))
    return ct_scan[slicing]


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
        for key in f.attrs:
            info[key]= f.attrs[key]
        info['pair_path'] = f['pair_path'][:]
        f.close()
        return {'data':data, 'info': info}


def write_file(path, dic, type='h5py'):
    """

    :param path: file path
    :param dic:  which has three item : numpy 'data',  dic 'info' , string list 'pair_path'
    :param type:
    :return:
    """
    if type == 'h5py':
        f = h5py.File(path, 'w')
        f.create_dataset('data',data=dic['data'])
        if dic['label'] is not None:
            f.create_dataset('label', data= dic['label'] )
        for key, value in dic['info'].items():
            f.attrs[key] = value
        asciiList = [[path.encode("ascii", "ignore") for path in pair] for pair in dic['pair_path']]
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
    :return:
    """
    dic = {'data': img_pair_list, 'info': info, 'pair_path':img_pair_path_list, 'label': label_pair_list}
    write_file(path, dic, type='h5py')
    if verbose:
        print('data saved: {}'.format(path))
        print(dic['info'])
        print("the shape of pair{}".format(dic['data'][:].shape))
        print('the location of the first file pair\n{}'.format(img_pair_path_list[0]))



