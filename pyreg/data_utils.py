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



def inter_pair(path, type, skip=False):
    """
    get the paired filename list
    :param path: dic path
    :param type: type filter, here should be [*1_a.bmp, *2_a.bmp]
    :param skip: if skip, return all possible pairs, if not, return pairs in increasing order
    :return: [N,2]
    """
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
            raise ValueError("Warnning, too many pairs, be sure the memory is big enough. Comment this line if you want to continue ")
            pair = []
            for i in range(f_num - 1):
                pair_tmp = [[f_filter[i], f_filter[idx + i]] for idx in range(1,f_num - i)]
                pair += pair_tmp
        pair_list += pair
    return pair_list





def intra_pair(path, dic_list, type, skip):
    """

    :param path: dic path
    :param dic_list: each elem in list contain the path of folder which contains the instance from the same person
    :param type: type filter, here should be [*1_a.bmp, *2_a.bmp]
    :param skip:  if skip, return all possible pairs, if not, return pairs in increasing order
    :return: [N,2]
    """
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
        img1 =read_img(pair[0])
        img2 =read_img(pair[1])

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


def check_same_size(img, standard):
    """
    make sure all the image are of the same size
    :param img: img to compare
    :param standard: standarded image size
    """
    assert img.shape == standard, "img size must be the same"

def normalize_img(image, ntp=False):
    """
    normalize image,
    warning, default [-1,1], which would be tricky when dealing with the bilinear,
    which default background is 0, corresponding corrections have been done in this application
    :param image:
    :param ntp: True: normalized into [-1,1], False: normalized into [0,1]
    :return: normalized image
    """
    if ntp:
        image[:] = image / np.percentile(image, 95) * 0.95
        image[:] = 2 * (image - np.min(image)) / (np.max(image) - np.min(image)) - 1
    else:
        image[:] = image / np.percentile(image, 95) * 0.95
        image[:] = (image - np.min(image)) / (np.max(image) - np.min(image))


def read_img(path):
    """
    :param path:
    :return: numpy image
    """
    itkimage = sitk.ReadImage(path)
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    return np.squeeze(ct_scan)



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

    :param path:
    :param dic:  which has three item : numpy 'data',  dic 'info' , string list 'pair_path'
    :param type:
    :return:
    """
    if type == 'h5py':
        f = h5py.File(path, 'w')
        f.create_dataset('data',data=dic['data'])
        for key, value in dic['info'].items():
            f.attrs[key] = value
        asciiList = [[path.encode("ascii", "ignore") for path in pair] for pair in dic['pair_path']]
        string_dt = h5py.special_dtype(vlen=str)
        f.create_dataset('pair_path', data=asciiList,dtype=string_dt)
        f.close()
    else:
        raise ValueError('only h5py supported currently')



def save_to_h5py(path, img_pair_list, info, img_pair_path_list):
    """

    :param path:
    :param img_pair_list: numpy data
    :param info: dictionary for data info
    :return:
    """
    dic = {'data': img_pair_list, 'info': info, 'pair_path':img_pair_path_list}
    write_file(path, dic, type='h5py')
    print('data saved: {}'.format(path))
    print(dic['info'])
    print("the shape of pair{}".format(dic['data'][:].shape))
    print('the location of the first file pair\n{}'.format(img_pair_path_list[0]))



