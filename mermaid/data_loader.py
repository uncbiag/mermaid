from __future__ import print_function, division
from __future__ import absolute_import
from builtins import object
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from .data_utils import *
from time import time



class RegistrationDataset(Dataset):
    """registration dataset."""

    def __init__(self, data_path, transform=None):
        """

        :param data_path:  string, path to processed data
        :param transform: function,   apply transform on data
        """
        self.data_path = data_path
        self.transform = transform
        self.data_type = '*.h5py'
        self.get_file_list()

    def get_file_list(self):
        """
        get the all files belonging to data_type from the data_path,
        :return: full file path list, file name list
        """
        self.path_list = read_txt_into_list(os.path.join(self.data_path,'pair_path_list.txt'))
        self.pair_name_list = read_txt_into_list(os.path.join(self.data_path, 'pair_name_list.txt'))
        if len(self.pair_name_list)==0:
            self.pair_name_list = ['pair_{}'.format(idx) for idx in range(len(self.path_list))]


    def __len__(self):
        return len(self.pair_name_list)

    def __getitem__(self, idx):
        """
        :param idx: id of the items
        :return: the processed data, return as type of dic
        """
        pair_path = self.path_list[idx]
        filename = self.pair_name_list[idx]
        pair_dic = [read_h5py_file(pt) for pt in pair_path]
        sample = {'image': np.asarray([pair_dic[0]['data'],pair_dic[1]['data']]),
                  'info': pair_dic[0]['info']}
        if pair_dic[0]['label'] is not None:
            sample ['label']= np.asarray([pair_dic[0]['label'], pair_dic[1]['label']])
        else:
            sample['label'] = None
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            if sample['label'] is not None:
                 sample['label'] = self.transform(sample['label'])
        sample['spacing'] = self.transform(sample['info']['spacing'])
        return sample,filename




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        n_tensor = torch.from_numpy(sample)
        if n_tensor.shape[0] != 1:
            n_tensor.unsqueeze_(0)
        return n_tensor
