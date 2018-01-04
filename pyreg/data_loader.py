from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_utils import *
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
        self.pair_path_list , self.pair_name_list= self.get_file_list()

    def get_file_list(self):
        """
        get the all files belonging to data_type from the data_path,
        :return: full file path list, file name list
        """
        f_filter = []
        import fnmatch
        filenames=None
        for root, dirnames, filenames in os.walk(self.data_path):
            for filename in fnmatch.filter(filenames, self.data_type):
                f_filter.append(os.path.join(root, filename))
        return f_filter, [os.path.splitext(filename)[0] for filename in filenames]

    def __len__(self):
        return len(self.pair_name_list)

    def retrieve_file_id(self, filename):
        """ get the index of the file in the filelist"""
        return self.pair_name_list.index(filename)

    def __getitem__(self, idx):
        """
        :param idx: id of the items
        :return: the processed data, return as type of dic
        """
        dic = read_file(self.pair_path_list[idx])
        sample = {'image': dic['data'][0], 'info': dic['info'], 'label':dic['label']}
        transformed={}
        if self.transform:
             transformed['image'] = self.transform(sample['image'])
             if sample['label'] is not None:
                transformed['label'] = self.transform(sample['label'][0])
             transformed['pair_path'] = self.retrieve_file_id(sample['info']['pair_path'][0])
             transformed['spacing'] = self.transform(sample['info']['spacing'])

        return transformed






class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        return torch.from_numpy(sample)
