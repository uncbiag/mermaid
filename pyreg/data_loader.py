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

######################################################################
# Dataset class
# -------------
#
# ``torch.utils.data.Dataset`` is an abstract class representing a
# dataset.
# Your custom dataset should inherit ``Dataset`` and override the following
# methods:
#
# -  ``__len__`` so that ``len(dataset)`` returns the size of the dataset.
# -  ``__getitem__`` to support the indexing such that ``dataset[i]`` can
#    be used to get :math:`i`\ th sample



class RegistrationDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path, transform=None):
        """
        Args:
            csv_file (string): Path to the saved data file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path = data_path
        self.transform = transform
        self.data_type = '*.h5py'
        self.pair_path_list , self.pair_name_list= self.get_file_list()

    def get_file_list(self):
        f_filter = []
        import fnmatch
        filenames=None
        for root, dirnames, filenames in os.walk(self.data_path):
            for filename in fnmatch.filter(filenames, self.data_type):
                f_filter.append(os.path.join(root, filename))
        return f_filter, [os.path.splitext(filename)[0] for filename in filenames]

    def __len__(self):
        return len(self.pair_name_list)

    def retriv_file_id(self, filename):
        return self.pair_name_list.index(filename)

    def __getitem__(self, idx):
        dic = read_file(self.pair_path_list[idx])
        sample = {'image': dic['data'][0], 'info': dic['info'], 'label':dic['label']}
        transformed={}
        if self.transform:
             transformed['image'] = self.transform(sample['image'])
             if sample['label'] is not None:
                transformed['label'] = self.transform(sample['label'][0])
             transformed['pair_path'] = self.retriv_file_id(sample['info']['pair_path'][0])
             transformed['spacing'] = self.transform(sample['info']['spacing'])

        return transformed


class Normalize(object):
    """-1,1 normalization , this method will not be used but remained, normalization has been done when reading data"""
    def __call__(self, sample):
        img_pair = sample['image']
        for image in img_pair:
            image[:]= 2*(image-np.min(image))/(np.max(image)-np.min(image)) -1
        return {'image': img_pair}





class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        return torch.from_numpy(sample)







