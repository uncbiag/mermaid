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

    def __init__(self, data_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the saved data file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dic = read_file(data_dir, type='h5py')
        self.data = dic['data']
        self.info = dic['info']
        self.root_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'image': self.data[idx], 'path': self.info['pair_path'][idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

#

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img_pair = sample['image']
        for image in img_pair:
            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            image[:] = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return {'image': img_pair}


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
        image = sample['image']

        # swap color axis because
        # numpy image: H x W
        # torch image: 1 X H X W
        image = np.stack([image], axis=0)
        return {'image': torch.from_numpy(image)}







