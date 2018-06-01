from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.serialization import load_lua
import prediction_network_2d
import util_2d
import numpy as np
import argparse
import time
import h5py
import SimpleITK as sitk
from torch.autograd import Variable

start = time.time()

print('config for testing')
print('Loading configuration and network')
config = torch.load("./fast_reg/checkpoints/checkpoint_30.pth.tar");
patch_size = config['patch_size']
network_feature = config['network_feature']

batch_size = 32 #32*8

step_size = 14 # 1 pixel overlap for 15x15 patches
use_multiGPU = False;

print('creating net')
net_single = prediction_network_2d.net(network_feature).cuda();
net_single_state = net_single.state_dict();

net_single.load_state_dict(config['state_dict'])

if use_multiGPU:
    print('multi GPU net')
    device_ids=range(0, 8)
    net = torch.nn.DataParallel(net_single, device_ids=device_ids).cuda()
else:
    net = net_single

net.train()

input_batch = torch.zeros(batch_size, 2, patch_size, patch_size).cuda()

base_idx = 1
image_from_dataset = util_2d.readHDF5("./fast_reg/results/train_Isource.h5").float()
image_to_dataset = util_2d.readHDF5("./fast_reg/results/train_Itarget.h5").float()
print(image_from_dataset.size(),image_to_dataset.size())
dataset_size = image_from_dataset.size()
prediction_results = []
for slice_idx in range(0, dataset_size[0]):
    image_from_slice = image_from_dataset[slice_idx];
    image_to_slice = image_to_dataset[slice_idx];
    predict_result = util_2d.predict_momentum(image_from_slice, image_to_slice, input_batch, batch_size, patch_size, net);
    predict_result = predict_result.numpy();
    prediction_results.append(np.expand_dims(predict_result, axis=0))
    #write out predicted momentum
    # save_path = "./fast_reg/test_results/predMom_trainset_image_" + str(base_idx) + ".nii.gz"
    # sitk.WriteImage(sitk.GetImageFromArray(predict_result), save_path)
    base_idx += 1
f = h5py.File("./fast_reg/results/predMom_trainset.h5", "w")
dset = f.create_dataset("dataset", data=prediction_results)
f.close()