import torch
import numpy as np
import h5py
import os, fnmatch
from torch.autograd import Variable

def calculateIdx1D(length, patch_length, step):
	one_dim_pos = torch.range(0, length-patch_length, step)
	if (length-patch_length) % step != 0:
		one_dim_pos = torch.cat((one_dim_pos, torch.ones(1) * (length-patch_length)))
	return one_dim_pos;

#given a flattened idx, return the position in the 2D image space (DONE)
def idx2pos(idx, image_size):
	pos_x = idx / image_size[1];
	pos_y = idx % image_size[1];
	return torch.LongTensor([pos_x, pos_y]);

#given a position in the 2D image space, return a flattened idx (DONE)
def pos2idx(pos, image_size):
	return (pos[0] * image_size[1]) + pos[1];

#given a flatterned idx for a 4D data (n_images * 3D image), return the position in the 4D space (DONE)
def idx2pos_4D(idx, image_size):
    image_slice = idx / (image_size[0] * image_size[1])
    single_image_idx = idx % (image_size[0] * image_size[1])
    single_image_pos = idx2pos(single_image_idx, image_size)
    return torch.cat((image_slice * torch.ones(1).long(), single_image_pos))

# calculate the idx of the patches for 3D dataset (DONE)
def calculatePatchIdx3D(num_image, patch_size, image_size, step_size):
    pos_idx = [calculateIdx1D(image_size[i], patch_size[i], step_size[i]).long() for i in range(0, 2)];
    pos_idx_flat = torch.zeros(pos_idx[0].size()[0] * pos_idx[1].size()[0]).long()
    flat_idx = 0;
    pos_2d = torch.zeros(2).long();
    for x_pos in range(0, pos_idx[0].size()[0]):
        for y_pos in range(0, pos_idx[1].size()[0]):
            pos_2d[0] = pos_idx[0][x_pos]
            pos_2d[1] = pos_idx[1][y_pos]
            pos_idx_flat[flat_idx] = pos2idx(pos_2d, image_size)
            flat_idx = flat_idx+1;           

    pos_idx_flat_all = pos_idx_flat.long();

    for i in range(1, num_image):
        pos_idx_flat_all = torch.cat((pos_idx_flat_all, pos_idx_flat.long() + i * (image_size[0] * image_size[1])));

    return pos_idx_flat_all;


# read HDF5 format file
def readHDF5(filename):
	f = h5py.File(filename, 'r')
	data = f['/dataset'][()]
	data = torch.from_numpy(data)
	f.close()
	return data

# predict the momentum given a moving and target image (DONE)
def predict_momentum(moving, target, input_batch, batch_size, patch_size, net, step_size=14):
    moving.cuda();
    target.cuda();
    data_size = moving.size();
    flat_idx = calculatePatchIdx3D(1, patch_size*torch.ones(2), data_size, step_size*torch.ones(2));
    flat_idx_select = torch.zeros(flat_idx.size());
    #remove the background patches
    for patch_idx in range(1, flat_idx.size()[0]):
        patch_pos = idx2pos_4D(flat_idx[patch_idx], data_size)
        moving_patch = moving[patch_pos[1]:patch_pos[1]+patch_size, patch_pos[2]:patch_pos[2]+patch_size]
        target_patch = target[patch_pos[1]:patch_pos[1]+patch_size, patch_pos[2]:patch_pos[2]+patch_size]
        if (torch.sum(moving_patch) + torch.sum(target_patch) != 0):
            flat_idx_select[patch_idx] = 1;

    flat_idx_select = flat_idx_select.byte();
    flat_idx = torch.masked_select(flat_idx, flat_idx_select);	

    momentum_predict = torch.zeros(2, data_size[0], data_size[1]).cuda()
    momentum_weight = torch.zeros(2, data_size[0], data_size[1]).cuda()

    batch_idx = 0;
    while(batch_idx < flat_idx.size()[0]):
        if (batch_idx + batch_size < flat_idx.size()[0]):
            cur_batch_size = batch_size;
        else:
            cur_batch_size = flat_idx.size()[0] - batch_idx

        for slices in range(0, cur_batch_size):
            patch_pos = idx2pos_4D(flat_idx[batch_idx+slices], data_size)
            input_batch[slices, 0] = moving[patch_pos[1]:patch_pos[1]+patch_size, patch_pos[2]:patch_pos[2]+patch_size]
            input_batch[slices, 1] = target[patch_pos[1]:patch_pos[1]+patch_size, patch_pos[2]:patch_pos[2]+patch_size]

        input_batch_variable = Variable(input_batch, volatile=True)
        recon_batch_variable = net(input_batch_variable)
        for slices in range(0, cur_batch_size):
            patch_pos = idx2pos_4D(flat_idx[batch_idx+slices], data_size)
            momentum_predict[:, patch_pos[1]:patch_pos[1]+patch_size, patch_pos[2]:patch_pos[2]+patch_size] += recon_batch_variable.data[slices]
            momentum_weight[:, patch_pos[1]:patch_pos[1]+patch_size, patch_pos[2]:patch_pos[2]+patch_size] += 1

        batch_idx += cur_batch_size

    #remove 0 weight areas
    momentum_weight += (momentum_weight == 0).float()

    return momentum_predict.div(momentum_weight).cpu();

#enddef

# find the prediction files in ADNI1 and ADNI2
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
#enddef
