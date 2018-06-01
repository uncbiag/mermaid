from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.serialization import load_lua
import prediction_network_2d
import util_2d
import gc
import numpy as np

scratch = True;

use_multiGPU = False;
n_GPUs = 1

print('config for training')
if scratch:
	config = Counter();
	config['learning_rate'] = 0.0002;
	config['batch_size'] = 50
	config['epochs'] = 30;
	config['start_epoch'] = 1;
	config['patch_size'] = 15
	config['network_feature'] = 64;
	config['train_start_datapart'] = 1;
else:
	print('Loading old checkpoint!')
	config = torch.load('./fast_reg/checkpoints/checkpoint_9.pth.tar');
	print('Finished loading!')

learning_rate = config['learning_rate']
batch_size = config['batch_size']
epochs = config['epochs']
start_epoch = config['start_epoch']
patch_size = config['patch_size']
network_feature = config['network_feature']
train_start_datapart = config['train_start_datapart']

print('prepare data structures')
train_image = util_2d.readHDF5("./fast_reg/results/train_Isource.h5");
dataset_size = train_image.size()
train_image = None;
gc.collect()

input_batch = torch.zeros(batch_size, 2, patch_size, patch_size).float().cuda()
output_batch = torch.zeros(batch_size, 2, patch_size, patch_size).float().cuda()

criterion = nn.L1Loss(False).cuda(); # False = don't average

print('creating net')
net_single = prediction_network_2d.net(network_feature).cuda();
if not scratch:
	print('Loading old checkpoint!')
	net_single.load_state_dict(config['state_dict'])
	print('Finished loading checkpoint!')
	
if use_multiGPU:
	print('multi GPU net')
	device_ids=range(0, n_GPUs)
	net = torch.nn.DataParallel(net_single, device_ids=device_ids).cuda()
else:
	net = net_single

net.train()

optimizer = optim.Adam(net.parameters(), learning_rate)


print('start training')

def train_cur_data(cur_epoch, datapart, net, criterion, optimizer):
    #read in
    # TO DO: h5py
	image_appear_trainset = util_2d.readHDF5("./fast_reg/results/train_Isource.h5").float()
	image_appear_trainset_target = util_2d.readHDF5("./fast_reg/results/train_Itarget.h5").float()
	train_m0 = util_2d.readHDF5("./fast_reg/results/train_Momentums.h5").float()
	print(image_appear_trainset.size(),image_appear_trainset_target.size(),train_m0.size())
	print(image_appear_trainset.size()[0])
	flat_idx = util_2d.calculatePatchIdx3D(image_appear_trainset.size()[0], patch_size*torch.ones(2), dataset_size[1:], 1*torch.ones(2));
	flat_idx_select = torch.zeros(flat_idx.size());

	print(flat_idx[-1])
	#remove the background patches
	for patch_idx in range(1, flat_idx.size()[0]):
		patch_pos = util_2d.idx2pos_4D(flat_idx[patch_idx], dataset_size[1:])
		moving_patch = image_appear_trainset[patch_pos[0], patch_pos[1]:patch_pos[1]+patch_size, patch_pos[2]:patch_pos[2]+patch_size]
		target_patch = image_appear_trainset_target[patch_pos[0], patch_pos[1]:patch_pos[1]+patch_size, patch_pos[2]:patch_pos[2]+patch_size]
		if (torch.sum(moving_patch) + torch.sum(target_patch) != 0):
			flat_idx_select[patch_idx] = 1;
	#select index of patch
	flat_idx_select = flat_idx_select.byte();

	flat_idx = torch.masked_select(flat_idx, flat_idx_select);
	print(flat_idx.size())
	N = flat_idx.size()[0] / batch_size; #nr of iterations N
	#put data into batch
	for iters in range(0, N):
		train_idx = (torch.rand(batch_size).double() * flat_idx.size()[0])
		train_idx = torch.floor(train_idx).long()
		for slices in range(0, batch_size):
			patch_pos = util_2d.idx2pos_4D(flat_idx[train_idx[slices]], dataset_size[1:])
			input_batch[slices, 0] =  image_appear_trainset[patch_pos[0], patch_pos[1]:patch_pos[1]+patch_size, patch_pos[2]:patch_pos[2]+patch_size].cuda()
			input_batch[slices, 1] =  image_appear_trainset_target[patch_pos[0], patch_pos[1]:patch_pos[1]+patch_size, patch_pos[2]:patch_pos[2]+patch_size].cuda()
			output_batch[slices] =  train_m0[patch_pos[0], :, patch_pos[1]:patch_pos[1]+patch_size, patch_pos[2]:patch_pos[2]+patch_size].cuda() #ground truth
		#pytorch variables
		input_batch_variable = Variable(input_batch).cuda()
		output_batch_variable = Variable(output_batch).cuda()
		#set all gradients to zero / initialization
		optimizer.zero_grad()
		recon_batch_variable = net(input_batch_variable) #do the prediction
		loss = criterion(recon_batch_variable, output_batch_variable) #use ground truth to predict the loss (with L1)
		loss.backward() #backward propagation with loss
		loss_value = loss.data[0]
		optimizer.step() #update parameters
		print('====> Epoch: {}, datapart: {}, iter: {}/{}, loss: {:.4f}'.format(
          	cur_epoch, datapart, iters, N, loss_value/batch_size))
		if iters % 100 == 0 : #save params every 100 steps in case of crash
			if use_multiGPU:
				cur_state_dict = net.module.state_dict();
			else:
				cur_state_dict = net.state_dict()			
			modal_name =  "./fast_reg/checkpoints/checkpoint_{}.pth.tar".format(cur_epoch)
			torch.save({
            	'learning_rate': learning_rate,
            	'batch_size': batch_size,
            	'epochs' : epochs,
            	'start_epoch': cur_epoch,
            	'patch_size' : patch_size,
            	'network_feature' : network_feature,
            	'train_start_datapart' : datapart,
            	'state_dict': cur_state_dict,
        	}, modal_name )
#enddef


for cur_epoch in range(start_epoch, epochs+1):
	start_datapart = 1;
	for datapart in range(1, 2) : #how many epochs, usually 10
		train_cur_data(cur_epoch, datapart, net, criterion, optimizer);
		gc.collect()