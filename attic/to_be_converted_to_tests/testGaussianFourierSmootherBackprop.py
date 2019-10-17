# first do the torch imports
from __future__ import print_function
import os
import sys
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
#os.chdir('../')

sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../mermaid'))
sys.path.insert(0,os.path.abspath('../mermaid/libraries'))
import torch
import time
import mermaid.module_parameters as pars
import numpy as np
import mermaid.smoother_factory as SF
import mermaid.example_generation as eg
import mermaid.custom_pytorch_extensions_new as ce
import mermaid.utils as utils
from mermaid.data_wrapper import USE_CUDA, FFTVal,AdaptVal, MyTensor
import ants

import torch.nn.functional as F

class SobelFilter(object):
    def __init__(self):

        dx = AdaptVal(torch.Tensor([[[-1., -3., -1.], [-3., -6., -3.], [-1., -3., -1.]],
                                                        [[0., 0., 0.], [0., 0, 0.], [0., 0., 0.]],
                                                        [[1., 3., 1.], [3., 6., 3.], [1., 3., 1.]]]
                                                       )).view(1,1,3,3,3)
        dy  = AdaptVal(
            torch.Tensor([[[1., 3., 1.], [0., 0., 0.], [-1., -3., -1.]],
                                    [[3., 6., 3.], [0., 0, 0.], [-3., -6., -3.]],
                                    [[1., 3., 1.], [0., 0., 0.], [-1., -3., -1.]]]
                                   )).view(1, 1, 3, 3, 3)
        dz = AdaptVal(
            torch.Tensor([[[-1., 0., 1.], [-3., 0., 3.], [-1., 0., 1.]],
                                    [[-3., 0., 3.], [-6., 0, 6.], [-3., 0., 3.]],
                                    [[-1., 0., 1.], [-3., 0., 3.], [-1., 0., 1.]]]
                                   )).view(1, 1, 3, 3, 3)
        self.spatial_filter = torch.cat((dx, dy,dz), 0)
        self.spatial_filter=self.spatial_filter.repeat(1,1,1,1,1)

    def __call__(self,disField):
        conv =  F.conv3d
        jacobiField = conv(disField, self.spatial_filter)

        return torch.mean(jacobiField ** 2)
class ImageReconst(nn.Module):
    def __init__(self, I0,dim,sz,spacing):
        super(ImageReconst, self).__init__()
        self.dim = dim
        self.sz = sz
        self.spacing = spacing
        self.fourier_smoother = self.__gen_fourier_smoother()
        self.Target = self.__get_smoothed_target(I0)
        self.Source = self.__init_rand_source()
        self.sobel_filter = SobelFilter()
        self.smooth_factor =0.1



    def __get_smoothed_target(self, I0):
        ITarget = AdaptVal(torch.from_numpy(I0.copy()))
        # cparams = pars.ParameterDict()
        # cparams[('smoother', {})]
        # cparams['smoother']['type'] = 'gaussianSpatial'
        # cparams['smoother']['gaussianStd'] = 0.005
        # s = SF.SmootherFactory(sz[2::], spacing).create_smoother(cparams)
        # ITarget = s.smooth(ITarget).detach()
        ITarget = self.fourier_smoother(ITarget).detach()
        return ITarget
    def __init_rand_source(self):
        ISource = nn.Parameter(torch.rand(self.Target.shape)*2-1)
        return ISource

    def __gen_fourier_smoother(self):
        gaussianStd = 0.05
        mus = np.zeros(self.dim)
        stds = gaussianStd * np.ones(self.dim)
        centered_id = utils.centered_identity_map(self.sz, self.spacing)
        g = utils.compute_normalized_gaussian(centered_id, mus, stds)
        FFilter, _ = ce.create_complex_fourier_filter(g, self.sz)
        fourier_smoother = ce.FourierConvolution(FFilter)
        return fourier_smoother
    def get_reconst_img(self):
        return self.Source.detach()
    def get_target(self):
        return self.Target


    def forward(self):
        diff = self.fourier_smoother(self.Source)-self.Target
        smoothness = self.sobel_filter(self.Source)
        loss =  torch.mean(torch.abs(diff))+smoothness* self.smooth_factor
        return loss

def update_learning_rate(optimizer, new_lr=-1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print(" the learning rate now is set to {}".format(new_lr))


#ce.check_fourier_conv()

dim = 3
params = pars.ParameterDict()
szEx = np.tile(80, dim)         # size of the desired images: (sz)^dim
params['square_example_images']=({},'Settings for example image generation')
params['square_example_images']['len_s'] = szEx.min()//6
params['square_example_images']['len_l'] = szEx.max()//4

I0,I1,spacing= eg.CreateSquares(dim).create_image_pair(szEx,params)
#I0, I1,spacing = eg.CreateRealExampleImages(dim).create_image_pair(szEx, params)  # create a default image size with two sample squares

sz = np.array(I0.shape)
assert( len(sz)==dim+2 )
saved_folder = '/playpen/zyshen/debugs/fft_grad_check'
if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)
ants.image_write(ants.from_numpy(np.squeeze(I0)),saved_folder+'/non_smoothed.nii.gz')
img_reconst = ImageReconst(I0,dim,sz[2::],spacing)
target = img_reconst.get_target()
target = np.squeeze(target.cpu().numpy())
ants.image_write( ants.from_numpy(target),saved_folder+'/target.nii.gz')

img_reconst = img_reconst.cuda()
lr = 0.0005
optimizer_ft = optim.Adam(img_reconst.parameters(), lr=lr)
start = time.time()
for i in range(80000):
    loss = img_reconst.forward()
    loss.backward()
    optimizer_ft.step()
    if (i+1)%8000==0:
        lr = lr #max(lr/5.,1e-5)
        update_learning_rate(optimizer_ft,lr)
    if i%100==0:
        print(" the current step is {} with reconstruction loss is {}".format(i,loss.item()))
    optimizer_ft.zero_grad()

print("the optimization finished in {} s".format(time.time()-start))
reconstructed_img = img_reconst.get_reconst_img()
reconstructed_img = np.squeeze(reconstructed_img.cpu().numpy())
reconstructed_img_ants = ants.from_numpy(reconstructed_img)
ants.image_write(reconstructed_img_ants,saved_folder+'/reconstucted.nii.gz')
