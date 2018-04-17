import set_pyreg_paths

import torch
import numpy as np
import pyreg.image_sampling as IS

d = torch.load('test_data_upsample.pt')

sampler = IS.ResampleImage()

sf = 4

tst_phi_lr = d['phi_lr'][:,:,0:-1:sf,0:-1:sf,0:-1:sf]
spacing = d['spacing']*sf
spline_order = d['spline_order']
desired_size = np.array(list(d['desiredSz']))/sf

#with torch.autograd.profiler.profile() as prof:
phiWarped, _ = sampler.upsample_image_to_size(tst_phi_lr, spacing, desired_size, spline_order)

#print(prof)

