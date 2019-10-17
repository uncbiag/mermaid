import os
import SimpleITK as sitk
from mermaid.utils import omt_boundary_weight_mask,momentum_boundary_weight_mask
import numpy as np
img_sz= [40,96,96]
spacing = 1./(np.array(img_sz)-1)

save_path = '/playpen/zyshen/debugs/masks/omt_mask.nii.gz'
os.makedirs('/playpen/zyshen/debugs/masks',exist_ok=True)
mask = omt_boundary_weight_mask(img_sz,spacing,mask_range=3,mask_value=15,smoother_std=0.04)
mask = mask[0,0].detach().cpu().numpy()
mask_sitk = sitk.GetImageFromArray(mask)
sitk.WriteImage(mask_sitk,save_path)




img_sz=[40,96,96]# [20,48,48]#[40,96,96]
spacing = 1./(np.array(img_sz)-1)
save_path = '/playpen/zyshen/debugs/masks/velocity_mask.nii.gz'
mask = momentum_boundary_weight_mask(img_sz,spacing,mask_range=4,smoother_std=0.04)
mask = mask**2
mask = mask[0,0].detach().cpu().numpy()
mask_sitk = sitk.GetImageFromArray(mask)
sitk.WriteImage(mask_sitk,save_path)