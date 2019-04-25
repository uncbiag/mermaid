import SimpleITK as sitk
from mermaid.pyreg.utils import omt_boundary_weight_mask
import numpy as np
img_sz= [40,96,96]
spacing = 1./(np.array(img_sz)-1)
save_path = '/playpen/zyshen/debugs/todel_/omt_mask.nii.gz'
mask = omt_boundary_weight_mask(img_sz,spacing,mask_range=3,mask_value=15,smoother_std=0.04)
mask = mask[0,0].detach().cpu().numpy()
mask_sitk = sitk.GetImageFromArray(mask)
sitk.WriteImage(mask_sitk,save_path)

