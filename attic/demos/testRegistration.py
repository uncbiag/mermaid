from __future__ import print_function
import set_pyreg_paths

# first do the torch imports
import torch
from mermaid.data_wrapper import AdaptVal, MyTensor
import numpy as np

import mermaid.example_generation as eg
import mermaid.module_parameters as pars
import mermaid.multiscale_optimizer as MO
import mermaid.smoother_factory as SF

params = pars.ParameterDict()
#params.load_JSON('../test/json/test_svf_image_single_scale_config.json')
#params.load_JSON('../test/json/to_rename_net_base_config.json')
#params.load_JSON('../test/json/lddmm_localadapt_net_base_config.json')
params.load_JSON('/playpen/zyshen/reg_clean/mermaid_settings/cur_settings_adpt_lddmm_for_synth.json')
#params.load_JSON('/playpen/zyshen/reg_clean/mermaid_settings/cur_settings_svf_fix.json')
#params['model']['deformation']['use_map'] = False
#params['model']['registration_model']['type'] = 'svf_scalar_momentum_image'
#params['model']['deformation']['use_map'] = True
#params['model']['registration_model']['type'] = 'svf_scalar_momentum_map'
#params['model']['deformation']['use_map'] = False
#params['model']['registration_model']['type'] = 'svf_vector_momentum_image'
params['model']['deformation']['use_map'] = True
params['model']['registration_model']['type'] = 'lddmm_adapt_smoother_map'#'lddmm_shooting_map'#'svf_vector_momentum_map'#'lddmm_adapt_smoother_map'



# example_img_len = 64
# dim = 3
# szEx = np.tile(example_img_len, dim)  # size of the desired images: (sz)^dim
# I0, I1,spacing = eg.CreateSquares(dim).create_image_pair(szEx, params)  # create a default image size with two sample squares




# s_path = '/playpen/zyshen/reg_clean/mermaid/demos/undertest_kernel_weighting_type_w_K_w/brain_affine_icbm/m1.nii'
# t_path = '/playpen/zyshen/reg_clean/mermaid/demos/undertest_kernel_weighting_type_w_K_w/brain_affine_icbm/m4.nii'
# I0, I1,spacing = eg.CreateRealExampleImages(dim=2,s_path=s_path,t_path=t_path).create_image_pair()  # create a default image size with two sample squares
# gt_m_path = '/playpen/zyshen/reg_clean/mermaid/demos/undertest_kernel_weighting_type_w_K_w/misc/gt_momentum_00001.pt'
# momentum = torch.load(gt_m_path)
# gt_w_path = '/playpen/zyshen/reg_clean/mermaid/demos/undertest_kernel_weighting_type_w_K_w/misc/gt_weights_00001.pt'
# weights = torch.load(gt_w_path)
# m_orig_abs = torch.norm(torch.Tensor(momentum), p=None, dim=1, keepdim=True)
from tools.visual_tools import plot_2d_img
#




s_path = '/playpen/zyshen/debugs/syn_expr_0405/id_000_debug_s_img.pt'
t_path = '/playpen/zyshen/debugs/syn_expr_0405/id_000_debug_t_img.pt'

w_path = '/playpen/zyshen/debugs/syn_expr_0405/id_000_debug_s_weight.pt'
I0= torch.load(s_path).numpy()
I1 = torch.load(t_path).numpy()
spacing = 1./(np.array(I0.shape[2:])-1)
weights =torch.load(w_path)






# visualize_m_and_v = True
# plot_2d_img(m_orig_abs[0, 0], 'localized_m')


# s_path = '../data/moving.png'
# t_path = '../data/target2.png'
# from tools.visual_tools import read_png_into_standard_form
# I0,spacing = read_png_into_standard_form(s_path,name='Source',visual=True)
# I1,spacing = read_png_into_standard_form(t_path,name = 'Target',visual=True)
#
#

sz = np.array(I0.shape)

# create the source and target image as pyTorch variables
ISource = AdaptVal(torch.from_numpy(I0.copy()))
ITarget = AdaptVal(torch.from_numpy(I1))

# smooth both a little bit
# params[('image_smoothing', {}, 'image smoothing settings')]
# params['image_smoothing'][('smooth_images', True, '[True|False]; smoothes the images before registration')]
# params['image_smoothing'][('smoother',{},'settings for the image smoothing')]
# params['image_smoothing']['smoother'][('gaussian_std', 0.05, 'how much smoothing is done')]
# params['image_smoothing']['smoother'][('type', 'gaussian', "['gaussianSpatial'|'gaussian'|'diffusion']")]

cparams = params['image_smoothing']
# s = SF.SmootherFactory(sz[2::], spacing).create_smoother(cparams)
# ISource = s.smooth(ISource)
# ITarget = s.smooth(ITarget)

so = MO.SimpleSingleScaleRegistration(ISource, ITarget, spacing,sz, params)
#so = MO.SimpleMultiScaleRegistration(ISource, ITarget, spacing,sz, params)

so.get_optimizer().set_visualization( True )
so.get_optimizer().set_visualize_step( 5 )
so.optimizer.set_model(params['model']['registration_model']['type'])
# so.optimizer.model.m.data = MyTensor(momentum)
# so.optimizer.model.freeze_momentum()
so.optimizer.model.local_weights.data = weights.cuda()
so.optimizer.model.freeze_adaptive_regularizer_param()
so.register()

energy = so.get_energy()

print( energy )

params.write_JSON( 'regTest_settings_clean.json')
params.write_JSON_comments( 'regTest_settings_comments.json')

