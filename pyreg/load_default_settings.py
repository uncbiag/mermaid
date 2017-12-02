import pyreg.config_parser as cp
from pyreg.config_parser import nr_of_threads

par_democonf = cp.get_democonf_settings()['democonf']
par_baseconf = cp.get_baseconf_settings()['baseconf']
par_algconf = cp.get_algconf_settings()['algconf']

use_map = par_algconf['model']['deformation']['use_map']
map_low_res_factor = par_algconf['model']['deformation']['map_low_res_factor']
model_name = par_algconf['model']['deformation']['name']
optimizer_name = par_algconf['optimizer']['name']
nr_of_iterations = par_algconf['optimizer']['single_scale']['nr_of_iterations']
smooth_images = par_algconf['image_smoothing']['smooth_images']
multi_scale_scale_factors = par_algconf['optimizer']['multi_scale']['scale_factors']
multi_scale_iterations_per_scale = par_algconf['optimizer']['multi_scale']['scale_iterations']

visualize = par_baseconf['visualize']
visualize_step = par_baseconf['visualize_step']
load_settings_from_file = par_baseconf['load_settings_from_file']
save_settings_to_file = par_baseconf['save_settings_to_file']

dim = par_democonf['dim']
example_img_len = par_democonf['example_img_len']
use_real_images = par_democonf['use_real_images']

