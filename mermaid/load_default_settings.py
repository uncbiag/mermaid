from . import config_parser as cp
from .config_parser import nr_of_threads

# read the baseconf config always from file (the same is true for the compute_settings)
par_baseconf = cp.get_baseconf_settings(cp.get_default_baseconf_settings_filenames()[0])['baseconf']

if par_baseconf['load_default_settings_from_default_setting_files']:
    # load all the settings from the default files in the settings directory
    par_democonf = cp.get_democonf_settings(cp.get_default_democonf_settings_filenames()[0])['democonf']
    par_algconf = cp.get_algconf_settings(cp.get_default_algconf_settings_filenames()[0])['algconf']
    par_respro = cp.get_respro_settings(cp.get_default_respro_settings_filenames()[0])['respro']
else:
    # otherwise get the default settings from config_parser.py
    par_democonf = cp.get_democonf_settings()['democonf']
    par_algconf = cp.get_algconf_settings()['algconf']
    par_respro = cp.get_respro_settings()['respro']

use_map = par_algconf['model']['deformation']['use_map']
map_low_res_factor = par_algconf['model']['deformation']['map_low_res_factor']
model_name = par_algconf['model']['deformation']['name']
optimizer_name = par_algconf['optimizer']['name']
nr_of_iterations = par_algconf['optimizer']['single_scale']['nr_of_iterations']
smooth_images = par_algconf['image_smoothing']['smooth_images']
multi_scale_scale_factors = par_algconf['optimizer']['multi_scale']['scale_factors']
multi_scale_iterations_per_scale = par_algconf['optimizer']['multi_scale']['scale_iterations']

visualize = par_respro['visualize']
visualize_step = par_respro['visualize_step']
load_settings_from_file = par_baseconf['load_settings_from_file']
save_settings_to_file = par_baseconf['save_settings_to_file']

dim = par_democonf['dim']
example_img_len = par_democonf['example_img_len']
use_real_images = par_democonf['use_real_images']

