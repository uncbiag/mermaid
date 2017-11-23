
from __future__ import print_function

import os
import pyreg.module_parameters as pars
import multiprocessing as mp

print(os.getcwd())
this_directory = os.path.dirname(__file__)
# __file__ is the absolute path to the current python file.

# First get the parameters for the general computational setup
global_config_file_name = os.path.join(this_directory,r'./base_settings.json')
global_params = pars.ParameterDict()
global_params.load_JSON(global_config_file_name)

global_params[('computation',{},'how computations are done')]
CUDA_ON = global_params['computation'][('CUDA_ON',True,'Determines if the code should be run on the GPU')]
USE_FLOAT16 = global_params['computation'][('USE_FLOAT16',False,'if set to True uses half-precision - not recommended')]
nr_of_threads = global_params['computation'][('nr_of_threads',mp.cpu_count(),'set the maximal number of threads')]

global_params[('configuration',{},'keeps track of general configurations')]
load_settings_from_file = global_params['configuration'][('load_settings_from_file',True,'if set to True configuration settings are loaded from file')]
save_settings_to_file = global_params['configuration'][('save_settings_to_file',True,'if set to True configuration settings are saved to file')]
visualize = global_params['configuration'][('visualize',False,'if set to true intermediate results are visualized')]
visualize_step = global_params['configuration'][('visualize_step',5,'visualziation after how many steps')]

global_params[('demo_examples',{},'settings for demo images')]
dim = global_params['demo_examples'][('dim',2,'Spatial dimension for demo examples 1/2/3')]
example_img_len = global_params['demo_examples'][('example_img_len',128,'side length of image cube for example')]
use_real_images = global_params['demo_examples'][('use_real_images',False,'if set to true using real and otherwise synthetic images')]

# Now get the base settings which are used for runs (unless they are overwritten later)
base_default_config_name = os.path.join(this_directory,r'./default_config.json')
base_default_params = pars.ParameterDict()
base_default_params.load_JSON(base_default_config_name)

base_default_params[('optimizer',{},'optimizer settings')]
optimizer_name = base_default_params['optimizer'][('name','lbfgs_ls','name of the optimizer: [lbfgs_ls|adam]')]
my_adam_EPS = base_default_params['optimizer'][('my_adam_EPS',0.0,'epsilon used for adam optimizer, if adam is used')]
base_default_params['optimizer'][('single_scale',{},'single scale settings')]
nr_of_iterations = base_default_params['optimizer']['single_scale'][('nr_of_iterations',100,'number of iterations')]
base_default_params['optimizer'][('multi_scale',{},'multi scale settings')]
multi_scale_scale_factors = base_default_params['optimizer']['multi_scale'][('scale_factors',[1.0, 0.5, 0.25],'how images are scaled')]
multi_scale_iterations_per_scale = base_default_params['optimizer']['multi_scale'][('scale_iterations',[10,20,20],'number of iterations per scale')]

base_default_params[('model',{},'general model settings')]
base_default_params['model'][('deformation',{},'model describing the deformation')]
model_name = base_default_params['model']['deformation'][('name','lddmm_shooting',"['svf'|'svf_quasi_momentum'|'lddmm_shooting'|'lddmm_shooting_scalar_momentum']")]
use_map = base_default_params['model']['deformation'][('use_map',True,'[True|False] either do computations via a map or directly using the image')]
base_default_params['model']['deformation'][('forward_model',{},'Holds the parameters for the forward model')]
base_default_params['model']['deformation']['forward_model'][('number_of_time_steps',10,'Number of time steps for integration (if applicable)')]
base_default_params['model']['deformation']['forward_model'][('smoother',{},'how the smoothing of velocity fields is done')]
base_default_params['model']['deformation']['forward_model']['smoother'][('type','gaussian','type of smoothing')]
base_default_params['model']['deformation']['forward_model']['smoother'][('gaussian_std',0.15,'standard deviation for smoothing')]

base_default_params['model'][('similarity',{},'model describing the similarity measure')]
sim_measure_sigma = base_default_params['model']['similarity'][('sim_measure_sigma',0.1,'1/sigma^2 weighting')]

base_default_params[('image_smoothing',{},'image smoothing settings')]
smooth_images = base_default_params['image_smoothing'][('smooth_images',True,'[True|False]; smoothes the images before registration')]
smooth_images_gaussian_std = base_default_params['image_smoothing'][('smooth_images_gaussian_std',0.05,'how much smoothing is done')]
image_smoothing_type = base_default_params['image_smoothing'][('smoothing_type', 'gaussian', "['gaussianSpatial'|'gaussian'|'diffusion']")]

# write out the configuration files
# Set to true only if these configuration files should be created from scratch
# otherwise this should be false
if False:
    global_params.write_JSON('base_settings.json')
    global_params.write_JSON_comments('base_settings_comments.json')
    base_default_params.write_JSON('default_config.json')
    base_default_params.write_JSON_comments('default_config_comments.json')




