from __future__ import print_function

import os
import module_parameters as pars
import multiprocessing as mp

# first define all the configuration filenames
this_directory = os.path.dirname(__file__)
# __file__ is the absolute path to the current python file.

compute_settings_filename = os.path.join(this_directory, r'../settings/compute_settings.json')
compute_settings_comments_filename = os.path.join(this_directory, r'../settings/compute_settings_comments.json')

baseconf_settings_filename = os.path.join(this_directory, r'../settings/baseconf_settings.json')
baseconf_settings_filename_comments = os.path.join(this_directory, r'../settings/baseconf_settings_comments.json')

algconf_settings_filename = os.path.join(this_directory, r'../settings/algconf_settings.json')
algconf_settings_filename_comments = os.path.join(this_directory, r'../settings/algconf_settings_comments.json')

democonf_settings_filename = os.path.join(this_directory, r'../settings/democonf_settings.json')
democonf_settings_filename_comments = os.path.join(this_directory, r'../settings/democonf_settings_comments.json')

datapro_settings_filename = os.path.join(this_directory, r'../settings/datapro_settings.json')
datapro_settings_filename_comments = os.path.join(this_directory, r'../settings/datapro_settings_comments.json')

respro_settings_filename = os.path.join(this_directory, r'../settings/respro_settings.json')
respro_settings_filename_comments = os.path.join(this_directory, r'../settings/respro_settings_comments.json')

# First get the computational settings that will be used in various parts of the library
compute_params = pars.ParameterDict()
compute_params.print_settings_off()
compute_params.load_JSON(compute_settings_filename)

compute_params[('compute',{},'how computations are done')]
CUDA_ON = compute_params['compute'][('CUDA_ON',False,'Determines if the code should be run on the GPU')]
USE_FLOAT16 = compute_params['compute'][('USE_FLOAT16',False,'if set to True uses half-precision - not recommended')]
nr_of_threads = compute_params['compute'][('nr_of_threads',mp.cpu_count(),'set the maximal number of threads')]

def get_democonf_settings( democonf_settings_filename = None ):

    # These are the parameters for the general I/O and example cases
    democonf_params = pars.ParameterDict()

    if democonf_settings_filename is None:
        this_directory = os.path.dirname(__file__)
        # __file__ is the absolute path to the current python file.
        democonf_settings_filename = os.path.join(this_directory, r'../settings/democonf_settings.json')

    democonf_params.load_JSON( democonf_settings_filename )

    democonf_params[('democonf', {}, 'settings for demo images/examples')]
    democonf_params['democonf'][('dim', 2, 'Spatial dimension for demo examples 1/2/3')]
    democonf_params['democonf'][('example_img_len', 128, 'side length of image cube for example')]
    democonf_params['democonf'][('use_real_images', False, 'if set to true using real and otherwise synthetic images')]

    return democonf_params

def get_baseconf_settings( baseconf_settings_filename = None ):

    # These are the parameters for the general I/O and example cases
    baseconf_params = pars.ParameterDict()

    if baseconf_settings_filename is None:
        this_directory = os.path.dirname(__file__)
        # __file__ is the absolute path to the current python file.
        baseconf_settings_filename = os.path.join(this_directory, r'../settings/baseconf_settings.json')

    baseconf_params.load_JSON( baseconf_settings_filename )
    baseconf_params[('baseconf',{},'determines if settings should be loaded from file and visualization options')]
    baseconf_params['baseconf'][('load_settings_from_file',True,'if set to True configuration settings are loaded from file')]
    baseconf_params['baseconf'][('save_settings_to_file',True,'if set to True configuration settings are saved to file')]


    return baseconf_params

def get_algconf_settings( algconf_settings_filename = None ):

    # These are the parameters for the general I/O and example cases
    algconf_params = pars.ParameterDict()

    if algconf_settings_filename is None:
        this_directory = os.path.dirname(__file__)
        # __file__ is the absolute path to the current python file.
        algconf_settings_filename = os.path.join(this_directory, r'../settings/algconf_settings.json')

    algconf_params.load_JSON( algconf_settings_filename )
    algconf_params[('algconf',{},'settings for the registration algorithms')]

    algconf_params['algconf'][('optimizer', {}, 'optimizer settings')]
    algconf_params['algconf']['optimizer'][('name', 'lbfgs_ls', 'name of the optimizer: [lbfgs_ls|adam]')]
    algconf_params['algconf']['optimizer'][('single_scale', {}, 'single scale settings')]
    algconf_params['algconf']['optimizer']['single_scale'][('nr_of_iterations', 20, 'number of iterations')]
    algconf_params['algconf']['optimizer'][('multi_scale', {}, 'multi scale settings')]
    algconf_params['algconf']['optimizer']['multi_scale'][('use_multiscale', False, 'use multi-scale optimizer')]
    algconf_params['algconf']['optimizer']['multi_scale'][('scale_factors', [1.0, 0.5, 0.25], 'how images are scaled')]
    algconf_params['algconf']['optimizer']['multi_scale'][('scale_iterations', [10, 20, 20], 'number of iterations per scale')]

    algconf_params['algconf'][('model', {}, 'general model settings')]
    algconf_params['algconf']['model'][('deformation', {}, 'model describing the desired deformation model')]
    algconf_params['algconf']['model']['deformation'][('name', 'lddmm_shooting', "['svf'|'svf_quasi_momentum'|'lddmm_shooting'|'lddmm_shooting_scalar_momentum'] all with '_map' or '_image' suffix")]

    algconf_params['algconf']['model']['deformation'][('use_map', True, '[True|False] either do computations via a map or directly using the image')]
    algconf_params['algconf']['model']['deformation'][('map_low_res_factor',1.0,'Set to a value in (0,1) if a map-based solution should be computed at a lower internal resolution (image matching is still at full resolution')]

    algconf_params['algconf']['model'][('registration_model', {}, 'general settings for the registration model')]
    algconf_params['algconf']['model']['registration_model'][('forward_model', {}, 'Holds the parameters for the forward model')]
    algconf_params['algconf']['model']['registration_model']['forward_model'][('number_of_time_steps', 10, 'Number of time steps for integration (if applicable)')]
    algconf_params['algconf']['model']['registration_model']['forward_model'][('smoother', {}, 'how the smoothing of velocity fields is done')]
    algconf_params['algconf']['model']['registration_model']['forward_model']['smoother'][('type', 'gaussian', 'type of smoothing')]
    algconf_params['algconf']['model']['registration_model']['forward_model']['smoother'][('gaussian_std', 0.1, 'standard deviation for smoothing')]

    algconf_params['algconf']['model']['registration_model'][('similarity_measure', {}, 'model describing the similarity measure')]
    algconf_params['algconf']['model']['registration_model']['similarity_measure'][('sigma', 0.1, '1/sigma^2 weighting')]
    algconf_params['algconf']['model']['registration_model']['similarity_measure'][('type', 'ssd', '[ssd|ncc]')]

    algconf_params['algconf']['model']['registration_model']['similarity_measure'][('develop_mod_on', False, 'if true would allow develop settings ')]
    algconf_params['algconf']['model']['registration_model']['similarity_measure'][('develop_mod', {}, 'developing mode ')]
    algconf_params['algconf']['model']['registration_model']['similarity_measure']['develop_mod'][('smoother', {}, 'how the smoothing of velocity fields is done ')]
    algconf_params['algconf']['model']['registration_model']['similarity_measure']['develop_mod']['smoother'][('type', 'gaussian', 'type of smoothing')]
    algconf_params['algconf']['model']['registration_model']['similarity_measure']['develop_mod']['smoother'][('gaussian_std', 0.1, 'standard deviation for smoothing')]
    algconf_params['algconf'][('image_smoothing', {}, 'image smoothing settings')]
    algconf_params['algconf']['image_smoothing'][('smooth_images', True, '[True|False]; smoothes the images before registration')]
    algconf_params['algconf']['image_smoothing'][('smoother',{},'settings for the image smoothing')]
    algconf_params['algconf']['image_smoothing']['smoother'][('gaussian_std', 0.01, 'how much smoothing is done')]
    algconf_params['algconf']['image_smoothing']['smoother'][('type', 'gaussian', "['gaussianSpatial'|'gaussian'|'diffusion']")]

    return algconf_params



def get_datapro_settings(datapro_settings_filename = None ):

    # These are the parameters for the general I/O and example cases
    datapro_params = pars.ParameterDict()

    if datapro_settings_filename is None:
        this_directory = os.path.dirname(__file__)
        # __file__ is the absolute path to the current python file.
        datapro_settings_filename = os.path.join(this_directory, r'../settings/datapro_settings.json')

    datapro_params.load_JSON( datapro_settings_filename )
    datapro_params[('datapro',{},'settings for the data process')]

    datapro_params['datapro'][('dataset', {}, 'general settings for dataset')]
    datapro_params['datapro']['dataset'][('name', 'lpba', 'name of the dataset: oasis2d, lpba, ibsr, cmuc' )]
    datapro_params['datapro']['dataset'][('task_name', 'lpba_affined', 'task name for data process' )]
    datapro_params['datapro']['dataset'][('data_path', None, "data path of the  dataset, default settings are in datamanger")]
    datapro_params['datapro']['dataset'][('label_path', None, "data path of the  dataset, default settings are in datamanger")]
    datapro_params['datapro']['dataset'][('output_path','/playpen/zyshen/data/', "the path to save the processed data")]
    datapro_params['datapro']['mode'][('prepare_data',False, 'prepare the data ')]
    datapro_params['datapro']['mode'][('sched','inter', "['inter'|'intra'], inter-personal or intra-personal")]
    datapro_params['datapro']['mode'][('all_comb',False, 'all possible pair combination ')]
    datapro_params['datapro']['mode'][('divided_ratio', (0.8, 0.1, 0.1), 'divided the dataset into train, val and test set by the divided_ratio')]
    datapro_params['datapro']['mode'][('slicing',100,'the index to be sliced from the 3d image dataset, support lpba, ibsr, cmuc')]
    datapro_params['datapro']['mode'][('axis',3,'which axis needed to be sliced')]
    datapro_params['datapro']['switch'][('switch_to_exist_task',False,'switch to existed task without modify other datapro settings')]

    datapro_params['datapro']['switch'][('task_root_path','/playpen/zyshen/data/oasis_inter_slicing90','path of existed processed data')]


    return datapro_params



def get_respro_settings(respro_settings_filename = None):
    respro_params = pars.ParameterDict()

    if respro_settings_filename is None:
        this_directory = os.path.dirname(__file__)
        # __file__ is the absolute path to the current python file.
        respro_settings_filename = os.path.join(this_directory, r'../settings/respro_settings.json')

    respro_params.load_JSON(respro_settings_filename)
    respro_params[('respro',{},'settings for the results process')]

    respro_params['respro'][('expr_name', 'reg', 'name of experiment')]
    respro_params['respro'][('visualize', True, 'if set to true intermediate results are visualized')]
    respro_params['respro'][('visualize_step', 5, 'Number of iterations between visualization output')]

    respro_params['respro'][('save_fig', False, 'save visualized results')]
    respro_params['respro'][('save_fig_path', '../data/saved_results', 'path of saved figures')]
    respro_params['respro'][('save_excel', True, 'save results in excel')]

    return respro_params




# write out the configuration files (when called as a script; in this way we can boostrap a new configuration)

if __name__ == "__main__":

    compute_params.write_JSON(compute_settings_filename)
    compute_params.write_JSON_comments(compute_settings_comments_filename)

    democonf_params = get_democonf_settings()
    democonf_params.write_JSON(democonf_settings_filename)
    democonf_params.write_JSON_comments(democonf_settings_filename_comments)

    baseconf_params = get_baseconf_settings()
    baseconf_params.write_JSON(baseconf_settings_filename)
    baseconf_params.write_JSON_comments(baseconf_settings_filename_comments)

    algconf_params = get_algconf_settings()
    algconf_params.write_JSON(algconf_settings_filename)
    algconf_params.write_JSON_comments(algconf_settings_filename_comments)

    datapro_params = get_datapro_settings()
    datapro_params.write_JSON(datapro_settings_filename)
    datapro_params.write_JSON_comments(datapro_settings_filename_comments)


    respro_params = get_respro_settings()
    respro_params.write_JSON(respro_settings_filename)
    respro_params.write_JSON_comments(respro_settings_filename_comments)




