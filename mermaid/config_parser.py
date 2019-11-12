from __future__ import print_function
from __future__ import absolute_import

import os
import mermaid.module_parameters as pars
import multiprocessing as mp


def _find_settings_directory(first_choice, second_choice, settings_name):
    if first_choice is not None:
        if os.path.exists(first_choice):
            first_choice_settings_name = os.path.join(first_choice,settings_name)
            if os.path.exists(first_choice_settings_name):
                print('Will read from {}'.format(first_choice_settings_name))
                return first_choice_settings_name

    if second_choice is not None:
        if os.path.exists(second_choice):
            second_choice_settings_name = os.path.join(second_choice, settings_name)
            if os.path.exists(second_choice_settings_name):
                print('Will read from {}'.format(second_choice_settings_name))
                return second_choice_settings_name

    print('Could not find a settings file for {}'.format(settings_name))
    return None


# first define all the configuration filenames
this_directory = os.path.dirname(__file__)
# __file__ is the absolute path to the current python file.

standard_settings_directory = os.path.join(this_directory, r'../mermaid_settings')

home_directory = os.path.expanduser('~')
putative_local_settings_directory = os.path.join(home_directory,r'.mermaid_settings')

if os.path.exists(putative_local_settings_directory):
    print('Found local settings directory; will read settings from there as available, otherwise from {}'.format(standard_settings_directory))
    local_settings_directory = putative_local_settings_directory
else:
    local_settings_directory = None

_compute_settings_filename = _find_settings_directory(local_settings_directory, standard_settings_directory, r'compute_settings.json')
_compute_settings_comments_filename = _find_settings_directory(local_settings_directory, standard_settings_directory, r'compute_settings_comments.json')

_baseconf_settings_filename = _find_settings_directory(local_settings_directory, standard_settings_directory, r'baseconf_settings.json')
_baseconf_settings_filename_comments = _find_settings_directory(local_settings_directory, standard_settings_directory, r'baseconf_settings_comments.json')

_algconf_settings_filename = _find_settings_directory(local_settings_directory, standard_settings_directory, r'algconf_settings.json')
_algconf_settings_filename_comments = _find_settings_directory(local_settings_directory, standard_settings_directory, r'algconf_settings_comments.json')

_democonf_settings_filename = _find_settings_directory(local_settings_directory, standard_settings_directory, r'democonf_settings.json')
_democonf_settings_filename_comments = _find_settings_directory(local_settings_directory, standard_settings_directory, r'democonf_settings_comments.json')

_respro_settings_filename = _find_settings_directory(local_settings_directory, standard_settings_directory, r'respro_settings.json')
_respro_settings_filename_comments = _find_settings_directory(local_settings_directory, standard_settings_directory, r'respro_settings_comments.json')

def get_default_compute_settings_filenames():
    """Returns the filename string where the compute settings will be read from.

    :return: filename string
    """
    return (_compute_settings_filename,_compute_settings_comments_filename)


def get_default_baseconf_settings_filenames():
    """Returns the filename string where the basic configuration will be read from.

    :return: filename string
    """
    return (_baseconf_settings_filename,_baseconf_settings_filename_comments)


def get_default_democonf_settings_filenames():
    """Returns the filename string where the configuration for demo datasets will be read from.

    :return: filename string
    """
    return (_democonf_settings_filename,_democonf_settings_filename_comments)


def get_default_algconf_settings_filenames():
    """Returns the filename string where the configuration for the registration algorithm will be read from.

    :return: filename string
    """
    return (_algconf_settings_filename,_algconf_settings_filename_comments)




def get_default_respro_settings_filenames():
    return (_respro_settings_filename,_respro_settings_filename_comments)

# First get the computational settings that will be used in various parts of the library
compute_params = pars.ParameterDict()
compute_params.print_settings_off()
compute_params.load_JSON(get_default_compute_settings_filenames()[0])

compute_params[('compute',{},'how computations are done')]
CUDA_ON = compute_params['compute'][('CUDA_ON',False,'Determines if the code should be run on the GPU')]
"""If set to True CUDA will be used, otherwise it will not be used"""

USE_FLOAT16 = compute_params['compute'][('USE_FLOAT16',False,'if set to True uses half-precision - not recommended')]
"""If set to True 16 bit computations will be used -- not recommended and not actively supported"""

nr_of_threads = compute_params['compute'][('nr_of_threads',mp.cpu_count(),'set the maximal number of threads')]
"""Specifies the number of threads"""

MATPLOTLIB_AGG = compute_params['compute'][('MATPLOTLIB_AGG',False,'Determines how matplotlib plots images. Set to True for remote debugging')]
"""If set to True matplotlib's AGG graphics renderer will be used; this should be set to True if run on a server and to False if visualization are desired as part of an interactive compute session"""


def get_baseconf_settings( baseconf_settings_filename = None ):
    """
    Returns the basic configuration settings as a parameter structure.

    :param baseconf_settings_filename: loads the settings from the specified filename, otherwise from the default filename or in the absence of such a file creates default settings from scratch.
    :return: parameter structure
    """

    # These are the parameters for the general I/O and example cases
    baseconf_params = pars.ParameterDict()
    baseconf_params[('baseconf',{},'determines if settings should be loaded from file and visualization options')]

    if baseconf_settings_filename is not None:
        print( 'Loading baseconf configuration from: ' + baseconf_settings_filename )
        baseconf_params.load_JSON( baseconf_settings_filename )
        return baseconf_params
    else:
        print( 'Using default baseconf settings from config_parser.py')

    baseconf_params['baseconf'][('load_default_settings_from_default_setting_files',False,'if set to True default configuration files (in settings directory) are first loaded')]
    baseconf_params['baseconf'][('load_settings_from_file',True,'if set to True configuration settings are loaded from file')]
    baseconf_params['baseconf'][('save_settings_to_file',True,'if set to True configuration settings are saved to file')]

    if not baseconf_params['baseconf']['load_default_settings_from_default_setting_files']:
        print('HINT: Only compute_settings.json and baseconf_settings.json will be read from file by default.')
        print('HINT: Set baseconf.load_default_settings_from_default_setting_files to True if you want to use the other setting files in directory settings.')
        print('HINT: Otherwise the defaults will be as defined in config_parser.py.')

    return baseconf_params


def get_democonf_settings( democonf_settings_filename = None ):
    """
    Returns the configuration settings for the demo data as a parameter structure.

    :param democonf_settings_filename: loads the settings from the specified filename, otherwise from the default filename or in the absence of such a file creates default settings from scratch.
    :return: parameter structure
    """

    # These are the parameters for the general I/O and example cases
    democonf_params = pars.ParameterDict()
    democonf_params[('democonf', {}, 'settings for demo images/examples')]

    if democonf_settings_filename is not None:
        print( 'Loading democonf configuration from: ' + democonf_settings_filename )
        democonf_params.load_JSON( democonf_settings_filename )
        return democonf_params
    else:
        print( 'Using default democonf settings from config_parser.py' )

    democonf_params['democonf'][('dim', 2, 'Spatial dimension for demo examples 1/2/3')]
    democonf_params['democonf'][('example_img_len', 128, 'side length of image cube for example')]
    democonf_params['democonf'][('use_real_images', False, 'if set to true using real and otherwise synthetic images')]

    return democonf_params


def get_algconf_settings( algconf_settings_filename = None ):
    """
    Returns the registration algorithm configuration settings as a parameter structure.

    :param algconf_settings_filename: loads the settings from the specified filename, otherwise from the default filename or in the absence of such a file creates default settings from scratch.
    :return: parameter structure
    """

    # These are the parameters for the general I/O and example cases
    algconf_params = pars.ParameterDict()
    algconf_params[('algconf',{},'settings for the registration algorithms')]

    if algconf_settings_filename is not None:
        print( 'Loading algconf configuration from: ' + algconf_settings_filename )
        algconf_params.load_JSON( algconf_settings_filename )
        return algconf_params
    else:
        print( 'Using default algconf settings from config_parser.py')

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
    #algconf_params['algconf']['model']['registration_model']['forward_model']['smoother'][('type', 'multiGaussian', 'type of smoothing')]
    #algconf_params['algconf']['model']['registration_model']['forward_model']['smoother'][('multi_gaussian_stds', [0.05,0.1,0.15,0.2,0.25], 'standard deviations for smoothing')]

    algconf_params['algconf']['model']['registration_model']['forward_model']['smoother'][
        ('type', 'gaussian', 'type of smoothing')]
    algconf_params['algconf']['model']['registration_model']['forward_model']['smoother'][
        ('gaussian_std', 0.15, 'standard deviations for smoothing')]

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




def get_respro_settings(respro_settings_filename = None):

    respro_params = pars.ParameterDict()
    respro_params[('respro', {}, 'settings for the results process')]

    if respro_settings_filename is not None:
        print( 'Loading respro configuration from: ' + respro_settings_filename )
        respro_params.load_JSON(respro_settings_filename)
        return respro_params
    else:
        print( 'Using default respro settings from config_parser.py')

    respro_params['respro'][('expr_name', 'reg', 'name of experiment')]
    respro_params['respro'][('visualize', True, 'if set to true intermediate results are visualized')]
    respro_params['respro'][('visualize_step', 5, 'Number of iterations between visualization output')]

    respro_params['respro'][('save_fig', False, 'save visualized results')]
    respro_params['respro'][('save_fig_path', '../data/saved_results', 'path of saved figures')]
    respro_params['respro'][('save_excel', True, 'save results in excel')]

    return respro_params

# write out the configuration files (when called as a script; in this way we can boostrap a new configuration)


if __name__ == "__main__":

    compute_params.write_JSON_and_JSON_comments(get_default_compute_settings_filenames())

    baseconf_params = get_baseconf_settings()
    baseconf_params.write_JSON_and_JSON_comments(get_default_baseconf_settings_filenames())

    democonf_params = get_democonf_settings()
    democonf_params.write_JSON_and_JSON_comments(get_default_democonf_settings_filenames())

    algconf_params = get_algconf_settings()
    algconf_params.write_JSON_and_JSON_comments(get_default_algconf_settings_filenames())

    respro_params = get_respro_settings()
    respro_params.write_JSON_and_JSON_comments(get_default_respro_settings_filenames())





