"""
This package provides a super-simple interface for standard registration tasks
"""

import multiscale_optimizer as MO
import module_parameters as pars
import model_factory as MF
import fileio
import numpy as np

import torch
from torch.autograd import Variable

from pyreg.data_wrapper import AdaptVal

class RegisterImagePair(object):

    def __init__(self):

        self.I1_warped = None
        self.phi = None
        self.params = None
        self.useMap = None

        self.available_models = MF.AvailableModels().get_models()

        self.ISource = None
        self.ITarget = None

        self.spacing = None

        self.normalize_intensity = True
        self.squeeze_image = True
        self.normalize_spacing = True

        self.visualize = True
        self.visualize_step = 5

    def print_available_models(self):
        MF.AvailableModels().print_available_models()

    def get_available_models(self):
        return MF.AvailableModels().get_models()

    def register_images_from_files(self,source_filename,target_filename,model_name,params=None):

        ISource,hdr0,spacing0,normalized_spacing0 = \
            fileio.ImageIO().read_to_nc_format(source_filename,
                                               intensity_normalize=self.normalize_intensities,
                                               squeeze_image=self.squeeze_image,
                                               normalize_spacing=self.normalize_spacing)

        ITarget,hdr1,spacing1,normalized_spacing1 = \
            fileio.ImageIO().read_to_nc_format(target_filename,
                                               intensity_normalize=self.normalize_intensities,
                                               squeeze_image=self.squeeze_image,
                                               normalize_spacing=self.normalize_spacing)

        assert (np.all(normalized_spacing0 == normalized_spacing1))
        spacing = normalized_spacing0

        self.register(ISource,ITarget,spacing,model_name,params)

    def register_images(self,ISource,ITarget,spacing,model_name,params=None):
        if params is None:
            self.params = pars.ParameterDict()
        else:
            self.params = params

        self.ISource = AdaptVal(Variable(torch.from_numpy(ISource.copy()), requires_grad=False))
        self.ITarget = AdaptVal(Variable(torch.from_numpy(ITarget), requires_grad=False))
        self.spacing = spacing

        if not self.available_models.has_key(model_name):
            print('Unknown model name: ' + model_name)
            MF.AvailableModels().print_available_models()
        else:
            # this model exists so let's use it
            self.params['model']['deformation']['use_map'] = self.available_models[model_name][2]
            self.params['model']['registration_model']['type'] = model_name
            #self.params['model']['deformation']['map_low_res_factor'] = 0.5
            self.params['optimizer']['single_scale']['rel_ftol'] = 1e-10
            self.params['optimizer']['single_scale']['nr_of_iterations'] = 201
            #self.params['registration_model']['forward_model']['smoother']['type']='multiGaussian'

            so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
            so.get_optimizer().set_visualization(self.visualize)
            so.get_optimizer().set_visualize_step(self.visualize_step)
            so.set_light_analysis_on(True)
            so.register()

            #self.params.load_JSON('./json/svf_momentum_based_config.json')
            self.params.write_JSON('test_simple_interface_' + model_name + '_settings_clean.json')
            self.params.write_JSON_comments('test_simple_interface_' + model_name + '_settings_comments.json')


