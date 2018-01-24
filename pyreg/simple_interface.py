"""
This package provides a super-simple interface for standard registration tasks
"""

import multiscale_optimizer as MO
import module_parameters as pars
import model_factory as MF
import fileio
import numpy as np
import utils

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

        self.opt = None

    def get_params(self):
        """
        Gets configuration parameters

        :return: ParameterDict instance holding the algorithm parameters
        """

        return self.params

    def print_available_models(self):
        MF.AvailableModels().print_available_models()

    def get_available_models(self):
        return MF.AvailableModels().get_models()

    def get_history(self):
        """
        Gets the optimizer history.

        :return: dictionary containing the optimizer history.
        """

        if self.opt is not None:
            return self.opt.get_history()
        else:
            return None

    def get_energy(self):
        """
        Returns the current energy
        :return: Returns a tuple (energy, similarity energy, regularization energy)
        """
        if self.opt is not None:
            return self.opt.get_energy()
        else:
            return None

    def get_warped_image(self):
        """
        Returns the warped image
        :return: the warped image
        """
        if self.opt is not None:
            if self.useMap:
                cmap = self.opt.get_map()
                # and now warp it
                return utils.compute_warped_image_multiNC(self.ISource, cmap, self.spacing)
            else:
                return self.opt.get_warped_image()

    def get_map(self):
        """
        Returns the deformation map
        :return: deformation map
        """
        if self.opt is not None:
            return self.opt.get_map()
        else:
            return None

    def get_model_parameters(self):
        """
        Returns the parameters of the model

        :return: model parameters
        """
        if self.opt is not None:
            return self.opt.get_model_parameters()
        else:
            return None

    def register_images_from_files(self,source_filename,target_filename,model_name,
                                   nr_of_iterations=None,
                                   similarity_measure_type=None,
                                   similarity_measure_sigma=None,
                                   map_low_res_factor=None,
                                   rel_ftol=None,
                                   smoother_type=None,
                                   optimize_over_smoother_parameters=None,
                                   json_config_out_filename=None,
                                   visualize_step=5,
                                   use_multi_scale=False,
                                   params=None):
        """
        Registers two images. Only ISource, ITarget, spacing, and model_name need to be specified.
        Default values will be used for all of the values that are not explicitly specified.

        :param source_filename: filename of the source image
        :param target_filename: filename for the target image
        :param model_name: name of the desired registration model [string]
        :param nr_of_iterations: nr of iterations
        :param similarity_measure_type: type of similarity measure ('ssd' or 'ncc')
        :param similarity_measure_sigma: similarity measures are weighted by 1/sigma^2
        :param map_low_res_factor: allows for parameterization of the registration at lower resolution than the image (0,1]
        :param rel_ftol: relative function tolerance for optimizer
        :param smoother_type: type of smoother (e.g., 'gaussian' or 'multiGaussian')
        :param optimize_over_smoother_parameters: for adaptive smoothers, weights/parameters will be optimized over if set to True
        :param json_config_out_filename: output file name for the used configuration.
        :param visualize_step: step at which the solution is visualized; if set to None, no visualizations will be created
        :param use_multi_scale: if set to True a multi-scale solver will be used
        :param params: parameter structure to pass settings or filename to load the settings from file.
        :return: n/a
        """

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

        self.register_images(ISource,ITarget,spacing,model_name,
                      nr_of_iterations=nr_of_iterations,
                      similarity_measure_type=similarity_measure_type,
                      similarity_measure_sigma=similarity_measure_sigma,
                      map_low_res_factor=map_low_res_factor,
                      rel_ftol=rel_ftol,
                      smoother_type=smoother_type,
                      optimize_over_smoother_parameters=optimize_over_smoother_parameters,
                      json_config_out_filename=json_config_out_filename,
                      visualize_step=visualize_step,
                      use_multi_scale=use_multi_scale,
                      params=params)

    def register_images(self,ISource,ITarget,spacing,model_name,
                        nr_of_iterations=None,
                        similarity_measure_type=None,
                        similarity_measure_sigma=None,
                        map_low_res_factor=None,
                        rel_ftol=None,
                        smoother_type=None,
                        optimize_over_smoother_parameters=None,
                        json_config_out_filename=None,
                        visualize_step=5,
                        use_multi_scale=False,
                        params=None):
        """
        Registers two images. Only ISource, ITarget, spacing, and model_name need to be specified.
        Default values will be used for all of the values that are not explicitly specified.

        :param ISource: source image
        :param ITarget: target image
        :param spacing: image spacing [dx,dy,dz]
        :param model_name: name of the desired registration model [string]
        :param nr_of_iterations: nr of iterations
        :param similarity_measure_type: type of similarity measure ('ssd' or 'ncc')
        :param similarity_measure_sigma: similarity measures are weighted by 1/sigma^2
        :param map_low_res_factor: allows for parameterization of the registration at lower resolution than the image (0,1]
        :param rel_ftol: relative function tolerance for optimizer
        :param smoother_type: type of smoother (e.g., 'gaussian' or 'multiGaussian')
        :param optimize_over_smoother_parameters: for adaptive smoothers, weights/parameters will be optimized over if set to True
        :param json_config_out_filename: output file name for the used configuration.
        :param visualize_step: step at which the solution is visualized; if set to None, no visualizations will be created
        :param use_multi_scale: if set to True a multi-scale solver will be used
        :param params: parameter structure to pass settings or filename to load the settings from file.
        :return: n/a
        """

        if params is None:
            self.params = pars.ParameterDict()
        elif type(params)==pars.ParameterDict:
            self.params = params
        elif type(params)==str:
            self.params = pars.ParameterDict()
            self.params.load_JSON(params)
        else:
            raise ValueError('Unknown parameter format: ' + str( type(params)))

        self.ISource = AdaptVal(Variable(torch.from_numpy(ISource.copy()), requires_grad=False))
        self.ITarget = AdaptVal(Variable(torch.from_numpy(ITarget), requires_grad=False))
        self.spacing = spacing

        if not self.available_models.has_key(model_name):
            print('Unknown model name: ' + model_name)
            MF.AvailableModels().print_available_models()
        else:
            # this model exists so let's use it
            self.useMap = self.available_models[model_name][2]
            self.params['model']['deformation']['use_map'] = self.useMap
            self.params['model']['registration_model']['type'] = model_name

            if nr_of_iterations is not None:
                self.params['optimizer']['single_scale']['nr_of_iterations'] = nr_of_iterations

            if similarity_measure_sigma is not None:
                self.params['model']['registration_model']['similarity_measure']['sigma'] = similarity_measure_sigma

            if similarity_measure_type is not None:
                self.params['model']['registration_model']['similarity_measure']['type'] = similarity_measure_type

            if map_low_res_factor is not None:
                self.params['model']['deformation']['map_low_res_factor'] = map_low_res_factor

            if rel_ftol is not None:
                self.params['optimizer']['single_scale']['rel_ftol'] = rel_ftol

            if smoother_type is not None:
                self.params['model']['registration_model']['forward_model']['smoother']['type'] = smoother_type

            if optimize_over_smoother_parameters is not None:
                self.params['model']['registration_model']['forward_model']['smoother']['optimize_over_smoother_parameters'] = optimize_over_smoother_parameters

            if use_multi_scale:
                self.opt = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
            else:
                self.opt = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)

            if visualize_step is not None:
                self.opt.get_optimizer().set_visualization(True)
                self.opt.get_optimizer().set_visualize_step(visualize_step)
            else:
                self.opt.get_optimizer().set_visualization(False)

            self.opt.set_light_analysis_on(True)
            self.opt.register()

            if json_config_out_filename is not None:
                self.params.write_JSON(json_config_out_filename)
