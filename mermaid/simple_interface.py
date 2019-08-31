"""
This package provides a super-simple interface for standard registration tasks
"""
from __future__ import print_function
from __future__ import absolute_import

# from builtins import str
# from builtins import object

from . import multiscale_optimizer as MO
from . import module_parameters as pars
from . import model_factory as MF
from . import fileio
import numpy as np

import torch
import torch.nn

from .data_wrapper import AdaptVal


class RegisterImagePair(object):

    def __init__(self):
        self.task_name = None
        self.par_respro = None
        self.recorder = None
        self.I1_warped = None
        self.phi = None
        self.params = None
        self.useMap = None

        self.available_models = MF.AvailableModels().get_models()

        self.ISource = None
        self.ITarget = None

        self.spacing = None
        self.sz = None

        self.normalize_intensity = True
        self.squeeze_image = True
        self.normalize_spacing = True

        self.opt = None

        self.delayed_model_parameters = None
        self.delayed_model_parameters_still_to_be_set = False

        self.delayed_initial_map = None
        self.delayed_initial_map_still_to_be_set = False
        self.delayed_initial_weight_map_still_to_be_set= False

        self.optimizer_has_been_initialized = False

    def get_params(self):
        """
        Gets configuration parameters

        :return: ParameterDict instance holding the algorithm parameters
        """

        return self.params

    def init_analysis_params(self, par_respro, task_name):
        self.par_respro=par_respro
        self.task_name = task_name

    def get_opt(self):
        return self.opt

    def set_opt(self,opt):
        self.opt= opt
        self.optimizer_has_been_initialized=True

    def _set_analysis(self, mo, extra_info):

        expr_name = extra_info[('expr_name','','experiment name')]
        save_fig = extra_info[('save_fig',False,'save intermid figure')]
        save_fig_num = extra_info[('save_fig_num',-1,'num of the fig to save per call, -1 to save all')]
        save_fig_path = extra_info[('save_fig_path','non_save_fig_path','path to save fig')]
        visualize = extra_info[('visualize',False,'visualiza the plot')]
        pair_name = extra_info[('pair_name',None,'list of img pair name')]
        mo.set_expr_name(expr_name)
        mo.set_save_fig(save_fig)
        mo.set_save_fig_path(save_fig_path)
        mo.set_save_fig_num(save_fig_num)
        mo.set_visualization(visualize)
        mo.set_pair_name(pair_name)


    @staticmethod
    def print_available_models():
        MF.AvailableModels().print_available_models()

    @staticmethod
    def get_available_models():
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
            return self.opt.get_warped_image()
        else:
            return None

    def get_warped_label(self):
        """
        Returns the warped label

        :return: the warped label
        """

        if self.opt is not None:
            return self.opt.get_warped_label()
        else:
            return None

    def get_map(self):
        """
        Returns the deformation map
        :return: deformation map
        """
        if self.opt is not None:
            return self.opt.get_map()
        else:
            return None

    def get_inverse_map(self):
        """
        Returns the inverse deformation map if available
        :return: deformation map
        """
        if self.opt is not None:
            return self.opt.get_inverse_map()
        else:
            return None

    def set_initial_map(self,map0,initial_inverse_map=None):
        """
        Sets the map that will be used as initial condition. By default this is the identity, but this can be
        overwritten with this method to for example allow concatenating multiple transforms.

        :param map0: initial map
        :param initial_inverse_map: inverse initial map; if this is not set, the inverse map cannot be computed on the fly
        :return:  n/a
        """

        if self.opt is not None:
            self.opt.set_initial_map(map0, initial_inverse_map)
            # self.opt.set_initial_inverse_map(initial_inverse_map)
            self.delayed_initial_map_still_to_be_set = False
        else:
            self.delayed_initial_map_still_to_be_set = True
            self.delayed_initial_map = map0
            self.delayed_initial_inverse_map = initial_inverse_map

    def set_weight_map(self,init_weight,freeze_weight=False):
        """
        Sets the map that will be used as initial condition. By default this is the identity, but this can be
        overwritten with this method to for example allow concatenating multiple transforms.

        :param map0: initial map
        :param initial_inverse_map: inverse initial map; if this is not set, the inverse map cannot be computed on the fly
        :return:  n/a
        """

        if self.opt is not None:
            self.opt.optimizer.set_initial_weight_map(init_weight, freeze_weight=freeze_weight)
            self.delayed_initial_map_still_to_be_set = False
        else:
            self.delayed_initial_weight_map_still_to_be_set = True
            self.delayed_initial_weight_map = init_weight
            self.delay_initial_weight_freeze_weight = freeze_weight

    def get_initial_map(self):
        """
        Returns the map that is used as initial condition (is typically identity)

        :return: returns the initial map
        """
        if self.opt is not None:
            return self.opt.get_initial_map()
        else:
            return None

    def get_initial_inverse_map(self):
        """
        Returns the inverse map that is used as initial condition (is typically identity)

        :return: returns the initial map
        """
        if self.opt is not None:
            return self.opt.get_initial_inverse_map()
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

    def set_model_parameters_to_zero(self):
        """
        Sets the model parameters to zero; typically not needed, but may be useful in certain scenarios.
        """

        p = self.get_model_parameters()
        if p is not None:
            for key in p:
                val = p[key]
                if torch.is_tensor(val):
                    val.zero_()
                elif type(val) == torch.nn.parameter.Parameter or type(val)==torch.Tensor:
                    val.data.zero_()

    def set_model_parameters(self,p):
        """
        Sets the model parameters

        :param p: model parameters
        :return:
        """

        if self.opt is not None:
            self.opt.set_model_parameters(p)
            self.delayed_model_parameters_still_to_be_set = False
        else:
            self.delayed_model_parameters_still_to_be_set = True
            self.delayed_model_parameters = p

    def _get_spacing_and_size_from_image_file(self,filename):
        example_ISource, hdr0, spacing0, normalized_spacing0 = \
            fileio.ImageIO().read_to_nc_format(filename,
                                               intensity_normalize=False,
                                               squeeze_image=self.squeeze_image,
                                               normalize_spacing=self.normalize_spacing)

        return normalized_spacing0,np.array(example_ISource.shape)

    def register_images_from_files(self,source_filename,target_filename,model_name,extra_info=None,
                                   lsource_filename=None, ltarget_filename=None,
                                   nr_of_iterations=None,
                                   learning_rate=None,
                                   similarity_measure_type=None,
                                   similarity_measure_sigma=None,
                                   compute_similarity_measure_at_low_res=None,
                                   map_low_res_factor=None,
                                   rel_ftol=None,
                                   smoother_type=None,
                                   json_config_out_filename=None,
                                   visualize_step=5,
                                   use_multi_scale=False,
                                   use_consensus_optimization=False,
                                   use_batch_optimization=False,
                                   checkpoint_dir='checkpoints',
                                   resume_from_last_checkpoint=False,
                                   optimizer_name=None,
                                   compute_inverse_map=False,
                                   params=None):
        """
        Registers two images. Only ISource, ITarget, spacing, and model_name need to be specified.
        Default values will be used for all of the values that are not explicitly specified.

        :param source_filename: filename of the source image
        :param target_filename: filename for the target image
        :param model_name: name of the desired registration model [string]
        :param nr_of_iterations: nr of iterations
        :param learning_rate: learning rate of optimizer
        :param similarity_measure_type: type of similarity measure ('ssd' or 'ncc')
        :param similarity_measure_sigma: similarity measures are weighted by 1/sigma^2
        :param compute_similarity_measure_at_low_res: allows computation of similarity measure at lower resolution, specified by map_low_res_factor
        :param map_low_res_factor: allows for parameterization of the registration at lower resolution than the image (0,1]
        :param rel_ftol: relative function tolerance for optimizer
        :param smoother_type: type of smoother (e.g., 'gaussian' or 'multiGaussian')
        :param json_config_out_filename: output file name for the used configuration.
        :param visualize_step: step at which the solution is visualized; if set to None, no visualizations will be created
        :param use_multi_scale: if set to True a multi-scale solver will be used
        :param use_consensus_optimization: if set to True, consensus optimization is used (i.e., independently optimized batches with the contraint that shared parameters are the same)
        :param use_batch_optimization: if set to True, batch optimization is used (i.e., optimization done w/ mini batches)
        :param checkpoint_dir: directory in which the checkpoints are written for consensus optimization
        :param resume_from_last_checkpoint: for consensus optimizer, resumes optimization from last checkpoint
        :param optimizer_name: name of the optimizer lbfgs_ls|adam|sgd
        :param compute_inverse_map: for map-based models that inverse map can optionally be computed
        :param params: parameter structure to pass settings or filename to load the settings from file.
        :return: n/a
        """

        if use_batch_optimization and use_consensus_optimization:
            raise ValueError('Cannot simultaneously select consensus AND batch optimization')

        LSource=None
        LTarget=None

        if not use_batch_optimization:
            ISource,hdr0,spacing0,normalized_spacing0 = \
                fileio.ImageIO().read_to_nc_format(source_filename,
                                                   intensity_normalize=self.normalize_intensity,
                                                   squeeze_image=self.squeeze_image,
                                                   normalize_spacing=self.normalize_spacing)

            ITarget,hdr1,spacing1,normalized_spacing1 = \
                fileio.ImageIO().read_to_nc_format(target_filename,
                                                   intensity_normalize=self.normalize_intensity,
                                                   squeeze_image=self.squeeze_image,
                                                   normalize_spacing=self.normalize_spacing)
            if lsource_filename and ltarget_filename:
                LSource, _, _, _ = \
                    fileio.ImageIO().read_to_nc_format(lsource_filename,
                                                       intensity_normalize=False,
                                                       squeeze_image=self.squeeze_image,
                                                       normalize_spacing=self.normalize_spacing)
                LTarget, _, _, _ = \
                    fileio.ImageIO().read_to_nc_format(ltarget_filename,
                                                       intensity_normalize=False,
                                                       squeeze_image=self.squeeze_image,
                                                       normalize_spacing=self.normalize_spacing)
            assert (np.all(normalized_spacing0 == normalized_spacing1))
            spacing = normalized_spacing0
            self.sz = np.array( ISource.size() )

        else:
            # batch normalization needs the filenames as input
            ISource = source_filename
            ITarget = target_filename
            if lsource_filename and ltarget_filename:
                LSource = lsource_filename
                LTarget = ltarget_filename

            # let's read one to get the spacing
            print('Reading one image to obtain the spacing information')
            if type(source_filename)==list:
                one_filename = source_filename[0]
            else:
                one_filename = source_filename

            spacing,self.sz = self._get_spacing_and_size_from_image_file(one_filename)

        self.register_images(ISource,ITarget,spacing,model_name,extra_info = extra_info,LSource=LSource,LTarget=LTarget,
                      nr_of_iterations=nr_of_iterations,
                      learning_rate=learning_rate,
                      similarity_measure_type=similarity_measure_type,
                      similarity_measure_sigma=similarity_measure_sigma,
                      compute_similarity_measure_at_low_res=compute_similarity_measure_at_low_res,
                      map_low_res_factor=map_low_res_factor,
                      rel_ftol=rel_ftol,
                      smoother_type=smoother_type,
                      json_config_out_filename=json_config_out_filename,
                      visualize_step=visualize_step,
                      use_multi_scale=use_multi_scale,
                      use_consensus_optimization=use_consensus_optimization,
                      use_batch_optimization=use_batch_optimization,
                      checkpoint_dir=checkpoint_dir,
                      resume_from_last_checkpoint=resume_from_last_checkpoint,
                      optimizer_name=optimizer_name,
                      compute_inverse_map=compute_inverse_map,
                      params=params)

    def register_images(self, ISource, ITarget, spacing, model_name=None, extra_info=None, LSource=None, LTarget=None,
                        nr_of_iterations=None,
                        learning_rate=None,
                        similarity_measure_type=None,
                        similarity_measure_sigma=None,
                        compute_similarity_measure_at_low_res=None,
                        map_low_res_factor=None,
                        rel_ftol=None,
                        smoother_type=None,
                        json_config_out_filename=None,
                        visualize_step=5,
                        use_multi_scale=False,
                        use_consensus_optimization=False,
                        use_batch_optimization=False,
                        checkpoint_dir=None,
                        resume_from_last_checkpoint=False,
                        optimizer_name=None,
                        compute_inverse_map=False,
                        params=None,
                        recording_step=None):
        """
        Registers two images. Only ISource, ITarget, spacing, and model_name need to be specified.
        Default values will be used for all of the values that are not explicitly specified.

        :param ISource: source image
        :param ITarget: target image
        :param spacing: image spacing [dx,dy,dz]
        :param model_name: name of the desired registration model [string]
        :param nr_of_iterations: nr of iterations
        :param learning_rate: learning rate of optimizer
        :param similarity_measure_type: type of similarity measure ('ssd' or 'ncc')
        :param similarity_measure_sigma: similarity measures are weighted by 1/sigma^2
        :param compute_similarity_measure_at_low_res: allows computation of similarity measure at lower resolution, specified by map_low_res_factor
        :param map_low_res_factor: allows for parameterization of the registration at lower resolution than the image (0,1]
        :param rel_ftol: relative function tolerance for optimizer
        :param smoother_type: type of smoother (e.g., 'gaussian' or 'multiGaussian')
        :param json_config_out_filename: output file name for the used configuration.
        :param visualize_step: step at which the solution is visualized; if set to None, no visualizations will be created
        :param use_multi_scale: if set to True a multi-scale solver will be used
        :param use_consensus_optimization: if set to True, consensus optimization is used (i.e., independently optimized batches with the contraint that shared parameters are the same)
        :param use_batch_optimization: if set to True, batch optimization is used (i.e., optimization done w/ mini batches)
        :param checkpoint_dir: directory in which the checkpoints are written for consensus optimization
        :param resume_from_last_checkpoint: for consensus optimizer, resumes optimization from last checkpoint
        :param optimizer_name: name of the optimizer lbfgs_ls|adam|sgd
        :param compute_inverse_map: for map-based models that inverse map can optionally be computed
        :param params: parameter structure to pass settings or filename to load the settings from file.
        :param recording_step: set tracking of all intermediate results in history each n-th step
        :return: n/a
        """

        if use_batch_optimization and use_consensus_optimization:
            raise ValueError('Cannot simultaneously select consensus AND batch optimization')

        if use_batch_optimization:
            if type(ISource)==np.ndarray or type(ITarget)==np.ndarray:
                raise ValueError('Batch normalization requires filename lists as inputs')

            if (self.sz is None) or spacing is None:
                # need to get it from the image
                if type(ISource)==list:
                    one_filename = ISource[0]
                    spacing_from_file, sz_from_file = self._get_spacing_and_size_from_image_file(one_filename)
                    if self.sz is None:
                        self.sz = sz_from_file
                    if spacing is None:
                        spacing = spacing_from_file
                else:
                    raise ValueError('Expected a list of filenames')

        else:
            if self.sz is None:
                if type(ISource)==np.ndarray or type(ISource)==torch.Tensor:
                    self.sz = np.array(ISource.shape)
                else:
                    raise ValueError('Input image needs to be a numpy array')

        if params is None:
            self.params = pars.ParameterDict()
        elif type(params) == pars.ParameterDict:
            self.params = params
        elif type(params) == type('acharacter'):
            self.params = pars.ParameterDict()
            self.params.load_JSON(params)
            model_name = self.params['model']['registration_model']['type']
        else:
            raise ValueError('Unknown parameter format: ' + str(type(params)))

        if use_batch_optimization or type(ISource) == torch.Tensor:
            self.ISource = ISource
            self.ITarget = ITarget
        else:
            self.ISource = AdaptVal(torch.from_numpy(ISource.copy()))
            self.ITarget = AdaptVal(torch.from_numpy(ITarget))

        self.spacing = spacing

        if model_name not in self.available_models:
            print('Unknown model name: ' + model_name)
            MF.AvailableModels().print_available_models()
        else:
            # this model exists so let's use it
            self.useMap = self.available_models[model_name][2]
            self.params['model']['deformation']['use_map'] = self.useMap
            self.params['model']['registration_model']['type'] = model_name

            if optimizer_name is not None:
                self.params['optimizer']['name'] = optimizer_name

            if nr_of_iterations is not None:
                self.params['optimizer']['single_scale']['nr_of_iterations'] = nr_of_iterations

            if similarity_measure_sigma is not None:
                self.params['model']['registration_model']['similarity_measure']['sigma'] = similarity_measure_sigma

            if similarity_measure_type is not None:
                self.params['model']['registration_model']['similarity_measure']['type'] = similarity_measure_type

            if compute_similarity_measure_at_low_res is not None:
                self.params['model']['deformation']['compute_similarity_measure_at_low_res'] = compute_similarity_measure_at_low_res

            if map_low_res_factor is not None:
                self.params['model']['deformation']['map_low_res_factor'] = map_low_res_factor

            if rel_ftol is not None:
                self.params['optimizer']['single_scale']['rel_ftol'] = rel_ftol

            if smoother_type is not None:
                self.params['model']['registration_model']['forward_model']['smoother']['type'] = smoother_type

            if (checkpoint_dir is not None) and use_consensus_optimization:
                self.params['optimizer']['consensus_settings']['checkpoint_output_directory'] = checkpoint_dir

            if resume_from_last_checkpoint and use_consensus_optimization:
                self.params['optimizer']['consensus_settings']['continue_from_last_checkpoint'] = True
            if use_multi_scale:
                if use_consensus_optimization or use_batch_optimization:
                    raise ValueError('Consensus or batch optimization is not yet supported for multi-scale registration')
                else:
                    self.opt = MO.SimpleMultiScaleRegistration(self.ISource,
                                                               self.ITarget,
                                                               self.spacing,
                                                               self.sz,
                                                               self.params,
                                                               compute_inverse_map=compute_inverse_map,
                                                               default_learning_rate=learning_rate)
            else:
                if use_consensus_optimization:
                    self.opt = MO.SimpleSingleScaleConsensusRegistration(self.ISource,
                                                                         self.ITarget,
                                                                         self.spacing,
                                                                         self.sz,
                                                                         self.params,
                                                                         compute_inverse_map=compute_inverse_map,
                                                                         default_learning_rate=learning_rate)
                elif use_batch_optimization:
                    self.opt = MO.SimpleSingleScaleBatchRegistration(self.ISource,
                                                                     self.ITarget,
                                                                     self.spacing,
                                                                     self.sz,
                                                                     self.params,
                                                                     compute_inverse_map=compute_inverse_map,
                                                                     default_learning_rate=learning_rate)
                else:
                    self.opt = MO.SimpleSingleScaleRegistration(self.ISource,
                                                                self.ITarget,
                                                                self.spacing,
                                                                self.sz,
                                                                self.params,
                                                                compute_inverse_map=compute_inverse_map,
                                                                default_learning_rate=learning_rate)

            if visualize_step is not None:
                self.opt.get_optimizer().set_visualization(True)
                self.opt.get_optimizer().set_visualize_step(visualize_step)
            else:
                self.opt.get_optimizer().set_visualization(False)
                self.opt.get_optimizer().set_visualize_step(visualize_step)

            if recording_step is not None:
                self.opt.get_optimizer().set_recording_step(recording_step)


            self.optimizer_has_been_initialized = True

            if self.delayed_model_parameters_still_to_be_set:
                self.set_model_parameters(self.delayed_model_parameters)

            if self.delayed_initial_map_still_to_be_set:
                self.set_initial_map(self.delayed_initial_map,initial_inverse_map=self.delayed_initial_inverse_map)
            if self.delayed_initial_weight_map_still_to_be_set:
                self.set_weight_map(self.delayed_initial_weight_map, self.delay_initial_weight_freeze_weight)

            if use_multi_scale and LSource is not None and LTarget is not None:
                LSource = AdaptVal(torch.from_numpy(LSource)) if not type(LSource) == torch.Tensor else LSource
                LTarget = AdaptVal(torch.from_numpy(LTarget)) if not type(LTarget) == torch.Tensor else LTarget
                self.opt.optimizer.set_source_label( AdaptVal(LSource))
                self.opt.optimizer.set_target_label( AdaptVal(LTarget))
            if extra_info is not None:
                self._set_analysis(self.opt.optimizer, extra_info)

            self.opt.register()

            if json_config_out_filename is not None:
                if type(json_config_out_filename) is tuple:
                    self.params.write_JSON_and_JSON_comments(json_config_out_filename)
                else:
                    self.params.write_JSON(json_config_out_filename)
