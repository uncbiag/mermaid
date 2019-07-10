"""
Package to quickly instantiate registration models by name.
"""
from __future__ import print_function
from __future__ import absolute_import

from builtins import object
from . import registration_networks as RN

def _print_models(models):
    print('\nKnown registration models are:')
    print('------------------------------')
    for key in models:
        print('{model_name:>40s}: {model_description}'.format(model_name=key, model_description=models[key][3]))


class AvailableModels(object):

    def __init__(self):
        # (model, loss, uses map, description)
        self.models = {
            'affine_map': (RN.AffineMapNet, RN.AffineMapLoss, True,
                           'map-based affine registration'),
            'diffusion_map': (RN.RegistrationNetDisplacement, RN.DiffusionMapLoss, True,
                              'displacement-based diffusion registration'),
            'curvature_map': (RN.RegistrationNetDisplacement, RN.CurvatureMapLoss, True,
                              'displacement-based curvature registration'),
            'total_variation_map': (RN.RegistrationNetDisplacement, RN.TotalVariationMapLoss, True,
                                    'displacement-based total variation registration'),
            'svf_map': (RN.SVFMapNet, RN.SVFMapLoss, True,
                        'map-based stationary velocity field'),
            'svf_image': (RN.SVFImageNet, RN.SVFImageLoss, False,
                          'image-based stationary velocity field'),
            'svf_scalar_momentum_image': (RN.SVFScalarMomentumImageNet, RN.SVFScalarMomentumImageLoss, False,
                                          'image-based stationary velocity field using the scalar momentum'),
            'svf_scalar_momentum_map': (RN.SVFScalarMomentumMapNet, RN.SVFScalarMomentumMapLoss, True,
                                        'map-based stationary velocity field using the scalar momentum'),
            'svf_vector_momentum_image': (RN.SVFVectorMomentumImageNet, RN.SVFVectorMomentumImageLoss, False,
                                          'image-based stationary velocity field using the vector momentum'),
            'svf_vector_momentum_map': (RN.SVFVectorMomentumMapNet, RN.SVFVectorMomentumMapLoss, True,
                                        'map-based stationary velocity field using the vector momentum'),
            'lddmm_shooting_map': (RN.LDDMMShootingVectorMomentumMapNet,
                                   RN.LDDMMShootingVectorMomentumMapLoss, True,
                                   'map-based shooting-based LDDMM using the vector momentum'),
            'lddmm_shooting_image': (RN.LDDMMShootingVectorMomentumImageNet,
                                     RN.LDDMMShootingVectorMomentumImageLoss, False,
                                     'image-based shooting-based LDDMM using the vector momentum'),
            'lddmm_shooting_scalar_momentum_map': (RN.LDDMMShootingScalarMomentumMapNet,
                                                   RN.LDDMMShootingScalarMomentumMapLoss, True,
                                                   'map-based shooting-based LDDMM using the scalar momentum'),
            'lddmm_shooting_scalar_momentum_image': (RN.LDDMMShootingScalarMomentumImageNet,
                                                     RN.LDDMMShootingScalarMomentumImageLoss, False,
                                                     'image-based shooting-based LDDMM using the scalar momentum'),
            'svf_quasi_momentum_image': (RN.SVFQuasiMomentumImageNet,
                                         RN.SVFQuasiMomentumImageLoss, False,
                                         'EXPERIMENTAL: Image-based SVF version which estimates velcocity based on a smoothed vetor field'),

            'lddmm_adapt_smoother_map': (RN.LDDMMAdaptiveSmootherMomentumMapNet,RN.LDDMMAdaptiveSmootherMomentumMapLoss,True,"map-based shooting-based Region specific diffemorphic mapping, with a spatio-temporal regularizer"),
            'svf_adapt_smoother_map': (RN.SVFVectorAdaptiveSmootherMomentumMapNet,RN.SVFVectorAdaptiveSmootherMomentumMapLoss,True,"map-based shooting-based vSVF, with a spatio regularizer"),
            'one_step_map': (RN.OneStepMapNet, RN.OneStepMapLoss, True, 'map based displacement field')
        }
        """dictionary defining all the models"""

    def get_models(self):
        """
        Returns all available models as a dictionary which has as keys the model name and tuple entries of the form
        (networkclass,lossclass,usesMap,explanation_string)
        :return: the model dictionary
        """
        return self.models

    def print_available_models(self):
        """
        Prints the models that are available and can be created with `create_registration_model`
        """

        _print_models(self.models)


class ModelFactory(object):
    """
    Factory class to instantiate registration models.
    """

    def __init__(self,sz_sim,spacing_sim,sz_model,spacing_model):
        """
        Constructor.

        :param sz_sim: image/map size to evaluate the similarity measure of the loss
        :param spacing_sim: image/map spacing to evaluate the similarity measure of the loss
        :param sz_model: sz of the model parameters (will only be different from sz_sim if computed at low res)
        :param spacing_model: spacing of model parameters (will only be different from spacing_sim if computed at low res)
        """

        self.sz_sim = sz_sim
        """size of the image (BxCxXxYxZ) format as used in the similarity measure part of the loss function"""
        self.spacing_sim = spacing_sim
        """spatial spacing as used in the similarity measure of the loss function"""
        self.sz_model = sz_model
        """size of the parameters (BxCxXxYxZ) as used in the model itself (and possibly in the loss function for regularization)"""
        self.spacing_model = spacing_model
        """spatial spacing as used in the model itself (and possibly in the loss function for regularization)"""
        self.dim = len( spacing_model )
        """spatial dimension"""
        self.models = AvailableModels().get_models()

        assert( len(spacing_model)==len(spacing_sim) )

    def get_models(self):
        """
        Returns all available models as a dictionary which has as keys the model name and tuple entries of the form
        (networkclass,lossclass,usesMap,explanation_string)
        :return: the model dictionary
        """
        return self.models

    def add_model(self,modelName,networkClass,lossClass,useMap,modelDescription='custom model'):
        """
        Allows to quickly add a new model.
        
        :param modelName: name for the model
        :param networkClass: network class defining the model
        :param lossClass: loss class being used by ty the model
        """
        print('Registering new model ' + modelName )
        self.models[modelName] = (networkClass,lossClass,useMap,modelDescription)

    def print_available_models(self):
        """
        Prints the models that are available and can be created with `create_registration_model`
        """

        _print_models(self.models)

    def create_registration_model(self, modelName, params, compute_inverse_map=False):
        """
        Performs the actual model creation including the loss function
        
        :param modelName: Name of the model to be created 
        :param params: parameter dictionary of type :class:`ParameterDict`
        :param compute_inverse_map: for a map-based model if this is turned on then the inverse map is computed on the fly
        :return: a two-tuple: model, loss
        """

        cparams = params[('registration_model',{},'specifies the registration model')]
        cparams['type']= (modelName,'Name of the registration model')
        external_env = cparams[('env', {},"env settings, typically are specificed by the external package, including the mode for solver or for smoother")]
        """settings for the task environment of the solver or smoother"""
        get_momentum_from_external_network = external_env[('get_momentum_from_external_network', False,"use external network to predict momentum, notice that the momentum network is not built in this package")]

        if modelName in self.models:

            uses_map = self.models[modelName][2]
            if uses_map:
                print('Using map-based ' + modelName + ' model')
                model = self.models[modelName][0](self.sz_model, self.spacing_model, cparams,compute_inverse_map)
            else:
                print('Using ' + modelName + ' model')
                model = self.models[modelName][0](self.sz_model, self.spacing_model, cparams)
            if get_momentum_from_external_network:
                loss = self.models[modelName][1](model.m, self.sz_sim, self.spacing_sim, self.sz_model,
                                                 self.spacing_model, cparams)
            else:
                print("works in mermaid iter mode")
                loss = self.models[modelName][1](list(model.parameters())[0], self.sz_sim, self.spacing_sim, self.sz_model, self.spacing_model, cparams)
            return model, loss
        else:
            self.print_available_models()
            raise ValueError('Registration model: ' + modelName + ' not known')
