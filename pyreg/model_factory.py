"""
Package to quickly instantiate registration models by name.
"""
from __future__ import print_function
from __future__ import absolute_import

from builtins import object
from . import registration_networks as RN
import pyreg.utils as utils
import pyreg.image_sampling as IS
from pyreg.data_wrapper import AdaptVal

import torch
from torch.autograd import Variable
import numpy as np


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
                                         'EXPERIMENTAL: Image-based SVF version which estimates velcocity based on a smoothed vetor field')
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


# todo: make sure this works; streamline parameters as we can get sz_sim from I0 and maybe sz_model from the model_parameters
def run_model(model_name, model_parameters, I0, sz_sim,spacing_sim,sz_model,spacing_model, params):
    """
    Runs the model forward

    :param model_name: name of the model
    :param model_parameters: parameters of the model (as gotten by model.get_model_parameters() )
    :param I0: source image
    :param sz_sim: size of similarity image (same as size of I0)
    :param spacing_sim: spacing of I0
    :param sz_model: size of the parameterization
    :param spacing_model: spacing of the parameteriation
    :param params: additonal parameters
    :return: either the map to warp the source image (for a map-based model) or the warped source image
    """

    available_models = AvailableModels().get_models()

    raise ValueError('Not tested yet. Check that this works first before using it. Feel free to comment out and try at your own risk.')

    if model_name not in available_models:
        _print_models(available_models)
        raise ValueError('Registration model: ' + model_name + ' not known')
    else:
        model, loss = ModelFactory(sz_sim, spacing_sim, sz_model, spacing_model).create_registration_model(model_name, params)
        model.set_registration_parameters(model_parameters, sz_model, spacing_model)

        sampler = IS.ResampleImage()

        uses_map = available_models[model_name][2]

        if uses_map:

            spline_order = params['model']['registration_model'][('spline_order', 1, 'Spline interpolation order; 1 is linear interpolation (default); 3 is cubic spline')]

            id = utils.identity_map_multiN(sz_sim, spacing_sim)
            identityMap = AdaptVal(Variable(torch.from_numpy(id), requires_grad=False))

            if not np.all(spacing_sim==spacing_model):
                lowres_id = utils.identity_map_multiN(sz_model, spacing_model)
                lowResIdentityMap = AdaptVal(Variable(torch.from_numpy(lowres_id), requires_grad=False))
                lowResISource,_ = sampler.downsample_image_to_size(I0, spacing_sim, sz_model, spline_order)

                rec_tmp = model(lowResIdentityMap, lowResISource)
                # now upsample to correct resolution
                phiWarped, _ = sampler.upsample_image_to_size(rec_tmp, spacing_model, spacing_sim, spline_order)

            else:
                phiWarped = model(identityMap, I0 )

            return phiWarped

        else: # does not use map
            IWarped = model(I0)

            return IWarped


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

    def create_registration_model(self, modelName, params):
        """
        Performs the actual model creation including the loss function
        
        :param modelName: Name of the model to be created 
        :param params: parameter dictionary of type :class:`ParameterDict`
        :return: a two-tupel: model, loss
        """

        cparams = params[('registration_model',{},'specifies the registration model')]
        cparams['type']= (modelName,'Name of the registration model')

        if modelName in self.models:
            print('Using ' + modelName + ' model')
            model = self.models[modelName][0](self.sz_model, self.spacing_model, cparams)
            loss = self.models[modelName][1](list(model.parameters())[0], self.sz_sim, self.spacing_sim, self.sz_model, self.spacing_model, cparams)
            return model, loss
        else:
            self.print_available_models()
            raise ValueError('Registration model: ' + modelName + ' not known')
