"""
Package to quickly instantiate registration models by name.
"""

import registration_networks as RN

class ModelFactory(object):
    """
    Factory class to instantiate registration models.
    """

    def __init__(self,sz,spacing):
        """
        Constructor.
        
        :param sz: image size, expectes BxCxXxYxZ format 
        :param spacing: spatial spacing as array (as many entries as there are spatial dimensions)
        """
        self.sz = sz
        """size of the image (BxCxXxYxZ) format"""
        self.spacing = spacing
        """spatial spacing"""
        self.dim = len( spacing )
        """spatial dimension"""

        self.models = {
            'svf_map': (RN.SVFMapNet,RN.SVFMapLoss),
            'svf_image': (RN.SVFImageNet,RN.SVFImageLoss),
            'svf_scalar_momentum_image': (RN.SVFScalarMomentumImageNet,RN.SVFScalarMomentumImageLoss),
            'svf_scalar_momentum_map': (RN.SVFScalarMomentumMapNet,RN.SVFScalarMomentumMapLoss),
            'svf_vector_momentum_image': (RN.SVFVectorMomentumImageNet, RN.SVFVectorMomentumImageLoss),
            'svf_vector_momentum_map': (RN.SVFVectorMomentumMapNet, RN.SVFVectorMomentumMapLoss),
            'lddmm_shooting_map': (RN.LDDMMShootingVectorMomentumMapNet,
                                   RN.LDDMMShootingVectorMomentumMapLoss),
            'lddmm_shooting_image': (RN.LDDMMShootingVectorMomentumImageNet,
                                     RN.LDDMMShootingVectorMomentumImageLoss),
            'lddmm_shooting_scalar_momentum_map': (RN.LDDMMShootingScalarMomentumMapNet,
                                                   RN.LDDMMShootingScalarMomentumMapLoss),
            'lddmm_shooting_scalar_momentum_image': (RN.LDDMMShootingScalarMomentumImageNet,
                                                     RN.LDDMMShootingScalarMomentumImageLoss),
            'svf_quasi_momentum_image': (RN.SVFQuasiMomentumImageNet,
                                         RN.SVFQuasiMomentumImageLoss)
        }
        """dictionary defining all the models"""

    def add_model(self,modelName,networkClass,lossClass):
        """
        Allows to quickly add a new model.
        
        :param modelName: name for the model
        :param networkClass: network class defining the model
        :param lossClass: loss class being used by ty the model
        """
        print('Registering new model ' + modelName )
        self.models[modelName] = (networkClass,lossClass)

    def print_available_models(self):
        """
        Prints the models that are available and can be created with `create_registration_model`
        """

        print('\nKnown registration models are:')
        print('   svf_map                              : map-based stationary velocity field')
        print('   svf_image                            : image-based stationary velocity field')
        print('   svf_scalar_momentum_image            : image-based stationary velocity field using the scalar momentum')
        print('   svf_scalar_momentum_map              : map-based stationary velocity field using the scalar momentum')
        print('   svf_vector_momentum_image            : image-based stationary velocity field using the vector momentum')
        print('   svf_vector_momentum_map              : map-based stationary velocity field using the vector momentum')
        print('   lddmm_shooting_map                   : map-based shooting-based LDDMM using the vector momentum')
        print('   lddmm_shooting_image                 : image-based shooting-based LDDMM using the vector momentum')
        print('   lddmm_shooting_scalar_momentum_map   : map-based shooting-based LDDMM using the scalar momentum')
        print('   lddmm_shooting_scalar_momentum_image : image-based shooting-based LDDMM using the scalar momentum')

        print('\nAll registered models are:')
        print(self.models)

    def create_registration_model(self, modelName, params):
        """
        Performs the actual model creation including the loss function
        
        :param modelName: Name of the model to be created 
        :param params: parameter dictionary of type :class:`ParameterDict`
        :return: a two-tupel: model, loss
        """

        cparams = params[('registration_model',{},'specifies the registration model')]
        cparams['type']= (modelName,'Name of the registration model')

        if self.models.has_key(modelName):
            print('Using ' + modelName + ' model')
            model = self.models[modelName][0](self.sz, self.spacing, cparams)
            loss = self.models[modelName][1](list(model.parameters())[0], self.sz, self.spacing, cparams)
            return model, loss
        else:
            self.print_available_models()
            raise ValueError('Registration model: ' + modelName + ' not known')
