import registration_networks as RN

class ModelFactory(object):

    def __init__(self,sz,spacing):
        self.sz = sz
        self.spacing = spacing
        self.dim = len( spacing )

        self.models = {
            'svf_map': (RN.SVFMapNet,RN.SVFMapLoss),
            'svf_image': (RN.SVFImageNet,RN.SVFImageLoss),
            'lddmm_shooting_map': (RN.LDDMMShootingVectorMomentumMapNet,
                                   RN.LDDMMShootingVectorMomentumMapLoss),
            'lddmm_shooting_image': (RN.LDDMMShootingVectorMomentumImageNet,
                                     RN.LDDMMShootingVectorMomentumImageLoss),
            'lddmm_shooting_scalar_momentum_map': (RN.LDDMMShootingScalarMomentumMapNet,
                                                   RN.LDDMMShootingScalarMomentumMapLoss),
            'lddmm_shooting_scalar_momentum_image': (RN.LDDMMShootingScalarMomentumImageNet,
                                                     RN.LDDMMShootingScalarMomentumImageLoss)
        }

    def add_model(self,modelName,networkClass,lossClass):
        print('Registering new model ' + modelName )
        self.models[modelName] = (networkClass,lossClass)

    def print_available_models(self):
        print('Known registration models are:')
        print('   SVF                        : stationary velocity field')
        print('   LDDMMShooting              : shooting-based LDDMM using the vector momentum')
        print('   LDDMMShootingScalarMomentum: shooting-based LDDMM using the scalar momentum')

        print('All registered models are:')
        print(self.models)

    def create_registration_model(self, modelName, params):
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
