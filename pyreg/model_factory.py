import registration_networks as RN
import module_parameters as pars

class ModelFactory(object):

    def __init__(self,sz,spacing):
        self.sz = sz
        self.spacing = spacing
        self.dim = len( spacing )

    def print_known_models(self):
        print('Known registration models are:')
        print('   SVF                        : stationary velocity field')
        print('   LDDMMShooting              : shooting-based LDDMM using the vector momentum')
        print('   LDDMMShootingScalarMomentum: shooting-based LDDMM using the scalar momentum')

    def createRegistrationModel(self,modelName,useMap,params):
        cparams = pars.setCurrentCategory(params,'registration_model')
        pars.setCurrentKey(cparams,'type',modelName,'Name of the registration model')
        if modelName=='SVF':
            if useMap:
                print('Using map-based SVF model')
                model = RN.SVFMapNet(self.sz,self.spacing,cparams)
                loss = RN.SVFMapLoss(list(model.parameters())[0], self.sz, self.spacing, cparams)
                return model,loss
            else:
                print('Using image-based SVF model')
                model = RN.SVFImageNet(self.sz,self.spacing,cparams)
                loss = RN.SVFImageLoss(list(model.parameters())[0], self.sz, self.spacing, cparams)
                return model,loss
        elif modelName=='LDDMMShooting':
            # TODO: Actually implement this
            if useMap:
                print('Using map-based shooting LDDMM')
                model = RN.LDDMMShootingVectorMomentumMapNet(self.sz, self.spacing, cparams)
                loss = RN.LDDMMShootingVectorMomentumMapLoss(list(model.parameters())[0], self.sz, self.spacing, cparams)
                return model, loss
            else:
                print('Using shooting-based LDDMM')
                model = RN.LDDMMShootingVectorMomentumImageNet(self.sz,self.spacing,cparams)
                loss = RN.LDDMMShootingVectorMomentumImageLoss(list(model.parameters())[0], self.sz, self.spacing, cparams)
                return model,loss
        elif modelName=='LDDMMShootingScalarMomentum':
            # TODO: Actually implement this
            if useMap:
                print('Using map-based shooting scalar-momentum LDDMM')
                model = RN.LDDMMShootingScalarMomentumMapNet(self.sz, self.spacing, cparams)
                loss = RN.LDDMMShootingScalarMomentumMapLoss(list(model.parameters())[0], self.sz, self.spacing, cparams)
                return model, loss
            else:
                print('Using shooting-based scalar-momentum LDDMM')
                model = RN.LDDMMShootingScalarMomentumImageNet(self.sz,self.spacing,cparams)
                loss = RN.LDDMMShootingScalarMomentumImageLoss(list(model.parameters())[0], self.sz, self.spacing, cparams)
                return model,loss
        else:
            self.print_known_models()
            raise ValueError( 'Registration model: ' + modelName + ' not known')

