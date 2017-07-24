import registration_networks as RN

class ModelFactory(object):

    def __init__(self,sz,spacing):
        self.sz = sz
        self.spacing = spacing
        self.dim = len( spacing )

    def print_known_models(self):
        print('Known registration models are:')
        print('   SVF: stationary velocity field')

    def createRegistrationModel(self,modelName='SVF',useMap=False,params=None):
        if modelName=='SVF':
            if useMap:
                print('Using map-based SVF model')
                model = RN.SVFNetMap(self.sz,self.spacing,params)
                loss = RN.SVFLossMap(list(model.parameters())[0], self.sz, self.spacing, params)
                return model,loss
            else:
                print('Using image-based SVF model')
                model = RN.SVFNet(self.sz,self.spacing,params)
                loss = RN.SVFLoss(list(model.parameters())[0], self.sz, self.spacing, params)
                return model,loss
        elif modelName=='LDDMM':
            raise ValueError('LDDMM not yet implemented')
        else:
            self.print_known_models()
            raise ValueError( 'Registration model: ' + modelName + ' not known')

