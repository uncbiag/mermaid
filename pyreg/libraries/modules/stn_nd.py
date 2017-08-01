from torch.nn.modules.module import Module
from functions.stn_nd import STNFunction_ND, STNFunction_ND_BCXYZ, STNFunction

class STN_ND(Module):
    def __init__(self, dim):
        super(STN_ND, self).__init__()
        self.dim = dim
        self.f = STNFunction_ND( self.dim )
    def forward(self, input1, input2):
        return self.f(input1, input2)

class STN_ND_BCXYZ(Module):
    def __init__(self, dim):
        super(STN_ND_BCXYZ, self).__init__()
        self.dim = dim
        self.f = STNFunction_ND_BCXYZ( self.dim )
    def forward(self, input1, input2):
        return self.f(input1, input2)

# old code starts here

class STN(Module):
    def __init__(self):
        super(STN, self).__init__()
        self.f = STNFunction()
    def forward(self, input1, input2):
        return self.f(input1, input2)
