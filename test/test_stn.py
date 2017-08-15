# start with the setup

import os
import sys

sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../pyreg'))
sys.path.insert(0,os.path.abspath('../pyreg/libraries'))

import numpy as np
import numpy.testing as npt
import utils
import torch
from torch.autograd import Variable

from libraries.modules.stn_nd import STN_ND_BCXYZ

import unittest
import imp

try:
    imp.find_module('HtmlTestRunner')
    foundHTMLTestRunner = True
    import HtmlTestRunner
except ImportError:
    foundHTMLTestRunner = False

# done with all the setup

# testing code starts here

class Test_stn_1d(unittest.TestCase):

    def setUp(self):
        self.stn = STN_ND_BCXYZ(1)

    def tearDown(self):
        pass

    def test_identity(self):
        I0 = Variable( torch.zeros([1,1,50]) )
        I0[0,0,12:33] = 1
        I1 = Variable( torch.zeros([1,1,50]) )
        I1[0,0,12:33] = 1
        id = Variable( torch.from_numpy( utils.identity_map_multiN([1,1,50]) ) )
        I1_warped = self.stn(I0,id)
        npt.assert_almost_equal(I1.data.numpy(),I1_warped.data.numpy(),decimal=5)

    def test_shift(self):
        I0 = Variable(torch.zeros([1, 1, 100]))
        I0[0, 0, 12:33] = 1
        I1 = Variable(torch.zeros([1, 1, 100]))
        I1[0, 0, 22:43] = 1
        id = Variable(torch.from_numpy(utils.identity_map_multiN([1, 1, 100])))
        id_shift = Variable(torch.zeros([1,1,100]))
        id_shift[0,0,:] = (id[0,0,:] - 10*(id[0,0,1]-id[0,0,0]))
        I1_warped = self.stn(I0, id_shift)
        npt.assert_almost_equal(I1.data.numpy(), I1_warped.data.numpy(), decimal=5)

    def test_expand(self):
        stn = STN_ND_BCXYZ(1)
        I0 = Variable(torch.zeros([1, 1, 11]))
        I0[0, 0, :] = torch.FloatTensor([0,0,0,0,1,1,1,0,0,0,0])
        I1 = Variable(torch.zeros([1, 1, 11]))
        I1[0, 0, :] = torch.FloatTensor([0,0,0.5,1,1,1,1,1,0.5,0,0])
        id = Variable(torch.from_numpy(utils.identity_map_multiN([1, 1, 11])))
        id_expand = Variable(torch.zeros([1, 1, 11]))
        id_expand[0, 0, :] = 0.5*id[0, 0, :]
        I1_warped = stn(I0, id_expand)
        npt.assert_almost_equal(I1.data.numpy(), I1_warped.data.numpy(), decimal=5)

if __name__ == '__main__':
    if foundHTMLTestRunner:
        unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='test_output'))
    else:
       unittest.main()
