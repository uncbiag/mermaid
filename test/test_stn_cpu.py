# start with the setup

import os
import sys

sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../mermaid'))
sys.path.insert(0,os.path.abspath('../mermaid/libraries'))

import numpy as np
import numpy.testing as npt
import mermaid.utils as utils
import torch

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
        pass

    def tearDown(self):
        pass

    def test_identity(self):
        spacing = np.array([1./(50.-1.)])
        self.stn = STN_ND_BCXYZ(spacing)
        I0 =  torch.zeros([1,1,50])
        I0[0,0,12:33] = 1
        I1 =  torch.zeros([1,1,50])
        I1[0,0,12:33] = 1
        id =  torch.from_numpy( utils.identity_map_multiN([1,1,50],spacing) )
        I1_warped = self.stn(I0,id)
        npt.assert_almost_equal(I1.data.numpy(),I1_warped.data.numpy(),decimal=4)

    def test_shift(self):
        spacing = np.array([1./(100.-1.)])
        self.stn = STN_ND_BCXYZ(spacing)
        I0 = torch.zeros([1, 1, 100])
        I0[0, 0, 12:33] = 1
        I1 = torch.zeros([1, 1, 100])
        I1[0, 0, 22:43] = 1
        id = torch.from_numpy(utils.identity_map_multiN([1, 1, 100],spacing))
        id_shift = torch.zeros([1,1,100])
        id_shift[0,0,:] = (id[0,0,:] - 10*(id[0,0,1]-id[0,0,0]))
        I1_warped = self.stn(I0, id_shift)
        npt.assert_almost_equal(I1.data.numpy(), I1_warped.data.numpy(), decimal=4)

    def test_expand(self):
        spacing = np.array([1./(11.-1.)])
        self.stn = STN_ND_BCXYZ(spacing)
        I0 = torch.zeros([1, 1, 11])
        I0[0, 0, :] = torch.FloatTensor([0,0,0,0,1,1,1,0,0,0,0])
        I1 = torch.zeros([1, 1, 11])
        I1[0, 0, :] = torch.FloatTensor([0,0,0.5,1,1,1,1,1,0.5,0,0])
        id = torch.from_numpy(utils.identity_map_multiN([1, 1, 11],spacing))
        id_expand = torch.zeros([1, 1, 11])
        id_expand[0, 0, :] = 0.5*(id[0, 0, :]-0.5)+0.5
        I1_warped = self.stn(I0, id_expand)
        npt.assert_almost_equal(I1.data.numpy(), I1_warped.data.numpy(), decimal=4)

class Test_stn_2d(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_identity(self):
        spacing = np.array([1./(50.-1.),1./(50.-1.)])
        self.stn = STN_ND_BCXYZ(spacing)
        I0 =  torch.zeros([1,1,50,50])
        I0[0,0,12:33,12:33] = 1
        I1 =  torch.zeros([1,1,50,50])
        I1[0,0,12:33,12:33] = 1
        id =  torch.from_numpy( utils.identity_map_multiN([1,1,50,50],spacing) )
        I1_warped = self.stn(I0,id)
        npt.assert_almost_equal(I1.data.numpy(),I1_warped.data.numpy(),decimal=4)

    def test_shift(self):
        spacing = np.array([1. / (10. - 1.), 1. / (10. - 1.)])
        self.stn = STN_ND_BCXYZ(spacing)
        I0 = torch.zeros([1, 1, 10,10])
        I0[0, 0, 2:6,2:6] = 1
        I1 = torch.zeros([1, 1, 10,10])
        I1[0, 0, 5:9,5:9] = 1
        id = torch.from_numpy(utils.identity_map_multiN([1, 1, 10,10],spacing))
        id_shift = torch.zeros([1,2,10,10])
        id_shift[0,:,:,:] = (id[0,:,:,:] - 3*(id[0,0,1,0]-id[0,0,0,0]))
        I1_warped = self.stn(I0, id_shift)
        npt.assert_almost_equal(I1.data.numpy(), I1_warped.data.numpy(), decimal=4)

    def test_expand(self):
        spacing = np.array([1. / (11. - 1.), 1. / (11. - 1.)])
        self.stn = STN_ND_BCXYZ(spacing)
        I0 = torch.zeros([1, 1, 11,11])
        I0[0, 0, 4:7,4:7] = 1
        I1 = torch.zeros([1, 1, 11,11])
        I1[0, 0, 3:8,3:8] = 1
        I1[0,0,2,3:8]=0.5
        I1[0,0,8,3:8]=0.5
        I1[0,0,3:8,2]=0.5
        I1[0,0,3:8,8]=0.5
        I1[0,0,2,2]=0.25
        I1[0,0,2,8]=0.25
        I1[0,0,8,2]=0.25
        I1[0,0,8,8]=0.25
        id = torch.from_numpy(utils.identity_map_multiN([1, 1, 11,11],spacing))
        id_expand = torch.zeros([1, 2, 11,11])
        id_expand[0, :, :,:] = 0.5*(id[0, :, :,:]-0.5)+0.5
        I1_warped = self.stn(I0, id_expand)
        npt.assert_almost_equal(I1.data.numpy(), I1_warped.data.numpy(), decimal=4)

class Test_stn_3d(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_identity(self):
        spacing = np.array([1. / (50. - 1.), 1. / (50. - 1.), 1./(50.-1.)])
        self.stn = STN_ND_BCXYZ(spacing)
        I0 =  torch.zeros([1,1,50,50,50])
        I0[0,0,12:33,12:33,12:33] = 1
        I1 =  torch.zeros([1,1,50,50,50])
        I1[0,0,12:33,12:33,12:33] = 1
        id =  torch.from_numpy( utils.identity_map_multiN([1,1,50,50,50],spacing) )
        I1_warped = self.stn(I0,id)
        npt.assert_almost_equal(I1.data.numpy(),I1_warped.data.numpy(),decimal=4)

    def test_shift(self):
        spacing = np.array([1. / (100. - 1.), 1. / (100. - 1.), 1. / (100. - 1.)])
        self.stn = STN_ND_BCXYZ(spacing)
        I0 = torch.zeros([1, 1, 100,100,100])
        I0[0, 0, 12:33,12:33,12:33] = 1
        I1 = torch.zeros([1, 1, 100,100,100])
        I1[0, 0, 22:43,22:43,22:43] = 1
        id = torch.from_numpy(utils.identity_map_multiN([1, 1, 100,100,100],spacing))
        id_shift = torch.zeros([1,3,100,100,100])
        id_shift[0,:,:,:] = (id[0,:,:,:] - 10*(id[0,0,1,0,0]-id[0,0,0,0,0]))
        I1_warped = self.stn(I0, id_shift)
        npt.assert_almost_equal(I1.data.numpy(), I1_warped.data.numpy(), decimal=4)

    def test_expand(self):
        spacing = np.array([1. / (11. - 1.), 1. / (11. - 1.), 1. / (11. - 1.)])
        self.stn = STN_ND_BCXYZ(spacing)
        I0 = torch.zeros([1, 1, 11,11,11])
        I0[0, 0, 4:7,4:7,4:7] = 1
        I1_np = np.array([[[[[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ]],
         [[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ]],
         [[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.125,  0.25 ,  0.25 ,  0.25 ,  0.25 ,  0.25 ,
            0.125,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.125,  0.25 ,  0.25 ,  0.25 ,  0.25 ,  0.25 ,
            0.125,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ]],
         [[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ]],
         [[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ]],
         [[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ]],
         [[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ]],
         [[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.5  ,  1.   ,  1.   ,  1.   ,  1.   ,  1.   ,
            0.5  ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ]],
         [[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.125,  0.25 ,  0.25 ,  0.25 ,  0.25 ,  0.25 ,
            0.125,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.25 ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,  0.5  ,
            0.25 ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.125,  0.25 ,  0.25 ,  0.25 ,  0.25 ,  0.25 ,
            0.125,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ]],
         [[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ]],
         [[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ],
          [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,
            0.   ,  0.   ,  0.   ]]]]])


        I1 = torch.from_numpy(I1_np)
        id = torch.from_numpy(utils.identity_map_multiN([1, 1, 11,11,11],spacing))
        id_expand = torch.zeros([1, 3, 11,11,11])
        id_expand[0, :, :,:,:] = 0.5*(id[0, :, :,:,:]-0.5)+0.5
        I1_warped = self.stn(I0, id_expand)
        npt.assert_almost_equal(I1.data.numpy(), I1_warped.data.numpy(), decimal=4)

def run_test_by_name_1d( testName ):
    suite = unittest.TestSuite()
    suite.addTest(Test_stn_1d(testName))
    runner = unittest.TextTestRunner()
    runner.run(suite)

def run_test_by_name_2d( testName ):
    suite = unittest.TestSuite()
    suite.addTest(Test_stn_2d(testName))
    runner = unittest.TextTestRunner()
    runner.run(suite)

def run_test_by_name_3d(testName):
    suite = unittest.TestSuite()
    suite.addTest(Test_stn_3d(testName))
    runner = unittest.TextTestRunner()
    runner.run(suite)

#run_test_by_name_1d('test_expand')

if __name__ == '__main__':
    if foundHTMLTestRunner:
        unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='test_output'))
    else:
       unittest.main()
