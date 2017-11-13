# start with the setup

import os
import sys

sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../pyreg'))
sys.path.insert(0,os.path.abspath('../pyreg/libraries'))

import numpy as np
import numpy.testing as npt
import torch
from torch.autograd import Variable

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

import finite_differences as FD

#TODO: add tests for non-Neumann boundary conditions (linear extrapolation)
#TODO: do experiments how the non-Neumann bounday conditions behave in practive

class Test_finite_difference_1d_neumann_numpy(unittest.TestCase):

    def setUp(self):
        self.spacing = np.array([0.1])
        self.fd_np = FD.FD_np(self.spacing)

    def tearDown(self):
        pass

    def test_xp(self):
        xp = self.fd_np.xp( np.array([[1,2,3]]) )
        npt.assert_almost_equal( xp, [[2,3,3]])
        
    def test_xm(self):
        xm = self.fd_np.xm( np.array([[1,2,3]]) )
        npt.assert_almost_equal( xm, [[1,1,2]])

    def test_dXb(self):
        dxb = self.fd_np.dXb( np.array([[1,2,3]]))
        npt.assert_almost_equal( dxb, [[0,10,10]])

    def test_dXf(self):
        dxf = self.fd_np.dXf( np.array([[1,2,3]]))
        npt.assert_almost_equal( dxf, [[10,10,0]])

    def test_dXc(self):
        dxc = self.fd_np.dXc(np.array([[1, 2, 3]]))
        npt.assert_almost_equal(dxc, [[5, 10, 5]])

    def test_ddXc(self):
        ddxc = self.fd_np.ddXc(np.array([[1, 0, 3]]))
        npt.assert_almost_equal(ddxc, [[-100, 400, -300]])

    def test_lap(self):
        lap = self.fd_np.lap(np.array([[1,0,3]]))
        npt.assert_almost_equal(lap, [[-100,400,-300]])


class Test_finite_difference_2d_neumann_numpy(unittest.TestCase):
    def setUp(self):
        self.spacing = np.array([0.1,0.2])
        self.fd_np = FD.FD_np(self.spacing)

    def tearDown(self):
        pass

    def test_xp(self):
        xp = self.fd_np.xp(np.array([[[1, 2, 3],[4,5,6],[7,8,9]]]))
        npt.assert_almost_equal(xp, [[[4, 5, 6],[7,8,9],[7,8,9]]])

    def test_xm(self):
        xm = self.fd_np.xm(np.array([[[1, 2, 3],[4,5,6],[7,8,9]]]))
        npt.assert_almost_equal(xm,[[[1,2,3],[1,2,3],[4,5,6]]])

    def test_yp(self):
        yp = self.fd_np.yp(np.array([[[1, 2, 3],[4,5,6],[7,8,9]]]))
        npt.assert_almost_equal(yp, [[[2, 3, 3],[5,6,6],[8,9,9]]])

    def test_ym(self):
        ym = self.fd_np.ym(np.array([[[1, 2, 3],[4,5,6],[7,8,9]]]))
        npt.assert_almost_equal(ym,[[[1,1,2],[4,4,5],[7,7,8]]])

    def test_dXb(self):
        dxb = self.fd_np.dXb(np.array([[[1, 2, 3],[4,5,6],[7,8,9]]]))
        npt.assert_almost_equal(dxb,[[[0, 0, 0],[30,30,30],[30,30,30]]])

    def test_dXf(self):
        dxf = self.fd_np.dXf(np.array([[[1, 2, 3],[4,5,6],[7,8,9]]]))
        npt.assert_almost_equal(dxf,[[[30, 30, 30],[30,30,30],[0,0,0]]])

    def test_dXc(self):
        dxc = self.fd_np.dXc(np.array([[[1, 2, 3],[4,5,6],[7,8,9]]]))
        npt.assert_almost_equal(dxc,[[[15,15, 15],[30,30,30],[15,15,15]]])

    def test_dYb(self):
        dyb = self.fd_np.dYb(np.array([[[1, 2, 3],[4,5,6],[7,8,9]]]))
        npt.assert_almost_equal(dyb,[[[0, 5, 5],[0,5,5],[0,5,5]]])

    def test_dYf(self):
        dyf = self.fd_np.dYf(np.array([[[1, 2, 3],[4,5,6],[7,8,9]]]))
        npt.assert_almost_equal(dyf,[[[5, 5, 0],[5,5,0],[5,5,0]]])

    def test_dYc(self):
        dyc = self.fd_np.dYc(np.array([[[1, 2, 3],[4,5,6],[7,8,9]]]))
        npt.assert_almost_equal(dyc,[[[2.5,5, 2.5],[2.5,5,2.5],[2.5,5,2.5]]])

    def test_ddXc(self):
        ddxc = self.fd_np.ddXc(np.array([[[1, 2, 3],[4,5,6],[7,8,9]]]))
        npt.assert_almost_equal(ddxc,[[[300, 300, 300],[0,0,0],[-300,-300,-300]]])

    def test_ddYc(self):
        ddyc = self.fd_np.ddYc(np.array([[[1, 2, 3],[4,5,6],[7,8,9]]]))
        npt.assert_almost_equal(ddyc,[[[25, 0, -25],[25,0,-25],[25,0,-25]]])

    def test_lap(self):
        lap = self.fd_np.lap(np.array([[[1, 2, 3],[4,5,6],[7,8,9]]]))
        npt.assert_almost_equal(lap,[[[325, 300, 275],[25,0,-25],[-275,-300,-325]]])

class Test_finite_difference_3d_neumann_numpy(unittest.TestCase):
    def setUp(self):
        self.spacing = np.array([0.1,0.2,0.5])
        self.fd_np = FD.FD_np(self.spacing)
        self.inArray = np.array([[[[ 0.,  1.,  2.],
                                      [ 3.,  4.,  5.],
                                      [ 6.,  7.,  8.]],
                                     [[ 9.,  10.,  11.],
                                      [ 12.,  13.,  14.],
                                      [ 15.,  16.,  17.]],
                                     [[ 18.,  19.,  20.],
                                      [ 21.,  22.,  23.],
                                      [ 24.,  25.,  26.]]]])

    def tearDown(self):
        pass

    def test_xp(self):
        xp = self.fd_np.xp( self.inArray )

        npt.assert_almost_equal(xp, [[[[ 9.,  10.,  11.],
                                      [ 12.,  13.,  14.],
                                      [ 15.,  16.,  17.]],
                                     [[ 18.,  19.,  20.],
                                      [ 21.,  22.,  23.],
                                      [ 24.,  25.,  26.]],
                                     [[18., 19., 20.],
                                      [21., 22., 23.],
                                      [24., 25., 26.]]
                                     ]])

    def test_xm(self):
        xm = self.fd_np.xm(self.inArray)
        npt.assert_almost_equal(xm,[[[[ 0.,  1.,  2.],
                                      [ 3.,  4.,  5.],
                                      [ 6.,  7.,  8.]],
                                    [[0., 1., 2.],
                                     [3., 4., 5.],
                                     [6., 7., 8.]],
                                     [[ 9.,  10.,  11.],
                                      [ 12.,  13.,  14.],
                                      [ 15.,  16.,  17.]]]])

    def test_yp(self):
        yp = self.fd_np.yp(self.inArray)
        npt.assert_almost_equal(yp, [[[[  3.,   4.,   5.],
                                    [  6.,   7.,   8.],
                                    [  6.,   7.,   8.]],
                                   [[ 12.,  13.,  14.],
                                    [ 15.,  16.,  17.],
                                    [ 15.,  16.,  17.]],
                                   [[ 21.,  22.,  23.],
                                    [ 24.,  25.,  26.],
                                    [ 24.,  25.,  26.]]]])

    def test_ym(self):
        ym = self.fd_np.ym(self.inArray)
        npt.assert_almost_equal(ym,[[[[  0.,   1.,   2.],
                                    [  0.,   1.,   2.],
                                     [  3.,   4.,   5.]],
                                    [[  9.,  10.,  11.],
                                     [  9.,  10.,  11.],
                                     [ 12.,  13.,  14.]],
                                    [[ 18.,  19.,  20.],
                                     [ 18.,  19.,  20.],
                                     [ 21.,  22.,  23.]]]])

    def test_zp(self):
        zp = self.fd_np.zp(self.inArray)
        npt.assert_almost_equal(zp,[[[[  1.,   2.,   2.],
                                    [  4.,   5.,   5.],
                                    [  7.,   8.,   8.]],
                                   [[ 10.,  11.,  11.],
                                    [ 13.,  14.,  14.],
                                    [ 16.,  17.,  17.]],
                                   [[ 19.,  20.,  20.],
                                    [ 22.,  23.,  23.],
                                    [ 25.,  26.,  26.]]]])

    def test_zm(self):
        zm = self.fd_np.zm(self.inArray)
        npt.assert_almost_equal(zm,[[[[  0.,   0.,   1.],
                                    [  3.,   3.,   4.],
                                    [  6.,   6.,   7.]],
                                   [[  9.,   9.,  10.],
                                    [ 12.,  12.,  13.],
                                    [ 15.,  15.,  16.]],
                                   [[ 18.,  18.,  19.],
                                    [ 21.,  21.,  22.],
                                    [ 24.,  24.,  25.]]]])

    def test_dXb(self):
        dxb = self.fd_np.dXb(self.inArray)
        npt.assert_almost_equal(dxb,[[[[  0.,   0.,   0.],
                                        [  0.,   0.,   0.],
                                        [  0.,   0.,   0.]],
                                       [[ 90.,  90.,  90.],
                                        [ 90.,  90.,  90.],
                                        [ 90.,  90.,  90.]],
                                       [[ 90.,  90.,  90.],
                                        [ 90.,  90.,  90.],
                                        [ 90.,  90.,  90.]]]])

    def test_dXf(self):
        dxf = self.fd_np.dXf(self.inArray)
        npt.assert_almost_equal(dxf,[[[[ 90.,  90.,  90.],
                                    [ 90.,  90.,  90.],
                                    [ 90.,  90.,  90.]],
                                   [[ 90.,  90.,  90.],
                                    [ 90.,  90.,  90.],
                                    [ 90.,  90.,  90.]],
                                   [[  0.,   0.,   0.],
                                    [  0.,   0.,   0.],
                                    [  0.,   0.,   0.]]]])

    def test_dXc(self):
        dxc = self.fd_np.dXc(self.inArray)
        npt.assert_almost_equal(dxc,[[[[ 45.,  45.,  45.],
                                    [ 45.,  45.,  45.],
                                    [ 45.,  45.,  45.]],
                                   [[ 90.,  90.,  90.],
                                    [ 90.,  90.,  90.],
                                    [ 90.,  90.,  90.]],
                                   [[ 45.,  45.,  45.],
                                    [ 45.,  45.,  45.],
                                    [ 45.,  45.,  45.]]]])

    def test_dYb(self):
        dyb = self.fd_np.dYb(self.inArray)
        npt.assert_almost_equal(dyb,[[[[  0.,   0.,   0.],
                                    [ 15.,  15.,  15.],
                                    [ 15.,  15.,  15.]],
                                   [[  0.,   0.,   0.],
                                    [ 15.,  15.,  15.],
                                    [ 15.,  15.,  15.]],
                                   [[  0.,   0.,   0.],
                                    [ 15.,  15.,  15.],
                                    [ 15.,  15.,  15.]]]])

    def test_dYf(self):
        dyf = self.fd_np.dYf(self.inArray)
        npt.assert_almost_equal(dyf,[[[[ 15.,  15.,  15.],
                                    [ 15.,  15.,  15.],
                                    [  0.,   0.,   0.]],
                                   [[ 15.,  15.,  15.],
                                    [ 15.,  15.,  15.],
                                    [  0.,   0.,   0.]],
                                   [[ 15.,  15.,  15.],
                                    [ 15.,  15.,  15.],
                                    [  0.,   0.,   0.]]]])

    def test_dYc(self):
        dyc = self.fd_np.dYc(self.inArray)
        npt.assert_almost_equal(dyc,[[[[  7.5,   7.5,   7.5],
                                    [ 15. ,  15. ,  15. ],
                                    [  7.5,   7.5,   7.5]],
                                   [[  7.5,   7.5,   7.5],
                                    [ 15. ,  15. ,  15. ],
                                    [  7.5,   7.5,   7.5]],
                                   [[  7.5,   7.5,   7.5],
                                    [ 15. ,  15. ,  15. ],
                                    [  7.5,   7.5,   7.5]]]])

    def test_dZb(self):
        dzb = self.fd_np.dZb(self.inArray)
        npt.assert_almost_equal(dzb,[[[[ 0.,  2.,  2.],
                                    [ 0.,  2.,  2.],
                                    [ 0.,  2.,  2.]],
                                   [[ 0.,  2.,  2.],
                                    [ 0.,  2.,  2.],
                                    [ 0.,  2.,  2.]],
                                   [[ 0.,  2.,  2.],
                                    [ 0.,  2.,  2.],
                                    [ 0.,  2.,  2.]]]])

    def test_dZf(self):
        dzf = self.fd_np.dZf(self.inArray)
        npt.assert_almost_equal(dzf,[[[[ 2.,  2.,  0.],
                                    [ 2.,  2.,  0.],
                                    [ 2.,  2.,  0.]],
                                   [[ 2.,  2.,  0.],
                                    [ 2.,  2.,  0.],
                                    [ 2.,  2.,  0.]],
                                   [[ 2.,  2.,  0.],
                                    [ 2.,  2.,  0.],
                                    [ 2.,  2.,  0.]]]])

    def test_dZc(self):
        dzc = self.fd_np.dZc(self.inArray)
        npt.assert_almost_equal(dzc,[[[[ 1.,  2.,  1.],
                                    [ 1.,  2.,  1.],
                                    [ 1.,  2.,  1.]],
                                   [[ 1.,  2.,  1.],
                                    [ 1.,  2.,  1.],
                                    [ 1.,  2.,  1.]],
                                   [[ 1.,  2.,  1.],
                                    [ 1.,  2.,  1.],
                                    [ 1.,  2.,  1.]]]])

    def test_ddXc(self):
        ddxc = self.fd_np.ddXc(self.inArray)
        npt.assert_almost_equal(ddxc,[[[[ 900.,  900.,  900.],
                                    [ 900.,  900.,  900.],
                                    [ 900.,  900.,  900.]],
                                   [[   0.,    0.,    0.],
                                    [   0.,    0.,    0.],
                                    [   0.,    0.,    0.]],
                                   [[-900., -900., -900.],
                                    [-900., -900., -900.],
                                    [-900., -900., -900.]]]])

    def test_ddYc(self):
        ddyc = self.fd_np.ddYc(self.inArray)
        npt.assert_almost_equal(ddyc,[[[[ 75.,  75.,  75.],
                                    [  0.,   0.,   0.],
                                    [-75., -75., -75.]],
                                   [[ 75.,  75.,  75.],
                                    [  0.,   0.,   0.],
                                    [-75., -75., -75.]],
                                   [[ 75.,  75.,  75.],
                                    [  0.,   0.,   0.],
                                    [-75., -75., -75.]]]])

    def test_ddZc(self):
        ddzc = self.fd_np.ddZc(self.inArray)
        npt.assert_almost_equal(ddzc,[[[[ 4.,  0., -4.],
                                    [ 4.,  0., -4.],
                                    [ 4.,  0., -4.]],
                                   [[ 4.,  0., -4.],
                                    [ 4.,  0., -4.],
                                    [ 4.,  0., -4.]],
                                   [[ 4.,  0., -4.],
                                    [ 4.,  0., -4.],
                                    [ 4.,  0., -4.]]]])

    def test_lap(self):
        lap = self.fd_np.lap(self.inArray)
        npt.assert_almost_equal(lap,[[[[ 979.,  975.,  971.],
                                    [ 904.,  900.,  896.],
                                    [ 829.,  825.,  821.]],
                                   [[  79.,   75.,   71.],
                                    [   4.,    0.,   -4.],
                                    [ -71.,  -75.,  -79.]],
                                   [[-821., -825., -829.],
                                    [-896., -900., -904.],
                                    [-971., -975., -979.]]]])


class Test_finite_difference_1d_neumann_torch(unittest.TestCase):

    def setUp(self):
        self.spacing = np.array([0.1])
        self.fd_torch = FD.FD_torch(self.spacing)

    def tearDown(self):
        pass

    def test_xp(self):
        xp = self.fd_torch.xp( Variable( torch.FloatTensor([[1,2,3]]) ) )
        npt.assert_almost_equal( xp.data.numpy(), [[2,3,3]])

    def test_xm(self):
        xm = self.fd_torch.xm( Variable(torch.FloatTensor([[1,2,3]]) ) )
        npt.assert_almost_equal( xm.data.numpy(), [[1,1,2]])

    def test_dXb(self):
        dxb = self.fd_torch.dXb( Variable( torch.FloatTensor([[1,2,3]])) )
        npt.assert_almost_equal( dxb.data.numpy(), [[0,10,10]])

    def test_dXf(self):
        dxf = self.fd_torch.dXf( Variable( torch.FloatTensor([[1,2,3]])) )
        npt.assert_almost_equal( dxf.data.numpy(), [[10,10,0]])

    def test_dXc(self):
        dxc = self.fd_torch.dXc( Variable( torch.FloatTensor([[1, 2, 3]])) )
        npt.assert_almost_equal(dxc.data.numpy(), [[5, 10, 5]])

    def test_ddXc(self):
        ddxc = self.fd_torch.ddXc( Variable( torch.FloatTensor([[1, 0, 3]])) )
        npt.assert_almost_equal(ddxc.data.numpy(), [[-100, 400, -300]])

    def test_lap(self):
        lap = self.fd_torch.lap( Variable( torch.FloatTensor([[1,0,3]])) )
        npt.assert_almost_equal(lap.data.numpy(), [[-100,400,-300]])


class Test_finite_difference_2d_neumann_torch(unittest.TestCase):
    def setUp(self):
        self.spacing = np.array([0.1,0.2])
        self.fd_torch = FD.FD_torch(self.spacing)

    def tearDown(self):
        pass

    def test_xp(self):
        xp = self.fd_torch.xp(Variable( torch.FloatTensor([[[1, 2, 3],[4,5,6],[7,8,9]]])))
        npt.assert_almost_equal(xp.data.numpy(), [[[4, 5, 6],[7,8,9],[7,8,9]]])

    def test_xm(self):
        xm = self.fd_torch.xm(Variable( torch.FloatTensor([[[1, 2, 3],[4,5,6],[7,8,9]]])))
        npt.assert_almost_equal(xm.data.numpy(),[[[1,2,3],[1,2,3],[4,5,6]]])

    def test_yp(self):
        yp = self.fd_torch.yp(Variable( torch.FloatTensor([[[1, 2, 3],[4,5,6],[7,8,9]]])))
        npt.assert_almost_equal(yp.data.numpy(), [[[2, 3, 3],[5,6,6],[8,9,9]]])

    def test_ym(self):
        ym = self.fd_torch.ym(Variable( torch.FloatTensor([[[1, 2, 3],[4,5,6],[7,8,9]]])))
        npt.assert_almost_equal(ym.data.numpy(),[[[1,1,2],[4,4,5],[7,7,8]]])

    def test_dXb(self):
        dxb = self.fd_torch.dXb(Variable( torch.FloatTensor([[[1, 2, 3],[4,5,6],[7,8,9]]])))
        npt.assert_almost_equal(dxb.data.numpy(),[[[0, 0, 0],[30,30,30],[30,30,30]]])

    def test_dXf(self):
        dxf = self.fd_torch.dXf(Variable( torch.FloatTensor([[[1, 2, 3],[4,5,6],[7,8,9]]])))
        npt.assert_almost_equal(dxf.data.numpy(),[[[30, 30, 30],[30,30,30],[0,0,0]]])

    def test_dXc(self):
        dxc = self.fd_torch.dXc(Variable( torch.FloatTensor([[[1, 2, 3],[4,5,6],[7,8,9]]])))
        npt.assert_almost_equal(dxc.data.numpy(),[[[15,15, 15],[30,30,30],[15,15,15]]])

    def test_dYb(self):
        dyb = self.fd_torch.dYb(Variable( torch.FloatTensor([[[1, 2, 3],[4,5,6],[7,8,9]]])))
        npt.assert_almost_equal(dyb.data.numpy(),[[[0, 5, 5],[0,5,5],[0,5,5]]])

    def test_dYf(self):
        dyf = self.fd_torch.dYf(Variable( torch.FloatTensor([[[1, 2, 3],[4,5,6],[7,8,9]]])))
        npt.assert_almost_equal(dyf.data.numpy(),[[[5, 5, 0],[5,5,0],[5,5,0]]])

    def test_dYc(self):
        dyc = self.fd_torch.dYc(Variable( torch.FloatTensor([[[1, 2, 3],[4,5,6],[7,8,9]]])))
        npt.assert_almost_equal(dyc.data.numpy(),[[[2.5,5, 2.5],[2.5,5,2.5],[2.5,5,2.5]]])

    def test_ddXc(self):
        ddxc = self.fd_torch.ddXc(Variable( torch.FloatTensor([[[1, 2, 3],[4,5,6],[7,8,9]]])))
        npt.assert_almost_equal(ddxc.data.numpy(),[[[300, 300, 300],[0,0,0],[-300,-300,-300]]])

    def test_ddYc(self):
        ddyc = self.fd_torch.ddYc(Variable( torch.FloatTensor([[[1, 2, 3],[4,5,6],[7,8,9]]])))
        npt.assert_almost_equal(ddyc.data.numpy(),[[[25, 0, -25],[25,0,-25],[25,0,-25]]])

    def test_lap(self):
        lap = self.fd_torch.lap(Variable( torch.FloatTensor([[[1, 2, 3],[4,5,6],[7,8,9]]])))
        npt.assert_almost_equal(lap.data.numpy(),[[[325, 300, 275],[25,0,-25],[-275,-300,-325]]])


class Test_finite_difference_3d_neumann_torch(unittest.TestCase):

    def setUp(self):
        self.spacing = np.array([0.1,0.2,0.5])
        self.fd_torch = FD.FD_torch(self.spacing)
        self.inArray = Variable( torch.FloatTensor([[[[ 0.,  1.,  2.],
                                      [ 3.,  4.,  5.],
                                      [ 6.,  7.,  8.]],
                                     [[ 9.,  10.,  11.],
                                      [ 12.,  13.,  14.],
                                      [ 15.,  16.,  17.]],
                                     [[ 18.,  19.,  20.],
                                      [ 21.,  22.,  23.],
                                      [ 24.,  25.,  26.]]]]))

    def tearDown(self):
        pass

    def test_xp(self):
        xp = self.fd_torch.xp( self.inArray )

        npt.assert_almost_equal(xp.data.numpy(), [[[[ 9.,  10.,  11.],
                                      [ 12.,  13.,  14.],
                                      [ 15.,  16.,  17.]],
                                     [[ 18.,  19.,  20.],
                                      [ 21.,  22.,  23.],
                                      [ 24.,  25.,  26.]],
                                     [[18., 19., 20.],
                                      [21., 22., 23.],
                                      [24., 25., 26.]]]
                                     ])

    def test_xm(self):
        xm = self.fd_torch.xm(self.inArray)
        npt.assert_almost_equal(xm.data.numpy(),[[[[ 0.,  1.,  2.],
                                      [ 3.,  4.,  5.],
                                      [ 6.,  7.,  8.]],
                                    [[0., 1., 2.],
                                     [3., 4., 5.],
                                     [6., 7., 8.]],
                                     [[ 9.,  10.,  11.],
                                      [ 12.,  13.,  14.],
                                      [ 15.,  16.,  17.]]]])

    def test_yp(self):
        yp = self.fd_torch.yp(self.inArray)
        npt.assert_almost_equal(yp.data.numpy(), [[[[  3.,   4.,   5.],
                                    [  6.,   7.,   8.],
                                    [  6.,   7.,   8.]],
                                   [[ 12.,  13.,  14.],
                                    [ 15.,  16.,  17.],
                                    [ 15.,  16.,  17.]],
                                   [[ 21.,  22.,  23.],
                                    [ 24.,  25.,  26.],
                                    [ 24.,  25.,  26.]]]])

    def test_ym(self):
        ym = self.fd_torch.ym(self.inArray)
        npt.assert_almost_equal(ym.data.numpy(),[[[[  0.,   1.,   2.],
                                    [  0.,   1.,   2.],
                                     [  3.,   4.,   5.]],
                                    [[  9.,  10.,  11.],
                                     [  9.,  10.,  11.],
                                     [ 12.,  13.,  14.]],
                                    [[ 18.,  19.,  20.],
                                     [ 18.,  19.,  20.],
                                     [ 21.,  22.,  23.]]]])

    def test_zp(self):
        zp = self.fd_torch.zp(self.inArray)
        npt.assert_almost_equal(zp.data.numpy(),[[[[  1.,   2.,   2.],
                                    [  4.,   5.,   5.],
                                    [  7.,   8.,   8.]],
                                   [[ 10.,  11.,  11.],
                                    [ 13.,  14.,  14.],
                                    [ 16.,  17.,  17.]],
                                   [[ 19.,  20.,  20.],
                                    [ 22.,  23.,  23.],
                                    [ 25.,  26.,  26.]]]])

    def test_zm(self):
        zm = self.fd_torch.zm(self.inArray)
        npt.assert_almost_equal(zm.data.numpy(),[[[[  0.,   0.,   1.],
                                    [  3.,   3.,   4.],
                                    [  6.,   6.,   7.]],
                                   [[  9.,   9.,  10.],
                                    [ 12.,  12.,  13.],
                                    [ 15.,  15.,  16.]],
                                   [[ 18.,  18.,  19.],
                                    [ 21.,  21.,  22.],
                                    [ 24.,  24.,  25.]]]])


    def test_dXb(self):
        dxb = self.fd_torch.dXb(self.inArray)
        npt.assert_almost_equal(dxb.data.numpy(),[[[[  0.,   0.,   0.],
                                        [  0.,   0.,   0.],
                                        [  0.,   0.,   0.]],
                                       [[ 90.,  90.,  90.],
                                        [ 90.,  90.,  90.],
                                        [ 90.,  90.,  90.]],
                                       [[ 90.,  90.,  90.],
                                        [ 90.,  90.,  90.],
                                        [ 90.,  90.,  90.]]]])

    def test_dXf(self):
        dxf = self.fd_torch.dXf(self.inArray)
        npt.assert_almost_equal(dxf.data.numpy(),[[[[ 90.,  90.,  90.],
                                    [ 90.,  90.,  90.],
                                    [ 90.,  90.,  90.]],
                                   [[ 90.,  90.,  90.],
                                    [ 90.,  90.,  90.],
                                    [ 90.,  90.,  90.]],
                                   [[  0.,   0.,   0.],
                                    [  0.,   0.,   0.],
                                    [  0.,   0.,   0.]]]])

    def test_dXc(self):
        dxc = self.fd_torch.dXc(self.inArray)
        npt.assert_almost_equal(dxc.data.numpy(),[[[[ 45.,  45.,  45.],
                                    [ 45.,  45.,  45.],
                                    [ 45.,  45.,  45.]],
                                   [[ 90.,  90.,  90.],
                                    [ 90.,  90.,  90.],
                                    [ 90.,  90.,  90.]],
                                   [[ 45.,  45.,  45.],
                                    [ 45.,  45.,  45.],
                                    [ 45.,  45.,  45.]]]])

    def test_dYb(self):
        dyb = self.fd_torch.dYb(self.inArray)
        npt.assert_almost_equal(dyb.data.numpy(),[[[[  0.,   0.,   0.],
                                    [ 15.,  15.,  15.],
                                    [ 15.,  15.,  15.]],
                                   [[  0.,   0.,   0.],
                                    [ 15.,  15.,  15.],
                                    [ 15.,  15.,  15.]],
                                   [[  0.,   0.,   0.],
                                    [ 15.,  15.,  15.],
                                    [ 15.,  15.,  15.]]]])

    def test_dYf(self):
        dyf = self.fd_torch.dYf(self.inArray)
        npt.assert_almost_equal(dyf.data.numpy(),[[[[ 15.,  15.,  15.],
                                    [ 15.,  15.,  15.],
                                    [  0.,   0.,   0.]],
                                   [[ 15.,  15.,  15.],
                                    [ 15.,  15.,  15.],
                                    [  0.,   0.,   0.]],
                                   [[ 15.,  15.,  15.],
                                    [ 15.,  15.,  15.],
                                    [  0.,   0.,   0.]]]])

    def test_dYc(self):
        dyc = self.fd_torch.dYc(self.inArray)
        npt.assert_almost_equal(dyc.data.numpy(),[[[[  7.5,   7.5,   7.5],
                                    [ 15. ,  15. ,  15. ],
                                    [  7.5,   7.5,   7.5]],
                                   [[  7.5,   7.5,   7.5],
                                    [ 15. ,  15. ,  15. ],
                                    [  7.5,   7.5,   7.5]],
                                   [[  7.5,   7.5,   7.5],
                                    [ 15. ,  15. ,  15. ],
                                    [  7.5,   7.5,   7.5]]]])

    def test_dZb(self):
        dzb = self.fd_torch.dZb(self.inArray)
        npt.assert_almost_equal(dzb.data.numpy(),[[[[ 0.,  2.,  2.],
                                    [ 0.,  2.,  2.],
                                    [ 0.,  2.,  2.]],
                                   [[ 0.,  2.,  2.],
                                    [ 0.,  2.,  2.],
                                    [ 0.,  2.,  2.]],
                                   [[ 0.,  2.,  2.],
                                    [ 0.,  2.,  2.],
                                    [ 0.,  2.,  2.]]]])

    def test_dZf(self):
        dzf = self.fd_torch.dZf(self.inArray)
        npt.assert_almost_equal(dzf.data.numpy(),[[[[ 2.,  2.,  0.],
                                    [ 2.,  2.,  0.],
                                    [ 2.,  2.,  0.]],
                                   [[ 2.,  2.,  0.],
                                    [ 2.,  2.,  0.],
                                    [ 2.,  2.,  0.]],
                                   [[ 2.,  2.,  0.],
                                    [ 2.,  2.,  0.],
                                    [ 2.,  2.,  0.]]]])

    def test_dZc(self):
        dzc = self.fd_torch.dZc(self.inArray)
        npt.assert_almost_equal(dzc.data.numpy(),[[[[ 1.,  2.,  1.],
                                    [ 1.,  2.,  1.],
                                    [ 1.,  2.,  1.]],
                                   [[ 1.,  2.,  1.],
                                    [ 1.,  2.,  1.],
                                    [ 1.,  2.,  1.]],
                                   [[ 1.,  2.,  1.],
                                    [ 1.,  2.,  1.],
                                    [ 1.,  2.,  1.]]]])

    def test_ddXc(self):
        ddxc = self.fd_torch.ddXc(self.inArray)
        npt.assert_almost_equal(ddxc.data.numpy(),[[[[ 900.,  900.,  900.],
                                    [ 900.,  900.,  900.],
                                    [ 900.,  900.,  900.]],
                                   [[   0.,    0.,    0.],
                                    [   0.,    0.,    0.],
                                    [   0.,    0.,    0.]],
                                   [[-900., -900., -900.],
                                    [-900., -900., -900.],
                                    [-900., -900., -900.]]]])

    def test_ddYc(self):
        ddyc = self.fd_torch.ddYc(self.inArray)
        npt.assert_almost_equal(ddyc.data.numpy(),[[[[ 75.,  75.,  75.],
                                    [  0.,   0.,   0.],
                                    [-75., -75., -75.]],
                                   [[ 75.,  75.,  75.],
                                    [  0.,   0.,   0.],
                                    [-75., -75., -75.]],
                                   [[ 75.,  75.,  75.],
                                    [  0.,   0.,   0.],
                                    [-75., -75., -75.]]]])

    def test_ddZc(self):
        ddzc = self.fd_torch.ddZc(self.inArray)
        npt.assert_almost_equal(ddzc.data.numpy(),[[[[ 4.,  0., -4.],
                                    [ 4.,  0., -4.],
                                    [ 4.,  0., -4.]],
                                   [[ 4.,  0., -4.],
                                    [ 4.,  0., -4.],
                                    [ 4.,  0., -4.]],
                                   [[ 4.,  0., -4.],
                                    [ 4.,  0., -4.],
                                    [ 4.,  0., -4.]]]])

    def test_lap(self):
        lap = self.fd_torch.lap(self.inArray)
        npt.assert_almost_equal(lap.data.numpy(),[[[[ 979.,  975.,  971.],
                                    [ 904.,  900.,  896.],
                                    [ 829.,  825.,  821.]],
                                   [[  79.,   75.,   71.],
                                    [   4.,    0.,   -4.],
                                    [ -71.,  -75.,  -79.]],
                                   [[-821., -825., -829.],
                                    [-896., -900., -904.],
                                    [-971., -975., -979.]]]])


if __name__ == '__main__':
    if foundHTMLTestRunner:
        unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='test_output'))
    else:
       unittest.main()
