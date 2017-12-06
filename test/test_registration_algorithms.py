# Runs various registration algorithms

# start with the setup

import os
import sys

sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../pyreg'))
sys.path.insert(0,os.path.abspath('../pyreg/libraries'))

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

# first do the torch imports
import torch
from torch.autograd import Variable
from pyreg.data_wrapper import AdaptVal
import numpy as np
import numpy.testing as npt

import pyreg.example_generation as eg
import pyreg.module_parameters as pars
import pyreg.multiscale_optimizer as MO
import pyreg.smoother_factory as SF

# test it

class Test_registration_algorithms(unittest.TestCase):

    def createImage(self,ex_len=64):

        example_img_len = ex_len
        dim = 2

        szEx = np.tile(example_img_len, dim)  # size of the desired images: (sz)^dim
        I0, I1 = eg.CreateSquares(dim).create_image_pair(szEx,
                                                         self.params)  # create a default image size with two sample squares
        sz = np.array(I0.shape)
        self.spacing = 1. / (sz[2::] - 1)  # the first two dimensions are batch size and number of image channels

        # create the source and target image as pyTorch variables
        self.ISource = AdaptVal(Variable(torch.from_numpy(I0.copy()), requires_grad=False))
        self.ITarget = AdaptVal(Variable(torch.from_numpy(I1), requires_grad=False))

        # smooth both a little bit
        self.params[('image_smoothing', {}, 'image smoothing settings')]
        self.params['image_smoothing'][('smooth_images', True, '[True|False]; smoothes the images before registration')]
        self.params['image_smoothing'][('smoother', {}, 'settings for the image smoothing')]
        self.params['image_smoothing']['smoother'][('gaussian_std', 0.05, 'how much smoothing is done')]
        self.params['image_smoothing']['smoother'][('type', 'gaussian', "['gaussianSpatial'|'gaussian'|'diffusion']")]

        cparams = self.params['image_smoothing']
        s = SF.SmootherFactory(sz[2::], self.spacing).create_smoother(cparams)
        self.ISource = s.smooth_scalar_field(self.ISource)
        self.ITarget = s.smooth_scalar_field(self.ITarget)

    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_svf_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_svf_image_single_scale_config.json')

        self.createImage( 32 )

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E=[ 2.49891615], similarityE=[ 0.78662372], regE=[ 1.71229231], relF=[ 0.00913821]

        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 2.49891615, decimal=4 )
        npt.assert_almost_equal( energy[1], 0.78662372, decimal=4 )
        npt.assert_almost_equal( energy[2], 1.71229231, decimal=4 )


    def test_lddmm_shooting_scalar_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_image_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E=[ 0.02742532], similarityE=[ 0.02321979], regE=[ 0.00420553], relF=[ 0.00610944]

        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.02742532, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.02321979, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.00420553, decimal=4 )

    def test_lddmm_shooting_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_image_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E=[ 0.01725787], similarityE=[ 0.01244521], regE=[ 0.00481266], relF=[ 0.]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.01725787, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.01244521, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.00481266, decimal=4 )

    def test_lddmm_shooting_scalar_momentum_image_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_image_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.register()

        # E=[ 0.08246735], similarityE=[ 0.07836073], regE=[ 0.00410663], relF=[ 0.01147533]

        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.08246735, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.07836073, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.00410663, decimal=4 )

    def test_lddmm_shooting_image_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_image_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.register()

        # E=[ 0.01843772], similarityE=[ 0.01369324], regE=[ 0.00474448], relF=[ 0.00353257]

        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.01843772, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.01369324, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.00474448, decimal=4 )

    def test_lddmm_shooting_scalar_momentum_map_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_map_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.register()

        # E = [0.05379514], similarityE = [0.05005502], regE = [0.00374013], relF = [0.02199404]

        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.05379514, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.05005502, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.00374013, decimal=4 )


    def test_lddmm_shooting_map_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_map_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.register()

        # E=[ 0.02130384], similarityE=[ 0.01675734], regE=[ 0.0045465], relF=[ 0.00699624]

        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.02130384, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.01675734, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.0045465, decimal=4 )

    def test_svf_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_svf_map_single_scale_config.json')

        self.createImage( 32 )

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E=[ 1.99103928], similarityE=[ 0.43856502], regE=[ 1.55247426], relF=[ 0.01893948]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 1.99103928, decimal=4 )
        npt.assert_almost_equal( energy[1], 0.43856502, decimal=4 )
        npt.assert_almost_equal( energy[2], 1.55247426, decimal=4 )

    def test_lddmm_shooting_scalar_momentum_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_map_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E=[ 0.0579784], similarityE=[ 0.05422445], regE=[ 0.00375395], relF=[ 0.01241212]

        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 0.0579784, decimal=4 )
        npt.assert_almost_equal( energy[1], 0.05422445, decimal=4 )
        npt.assert_almost_equal( energy[2], 0.00375395, decimal=4 )

    def test_lddmm_shooting_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_map_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E=[ 0.01588291], similarityE=[ 0.01131474], regE=[ 0.00456817], relF=[ 0.00034867]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 0.01588291, decimal=4 )
        npt.assert_almost_equal( energy[1], 0.01131474, decimal=4 )
        npt.assert_almost_equal( energy[2], 0.00456817, decimal=4 )

    def test_svf_scalar_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_based_config.json')

        self.params['model']['deformation']['use_map'] = False
        self.params['model']['registration_model']['type'] = 'svf_scalar_momentum_image'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E=[ 0.59089059], similarityE=[ 0.58405572], regE=[ 0.00683488], relF=[ 0.00551791]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.59089059, decimal=4)
        npt.assert_almost_equal(energy[1], 0.58405572, decimal=4)
        npt.assert_almost_equal(energy[2], 0.00683488, decimal=4)

    def test_svf_scalar_momentum_map_single_scale(self):
        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_based_config.json')

        self.params['model']['deformation']['use_map'] = True
        self.params['model']['registration_model']['type'] = 'svf_scalar_momentum_map'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E=[ 0.50193405], similarityE=[ 0.49500716], regE=[ 0.00692687], relF=[ 0.00335134]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.50193405, decimal = 4)
        npt.assert_almost_equal(energy[1], 0.49500716, decimal = 4)
        npt.assert_almost_equal(energy[2], 0.00692687, decimal = 4)

    def test_svf_vector_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_based_config.json')

        self.params['model']['deformation']['use_map'] = False
        self.params['model']['registration_model']['type'] = 'svf_vector_momentum_image'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        #  E=[ 0.01570327], similarityE=[ 0.00999644], regE=[ 0.00570683], relF=[ 0.00794008]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.01570327, decimal = 4)
        npt.assert_almost_equal(energy[1], 0.00999644, decimal = 4)
        npt.assert_almost_equal(energy[2], 0.00570683, decimal = 4)

    def test_svf_vector_momentum_map_single_scale(self):
        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_based_config.json')

        self.params['model']['deformation']['use_map'] = True
        self.params['model']['registration_model']['type'] = 'svf_vector_momentum_map'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E=[ 0.01703692], similarityE=[ 0.01153301], regE=[ 0.00550391], relF=[ 0.0029047]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.01703692, decimal = 4)
        npt.assert_almost_equal(energy[1], 0.01153301, decimal = 4)
        npt.assert_almost_equal(energy[2], 0.00550391, decimal = 4)

if __name__ == '__main__':
    if foundHTMLTestRunner:
        unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='test_output'))
    else:
       unittest.main()

