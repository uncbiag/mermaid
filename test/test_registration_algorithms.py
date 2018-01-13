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
        I0, I1, self.spacing = eg.CreateSquares(dim).create_image_pair(szEx,self.params)  # create a default image size with two sample squares
        sz = np.array(I0.shape)

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
        self.ISource = s.smooth(self.ISource)
        self.ITarget = s.smooth(self.ITarget)

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
        so.set_light_analysis_on(True)
        so.register()

        # E=[ 2.3217175], similarityE=[ 0.73642945], regE=[ 1.58528805], relF=[ 0.02184414]

        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 2.3217175, decimal=2 )
        npt.assert_almost_equal( energy[1], 0.73642945, decimal=2 )
        npt.assert_almost_equal( energy[2], 1.58528805, decimal=2 )


    def test_lddmm_shooting_scalar_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_image_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E = [0.10526231], similarityE = [0.10136371], regE = [0.0038986], relF = [0.00578881]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.10526231, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.10136371, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.0038986, decimal=4 )

    def test_lddmm_shooting_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_image_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E = [0.02123317], similarityE = [0.01606764], regE = [0.00516553], relF = [0.00412761]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.02123317, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.01606764, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.00516553, decimal=4 )

    def test_lddmm_shooting_scalar_momentum_image_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_image_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.set_light_analysis_on(True)
        mo.register()

        # E = [0.11112733], similarityE = [0.10722754], regE = [0.00389979], relF = [0.01953959]
        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.11112733, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.10722754, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.00389979, decimal=4 )

    def test_lddmm_shooting_image_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_image_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.set_light_analysis_on(True)
        mo.register()

        # E = [0.02945824], similarityE = [0.02432526], regE = [0.00513299], relF = [0.00153344]
        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.02945824, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.02432526, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.00513299, decimal=4 )

    def test_lddmm_shooting_scalar_momentum_map_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_map_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.set_light_analysis_on(True)
        mo.register()

        # E = [0.31037834], similarityE = [0.30713284], regE = [0.0032455], relF = [0.02254277]
        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.31037834, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.30713284, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.0032455, decimal=4 )


    def test_lddmm_shooting_map_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_map_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.set_light_analysis_on(True)
        mo.register()

        # E = [0.02877549], similarityE = [0.02381846], regE = [0.00495704], relF = [0.00946863]
        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.02877549, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.02381846, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.00495704, decimal=4 )

    def test_svf_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_svf_map_single_scale_config.json')

        self.createImage( 32 )

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E = [18.60135269], similarityE = [10.62093163], regE = [7.98042107], relF = [0.2880429]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 18.60135269, decimal=4 )
        npt.assert_almost_equal( energy[1], 10.62093163, decimal=4 )
        npt.assert_almost_equal( energy[2], 7.98042107, decimal=4 )

    def test_lddmm_shooting_scalar_momentum_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_map_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E = [0.27815577], similarityE = [0.27455819], regE = [0.00359759], relF = [0.00780544]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 0.27815577, decimal=4 )
        npt.assert_almost_equal( energy[1], 0.27455819, decimal=4 )
        npt.assert_almost_equal( energy[2], 0.00359759, decimal=4 )

    def test_lddmm_shooting_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_map_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E = [0.03226972], similarityE = [0.02630465], regE = [0.00596507], relF = [0.01728182]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 0.03226972, decimal=4 )
        npt.assert_almost_equal( energy[1], 0.02630465, decimal=4 )
        npt.assert_almost_equal( energy[2], 0.00596507, decimal=4 )

    def test_svf_scalar_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_based_config.json')

        self.params['model']['deformation']['use_map'] = False
        self.params['model']['registration_model']['type'] = 'svf_scalar_momentum_image'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E=[ 0.39766905], similarityE=[ 0.38561717], regE=[ 0.01205187], relF=[ 0.0440598]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.39766905, decimal=4)
        npt.assert_almost_equal(energy[1], 0.38561717, decimal=4)
        npt.assert_almost_equal(energy[2], 0.01205187, decimal=4)

    def test_svf_scalar_momentum_map_single_scale(self):
        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_based_config.json')

        self.params['model']['deformation']['use_map'] = True
        self.params['model']['registration_model']['type'] = 'svf_scalar_momentum_map'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E=[ 0.44303086], similarityE=[ 0.42885712], regE=[ 0.01417374], relF=[ 0.01893919]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.44303086, decimal = 4)
        npt.assert_almost_equal(energy[1], 0.42885712, decimal = 4)
        npt.assert_almost_equal(energy[2], 0.01417374, decimal = 4)

    def test_svf_vector_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_based_config.json')

        self.params['model']['deformation']['use_map'] = False
        self.params['model']['registration_model']['type'] = 'svf_vector_momentum_image'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E = [0.04093754], similarityE = [0.03253355], regE = [0.00840399], relF = [0.00446913]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.04093754, decimal = 4)
        npt.assert_almost_equal(energy[1], 0.03253355, decimal = 4)
        npt.assert_almost_equal(energy[2], 0.00840399, decimal = 4)

    def test_svf_vector_momentum_map_single_scale(self):
        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_based_config.json')

        self.params['model']['deformation']['use_map'] = True
        self.params['model']['registration_model']['type'] = 'svf_vector_momentum_map'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E = [0.11191304], similarityE = [0.10329356], regE = [0.00861948], relF = [0.0246245]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.11191304, decimal = 4)
        npt.assert_almost_equal(energy[1], 0.10329356, decimal = 4)
        npt.assert_almost_equal(energy[2], 0.00861948, decimal = 4)


def run_test_by_name( testName ):
    suite = unittest.TestSuite()
    suite.addTest(Test_registration_algorithms(testName))
    runner = unittest.TextTestRunner()
    runner.run(suite)

#run_test_by_name('test_svf_scalar_momentum_image_single_scale')


if __name__ == '__main__':
    if foundHTMLTestRunner:
        unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='test_output'))
    else:
       unittest.main()

