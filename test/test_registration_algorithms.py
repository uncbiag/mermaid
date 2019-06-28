# Runs various registration algorithms

# start with the setup

import importlib.util
import os
import sys
import unittest

import torch
from mermaid.data_wrapper import AdaptVal
import numpy as np
import numpy.testing as npt

import mermaid.example_generation as eg
import mermaid.module_parameters as pars
import mermaid.multiscale_optimizer as MO
import mermaid.smoother_factory as SF

# ensure deterministic behavior
torch.manual_seed(0)
np.random.seed(1)

sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../mermaid'))
sys.path.insert(0,os.path.abspath('../mermaid/libraries'))

try:
    importlib.util.find_spec('HtmlTestRunner')
    foundHTMLTestRunner = True
    import HtmlTestRunner
except ImportError:
    foundHTMLTestRunner = False


class Test_registration_algorithms(unittest.TestCase):

    def createImage(self,ex_len=64):

        example_img_len = ex_len
        dim = 2

        szEx = np.tile(example_img_len, dim)  # size of the desired images: (sz)^dim
        I0, I1, self.spacing = eg.CreateSquares(dim).create_image_pair(szEx,self.params)  # create a default image size with two sample squares
        self.sz = np.array(I0.shape)

        # create the source and target image as pyTorch variables
        self.ISource = AdaptVal(torch.from_numpy(I0.copy()))
        self.ITarget = AdaptVal(torch.from_numpy(I1))

        # smooth both a little bit
        self.params[('image_smoothing', {}, 'image smoothing settings')]
        self.params['image_smoothing'][('smooth_images', True, '[True|False]; smoothes the images before registration')]
        self.params['image_smoothing'][('smoother', {}, 'settings for the image smoothing')]
        self.params['image_smoothing']['smoother'][('gaussian_std', 0.05, 'how much smoothing is done')]
        self.params['image_smoothing']['smoother'][('type', 'gaussian', "['gaussianSpatial'|'gaussian'|'diffusion']")]

        cparams = self.params['image_smoothing']
        s = SF.SmootherFactory(self.sz[2::], self.spacing).create_smoother(cparams)
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

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 1.81877946, decimal=2 )
        npt.assert_almost_equal( energy[1], 0.59742701, decimal=2 )
        npt.assert_almost_equal( energy[2], 1.23347449, decimal=2 )

    def test_lddmm_shooting_scalar_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_image_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.03198373, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.0210261, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.01095762, decimal=4 )

    def test_lddmm_shooting_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_image_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.02896098, decimal=2 )
        npt.assert_almost_equal(energy[1], 0.0170299, decimal=2 )
        npt.assert_almost_equal(energy[2], 0.01193108, decimal=2 )

    def test_lddmm_shooting_scalar_momentum_image_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_image_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.set_light_analysis_on(True)
        mo.register()

        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.03197587, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.02071979, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.01110199, decimal=4 )

    def test_lddmm_shooting_image_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_image_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.set_light_analysis_on(True)
        mo.register()

        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.04366652, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.03095276, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.01267911, decimal=4 )

    def test_lddmm_shooting_scalar_momentum_map_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_map_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.set_light_analysis_on(True)
        mo.register()

        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.04333755, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.03237363, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.010963925, decimal=4 )

    def test_lddmm_shooting_map_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_map_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.set_light_analysis_on(True)
        mo.register()

        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.07099805, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.05634699, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.01465014, decimal=4 )

    def test_svf_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_svf_map_single_scale_config.json')

        self.createImage(32)

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 16.89888, decimal=4
        npt.assert_almost_equal( energy[1], 5.101526, decimal=4 )
        npt.assert_almost_equal( energy[2], 11.79735, decimal=4 )

    def test_lddmm_shooting_scalar_momentum_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_map_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 0.04196917, decimal=4 )
        npt.assert_almost_equal( energy[1], 0.03112457, decimal=4 )
        npt.assert_almost_equal( energy[2], 0.0108446, decimal=4 )

    def test_lddmm_shooting_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_map_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.06821306, decimal=3)
        npt.assert_almost_equal(energy[1], 0.05349965, decimal=3)
        npt.assert_almost_equal(energy[2], 0.01346329, decimal=3)

    def test_svf_scalar_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_base_config.json')

        self.params['model']['deformation']['use_map'] = False
        self.params['model']['registration_model']['type'] = 'svf_scalar_momentum_image'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.12413108, decimal=4)
        npt.assert_almost_equal(energy[1], 0.11151054, decimal=4)
        npt.assert_almost_equal(energy[2], 0.012620546855032444, decimal=4)

    def test_svf_scalar_momentum_map_single_scale(self):
        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_base_config.json')

        self.params['model']['deformation']['use_map'] = True
        self.params['model']['registration_model']['type'] = 'svf_scalar_momentum_map'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.16180025, decimal = 4)
        npt.assert_almost_equal(energy[1], 0.14811447, decimal = 4)
        npt.assert_almost_equal(energy[2], 0.013685783, decimal = 4)

    def test_svf_vector_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_base_config.json')

        self.params['model']['deformation']['use_map'] = False
        self.params['model']['registration_model']['type'] = 'svf_vector_momentum_image'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.02504558, decimal = 2)
        npt.assert_almost_equal(energy[1], 0.01045385, decimal = 2)
        npt.assert_almost_equal(energy[2], 0.01459173, decimal = 2)

    def test_svf_vector_momentum_map_single_scale(self):
        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_base_config.json')

        self.params['model']['deformation']['use_map'] = True
        self.params['model']['registration_model']['type'] = 'svf_vector_momentum_map'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.03706806, decimal=4)
        npt.assert_almost_equal(energy[1], 0.02302469, decimal=4)
        npt.assert_almost_equal(energy[2], 0.01404336, decimal=4)


def run_test_by_name(test_name):
    suite = unittest.TestSuite()
    suite.addTest(Test_registration_algorithms(test_name))
    runner = unittest.TextTestRunner()
    runner.run(suite)

# run_test_by_name('test_svf_image_single_scale')
# run_test_by_name('test_svf_map_single_scale')
# run_test_by_name('test_lddmm_shooting_map_single_scale')
# run_test_by_name('test_lddmm_shooting_image_single_scale')

if __name__ == '__main__':
    if foundHTMLTestRunner:
        unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='test_output'))
    else:
       unittest.main()

