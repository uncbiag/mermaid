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
        self.sz = np.array(I0.shape)

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

        # E=[ 1.80229616], similarityE=[ 0.71648604], regE=[ 1.08581007], relF=[ 0.0083105]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 1.80229616, decimal=2 )
        npt.assert_almost_equal( energy[1], 0.71648604, decimal=2 )
        npt.assert_almost_equal( energy[2], 1.08581007, decimal=2 )


    def test_lddmm_shooting_scalar_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_image_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E = [0.03198373], similarityE = [0.0210261], regE = [0.01095762], relF = [0.]
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

        # E=[ 0.02896098], similarityE=[ 0.0170299], regE=[ 0.01193108], relF=[ 0.00193194]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.02896098, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.0170299, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.01193108, decimal=4 )

    def test_lddmm_shooting_scalar_momentum_image_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_image_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.set_light_analysis_on(True)
        mo.register()

        # E=[ 0.03197587], similarityE=[ 0.02087387], regE=[ 0.01110199], relF=[ 0.00138645]
        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.03197587, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.02087387, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.01110199, decimal=4 )

    def test_lddmm_shooting_image_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_image_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.set_light_analysis_on(True)
        mo.register()

        # E = [0.04338037], similarityE = [0.03070126], regE = [0.01267911], relF = [0.01936091]
        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.04338037, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.03070126, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.01267911, decimal=4 )

    def test_lddmm_shooting_scalar_momentum_map_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_map_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.set_light_analysis_on(True)
        mo.register()

        # E = [0.08930502], simE = [0.08034889], regE = [0.00895613], optParE = [0.], relF = [0.03883468]
        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.08930502, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.08034889, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.00895613, decimal=4 )


    def test_lddmm_shooting_map_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_map_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.set_light_analysis_on(True)
        mo.register()

        # E = [0.07970674], simE = [0.06657108], regE = [0.01313565], optParE = [0.], relF = [0.02088663]
        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.07970674, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.06657108, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.01313565, decimal=4 )

    def test_svf_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_svf_map_single_scale_config.json')

        self.createImage( 32 )

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E = [36.42594528], similarityE = [16.22630882], regE = [20.19963646], relF = [0.0422723]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 36.42594528, decimal=4 )
        npt.assert_almost_equal( energy[1], 16.22630882, decimal=4 )
        npt.assert_almost_equal( energy[2], 20.19963646, decimal=4 )

    def test_lddmm_shooting_scalar_momentum_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_map_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E=[ 0.04196917], similarityE=[ 0.03112457], regE=[ 0.0108446], relF=[  5.37358646e-05]
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

        # E = [0.05674197], similarityE = [0.04364978], regE = [0.01309219], relF = [0.01391943]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 0.05674197, decimal=4 )
        npt.assert_almost_equal( energy[1], 0.04364978, decimal=4 )
        npt.assert_almost_equal( energy[2], 0.01309219, decimal=4 )

    def test_svf_scalar_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_based_config.json')

        self.params['model']['deformation']['use_map'] = False
        self.params['model']['registration_model']['type'] = 'svf_scalar_momentum_image'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E = [0.12390037], similarityE = [0.11124722], regE = [0.01265315], relF = [0.01015974]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.12390037, decimal=4)
        npt.assert_almost_equal(energy[1], 0.11124722, decimal=4)
        npt.assert_almost_equal(energy[2], 0.01265315, decimal=4)

    def test_svf_scalar_momentum_map_single_scale(self):
        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_based_config.json')

        self.params['model']['deformation']['use_map'] = True
        self.params['model']['registration_model']['type'] = 'svf_scalar_momentum_map'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E = [0.16383654], similarityE = [0.15004958], regE = [0.01378695], relF = [0.00207442]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.16383654, decimal = 4)
        npt.assert_almost_equal(energy[1], 0.15004958, decimal = 4)
        npt.assert_almost_equal(energy[2], 0.01378695, decimal = 4)

    def test_svf_vector_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_based_config.json')

        self.params['model']['deformation']['use_map'] = False
        self.params['model']['registration_model']['type'] = 'svf_vector_momentum_image'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E=[ 0.02504558], similarityE=[ 0.01045385], regE=[ 0.01459173], relF=[ 0.00203472]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.02504558, decimal = 4)
        npt.assert_almost_equal(energy[1], 0.01045385, decimal = 4)
        npt.assert_almost_equal(energy[2], 0.01459173, decimal = 4)

    def test_svf_vector_momentum_map_single_scale(self):
        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_based_config.json')

        self.params['model']['deformation']['use_map'] = True
        self.params['model']['registration_model']['type'] = 'svf_vector_momentum_map'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E = [0.03567605], similarityE = [0.02147852], regE = [0.01419752], relF = [0.0002105]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.03567605, decimal = 4)
        npt.assert_almost_equal(energy[1], 0.02147852, decimal = 4)
        npt.assert_almost_equal(energy[2], 0.01419752, decimal = 4)


def run_test_by_name( testName ):
    suite = unittest.TestSuite()
    suite.addTest(Test_registration_algorithms(testName))
    runner = unittest.TextTestRunner()
    runner.run(suite)

#run_test_by_name('test_svf_vector_momentum_image_single_scale')


if __name__ == '__main__':
    if foundHTMLTestRunner:
        unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='test_output'))
    else:
       unittest.main()

