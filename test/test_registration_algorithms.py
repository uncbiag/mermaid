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

        # E=[ 1.80229616], similarityE=[ 0.71648604], regE=[ 1.08581007], relF=[ 0.0083105]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 1.80229616, decimal=2 )
        npt.assert_almost_equal( energy[1], 0.71648604, decimal=2 )
        npt.assert_almost_equal( energy[2], 1.08581007, decimal=2 )


    def test_lddmm_shooting_scalar_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_image_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
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

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
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

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
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

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
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

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.set_light_analysis_on(True)
        mo.register()

        # E = [0.04164727], similarityE = [0.03095263], regE = [0.01069464], relF = [0.00020597]
        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.04164727, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.03095263, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.01069464, decimal=4 )


    def test_lddmm_shooting_map_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_map_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.set_light_analysis_on(True)
        mo.register()

        # E=[ 0.07468626], similarityE=[ 0.06153471], regE=[ 0.01315155], relF=[ 0.00284894]
        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.07468626, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.06153471, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.01315155, decimal=4 )

    def test_svf_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_svf_map_single_scale_config.json')

        self.createImage( 32 )

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E=[ 12.94093132], similarityE=[ 6.507792], regE=[ 6.4331398], relF=[ 0.24289775]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 12.94093132, decimal=4 )
        npt.assert_almost_equal( energy[1], 6.507792, decimal=4 )
        npt.assert_almost_equal( energy[2], 6.4331398, decimal=4 )

    def test_lddmm_shooting_scalar_momentum_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_map_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E=[ 0.0418122], similarityE=[ 0.03110093], regE=[ 0.01071127], relF=[ 0.0003651]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 0.0418122, decimal=4 )
        npt.assert_almost_equal( energy[1], 0.03110093, decimal=4 )
        npt.assert_almost_equal( energy[2], 0.01071127, decimal=4 )

    def test_lddmm_shooting_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_map_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.params)
        so.get_optimizer().set_visualization(False)
        so.set_light_analysis_on(True)
        so.register()

        # E = [0.04017303], similarityE = [0.02881156], regE = [0.01136147], relF = [0.004799]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 0.04017303, decimal=4 )
        npt.assert_almost_equal( energy[1], 0.02881156, decimal=4 )
        npt.assert_almost_equal( energy[2], 0.01136147, decimal=4 )

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

        # E=[ 0.11110511], similarityE=[ 0.09144293], regE=[ 0.01966219], relF=[ 0.00052108]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.11110511, decimal=4)
        npt.assert_almost_equal(energy[1], 0.09144293, decimal=4)
        npt.assert_almost_equal(energy[2], 0.01966219, decimal=4)

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

        # E=[ 0.12077859], similarityE=[ 0.10117345], regE=[ 0.01960514], relF=[ 0.05502199]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.12077859, decimal = 4)
        npt.assert_almost_equal(energy[1], 0.10117345, decimal = 4)
        npt.assert_almost_equal(energy[2], 0.01960514, decimal = 4)

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

        # E=[ 0.03664086], similarityE=[ 0.01524722], regE=[ 0.02139364], relF=[ 0.00321996]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.03664086, decimal = 4)
        npt.assert_almost_equal(energy[1], 0.01524722, decimal = 4)
        npt.assert_almost_equal(energy[2], 0.02139364, decimal = 4)

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

        # E=[ 0.05018119], similarityE=[ 0.03107212], regE=[ 0.01910907], relF=[ 0.01229894]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.05018119, decimal = 4)
        npt.assert_almost_equal(energy[1], 0.03107212, decimal = 4)
        npt.assert_almost_equal(energy[2], 0.01910907, decimal = 4)


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

