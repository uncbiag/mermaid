# Runs various registration algorithms

# start with the setup
import importlib.util
import os
import sys

sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../mermaid'))
sys.path.insert(0,os.path.abspath('../mermaid/libraries'))

import unittest
import torch
from mermaid.data_wrapper import AdaptVal
import numpy as np
import numpy.testing as npt
import random
import mermaid.example_generation as eg
import mermaid.module_parameters as pars
import mermaid.multiscale_optimizer as MO
import mermaid.smoother_factory as SF


try:
    importlib.util.find_spec('HtmlTestRunner')
    foundHTMLTestRunner = True
    import HtmlTestRunner
except ImportError:
    foundHTMLTestRunner = False

# test it


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
        torch.manual_seed(2019)
        torch.cuda.manual_seed(2019)
        np.random.seed(2019)
        random.seed(2019)

    def tearDown(self):
        pass

    def test_svf_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_svf_image_single_scale_config.json')

        self.createImage( 32 )

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E=[ 1.80229616], similarityE=[ 0.71648604], regE=[ 1.08581007], relF=[ 0.0083105]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 1.7919, decimal=1 )
        npt.assert_almost_equal( energy[1], 0.5309, decimal=1 )
        npt.assert_almost_equal( energy[2], 1.2610, decimal=1 )

    def test_lddmm_shooting_scalar_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_image_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E = [0.03198373], similarityE = [0.0210261], regE = [0.01095762], relF = [0.]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.0319, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.0210, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.0110, decimal=4 )

    def test_lddmm_shooting_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_image_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E=[ 0.02896098], similarityE=[ 0.0170299], regE=[ 0.01193108], relF=[ 0.00193194]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.0308, decimal=2 )
        npt.assert_almost_equal(energy[1], 0.0187, decimal=2 )
        npt.assert_almost_equal(energy[2], 0.0121, decimal=2 )

    def test_lddmm_shooting_scalar_momentum_image_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_image_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.register()

        # E=[ 0.03197587], similarityE=[ 0.02087387], regE=[ 0.01110199], relF=[ 0.00138645]
        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.0318, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.0207, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.0111, decimal=4 )

    def test_lddmm_shooting_image_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_image_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.register()

        # E = [0.04338037], similarityE = [0.03070126], regE = [0.01267911], relF = [0.01936091]
        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.0432, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.0306, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.0127, decimal=4 )

    def test_lddmm_shooting_scalar_momentum_map_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_map_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.register()

        # E = [0.08930502], simE = [0.08034889], regE = [0.00895613], optParE = [0.], relF = [0.03883468]
        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.0434, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.0324, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.0110, decimal=4 )


    def test_lddmm_shooting_map_multi_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_map_multi_scale_config.json')

        self.createImage()

        mo = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        mo.get_optimizer().set_visualization(False)
        mo.register()

        # E = [0.07970674], simE = [0.06657108], regE = [0.01313565], optParE = [0.], relF = [0.02088663]
        energy = mo.get_energy()

        npt.assert_almost_equal(energy[0], 0.0721, decimal=4 )
        npt.assert_almost_equal(energy[1], 0.0580, decimal=4 )
        npt.assert_almost_equal(energy[2], 0.0141, decimal=4 )

    def test_svf_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_svf_map_single_scale_config.json')

        self.createImage( 32 )

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E = [36.42594528], similarityE = [16.22630882], regE = [20.19963646], relF = [0.0422723]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 16.9574, decimal=0 )
        npt.assert_almost_equal( energy[1], 6.7187, decimal=0 )
        npt.assert_almost_equal( energy[2], 10.2387, decimal=0 )

    def test_lddmm_shooting_scalar_momentum_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_scalar_momentum_map_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E=[ 0.04196917], similarityE=[ 0.03112457], regE=[ 0.0108446], relF=[  5.37358646e-05]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 0.0419, decimal=4 )
        npt.assert_almost_equal( energy[1], 0.0311, decimal=4 )
        npt.assert_almost_equal( energy[2], 0.0108, decimal=4 )

    def test_lddmm_shooting_map_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/test_lddmm_shooting_map_single_scale_config.json')

        self.createImage()

        so = MO.SimpleSingleScaleRegistration( self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E = [0.05674197], similarityE = [0.04364978], regE = [0.01309219], relF = [0.01391943]
        energy = so.get_energy()

        npt.assert_almost_equal( energy[0], 0.0549, decimal=3)
        npt.assert_almost_equal( energy[1], 0.0415, decimal=3)
        npt.assert_almost_equal( energy[2], 0.0133, decimal=3)

    def test_svf_scalar_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_base_config.json')

        self.params['model']['deformation']['use_map'] = False
        self.params['model']['registration_model']['type'] = 'svf_scalar_momentum_image'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()
        # E=[0.12413108], simE=[0.11151054], regE=0.012620546855032444
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.1242, decimal=4)
        npt.assert_almost_equal(energy[1], 0.1116, decimal=4)
        npt.assert_almost_equal(energy[2], 0.0126, decimal=4)

    def test_svf_scalar_momentum_map_single_scale(self):
        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_base_config.json')

        self.params['model']['deformation']['use_map'] = True
        self.params['model']['registration_model']['type'] = 'svf_scalar_momentum_map'

        self.createImage()
        self.setUp()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E=[0.16388921], simE=[0.15010326], regE=0.013785961084067822
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.1618, decimal = 4)
        npt.assert_almost_equal(energy[1], 0.1481, decimal = 4)
        npt.assert_almost_equal(energy[2], 0.0137, decimal = 4)

    def test_svf_vector_momentum_image_single_scale(self):

        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_base_config.json')

        self.params['model']['deformation']['use_map'] = False
        self.params['model']['registration_model']['type'] = 'svf_vector_momentum_image'

        self.createImage()
        self.setUp()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E=[ 0.02504558], similarityE=[ 0.01045385], regE=[ 0.01459173], relF=[ 0.00203472]
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.0264, decimal = 2)
        npt.assert_almost_equal(energy[1], 0.0117, decimal = 2)
        npt.assert_almost_equal(energy[2], 0.0147, decimal = 2)

    def test_svf_vector_momentum_map_single_scale(self):
        self.params = pars.ParameterDict()
        self.params.load_JSON('./json/svf_momentum_base_config.json')

        self.params['model']['deformation']['use_map'] = True
        self.params['model']['registration_model']['type'] = 'svf_vector_momentum_map'

        self.createImage()

        so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
        so.get_optimizer().set_visualization(False)
        so.register()

        # E=[0.03567663], simE=[0.02147915], regE=0.01419747807085514
        energy = so.get_energy()

        npt.assert_almost_equal(energy[0], 0.0371, decimal = 4)
        npt.assert_almost_equal(energy[1], 0.0230, decimal = 4)
        npt.assert_almost_equal(energy[2], 0.0140, decimal = 4)


    # def test_rddmm_shooting_map_single_scale(self):
    #     self.params = pars.ParameterDict()
    #     self.params.load_JSON('./json/test_rddmm_shooting_map_single_scale_config.json')
    #
    #     self.params['model']['deformation']['use_map'] = True
    #     self.params['model']['registration_model']['type'] = 'lddmm_adapt_smoother_map'
    #
    #     self.setUp()
    #     self.createImage()
    #
    #     so = MO.SimpleSingleScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
    #     so.get_optimizer().set_visualization(False)
    #     so.register()
    #
    #     # E=[0.03567663], simE=[0.02147915], regE=0.01419747807085514
    #     energy = so.get_energy()
    #
    #     npt.assert_almost_equal(energy[0], 0.01054801, decimal = 4)
    #     npt.assert_almost_equal(energy[1], 0.00114554, decimal = 4)
    #     npt.assert_almost_equal(energy[2], 0.00946710, decimal = 4)
    #
    # def test_rddmm_shooting_map_multi_scale(self):
    #     self.params = pars.ParameterDict()
    #     self.params.load_JSON('./json/test_rddmm_shooting_map_multi_scale_config.json')
    #
    #     self.params['model']['deformation']['use_map'] = True
    #     self.params['model']['registration_model']['type'] = 'lddmm_adapt_smoother_map'
    #
    #     self.setUp()
    #     self.createImage()
    #
    #     so = MO.SimpleMultiScaleRegistration(self.ISource, self.ITarget, self.spacing, self.sz, self.params)
    #     so.get_optimizer().set_visualization(False)
    #     so.register()
    #
    #     energy = so.get_energy()
    #
    #     npt.assert_almost_equal(energy[0], 0.01049348, decimal=4)
    #     npt.assert_almost_equal(energy[1], 0.00187106, decimal=4)
    #     npt.assert_almost_equal(energy[2], 0.00871814, decimal=4)


def run_test_by_name(test_name):
    suite = unittest.TestSuite()
    suite.addTest(Test_registration_algorithms(test_name))
    runner = unittest.TextTestRunner()
    runner.run(suite)

#run_test_by_name('test_svf_vector_momentum_map_single_scale')
# run_test_by_name('test_rddmm_shooting_map_single_scale')
# run_test_by_name('test_rddmm_shooting_map_multi_scale')
# run_test_by_name('test_svf_scalar_momentum_map_single_scale')
#run_test_by_name('test_svf_vector_momentum_image_single_scale')
#run_test_by_name('test_lddmm_shooting_map_single_scale')
# run_test_by_name('test_lddmm_shooting_image_single_scale')

if __name__ == '__main__':
    if foundHTMLTestRunner:
        unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(output='test_output'))
    else:
       unittest.main()

