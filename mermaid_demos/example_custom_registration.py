"""

Custom registration example
===========================

"""

############################################################
#
# This tutorial whill show how to implement a custom registration model and a custom similarity measure.
#
# .. contents::
#

#########################
# Introduction
# ^^^^^^^^^^^^
#
# Following *pytorch* conventions all images in *mermaid* have to be in the format
# BxCxXxYxZ (BxCxX in 1D and BxCxXxY in 2D) I.e., in 1D, 2D, 3D we are dealing with 3D, 4D, 5D tensors.
# B is the batchsize, and C are the channels
# (for example to support color-images or general multi-modal registration scenarios)
#

###############################3
# Importing modules
# ^^^^^^^^^^^^^^^^^
#
# We start by importing some necessary modules
#
# The used mermaid modules are as follows:
#
# - ``mermaid.example_generation`` allows to generate simply synthetic data and some real image pairs to test the registration algorithms
# - ``mermaid.module_parameters`` allows generating mermaid parameter structures which are used to keep track of all the parameters
# - ``mermaid.smoother_factory`` allows to generate various types of smoothers for images and vector fields
# - ``mermaid.data_wrappter`` allows to put tensors into the right data format (for example CPU/GPU via ``AdaptVal``; makes use of the mermaid setting files for configuration)
# - ``mermaid.multiscale_optimizer`` allows single and multi-scale optimization of all the different registration models
# - ``mermaid.load_default_settings`` is a module from which various default settings can be loaded (without the need for providing your own json configuration file; these are written from the mermaid configuration setting files, but one can overwrite them by placing modified copies in ``.mermaid_settings``).

# standard modules for str, torch, and numpy
from __future__ import print_function
from builtins import str

# first do the torch imports
import torch
import numpy as np

# the mermaid modules
import mermaid.example_generation as eg
import mermaid.module_parameters as pars
import mermaid.smoother_factory as SF
from mermaid.data_wrapper import AdaptVal
import mermaid.multiscale_optimizer as MO
import mermaid.load_default_settings as ds

####################################3
# Specifying settings
# ^^^^^^^^^^^^^^^^^^^
#
# Before we start we name our own new model, load some default parameters to configure the registration algorithm and settings from where to load data
#

# general parameters
model_name = 'mySVFNet'

# creating some initial parameters 
params = pars.ParameterDict(ds.par_algconf)

# these are the default parameters we loaded
print(params)

#######################
# Creating some data
# ^^^^^^^^^^^^^^^^^^
#
# We are now ready to create some data. Either synthetic or real, depending on what was specified via the configuration file.
#

if ds.load_settings_from_file:
    settingFile = 'test_custom_registration_' + model_name + '_settings.json'
    params.load_JSON(settingFile)

if ds.use_real_images:
    I0,I1,spacing = eg.CreateRealExampleImages(ds.dim).create_image_pair()

else:
    szEx = np.tile( 50, ds.dim )         # size of the desired images: (sz)^dim

    params['square_example_images']=({},'Settings for example image generation')
    params['square_example_images']['len_s'] = int(szEx.min()//6)
    params['square_example_images']['len_l'] = int(szEx.max()//4)

    # create a default image size with two sample squares
    I0,I1,spacing= eg.CreateSquares(ds.dim).create_image_pair(szEx,params)

sz = np.array(I0.shape)

assert( len(sz)==ds.dim+2 )

print ('Spacing = ' + str( spacing ) )

# create the source and target image as pyTorch variables
ISource = AdaptVal(torch.from_numpy( I0.copy() ))
ITarget = AdaptVal(torch.from_numpy( I1 ))

# if desired we smooth them a little bit
if ds.smooth_images:
    # smooth both a little bit
    params['image_smoothing'] = ds.par_algconf['image_smoothing']
    cparams = params['image_smoothing']
    s = SF.SmootherFactory(sz[2::], spacing).create_smoother(cparams)
    ISource = s.smooth(ISource)
    ITarget = s.smooth(ITarget)

##############################3
# Setting up the optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# We instantiate the multi-scale optimizer, which requires knowledge about image size and spacing,
# as well as if a map will be used for computation (for this example we do not use one).
#

# this custom registration algorithm does not use a map, so set it to False
use_map = False
# If a map would be used we could compute at a lower resolution internally. 
map_low_res_factor = None
# Instantiate the multi-scale optimizer
mo = MO.MultiScaleRegistrationOptimizer(sz,spacing,use_map,map_low_res_factor,params)

##########################################
# Specifiying a custom similarity measure
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We first implement a custom similarity measure. We just pretend SSD would not already exist and reimplement it.
#
# Since we specify a new similarity measure we derive it from the general similiary measure class as specified in ``mermaid.similarity_measure_factory``.
#

# we name our own model ``mySSD`` and add it to the parameter object, so that the model will know which similarity measure to use
params['registration_model']['similarity_measure']['type'] = 'mySSD'

import mermaid.similarity_measure_factory as SM

# this implements the similarity measure (I0Source and phi will most often not be used, but are useful in some special cases)
class MySSD(SM.SimilarityMeasure):
    def compute_similarity(self,I0,I1, I0Source=None, phi=None):
        print('Computing my SSD')
        return ((I0 - I1) ** 2).sum() / (0.1**2) * self.volumeElement

######################################
# Specifying a registration model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We are now ready to specify the registration model itself. Here we implment a stationary velocity field which directly advects
# the image (given this velocity field), i.e., no map is used.
#
# We make use of various mermaid modules here
#
# - ``mermaid.registration_networks``: This module contains all the existing registration methods as well as the base classes from which we will derive our own model.
# - ``mermaid.utils``: This module contains various utility functions. E.g., to apply maps, deal with registration parameter dictionaries, etc.
# - ``mermaid.image_sampling``: This module is used to resample images (for example to higher or lower resolution as needed by a multi-scale approach).
# - ``mermaid.rungekutta_integrators``: This module contains Runge-Kutta integrators to integrate differential equations/
# - ``mermaid.forward_models``: This module contains various forward models. For example also an advection model, which we use here.
# - ``mermai.regularizer_factory``: This module implements various regularizers.
#

import mermaid.registration_networks as RN
import mermaid.utils as utils
import mermaid.image_sampling as IS
import mermaid.rungekutta_integrators as RK
import mermaid.forward_models as FM
import mermaid.regularizer_factory as RF

# We define our own image-based SVF class
class MySVFNet(RN.RegistrationNetTimeIntegration):
    def __init__(self,sz,spacing,params):
        super(MySVFNet, self).__init__(sz,spacing,params)
        # we create the parameters (here simply a vector field)
        self.v = self.create_registration_parameters()
        # and an integrator to integrate the advection equation
        self.integrator = self.create_integrator()

    def create_registration_parameters(self):
        """
        Creates the vector field over which we optimize
        """
        return utils.create_ND_vector_field_parameter_multiN(self.sz[2::], self.nrOfImages)

    def get_registration_parameters(self):
        """
        Returns the vector field over which we optimize
        """
        return self.v

    def set_registration_parameters(self, p, sz, spacing):
        """
        Allows setting the registration parameters. This is neede for the multi-scale solver to go between the scales.

        :param p: Registration parameter (here the vector field)
        :param sz: size at current scale
        :param spacing: spacing at current scale
        :return: n/a
        """
        self.v.data = p.data
        self.sz = sz
        self.spacing = spacing

    def create_integrator(self):
        """
        Creates an instance of the integrator (here RK4)
        """
        # here is where parameters for the forward model are always stored
        cparams = self.params[('forward_model',{},'settings for the forward model')]
        # we create an advection forward model
        advection = FM.AdvectImage(self.sz, self.spacing)
        # create the parameters to be passed to the integrator as a dictionary (including default parameters that can be passed)
        pars_to_pass = utils.combine_dict({'v': self.v}, self._get_default_dictionary_to_pass_to_integrator())
        # now we create the integrator and return it
        return RK.RK4(advection.f, advection.u, pars_to_pass, cparams)

    def forward(self, I, variables_from_optimizer=None):
        """
        The forward method simply applies the current registration transform (here integrates the advection equation)

        :param I: input image
        :param variable_from_optimizer: additional parameters that can be passed from the optimizer
        :return: returns the warped image I
        """

        # as we derived our class from RegistrationNetTimeIntegration we have access to member variables tFrom and tTo
        # specifying the desired integration time interval
        I1 = self.integrator.solve([I], self.tFrom, self.tTo)
        return I1[0]

    def upsample_registration_parameters(self, desiredSz):
        """
        Upsamples the registration parameters (needed for multi-scale solver).

        :param desiredSz: desired size to which we want to upsample.
        :return: returns the upsampled parameter (the upsampled velocity field) and the corresponding spacing as a tuple
        """
        sampler = IS.ResampleImage()
        vUpsampled,upsampled_spacing=sampler.upsample_image_to_size(self.v,self.spacing,desiredSz,spline_order=1)
        return vUpsampled,upsampled_spacing

    def downsample_registration_parameters(self, desiredSz):
        """
        Downsamples the registration parameters (needed for multi-scale solver).

        :param desiredSz: desired size to which we want to downsample.
        :return: returns the downsampled parameter (the upsampled velocity field) and the corresponding spacing as a tuple
        """
        sampler = IS.ResampleImage()
        vDownsampled,downsampled_spacing=sampler.downsample_image_to_size(self.v,self.spacing,desiredSz,spline_order=1)
        return vDownsampled,downsampled_spacing

###################################
# Specifying the loss function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Lastly, we write out custom loss function to penalize deformations that are not smooth and image mismatch.
# As we already defined the similarity measure (which will get called behind the scenes) we only need to deal
# with implementing the regularization energy here.
#
# We derive this class from ``RegistrationImageLoss``.
#
    
class MySVFImageLoss(RN.RegistrationImageLoss):
    def __init__(self,v,sz_sim,spacing_sim,sz_model,spacing_model,params):
        super(MySVFImageLoss, self).__init__(sz_sim,spacing_sim,sz_model,spacing_model,params)
        # the registration parameters
        self.v = v
        # create a new parameter category
        cparams = params[('loss',{},'settings for the loss function')]
        # under this category the regularizer settings will be saved
        self.regularizer = (RF.RegularizerFactory(self.spacing_sim).
                            create_regularizer(cparams))

    def compute_regularization_energy(self, I0_source, variables_from_forward_model=None, variables_from_optimizer=False):
        # here we compute the regularization energy, by simply evaluating the regularizer
        return self.regularizer.compute_regularizer_multiN(self.v)

#############################################
# Multi-scale optimizing our own custom model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We are now ready to optimize over our model. The only thing left is to associate the similarity measure, the registration model, and the loss
# function with the multi-scale optimizer. Note that we only had to implement upsampling and downsampling methods for the registration parameters.
# THe entire multi-scale framework now comes for free
#
    
# We have now created our own similiary measure (``MySSD``). We know register it with the multi-scale optimizer, so it knows it exists.
mo.add_similarity_measure('mySSD', MySSD)

# We do the same with the registration model and its loss
mo.add_model(model_name,MySVFNet,MySVFImageLoss,use_map=False)
# and set the model name, so we can refer to it
mo.set_model(model_name)

# here we set some visualization options (if to visualize and how frequently)
mo.set_visualization( ds.visualize )
mo.set_visualize_step( ds.visualize_step )

# we let the optimizer know what is the source and what is the target image
mo.set_source_image(ISource)
mo.set_target_image(ITarget)

# we set the scale factors and the number of iterations
mo.set_scale_factors( ds.multi_scale_scale_factors )
mo.set_number_of_iterations_per_scale( ds.multi_scale_iterations_per_scale )

# and while we are at it we also pick a custom optimizer
mo.set_optimizer(torch.optim.Adam)
mo.set_optimizer_params(dict(lr=0.01))

# and now we do the optimization

mo.optimize()

# If desired we can write out the registration parameters. Which keeps track of all parameters that were used.
# These parameters are as follows (and we can see that out own model was used).
#

print(params)

if ds.save_settings_to_file:
    params.write_JSON( 'test_custom_registration_' + model_name + '_settings_clean.json')
    params.write_JSON_comments( 'test_custom_registration_' + model_name + '_settings_comments.json')

###########################################3
# Conclusion
# ^^^^^^^^^^
#
# Given this example it is possible to set up a custom registration model without too much overhead and without being exposed to the complex *mermaid* internals.
#
