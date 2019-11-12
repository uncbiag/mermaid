"""

Minimal registration example
============================

"""

######################################################
#
# This tutorial will walk you through a very minimalistic example for image registration via mermaid.
#
# .. contents::
#

##########################
# Introduction
# ^^^^^^^^^^^^
#
# There are essentially three different ways of how to do registration with pytorch.
# Below are these options in increasing level of difficulty, but in order of increasing flexibility.
#
# 1. Use the simple registration interface (see ``example_simple_interface.py`` in the demos directory.
# 2. Manually instantiating the mermaid optimizer and setting up the registration parameters (which we are going to do here).
# 3. Instantiating the mermaid registration models and writing your own optimizer for them.
#

##########################
# Importing modules
# ^^^^^^^^^^^^^^^^^
#
# We start by importing some popular modules. The used mermaid modules are:
#
# - ``mermaid.multiscale_optimizer`` provides single scale and multi-scale optimizing functionality for all the registration models.
# - ``mermaid.example_generation`` allows to generate simply synthetic data and some real image pairs to test the registration algorithms
# - ``mermaid.module_parameters`` allows generating mermaid parameter structures which are used to keep track of all the parameters
#

import torch
from mermaid.data_wrapper import AdaptVal
import numpy as np

import mermaid.example_generation as eg
import mermaid.module_parameters as pars
import mermaid.multiscale_optimizer as MO

############################
# Specifying registration model properties
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Next we specify some settings which will need to be specified when setting up the registration model.
# These are the meanings of the following variables:
#
# - ``use_map``: Most image registration models in mermaid (and all the ones that are based on integration through velocity fields) support solutions by directly solving the associated equations on the images or alternatively via evolution of the transformation map (if ``use_map=True``). In general, a map-based solution is preferred as it avoids numerical discretization issues when, for example, directly advecting an image and because it directly results in the desired transformation map. Note that within mermaid a map phi denotes a map in target space (i.e., to map from source to target space) and its inverse map is a map defined in the coordinates of the source image (to map from the target space to the source space).
# - ``model_name``: Simply the name of the desired registration model. For example, ``lddmm_shooting`` is an LDDMM implementation which optimizes over the initial vector momentum and operates either on maps or directly on the image (depening on ``use_map``).
# - ``map_low_res_factor``: Especially when computing in 3D memory sometimes becomes an issue. Hence, for map-based solutions mermaid supports computing solutions of the evolution equations at a lower resolution. ``map_low_res_factor`` specifies how low this resoultion should be. ``None`` or ``1.0`` is the original resolution. ``0.5``, for example, uses half the resolution in each direction. In any case, the similarity measure is always evaluated at the original resolution via upsampling of the map.

use_map = True
model_name = 'lddmm_shooting'
map_low_res_factor = 0.5

if use_map:
    model_name = model_name + '_map'
else:
    model_name = model_name + '_image'


############################
# Specifying the optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# *mermaid* support mostly ``sgd`` or ``lbfgs_ls`` (an LBFGS method with linesearch). Other optimizers can be used, but this is not well supported at the moment. As each of these optimizers iterates the number of desired iterations needs to be specified. The optimizer also suppport visualizing intermediate output at given iteration intervals.
#

optimizer_name = 'sgd'
nr_of_iterations = 50
visualize = True
visualize_step = 10

#########################
# Creating the parameter structure
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# AS typical in mermaid, we create a parameters structure (``params``) that will keep track of all the parameters used during a run.
#

# keep track of general parameters
params = pars.ParameterDict()

#############################
# Creating an example
# ^^^^^^^^^^^^^^^^^^^
#
# Now we are ready to create some example data. We create squares for source and target images in two dimensions.

dim = 2
example_img_len = 64

szEx = np.tile( example_img_len, dim )         # size of the desired images: (sz)^dim
I0,I1,spacing= eg.CreateSquares(dim).create_image_pair(szEx,params) # create a default image size with two sample squares
sz = np.array(I0.shape)

##############################
# Moving from numpy to pytorch
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The example generation produces numpy arrays. As *mermaid* uses pytorch these need to be converted to pytorch arrays. Also, we support running *mermaid* on the CPU and the GPU. The convenience function ``AdaptVal`` takes care of this by moving an array either to the GPU or leaving it on the CPU.

# create the source and target image as pyTorch variables
ISource = AdaptVal(torch.from_numpy(I0.copy()))
ITarget = AdaptVal(torch.from_numpy(I1))

#############################
# Instantiating the optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now we are ready to instantiate the optimizer with the parameters defined above.
# We instantiate a single-scale optimizer here, but instantiating a multi-scale optimizer
# proceeds similarly. We then start the optimization (via ``so.optimizer()``).
#

so = MO.SingleScaleRegistrationOptimizer(sz,spacing,use_map,map_low_res_factor,params)
so.set_model(model_name)
so.set_optimizer_by_name( optimizer_name )
so.set_visualization( visualize )
so.set_visualize_step( visualize_step )

so.set_number_of_iterations(nr_of_iterations)

so.set_source_image(ISource)
so.set_target_image(ITarget)

# and now do the optimization
so.optimize()

##############################
# Writing out the parameter configuration file
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To inspect what settings were in fact used during the run we can write them out using json.
# We can write out both the actual settings as well as a json file describing what the individual
# settings mean. If desired one can then modify these settings and load them back in as the
# new desired setting for a new optimization run.

params.write_JSON( 'test_minimal_registration_' + model_name + '_settings_clean.json')
params.write_JSON_comments( 'test_minimal_registration_' + model_name + '_settings_comments.json')

