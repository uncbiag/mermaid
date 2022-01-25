# -*- coding: utf-8 -*-
"""

Simple registration interface example
=====================================

"""


######################################################################
# 
# This tutorial will show how to use the simple registration interface of *mermaid*.
#
# .. contents::
#


##############################
# Introduction
# ^^^^^^^^^^^^
#
# Mermaid can be a rather complex registration package.
# Hence it provides various convenience functions to make registration reasonably easy.
# One of them is a simple registration interface that allows setting the most popular parameters.
# But it also allows passing through all the advanced registration parameters as desired.
#

##############################
# Importing modules
# ^^^^^^^^^^^^^^^^^
#
# We start by importing some popular modules. The used mermaid modules are:
#
# - ``mermaid.simple_interface`` provides a very high level interface to the image registration algorithms
# - ``mermaid.example_generation`` allows to generate simply synthetic data and some real image pairs to test the registration algorithms
# - ``mermaid.module_parameters`` allows generating mermaid parameter structures which are used to keep track of all the parameters
#

from __future__ import print_function
import torch
# torch.cuda.set_device("cuda:2")
import numpy as np
import matplotlib.pyplot as plt

import mermaid.simple_interface as SI
import mermaid.example_generation as EG
import mermaid.module_parameters as pars

################################
# Creating some example data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can use ``mermaid.example_generation`` to create some test data as follows:
#

# We first generate a parameter object to hold settings
params = pars.ParameterDict()

# We select if we want to create synthetic image pairs or would prefer a real image pair
use_synthetic_test_case = True

# Desired dimension (mermaid supports 1D, 2D, and 3D registration)
dim = 2

# If we want to add some noise to the background (for synthetic examples)
add_noise_to_bg = True

# and now create it
if use_synthetic_test_case:
    length = 64
    # size of the desired images: (sz)^dim
    szEx = np.tile(length, dim )
    # create a default image size with two sample squares
    I0, I1, spacing = EG.CreateSquares(dim,add_noise_to_bg).create_image_pair(szEx, params)
else:
    # return a real image example
    I0, I1, spacing = EG.CreateRealExampleImages(dim).create_image_pair() # create a default image size with two sample squares

##################################
# Creating the registration algorithm
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We simply instantiate a simple interface object for the registration of image pairs.
# We can then query it as to what models registration models are currently supported.
#

# create a simple interface object for pair-wise image registration
si = SI.RegisterImagePair()

# print possible model names
si.print_available_models()

################################
# Doing the registration
# ^^^^^^^^^^^^^^^^^^^^^^
#
# We are now ready to perform the registration (picking one of the registration model options printed above).
#
# Here, we use a shooting-based LDDMM algorithm which works directly with transformation maps.
# The simple interface allows setting various registration settings.
#
# Of note, we read in a parameter file (``test2d_tst.json``) to parametrize the registration algorithm and
# write out the used parameters (after the run) into the same file as well as into a file with comments explaining
# all the parameter settings. If ``params`` is not specified default settings will be used. By running for one
# iteration for example, this allows writing out a fresh settings template which can then be edited.

si.register_images(I0, I1, spacing, model_name='lddmm_shooting_map',
                   nr_of_iterations=100,
                   use_multi_scale=False,
                   visualize_step=5,
                   optimizer_name='sgd',
                   learning_rate=0.1,
                   rel_ftol=1e-7,
                   json_config_out_filename=('test2d_tst.json',
                                             'test2d_tst_with_comments.json'),
                   params='test2d_tst.json',
                   recording_step=None)



############################
# Plotting some results
# ^^^^^^^^^^^^^^^^^^^^^
#
# We can query the energies over the iterations. Note that this code need to be modified for a multi-scale solution as
# energies will be returned at each scale.
#

h = si.get_history()

plt.clf()
e_p, = plt.plot(h['energy'], label='energy')
s_p, = plt.plot(h['similarity_energy'], label='similarity_energy')
r_p, = plt.plot(h['regularization_energy'], label='regularization_energy')
plt.legend(handles=[e_p, s_p, r_p])
plt.show()