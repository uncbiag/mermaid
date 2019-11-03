"""

Step-by-step registration example
=================================

"""

#####################################
#
# This tutorial will take you step-by-step through a very simple registration example.
#
# .. contents::
#


######################################
# Introduction
# ^^^^^^^^^^^^
#
# Starting with *mermaid* can initially be a little overwhelming. We therefore walk through a simple step-by-step registration example here. This example should teach you:
#
# - how to import the most essential *mermaid* packages
# - how to generate some example data
# - how to run instantiate and run a simple registration
# - how to work with mermaid parameter structures (these are important as they keep track of all the mermaid settings and can also be edited)
#

##################################
# Importing modules
# ^^^^^^^^^^^^^^^^^
#
# First we import some important mermaid modules. The used mermaid modules are:
#
# - ``mermaid.simple_interface`` provides a very high level interface to the image registration algorithms
# - ``mermaid.example_generation`` allows to generate simply synthetic data and some real image pairs to test the registration algorithms
# - ``mermaid.module_parameters`` allows generating mermaid parameter structures which are used to keep track of all the parameters
#
# Some of the high-level mermaid code and also the plotting depends on numpy (we will phase much of this out in the future). Hence we also ``import numpy``.

# first the simple registration interface (which provides basic, easy to use registration functionality)
import mermaid.simple_interface as SI
# the parameter module which will keep track of all the registration parameters
import mermaid.module_parameters as pars
# and some mermaid functionality to create test data
import mermaid.example_generation as EG

# also import numpy
import numpy as np

##########################
# mermaid parameters
# ^^^^^^^^^^^^^^^^^^
#
# Registration algorithms tend to have a lot of settings. Starting from the registration model, over
# the selection and settings for the optimizer, to general compute settings (for example, if mermaid
# should be run on the GPU or CPU). All the non-compute settings that affect registration results are
# automatically kept track inside a parameters structure.
#
# So let's first create an empty *mermaid* parameter object

# first we create a parameter structure to keep track of all registration settings
params = pars.ParameterDict()

#############################
# Generating registration example data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now we create some example data (source and target image examples for a two-dimensional square,
# of size 64x64) and keep track of the generated settings via this parameter object.

# and let's create two-dimensional squares
I0,I1,spacing = EG.CreateSquares(dim=2,add_noise_to_bg=True).create_image_pair(np.array([64,64]),params=params)

###############################
# Writing out parameters
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Parameters can easily be written to a file (or displayed via print). We can write it out including comments for what these settings are as follows. The first command writes out the actual json configuration, the second one comments that explain what the settings are (as json does not allow commented files by default).

params.write_JSON('step_by_step_example_data.json')
params.write_JSON_comments('step_by_step_example_data_with_comments.json')

###############################
# Resulting parameters after the example generation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The resulting output in this case looks as follows. This is contained in the files, but we simply print the
# parameters here.

print(params)

#################################
# Performing the registration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now we are ready to instantiate the registration object from mermaid. As we are not sure what settings to use, let alone know what settings exist, we simply run it first for one step and ask for the json configuration to be written out.

# now we instantiate the registration class
si = SI.RegisterImagePair()

# and use it for registration
# as registration algorithms can have many settings we simply run it first for one iteration and ask it to
# write out a configuration file, including comments to explain the settings.
si.register_images(I0, I1, spacing,
                   model_name='lddmm_shooting_map',
                   nr_of_iterations=1,
                   optimizer_name='sgd',
                   visualize_step=None,
                   json_config_out_filename=('step_by_step_basic_settings.json','step_by_step_basic_settings_with_comments.json'))

