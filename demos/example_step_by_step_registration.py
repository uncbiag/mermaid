"""

Step-by-step registration example
=================================

This example script walks through a mermaid registration example step by step.
"""


# before we start let's import some mermaid libraries

# first the simple registration interface (which provides basic, easy to use registration functionality)
import mermaid.simple_interface as SI
# the parameter module which will keep track of all the registration parameters
import mermaid.module_parameters as pars
# and some mermaid functionality to create test data
import mermaid.example_generation as EG

# also import numpy
import numpy as np

# first we create a parameter structure to keep track of all registration settings
params = pars.ParameterDict()

# and let's create two-dimensional squares
I0,I1,spacing = EG.CreateSquares(dim=2,add_noise_to_bg=True).create_image_pair(np.array([64,64]),params=params)
params.write_JSON('step_by_step_example_data.json')
params.write_JSON_comments('step_by_step_example_data_with_comments.json')

# now we instantiate the registration class
si = SI.RegisterImagePair()

# and use it for registration
# as registration algorithms can have many settings we simply run it first for one iteration and ask it to
# write out a configuration file, including comments to explain the settings
si.register_images(I0, I1, spacing,
                   model_name='lddmm_shooting_map',
                   nr_of_iterations=1,
                   optimizer_name='sgd',
                   json_config_out_filename=('step_by_step_basic_settings.json','step_by_step_basic_settings_with_comments.json')
                   )

