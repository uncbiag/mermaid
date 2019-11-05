Step by step registration example
=================================

Starting with *mermaid* can initially be a little overwhelming. We therefore walk through a simple step-by-step
registration example here. This example should teach you

- how to import the most essential *mermaid* packages
- how to generate some example data
- how to run instantiate and run a simple registration
- how to work with mermaid parameter structures (these are important as they keep track of all the mermaid settings and can also be edited)


.. contents::

mermaid imports
^^^^^^^^^^^^^^^

First we import some important mermaid modules

.. code:: python

  # first the simple registration interface (which provides basic, easy to use registration functionality)
  import mermaid.simple_interface as SI
  # the parameter module which will keep track of all the registration parameters
  import mermaid.module_parameters as pars
  # and some mermaid functionality to create test data
  import mermaid.example_generation as EG

Some of the high-level mermaid code and also the plotting depends on numpy (we will phase much of this out in the future).
Hence we als import numpy

.. code:: python

  import numpy

mermaid parameters
^^^^^^^^^^^^^^^^^^
  
Registration algorithms tend to have a lot of settings. Starting from the registration model, over the selection and settings
for the optimizer, to general compute settings (for example, if mermaid should be run on the GPU or CPU).
All the non-compute settings that affect registration results are automatically kept track inside a parameters structure.

So let's first create an empty *mermaid* parameter object

.. code:: python

  # first we create a parameter structure to keep track of all registration settings
  params = pars.ParameterDict()

Generating registration example data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  
Now we create some example data (source and target image examples for a two-dimensional square, of size 64x64) and keep track of the generated settings via this parameter object

.. code:: python

  # and let's create two-dimensional squares
  I0,I1,spacing = EG.CreateSquares(dim=2,add_noise_to_bg=True).create_image_pair(np.array([64,64]),params=params)

Parameters can easily be written to a file (or displayed via print). We can write it out including comments for what these
settings are as

.. code:: python

  params.write_JSON('step_by_step_example_data.json')
  params.write_JSON_comments('step_by_step_example_data_with_comments.json')

The first command writes out the actual json configuration, the second one comments that explain what the settings are
(as json does not allow commented files by default). The resulting output in this case looks as follows.

For step_by_step_example_data.json:

.. literalinclude:: step_by_step_example_data.json
  :language: JSON

And for its commented version step_by_step_example_data_with_comments.json:

.. literalinclude:: step_by_step_example_data_with_comments.json
  :language: JSON

Performing the registration
^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
Now we are ready to instantiate the registration object from mermaid

.. code:: python

  # now we instantiate the registration class
  si = SI.RegisterImagePair()

As we are not sure what settings to use, let alone know what settings exist, we simply run it first for one
step and ask for the json configuration to be written out.

.. code:: python

  si.register_images(I0, I1, spacing,
                   model_name='lddmm_shooting_map',
                   nr_of_iterations=1,
                   optimizer_name='sgd',
                   json_config_out_filename=('step_by_step_basic_settings.json','step_by_step_basic_settings_with_comments.json')
                   )

The resulting (entirely auto-generated) json configuration files look like this. We can then edit them and run
a registration with the settings we care about.

The actual settings in step_by_step_basic_settings.json are:

.. literalinclude:: step_by_step_basic_settings.json
  :language: JSON

These settings are explained in step_by_step_basic_settings_with_comments.json are:

.. literalinclude:: step_by_step_basic_settings_with_comments.json
  :language: JSON

Running the actual registration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can now edit the generated json file and modify the desired settings. The most important ones are proably the similiarty measure as well as the settings for the multi-Gaussian smoother. These can then be spefified via keyword *params*, i.e., something like

.. code:: python

   si.register_images(I0, I1, spacing, model_name='lddmm_shooting_map',
                       nr_of_iterations=50,
                       use_multi_scale=False,
                       visualize_step=10,
                       optimizer_name='sgd',
                       learning_rate=0.02,
                       rel_ftol=1e-7,
                       json_config_out_filename=('my_used_params.json','my_used_params_with_comments.json'),
                       params='my_params.json')
