Steb by step registration example
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

.. code::

  # first the simple registration interface (which provides basic, easy to use registration functionality)
  import mermaid.simple_interface as SI
  # the parameter module which will keep track of all the registration parameters
  import mermaid.module_parameters as pars
  # and some mermaid functionality to create test data
  import mermaid.example_generation as EG

Some of the high-level mermaid code and also the plotting depends on numpy (we will phase much of this out in the future).
Hence we als import numpy

.. code::

  import numpy

mermaid parameters
^^^^^^^^^^^^^^^^^^
  
Registration algorithms tend to have a lot of settings. Starting from the registration model, over the selection and settings
for the optimizer, to general compute settings (for example, if mermaid should be run on the GPU or CPU).
All the non-compute settings that affect registration results are automatically kept track inside a parameters structure.

So let's first create an empty *mermaid* parameter object

.. code::

  # first we create a parameter structure to keep track of all registration settings
  params = pars.ParameterDict()

Generating registration example data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  
Now we create some example data (source and target image examples for a two-dimensional square, of size 64x64) and keep track of the generated settings via this parameter object

.. code::

  # and let's create two-dimensional squares
  I0,I1,spacing = EG.CreateSquares(dim=2,add_noise_to_bg=True).create_image_pair(np.array([64,64]),params=params)

Parameters can easily be written to a file (or displayed via print). We can write it out including comments for what these
settings are as

.. code::

  params.write_JSON('step_by_step_example_data.json')
  params.write_JSON_comments('step_by_step_example_data_with_comments.json')

The first command writes out the actual json configuration, the second one comments that explain what the settings are
(as json does not allow commented files by default). The resulting output in this case looks as follows.

For step_by_step_example_data.json:

.. code-block:: json

    {
      "square_example_images": {
          "len_l": 16,
          "len_s": 10
      }
    }

And for its commented version step_by_step_example_data_with_comments.json:

.. code-block:: json

    {
        "square_example_images": {
            "__doc__": "Controlling the size of a nD cube",
            "len_l": "Maximum side-length of square",
            "len_s": "Mimimum side-length of square"
        }
    }

Performing the registration
^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
Now we are ready to instantiate the registration object from mermaid

.. code::

  # now we instantiate the registration class
  si = SI.RegisterImagePair()

As we are not sure what settings to use, let alone know what settings exist, we simply run it first for one
step and ask for the json configuration to be written out.

.. code::

  si.register_images(I0, I1, spacing,
                   model_name='lddmm_shooting_map',
                   nr_of_iterations=1,
                   optimizer_name='sgd',
                   json_config_out_filename=('step_by_step_basic_settings.json','step_by_step_basic_settings_with_comments.json')
                   )

The resulting (entirely auto-generated) json configuration files look like this. We can then edit them and run
a registration with the settings we care about.

The actual settings in step_by_step_basic_settings.json are:

.. code-block:: json

    {
        "model": {
            "deformation": {
                "compute_similarity_measure_at_low_res": false,
                "map_low_res_factor": 1.0,
                "use_map": true
            },
            "registration_model": {
                "forward_model": {
                    "adjoin_on": true,
                    "atol": 1e-05,
                    "number_of_time_steps": 20,
                    "rtol": 1e-05,
                    "smoother": {
                        "multi_gaussian_stds": [
                            0.05,
                            0.1,
                            0.15,
                            0.2,
                            0.25
                        ],
                        "multi_gaussian_weights": [
                            0.06666666666666667,
                            0.13333333333333333,
                            0.19999999999999998,
                            0.26666666666666666,
                            0.3333333333333333
                        ],
                        "type": "multiGaussian"
                    },
                    "solver": "rk4"
                },
                "loss": {
                    "display_max_displacement": false,
                    "limit_displacement": false,
                    "max_displacement": 0.05
                },
                "similarity_measure": {
                    "develop_mod_on": false,
                    "sigma": 0.1,
                    "type": "ssd"
                },
                "spline_order": 1,
                "type": "lddmm_shooting_map",
                "use_CFL_clamping": true
            }
        },
        "optimizer": {
            "gradient_clipping": {
                "clip_display": true,
                "clip_individual_gradient": false,
                "clip_individual_gradient_value": 1.0158730158730158,
                "clip_shared_gradient": true,
                "clip_shared_gradient_value": 1.0
            },
            "name": "sgd",
            "scheduler": {
                "factor": 0.5,
                "patience": 10,
                "verbose": true
            },
            "sgd": {
                "individual": {
                    "dampening": 0.0,
                    "lr": 0.01,
                    "momentum": 0.9,
                    "nesterov": true,
                    "weight_decay": 0.0
                },
                "shared": {
                    "dampening": 0.0,
                    "lr": 0.01,
                    "momentum": 0.9,
                    "nesterov": true,
                    "weight_decay": 0.0
                }
            },
            "single_scale": {
                "nr_of_iterations": 1,
                "rel_ftol": 0.0001
            },
            "use_step_size_scheduler": true,
            "weight_clipping_type": "none",
            "weight_clipping_value": 1.0
        }
    }

These settings are explained in step_by_step_basic_settings_with_comments.json are:

.. code-block:: json

    {
        "model": {
            "deformation": {
                "compute_similarity_measure_at_low_res": "If set to true map is not upsampled and the entire computations proceeds at low res",
                "map_low_res_factor": "Set to a value in (0,1) if a map-based solution should be computed at a lower internal resolution (image matching is still at full resolution",
                "use_map": "use a map for the solution or not True/False"
            },
            "registration_model": {
                "forward_model": {
                    "__doc__": "settings for the forward model",
                    "adjoin_on": "use adjoint optimization",
                    "atol": "absolute error torlance for dopri5",
                    "number_of_time_steps": "Number of time-steps to per unit time-interval integrate the PDE",
                    "rtol": "relative error torlance for dopri5",
                    "smoother": {
                        "multi_gaussian_stds": "std deviations for the Gaussians",
                        "multi_gaussian_weights": "weights for the multiple Gaussians",
                        "type": "type of smoother (diffusion|gaussian|adaptive_gaussian|multiGaussian|adaptive_multiGaussian|gaussianSpatial|adaptiveNet)"
                    },
                    "solver": "ode solver"
                },
                "loss": {
                    "__doc__": "settings for the loss function",
                    "display_max_displacement": "displays the current maximal displacement",
                    "limit_displacement": "[True/False] if set to true limits the maximal displacement based on the max_displacement_setting",
                    "max_displacement": "Max displacement penalty added to loss function of limit_displacement set to True"
                },
                "similarity_measure": {
                    "develop_mod_on": "developing mode",
                    "sigma": "1/sigma^2 is the weight in front of the similarity measure",
                    "type": "type of similarity measure (ssd/ncc)"
                },
                "spline_order": "Spline interpolation order; 1 is linear interpolation (default); 3 is cubic spline",
                "type": "Name of the registration model",
                "use_CFL_clamping": "If the model uses time integration, CFL clamping is used"
            }
        },
        "optimizer": {
            "gradient_clipping": {
                "__doc__": "clipping settings for the gradient for optimization",
                "clip_display": "If set to True displays if clipping occurred",
                "clip_individual_gradient": "If set to True, the gradient for the individual parameters will be clipped",
                "clip_individual_gradient_value": "Value to which the gradient for the individual parameters is clipped",
                "clip_shared_gradient": "If set to True, the gradient for the shared parameters will be clipped",
                "clip_shared_gradient_value": "Value to which the gradient for the shared parameters is clipped"
            },
            "name": "Optimizer (lbfgs|adam|sgd)",
            "scheduler": {
                "__doc__": "parameters for the ReduceLROnPlateau scheduler",
                "factor": "reduction factor",
                "patience": "how many steps without reduction before LR is changed",
                "verbose": "if True prints out changes in learning rate"
            },
            "sgd": {
                "individual": {
                    "dampening": "sgd dampening",
                    "lr": "desired learning rate",
                    "momentum": "sgd momentum",
                    "nesterov": "use Nesterove scheme",
                    "weight_decay": "sgd weight decay"
                },
                "shared": {
                    "dampening": "sgd dampening",
                    "lr": "desired learning rate",
                    "momentum": "sgd momentum",
                    "nesterov": "use Nesterove scheme",
                    "weight_decay": "sgd weight decay"
                }
            },
            "single_scale": {
                "nr_of_iterations": "number of iterations",
                "rel_ftol": "relative termination tolerance for optimizer"
            },
            "use_step_size_scheduler": "If set to True the step sizes are reduced if no progress is made",
            "weight_clipping_type": "Type of weight clipping that should be used [l1|l2|l1_individual|l2_individual|l1_shared|l2_shared|None]",
            "weight_clipping_value": "Value to which the norm is being clipped"
        }
    }

Running the actual registration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can now edit the generated json file and modify the desired settings. The most important ones are proably the similiarty measure as well as the settings for the multi-Gaussian smoother. These can then be spefified via keyword *params*, i.e., something like

.. code::

   si.register_images(I0, I1, spacing, model_name='lddmm_shooting_map',
                       nr_of_iterations=50,
                       use_multi_scale=False,
                       visualize_step=10,
                       optimizer_name='sgd',
                       learning_rate=0.02,
                       rel_ftol=1e-7,
                       json_config_out_filename=('my_used_params.json','my_used_params_with_comments.json'),
                       params='my_params.json')
