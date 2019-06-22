.. pytorchRegistration documentation master file, created by
   sphinx-quickstart on Sat Jul 29 08:41:36 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mermaid: iMagE Registration via autoMAtIc Differentiation [*]_
==============================================================

*mermaid* is a registration toolbox, which supports various image registration methods. In particular, it focuses on nonparametric registration approaches (including stationary velocity fields and large discplacement diffeomorphic metric mapping models) though simple affine registration is also possible. As it is entirely written in *pyTorch* it allows for rapid prototyping of new image registration approaches and similiarity measures. To keep track of registration parameters it makes use of json configuration files which entirely describe registration algorithms.

*mermaid* was primarily developed by:

  - Marc Niethammer
  - Zhengyang Shen
  - Roland Kwitt


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/installation.rst
   notes/parameters.rst
   notes/simple_example.rst
   notes/howto_own_registration.rst
   notes/todos.rst

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   config_parser
   custom_optimizers
   custom_pytorch_extensions
   data_wrapper
   deep_networks
   deep_smoothers
   example_generation
   fileio
   finite_differences
   forward_models
   image_manipulations
   image_sampling
   load_default_settings
   model_evaluation
   model_factory
   module_parameters
   multiscale_optimizer
   noisy_convolution
   optimizer_data_loaders
   registration_networks
   regularizer_factory
   rungekutta_integrators
   similarity_helper_omt
   similarity_measure_factory
   simple_interface
   spline_interpolation
   smoother_factory
   stn
   utils
   viewers
   visualize_registration_results

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. [*] OK, we just liked the acronym.

