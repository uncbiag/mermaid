.. pytorchRegistration documentation master file, created by
   sphinx-quickstart on Sat Jul 29 08:41:36 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mermaid: iMagE Registration via autoMAtIc Differentiation [*]_
==============================================================

*mermaid* is a registration toolbox, which supports various image registration methods. In particular, it focuses on nonparametric registration approaches (including stationary velocity fields and large discplacement diffeomorphic metric mapping models) though simple affine registration is also possible. As it is entirely written in *pyTorch* it allows for rapid prototyping of new image registration approaches and similiarity measures. To keep track of registration parameters it makes use of json configuration files which entirely describe registration algorithms. *mermaid* provides optimization-based registration approaches, but the companion-package *easyreg* (https://github.com/uncbiag/easyreg) adds support for deep-learning registration models by building on top of *mermaid's* transformation models.

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
   notes/settings.rst
   notes/todos.rst
   notes/howto_own_registration.rst
   notes/simple_example.rst
   notes/step_by_step_registration_example.rst
   notes/rdmm_example.rst
   example_gallery.rst
   
.. toctree::
   :maxdepth: 1
   :caption: Jupyter notebook examples

   jupyter/example_simple_interface.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Package reference

   simple_interface
   registration_models
   similarity_measures
   basic_numerics
   optimizers
   visualization
   configurations
   image_io
   regularization_and_smoothing
   utilities
   deep_networks
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`

.. [*] OK, we just liked the acronym.

