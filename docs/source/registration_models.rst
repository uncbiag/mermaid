Registration models
===================

This section describes the most important registration modules of *mermaid*. These

- generate a registration model (see :ref:`model-factory-label`),
- specify dynamic forward models, i.e., what gets integrated (see :ref:`forward-models-label`),
- allow evaluation of the forward models, e.g., when parameters are already given, or when integrating into deep learning models (see :ref:`model-evaluation-label`)
- implement the different regisration models (see :ref:`registration-networks-label`)

.. contents::

.. _model-factory-label:

Model factory
^^^^^^^^^^^^^

The model factory provides convenience functionality to instantiate different registration models.

.. inheritance-diagram:: mermaid.model_factory
.. automodule:: mermaid.model_factory
	:members:
	:undoc-members:

.. _forward-models-label:

Forward models
^^^^^^^^^^^^^^

The forward models are implementations of the dynamic equations for the registration models.

.. inheritance-diagram:: mermaid.forward_models_wrap
.. automodule:: mermaid.forward_models_wrap
	:members:
	:undoc-members:

.. inheritance-diagram:: mermaid.forward_models
.. automodule:: mermaid.forward_models
	:members:
	:undoc-members:

.. _model-evaluation-label:

Model evaluation
^^^^^^^^^^^^^^^^

Given registration parameters the model evaluation module allows to evaluate a given model.
That is, it performs setup and integration of the forward models. This can also be used to easily combine the forward models with deep learning approaches.

.. automodule:: mermaid.model_evaluation
	:members:
	:undoc-members:

.. _registration-networks-label:

Registration networks
^^^^^^^^^^^^^^^^^^^^^

The registration network module implements the different registration algorithms.

.. inheritance-diagram:: mermaid.registration_networks
.. automodule:: mermaid.registration_networks
	:members:
	:undoc-members:
