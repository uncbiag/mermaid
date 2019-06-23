Optimizers
==========

*mermaid* currently makes primarily use of SGD with momentum. However, it also supports optimization with lBFGS and some limited support for adam. The optimization code is part of the multiscale optimizer (described here :ref:`multi-scale-optimizer-label`). The customized lBFGS optimizer (to support line-search) is described here: :ref:`custom-optimizers-label`.

.. contents::

.. _multi-scale-optimizer-label:

Multiscale optimizer
^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: mermaid.multiscale_optimizer
.. automodule:: mermaid.multiscale_optimizer
	:members:
	:undoc-members:

.. _custom-optimizers-label:

Custom optimizers
^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: mermaid.custom_optimizers
.. automodule:: mermaid.custom_optimizers
	:members:
	:undoc-members:

.. _optimizer-data-loaders-label:

Optimizer data loaders
^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: mermaid.optimizer_data_loaders
.. automodule:: mermaid.optimizer_data_loaders
	:members:
	:undoc-members:
