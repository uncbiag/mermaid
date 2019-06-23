Basic Numerics
==============

This section describes various different numerical considerations and modules of *mermaid*. Specifically,

- the finite difference module, which is at the core of the discretization of the ordinary (ODE) and partial differential equations (PDE) and operators of mermaid (see :ref:`finite-differences-label`),
- the Runge-Kutta integrators to integrate the ODEs and PDEs of the registration models (see :ref:`runge-kutta-label`),
- the higher-order spline interpolations, which can be used instead of pytorch's built-in bilinear and trilinear interpolation. These interpolators are relatively general, but are currently not heavily used or supported by *mermaid* (see :ref:`spline-interpolation-label`).

.. contents::
  
.. _finite-differences-label:
  
Finite differences
^^^^^^^^^^^^^^^^^^

This module implements all of *mermaid's* finite difference functionality.

.. inheritance-diagram:: mermaid.finite_differences
.. automodule:: mermaid.finite_differences
	:members:
	:undoc-members:

.. _runge-kutta-label:
	   
Runge-Kutta integrators
^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: mermaid.rungekutta_integrators
.. automodule:: mermaid.rungekutta_integrators
	:members:
	:undoc-members:
	   
.. _spline-interpolation-label:

Spline interpolation
^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: mermaid.spline_interpolation
.. automodule:: mermaid.spline_interpolation
	:members:
	:undoc-members:
