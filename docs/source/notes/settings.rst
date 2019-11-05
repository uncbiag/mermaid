Mermaid configurations
======================

Settings for algorithms are passed via the *params* parameter.
There are also more global settings and reasonable default settings available. These live in the settings directory,
but can be overwritten by placing custom setting files (of the same names) in a local *.mermaid_settings* directory
in a user's home directory. More details can be found in :ref:`configurations`.

.. contents::

Computational settings
^^^^^^^^^^^^^^^^^^^^^^

The main computational settings are set via the ``compute_settings.json`` file and then directly translate
into the module variables ``CUDA_ON``, ``USE_FLOAT16``, ``nher_of_threads``, ``MATPLOTLIB_AGG``.
These can for example be imported as

.. code:: python
   
   from mermaid.config_parser import CUDA_ON, USE_FLOAT16

The settings filename can be queried via

.. currentmodule:: mermaid.config_parser
.. autofunction:: get_default_compute_settings_filenames
    :noindex:


The default settings are
   
.. literalinclude:: mermaid_settings/compute_settings.json
  :language: JSON

.. literalinclude:: mermaid_settings/compute_settings_comments.json
  :language: JSON

Algorithm settings
^^^^^^^^^^^^^^^^^^

Reasonable initial settings for registration algorithms can be obtained via the provided ``algconf_settings.json`` settings file.

The settings filename can be queried via

.. currentmodule:: mermaid.config_parser
.. autofunction:: get_default_algconf_settings_filenames
    :noindex:

The settings can be loaded into a parameter structure via

.. currentmodule:: mermaid.config_parser
.. autofunction:: get_algconf_settings
    :noindex:

The default settings (for these confgurations are)
		  
.. literalinclude:: mermaid_settings/algconf_settings.json
  :language: JSON

.. literalinclude:: mermaid_settings/algconf_settings_comments.json
  :language: JSON

Basic settings
^^^^^^^^^^^^^^

If saving and reading of configuration files is desired as an option those can be obtained via the provided ``baseconf_settings.json`` settings file.

The settings filename can be queried via

.. currentmodule:: mermaid.config_parser
.. autofunction:: get_default_baseconf_settings_filenames
    :noindex:

The settings can be loaded into a parameter structure via

.. currentmodule:: mermaid.config_parser
.. autofunction:: get_baseconf_settings
    :noindex:

The default settings (for these confgurations are)

.. literalinclude:: mermaid_settings/baseconf_settings.json
  :language: JSON

.. literalinclude:: mermaid_settings/baseconf_settings_comments.json
  :language: JSON

Democonf settings
^^^^^^^^^^^^^^^^^

Configurations for the creation of demo data should go into the provided ``democonf_settings.json`` settings file.

The settings filename can be queried via

.. currentmodule:: mermaid.config_parser
.. autofunction:: get_default_democonf_settings_filenames
    :noindex:

The settings can be loaded into a parameter structure via

.. currentmodule:: mermaid.config_parser
.. autofunction:: get_democonf_settings
    :noindex:

The default settings (for these confgurations are)

.. literalinclude:: mermaid_settings/democonf_settings.json
  :language: JSON

.. literalinclude:: mermaid_settings/democonf_settings_comments.json
  :language: JSON


Respro settings
^^^^^^^^^^^^^^^

.. todo:: These settings need more explanation. Where are they used?

The settings filename can be queried via

.. currentmodule:: mermaid.config_parser
.. autofunction:: get_default_respro_settings_filenames
    :noindex:

The settings can be loaded into a parameter structure via

.. currentmodule:: mermaid.config_parser
.. autofunction:: get_respro_settings
    :noindex:

The default settings (for these confgurations are)
	  
.. literalinclude:: mermaid_settings/respro_settings.json
  :language: JSON

.. literalinclude:: mermaid_settings/respro_settings_comments.json
  :language: JSON
