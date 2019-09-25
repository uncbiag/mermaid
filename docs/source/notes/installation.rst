Installation
============

This note briefly describes how to install and use *mermaid*. We recommend installing *mermaid* using `conda <http://docs.conda.io>`__. We also recommend using a conda virtual environment to keep the installation clean.

mermaid requirements
^^^^^^^^^^^^^^^^^^^^

*mermaid* is based on the following:

  - python >= 3.6
  - pytorch >= 1.0

It runs on Linux and on OSX. Windows installs might be possible, but have not been tested (hence will likely not work out of the box) and are not officially supported.
    
Anaconda installation
^^^^^^^^^^^^^^^^^^^^^^

If you have not done so yet, install anaconda. Simply follow the installation instructions `here <https://www.anaconda.com/download>`__. This will provide you with the conda package manager which will be used to create a virtual environment (if desired, we highly recommend it) and to install all python packages that *mermaid* depends on.

Creating a conda virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is best to install everything into a conda virtual environment. This can be done as follows.

.. code::

   conda create --name mermaid python=3.7 pip
   conda activate mermaid

If you later want to remove this environment again, this can be done by executing

.. code::

   conda remove --name mermaid --all
   
   
Mermaid installation via conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. todo::
   Add the actual conda channel here, once we have it

*mermaid* can conveniently be installed via *conda*. Once you have activated your desired conda environment (for example, the conda virtual environment created above) *mermaid* can be installed by executing

.. code::
   
   conda install -vv -k -c http://wwwx.cs.unc.edu/~mn/download/local_conda_channel -c pytorch -c conda-forge -c anaconda mermaid=0.2.0

Once installed *mermaid* simply needs to be imported in your python program via

.. code::
   
   import mermaid
   

Mermaid development installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it is desirable to install *mermaid* for development purposes. To do this, first download the git repository

.. code::

   git clone https://github.com/uncbiag/mermaid.git

The repository's main folder contains a setup.py file (see `python setup file <https://github.com/kennethreitz/setup.py>`_) for the setup file *mermaid's* is based on and a general explanation of its use. For development purposes then simply execute

.. code::

   cd mermaid
   python setup.py develop

This will install all library links and all missing packages and will allow mermaid imports.


Creating the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The documentation is created via `sphinx <http://www.sphinx-doc.org/>`__. To build it first install graphviz (on OSX: `brew install graphviz` or via conda, see below). If you installed via the developer option (via `setup.py`) you will also need to install *pandoc* (should be auto installed via conda). This can be done by following the instructions `here <https://pypi.org/project/pypandoc/>`__ (pypandoc will be auto installed via `setup.py`) or by installing it manually via conda:

.. code::

   conda install -c conda-forge pandoc

Graphviz can also be installed via conda if desired:

.. code::

   conda install -c anaconda graphviz

Then execute the following to make the documentation

.. code::

   cd mermaid
   cd docs
   make html


This will create the docs in `build/html`.

Running the code
^^^^^^^^^^^^^^^^

The simplest way to start is to look example script `demos/test_simple_interface,py`, or to run the examples from the `jupyter` directory which contains various example jupyter notebooks. You can run the jupyter notebooks as follows (should be intalled if you installed via conda or `python setup.py develop` as described above):

.. code::

   cd mermaid
   cd jupyter
   jupyter notebook



   
