.. todo::
   This documentation needs to be updated. Have a look at README.md which will have a more up-to-date intruction on the installation.

Installation
============

This note briefly describes how to install and use *mermaid*. We recommend installing *mermaid* using `conda <http://docs.conda.io>`_. We also recommend using a conda virtual environment to keep the installation clean.

mermaid requirements
^^^^^^^^^^^^^^^^^^^^

*mermaid* is based on the following:

  - python 3
  - pytorch > 1.0

It runs on Linux and on OSX. Windows installs might be possible, but have not been tested (hence will likely not work out of the box) and are not officially supported.
    
anaconda installation
^^^^^^^^^^^^^^^^^^^^^

If you have not done so yet, install anaconda. Simply follow the installation instructions `here <https://www.anaconda.com/download>`_. This will provide you with the conda package manager which will be used to create a virtual environment (if desired, we highly recommend it) and to install all python packages that *mermaid* depends on.

Creating a conda virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is best to install everything into a conda virtual environment. This can be done as follows.

.. code::

   conda create --name mermaid python=3.7 pip
   conda activate mermaid

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it is desirable to install *mermaid* for development purposes. To do this, first download the git repository

.. code::

   git clone https://github.com/uncbiag/mermaid.git

The repository's main folder contains a setup.py file (see `python setup file <https://github.com/kennethreitz/setup.py>`_) for the setup file *mermaid's* is based on and a general explanation of its use. For development purposes then simply execute

.. code::

   cd mermaid
   python setup.py develop

This will install all library links.


Creating the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The documentation is created via `sphinx <http://www.sphinx-doc.org/>`_. To build it first install graphviz (on OSX: `brew install graphviz`). Then execute the following

.. code::

   cd mermaid
   cd docs
   make html


This will create the docs in `build/html`.

Running the code
^^^^^^^^^^^^^^^^

.. todo::
   Not sure if this is still needed. It is not needed with the mermaid conda installation. Test if it is needed when one does the development setup.

The simplest way to start is to look at the two example scripts `testRegistrationGeneric.py` and `testRegistrationGenericMultiscale.py` at the top direcory. To mak sure that they all find the paths to the libraries simply do

.. code::

   import set_pyreg_paths

   
