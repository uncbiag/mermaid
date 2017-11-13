Installation
============

This note briefly describes how to install and use *mermaid*. Since this is all based on pyTorch, the first step if you have not done so yet is to install pyTorch itself. These installation notes are currently for OSX.

pyTorch installation
^^^^^^^^^^^^^^^^^^^^

See `pyTorch <http://pytorch.org/>`_.

To install via anaconda execute

.. code::

   conda install pytorch torchvision cuda80 -c soumith


However, as pyTorch is still actively developed it makes sense to install it from the git repository. At least at the time of writing of this document the repository had some autograd bugs fixed that were not fixed in the official release yet.

To install pyTorch from source do the following

.. code::

   git clone https://github.com/pytorch/pytorch.git


This will create a `pytorch` directory. On OSX do the following. (Set NO_CUDA=1 if you want to compile without CUDA support and NO_CUDA=0 if you want CUDA support.)

.. code::

   export NO_CUDA=1
   export CMAKE_PREFIX_PATH=[/path/to/anaconda/install]
   MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install


The `/path/to/anaconda/install` is for me for example `/Users/mn/anaconda`

Installing mermaid
^^^^^^^^^^^^^^^^^^

To install the registration package. First check it out from bitbucket:

.. code::

   git clone git@bitbucket.org:marcniethammer/pytorchregistration.git

Currently, there is only CPU support. But since we want to work with 1D, 2D, and 3D images we need a custom version of a spatial transformer network (STN).

To compile this do the following

.. code::

   cd pytorchregistration
   cd pyreg/libraries
   python build.py
   

You may need to adapt the build.py script (working on making this simpler). If it is compiled with gcc (you can install it on OSX via `brew install gcc` and then follow the intructions to give it the right permissions) it will support multi-threading via openmp. If compiled by clang, multi-threading support will not be available. However, in practice this part of the code is not the bottleneck, so it may not matter much.

Creating the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The documentation is created via `sphinx <http://www.sphinx-doc.org/>`_. To build it first install graphviz (on OSX: `brew install graphviz`). Then execute the following

.. code::

   cd pytorchregistration
   cd docs
   make html


This will create the docs in `build/html`.

Running the code
^^^^^^^^^^^^^^^^

The simplest way to start is to look at the two example scripts `testRegistrationGeneric.py` and `testRegistrationGenericMultiscale.py` at the top direcory. To mak sure that they all find the paths to the libraries simply do

.. code::

   import set_pyreg_paths

   
