.. todo::
   This documentation needs to be updated. Have a look at README.md which has a more up-to-date intruction on the installation.

Installation
============

This note briefly describes how to install and use *mermaid*. Since this is all based on pyTorch, the first step if you have not done so yet is to install pyTorch itself. These installation notes are currently for OSX and Linux. We have not tested mermaid on windows, so chances are it will not currently run there. 

anaconda installation
^^^^^^^^^^^^^^^^^^^^^

If you have not done so yet, install anaconda. Simply follow the installation instructions `here <https://www.anaconda.com/download>`_.

Creating a conda virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is best to install everything into a conda virtual environment. This can be done as follows.

.. code::

   conda create -n mermaid python=3.6 pip
   conda activate mermaid


pyTorch installation
^^^^^^^^^^^^^^^^^^^^

See `pyTorch <http://pytorch.org/>`_.

An easier way of installation (that includes pytorch) in a conda virtual environment is described in a section below.

To install via anaconda execute on the Mac

.. code::

   conda install pytorch torchvision -c pytorch

and on Linux

.. code::
   
   conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
   
However, as pyTorch is still actively developed it may make sense to install it from the git repository. At least at the time of writing of this document the repository had some autograd bugs fixed that were not fixed in the official release yet.

To install pyTorch from source do the following

.. code::

   git clone https://github.com/pytorch/pytorch.git


This will create a `pytorch` directory. On OSX do the following. (Set NO_CUDA=1 if you want to compile without CUDA support and NO_CUDA=0 if you want CUDA support.)

.. code::

   export NO_CUDA=1
   export CMAKE_PREFIX_PATH=[/path/to/anaconda/install]
   MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install


The `/path/to/anaconda/install` is for me for example `/Users/mn/anaconda`.

It is also possible to install everything in a virtual conda environment. This may be simpler to do (see below).

Downloading mermaid
^^^^^^^^^^^^^^^^^^^
To install the registration package. First check it out from github:

.. code::

   git clone https://github.com/uncbiag/mermaid.git

Installing all the packages required to run and compile mermaid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a variety of packages that are needed to run mermaid. Assuming you have anaconda installed simply execute

.. code::

    conda install cffi
    conda install -c conda-forge itk
    conda install sphinx
    pip install pynrrd

If the GPU should be supported we also need to get the package for pytorch FFT support. Unfortunately, it is not available via anaconda, but we can use pip

.. code::

    pip install pytorch-fft

Installing all the packages required to run and compile mermaid in a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an alternative way of installation, which is even easier. It installs into a virtual conda environment. To do so exectute the follwing (inside the mermaid directory).

.. code::

   conda create -n mermaid-py27 python=2.7 pip
   source activate mermaid-py27
   cd install
   pip install -r requirements_python2_osx.txt  [Pick the right one for your operating system]
   pip install -r requirements_pytorch_fft.txt

We are working on making the code compatible with python3, which is currently *not supported*. Once it is supported it will be possible to install doing the following:

.. code::
   
   conda create -n mermaid-py36 python=3.6 pip
   source activate mermaid-py36
   cd install
   pip install -r requirements_python3_osx.txt  [Pick the right one for your operating system]
   pip install -r requirements_pytorch_fft.txt


Installing mermaid
^^^^^^^^^^^^^^^^^^

Now we are ready to do some additional compilation for CPU and GPU support. Specicially, since we want to work with 1D, 2D, and 3D images we need a custom version of a spatial transformer network (STN).

To compile this do the following

.. code::

   cd mermaid
   cd pyreg/libraries
   sh make_cpu.sh

If you want to also compile for GPU support (assuming you have the appropriate NVIDA CUDA libraries installed execute

.. code::

    sh make_gpu.sh

You may need to adapt the build.py script (working on making this simpler). If it is compiled with gcc (you can install it on OSX via `brew install gcc` and then follow the intructions to give it the right permissions) it will support multi-threading via openmp. If compiled by clang, multi-threading support will not be available. However, in practice this part of the code is not the bottleneck, so it may not matter much.


Creating the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The documentation is created via `sphinx <http://www.sphinx-doc.org/>`_. To build it first install graphviz (on OSX: `brew install graphviz`). Then execute the following

.. code::

   cd mermaid
   cd docs
   make html


This will create the docs in `build/html`.

Experimental install option
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are currently working on creating an anaconda recipe for mermaid. This is included in the file `mermaid.yaml` in the top directory.
It shows what is necessary to install mermaid and can be used with 'conda-build` to create an anaconda package.
You can find more on how to use and build these packages `here <https://conda.io/docs/user-guide/tutorials/index.html>`_.

Running the code
^^^^^^^^^^^^^^^^

The simplest way to start is to look at the two example scripts `testRegistrationGeneric.py` and `testRegistrationGenericMultiscale.py` at the top direcory. To mak sure that they all find the paths to the libraries simply do

.. code::

   import set_pyreg_paths

   
