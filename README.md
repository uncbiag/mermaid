# Image registration using pyTorch

This is a project to integrate image registration algorithms with pyTorch.

The goal is to: 

1. Initially start with a stationary velocity field (SVF) implementation on images in 2D (done)
2. Create a map-based SVF implementation inspired by the map code of the spatial transformer networks (done)
3. Create an SVF model paramterized by momentum
4. Port everything from 2D to 3D (done)
5. Implement LDDMM using pyTorch (done for scalar- and vector-valued momentum)

# Setup

* To compile the spatial transformer code simply go to the pyreg/libraries directory and exectute 'python build.py'
* To run the code import set_pyreg_paths (which will take care of the paths for the libraries)
* The main registation code examples are in testRegistrationGeneric.py and testRegistrationGenericMultiscale.py
* To compile the documentation, simply execute 'make html' in the docs directory. This requires an installation of graphviz to render the inheritence diagrams. On OSX this can for example be intalled by 'brew install graphviz'. *This documentation also contains slightly more detailed installation instruction*

# TODO

There is a lot still to do. Here is an incomplete list:

* CUDA implementation of the spatial transformer parts (this can follow the existing 2D spatial transformer code in CUDA)
- image-smoothing based on CUDA (I wrote a pyTorch function to do it on the CPU in the Fourier domain)
- In general, make sure that everything can also run on the GPU (as most of it is standard pyTorch, this should hopefully not be too difficult)
- Write a more complete documentation (with sphinx and as part of the code)
- Extend the lBFGS optimizer so that it can deal with parameter groups (currently one one set of parameters seems to be supported; while this is sufficient for many standard registration tasks, it is not for more advanced models.)
- Add more tests
- Add some time-series models (this should be super-easy, because we no longer need to worry about the adjoint equations)
