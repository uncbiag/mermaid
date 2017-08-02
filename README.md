# Image registration using pyTorch

This is a project to integrate image registration algorithms with pyTorch.

The goal is to: 

1. Initially start with a stationary velocity field (SVF) implementation on images in 2D (done)
2. Create a map-based SVF implementation inspired by the map code of the spatial transformer networks (done)
3. Create an SVF model paramterized by momentum
4. Port everything from 2D to 3D (done)
5. Implement LDDMM using pyTorch (done for scalar- and vector-valued momentum)

# Setup

* To compile the spatial transformer code simply go to the pyreg/libraries directory and exectute 'python build.by'
* To run the code import set_pyreg_paths (which will take care of the paths for the libraries)
* The main registation code example is in testRegistrationGeneric.py
* To compile the documentation, simply execute 'make html' in the docs directory. This requires an installation of graphviz to render the inheritence diagrams. On OSX this can for example be intalled by 'brew install graphviz'
