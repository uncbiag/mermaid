# Spatial transformer code for 1D, 2D, and 3D

How to compile:

$ sh make_cuda.sh


This spatial transformer code is heavily inspired by the pyTorch spatial transformer code by Fei Xia found here:

https://github.com/fxia22/stn.pytorch

However, this code extends it so that now also 1D and 3D images can be spatially transformed, 2D and 3D cuda have also been implemented.

If you want to compile the code on osx with multithreading you will need to use gcc. You can install it with homebrew (brew install gcc)

Once installed set the following two environment variables

export CC=/usr/local/bin/gcc-7
export CXX=/usr/local/bin/g++-7

