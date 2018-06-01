# Spatial transformer code for 1D, 2D, and 3D

How to compile:

$ sh make_cuda.sh


Make sure the arch flag (‘-arch=‘) in make_cuda.sh matches the NVIDIA GPU architecture that the CUDA files will be compiled for. The default is '-arch=sm_61', older architectures might require different flags (e.g. -arch=sm_52). Corresponding arch flags can be looked up here:

http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/


This spatial transformer code is heavily inspired by the pyTorch spatial transformer code by Fei Xia found here:

https://github.com/fxia22/stn.pytorch

However, this code extends it so that now also 1D and 3D images can be spatially transformed, 2D and 3D cuda have also been implemented.

If you want to compile the code on osx with multithreading you will need to use gcc. You can install it with homebrew (brew install gcc)

Once installed set the following two environment variables

export CC=/usr/local/bin/gcc-7
export CXX=/usr/local/bin/g++-7

