
 <pre>
                                      _     _ 
                                     (_)   | |
  _ __ ___   ___ _ __ _ __ ___   __ _ _  __| |
 | '_ ` _ \ / _ \ '__| '_ ` _ \ / _` | |/ _` |
 | | | | | |  __/ |  | | | | | | (_| | | (_| |
 |_| |_| |_|\___|_|  |_| |_| |_|\__,_|_|\__,_|
                                                                                      
 </pre>                                       

[![Documentation Status](https://readthedocs.org/projects/mermaid/badge/?version=latest)](https://mermaid.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/uncbiag/mermaid.svg?branch=master)](https://travis-ci.org/uncbiag/mermaid)
![](https://anaconda.org/uncbiag/mermaid/badges/platforms.svg)
![](https://anaconda.org/uncbiag/mermaid/badges/version.svg)
![](https://anaconda.org/uncbiag/mermaid/badges/license.svg)


# iMagE Registration via autoMAtIc Differentiation

Mermaid is a registration toolkit making use of automatic differentiation for rapid prototyping. It is written in [PyTorch](https://pytorch.org/) and runs on the CPU and the GPU. Though GPU acceleration only becomes obvious for large images or 3D volumes. It supports registration of 1D (functions), 2D, and 3D images.

The easiest way to install a development version is to clone the repository, create a virtual conda environment and install it in there. This can be done as follows for a development installation:

```
conda create --name mermaid python=3.7 pip
conda activate mermaid
python setup.py develop
```

Or like this if you want to do a standard installation of mermaid:

```
conda create --name mermaid python=3.7 pip
conda activate mermaid
python setup.py install
```

There is also a nice documentation which can be built by executing

```
cd mermaid
cd docs
make html
```

You can also find the latest version on readthedocs:

https://mermaid.readthedocs.io/en/latest/index.html

In the near future there will also be a conda installer available. This will then allow installations via

```
conda install -c pytorch -c conda-forge -c anaconda -c uncbiag mermaid
```

There are already initial OSX/Linux versions available which can be installed via conda, but there are still some issues that need to be ironed out, so they might not be fully functional yet. Stay tuned.

**Supported transformation models**:
* affine_map: map-based affine registration
* diffusion_map: displacement-based diffusion registration
* curvature_map: displacement-based curvature registration
* total_variation_map: displacement-based total variation registration
* svf_map: map-based stationary velocity field
* svf_image: image-based stationary velocity field
* svf_scalar_momentum_image: image-based stationary velocity field using the scalar momentum
* svf_scalar_momentum_map: map-based stationary velocity field using the scalar momentum
* svf_vector_momentum_image: image-based stationary velocity field using the vector momentum
* svf_vector_momentum_map: map-based stationary velocity field using the vector momentum
* lddmm_shooting_map: map-based shooting-based LDDMM using the vector momentum
* lddmm_shooting_image: image-based shooting-based LDDMM using the vector momentum
* lddmm_shooting_scalar_momentum_map: map-based shooting-based LDDMM using the scalar momentum
* lddmm_shooting_scalar_momentum_image: image-based shooting-based LDDMM using the scalar momentum
* lddmm_adapt_smoother_map: map-based shooting-based Region specific diffemorphic mapping, with a spatio-temporal regularizer
* svf_adapt_smoother_map: map-based shooting-based vSVF, with a spatio regularizer

**Supported similarity measures**:
* ssd: sum of squared differences
* ncc: normalize cross correlation
* ncc_positive: positive normalized cross-correlation
* ncc_negative: negative normalized cross-correlation
* lncc: localized normalized cross correlation (multi-scale)

**Supported solvers**:
* embedded RK4
* torchdiffeq: explicit_adams, fixed_adams, tsit5, dopri5, euler, midpoint, rk4

**Optimizer**:
* support single/multi-scale optimizer
* support SGD, l-BFGS and some limited support for adam

<hr>


# easyreg

We also wrote a companion python package, easyreg, which allows training deep networks for image registration based on the registration models available in mermaid. I.e., easyreg allows training networks that backpropagate through the mermaid transformation models (SVF, LDDMM, ...). You can have a look at the package here:

https://github.com/uncbiag/easyreg

# Our other registration work

See https://github.com/uncbiag/registration for an overview of other registration approaches of our group and a short summary of how the approaches relate.



