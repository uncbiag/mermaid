
 <pre>
                                      _     _ 
                                     (_)   | |
  _ __ ___   ___ _ __ _ __ ___   __ _ _  __| |
 | '_ ` _ \ / _ \ '__| '_ ` _ \ / _` | |/ _` |
 | | | | | |  __/ |  | | | | | | (_| | | (_| |
 |_| |_| |_|\___|_|  |_| |_| |_|\__,_|_|\__,_|
                                                                                      
 </pre>                                       

# iMagE Registration via autoMAtIc Differentiation

Mermaid is a registration toolkit making use of automatic differentiation for rapid prototyping. It runs on the CPU and the GPU, though GPU acceleration only becomes obvious for large images or 3D volumes. 

The easiest way to install a development version is to clone the repository, create a virtual conda environment and install it in there. This can be done as follows:

```
conda create --name mermaid python=3.7 pip
conda activate mermaid
python setup.py develop
```

There is also a nice documentation which can be build by executing

```
cd mermaid
cd docs
make html
```

We are in the process of hosting this on readthedocs, but there are currently some memory issues which prevent running the sphinx script to create the documentation (any suggestions on how to fix this are welcome). In the meantime you can find it here:

http://wwwx.cs.unc.edu/~mn/mermaid/index.html

In the near future there will also be a conda installer available. This will then allow installations via

```
conda install -c pytorch -c conda-forge -c anaconda -c uncbiag mermaid=0.2.0
```

There are already initial OSX/Linux versions available which can be installed via conda, but there are still some issues that need to be ironed out, so they might not be fully functional yet. Stay tuned.

# easyreg

We also wrote a companion python package, easyreg, which allows training deep networks for image registration based on the registration models available in mermaid. I.e., easyreg allows training networks that backpropagate through the mermaid transformation models (SVF, LDDMM, ...). You can have a look at the package here:

https://github.com/uncbiag/easyreg

