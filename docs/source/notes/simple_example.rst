.. todo::
   This documentation needs to be updated. For now have a look at testSimpleInterface.py in the demos directory for a simple example on how to compute registrations.

Simple example
==============

This is a very simple example demonstrating how to use *mermaid*. This example is also available as
*testMinimalRegistration.py*.

To start off we import some important modules. First the pyTorch modules and numpy.

.. code:: python

    # first do the torch imports
    import torch
    import numpy as np


Next let's import some of the *mermaid* modules contained in the *pyreg* directory

.. code:: python

    import set_pyreg_paths                  # import the required paths
    import mermaid.example_generation as eg   # load the module to generate examples
    import mermaid.module_parameters as pars  # load the module to support parameters
    import mermaid.multiscale_optimizer as MO # load the optimizer module (which also supports single scale optimization)


Now let's choose a model, specify if it uses a map for the solution (i.e., warps the source image via a map
instead of solving directly an advection equation for an image), specify the desired dimension (can be 1, 2, or 3 --
play around with it yourself), and pick a maximum number of iterations for the solver. We also create an empty
parameter structure, so that *mermaid* can keep track of the parameters used. These can be written out layer by using
:meth:`pars.write_JSON`, but we ignore this for this simple example.

.. code:: python

    modelName = 'lddmm_shooting_map'
    useMap = True
    mapLowResFactor = 1.
    dim = 2
    nrOfIterations = 500 # number of iterations for the optimizer
    params = pars.ParameterDict()


Now let's create some example data, including variables holding the image size (sz) and spacing information.
In reality, of course, you would simply load your own data, which should already come with spacing information
and size information (size is in BCXYZ format: batch size, number of channels, X, Y, and Z coordinates; or only X
coordinates in 1D or X and Y coordinates in 2D).

.. code:: python

    szEx = np.tile( 50, dim )         # size of the desired images: (sz)^dim
    I0,I1= eg.CreateSquares(dim).create_image_pair(szEx,params) # create a default image size with two sample squares
    sz = np.array(I0.shape)
    spacing = 1./(sz[2::]-1) # the first two dimensions are batch size and number of image channels


To be able to communicate with pyTorch's autograd functionality, let's make these images pyTorch variables.

.. code:: python

    # create the source and target image as pyTorch variables
    ISource = torch.from_numpy( I0.copy() )
    ITarget = torch.from_numpy( I1 )


Now we are ready to set up the optimizer and to optimize. By default some visual output will be created.
Close each figure for the optimizer to advance.

.. code:: python

    so = MO.SingleScaleRegistrationOptimizer(sz,spacing,useMap,mapLowResFactor,params)
    so.set_model(modelName)

    so.set_number_of_iterations(nrOfIterations)

    so.set_source_image(ISource)
    so.set_target_image(ITarget)

    # and now do the optimization
    so.optimize()


That's it. Pretty easy, no?

There are also now a few convenience functions to make everything even easier. So intead of manually creating optimizers and such, you can use the following functions

.. code:: python

   so = MO.SimpleSingleScaleRegistration(ISource,ITarget,spacing,params)
   so.register()

or

.. code:: python

   so = MO.SimpleMultiScaleRegistration(ISource,ITarget,spacing,params)
   so.register()

   
See *testMinimalSimpleRegistration.py* and *testMinimalSimpleRegistrationMultiScale.py* in the *demos* directory for details.


