.. todo::
   This documentation needs to be updated. 

How to write your own registration model
========================================

This note explains how to write your own registration model.
For simplicity we assume that the stationary velocity field registration (SVF) does not exist. Here we explain, step-by-step, how to recreate it. We also assume that a new similarity measure should be added. We will start with the similarity measure as it is the easiest to implement.

Writing a similarity measure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similarity measures are derived from :class:`SimilarityMeasureSingleImage`. To create a new similarity measure simply derive your own class from :class:`SimilarityMeasureSingleImage` and implement the method :meth:`compute_similarity`. Assume you want to rewrite the sum of squared difference SSD similarity measure then this class will look as follows:

.. code:: python

   class MySSD(SimilarityMeasureSingleImage):
       def compute_similarity(self,I0,I1,I0Source=None,phi=None):
           sigma = 0.1
           return ((I0 - I1) ** 2).sum() / (sigma**2) * self.volumeElement

Here, `self.volumeElement` is defined in the base class :class:`SimilarityMeasureSingleImage` and indicates the volume occupied by a pixel or voxel.

Note that the parameter I0 and I1 have the format of XxYxZ. Check out base class :class`SimilarityMeasure` if parameter format of BxCxXxYxZ is needed, where B means batch and C means Channel.

As the machinery to include the similarity measure into all available registration methods is rather heavy, there is a convenience method which can be accessed through the optimizer interface.

Assuming the parameter stucture being used is called `params` (a :class:`ParameterDict` object), we can first tell that we want to use our own similarity measure via

.. code:: python
   
   params['registration_model']['similarity_measure']['type'] = 'mySSD'  

Now, once we have a multi-scale optimizer

.. code:: python
   
   import mermaid.multiscale_optimizer as MO
   mo = MO.MultiScaleRegistrationOptimizer(modelName,sz,spacing,useMap,mapLowResFactor,params)


we can simply instruct it to use our new similarity measure

.. code:: python
    
   mo.add_similarity_measure('mySSD', MySSD)
   

This will propagate through all the registration models. Hence, all of them will instantly be able to use the new similarity measure.

Writing a new registration model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The goal of this package is to make writing new models as easy as possible, while still providing an as simple to
use package as possible. These are obviously somewhat contradictory goals. As a compromise, there is also a relatively
easy interface which allows definitions of new models without integrating them into the overall machinery.

Let's first import a few packages that are needed to write the new network module

.. code:: python

    import registration_networks as RN
    import utils
    import image_sampling as IS
    import rungekutta_integrators as RK
    import forward_models as FM
    import regularizer_factory as RF


A new network is derived from the abstract class :class:`RegistrationNet`. To create a working new class, it is required
to define the following methods:

- :meth:`create_registration_parameters`: To set up the registration parameters required by the model. Needs to be torch `Parameter` type as defined in `torch.autograd`
- :meth:`get_registration_parameter`: simply return the registration parameter
- :meth:`set_registration_paramters`: to set the parameters, will be needed by the multi-scale optimizer to propagate parameters from one level to the next.
- :meth:`create_integrator`: since we are dealing with time-dependent problems here, this is to set up (and return!) an integrator for the system that is to be solved.
- :meth:`forward`: this is the method where all the magic happens. I.e., where we solve the forward problem by integrating the model forward in time.
- :meth:`upsample_registration_parameters`: method to spatially upsample the registration parameters. Needs to be defined if the multi-scale solver should be used. Does not need to be defined when solving on a single scale.


Let's start with the simplest possible class first

.. code:: python

    class MySVFNet(RN.RegistrationNet):
        def __init__(self,sz,spacing,params):
            super(MySVFNet, self).__init__(sz,spacing,params)
            self.v = self.create_registration_parameters()
            self.integrator = self.create_integrator()

        def create_registration_parameters(self):
            return utils.create_ND_vector_field_parameter_multiN(self.sz[2::], self.nrOfImages)

        def get_registration_parameters(self):
            return self.v

        def set_registration_parameters(self, p, sz, spacing):
            self.v.data = p.data
            self.sz = sz
            self.spacing = spacing

        def create_integrator(self):
            cparams = self.params[('forward_model',{},'settings for the forward model')]
            advection = FM.AdvectImage(self.sz, self.spacing)
            return RK.RK4(advection.f, advection.u, self.v, cparams)

        def forward(self, I):
            I1 = self.integrator.solve([I], self.tFrom, self.tTo)
            return I1[0]


If desired (for the multi-scale optimizer), also define

.. code:: python

    def upsample_registration_parameters(self, desiredSz):
        sampler = IS.ResampleImage()
        vUpsampled,upsampled_spacing=sampler.upsample_image_to_size(self.v,self.spacing,desiredSz)
        return vUpsampled,upsampled_spacing


Lastly, we also need to define our own loss function. Loss functions are derived from :class:`RegistrationImageLoss` or
:class:`RegistrationMapLoss` depending on if the source image is warped directly or via a coordinate map. The only
method that needs to be defined is :meth:`compute_regularization_energy`. For the SVF model we just created this could
for example look like this

.. code:: python

    class MySVFImageLoss(RN.RegistrationImageLoss):
    def __init__(self,v,sz,spacing,params):
        super(MySVFImageLoss, self).__init__(sz,spacing,params)
        self.v = v
        cparams = params[('loss',{},'settings for the loss function')]
        self.regularizer = (RF.RegularizerFactory(self.spacing).
                            create_regularizer(cparams))

    def compute_regularization_energy(self, I0_source):
        return self.regularizer.compute_regularizer_multiN(self.v)


Now that the models are defined, we need to use them. Just as for the custom similarity measure above, we can
do this by adding it to the multi-scale solver and then setting it (to be used for the solution).

.. code:: python

    myModelName = 'mySVF'
    mo.add_model(myModelName,MySVFNet,MySVFImageLoss)
    mo.set_model(myModelName)


If desired, it is possible to choose a custom optimizer (the default is LBFGS, with some default settings).
The following selects `adam` as an optimizer and sets one of its optimization parameters. Any optimizer supported
by pyTorch works in principle. However, be advised that especially the shooting formulations for registration may
require reasonably sophisticated optimizers for convergence.

.. code:: python

    mo.set_optimizer(torch.optim.Adam)
    mo.set_optimizer_params(dict(lr=0.01))

By default visualization output is turned on. But this can be set manually by

.. code:: python

    mo.set_visualization(True)
    mo.set_visualize_step(10)

And again as before the model can then be solved

.. code:: python

    mo.set_source_image(ISource)
    mo.set_target_image(ITarget)

    mo.set_scale_factors([1.0, 0.5, 0.25])
    mo.set_number_of_iterations_per_scale([5, 10, 10])

    mo.optimize()
