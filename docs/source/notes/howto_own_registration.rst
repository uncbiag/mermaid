How to write your own registration model
========================================

This note explains how to write your own registration model.
For simplicity we assume that the stationary velocity field registration (SVF) does not exist. Here we explain, step-by-step, how to recreate it. We also assume that a new similarity measure should be added. We will start with the similarity measure as it is the easiest to implement.

Writing a similarity measure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similarity measures are derived from :class:`SimilarityMeasure`. To create a new similarity measure simply derive your own class from :class:`SimilarityMeasure` and implement the method :meth:`compute_similarity`. Assume you want to rewrite the sum of squared difference SSD similarity measure then this class will look as follows:

.. code::

   class MySSD(SimilarityMeasure):
       def compute_similarity(self,I0,I1):
           sigma = 0.1
           return ((I0 - I1) ** 2).sum() / (sigma**2) * self.volumeElement

Here, `self.volumeElement` is defined in the base class :class:`SimilarityMeasure` and indicates the volume occupied by a pixel or voxel.

As the machinery to include the similarity measure into all available registration methods is rather heavy, there is a convenience method which can be accessed through the optimizer interface.

Assuming the parameter stucture being used is called `params` (a :class:`ParameterDict` object), we can first tell that we want to use our own similarity measure via

.. code::
   
   params['registration_model']['similarity_measure']['type'] = 'mySSD'  

Now, once we have a multi-scale optimizer

.. code::
   
   import pyreg.multiscale_optimizer as MO
   mo = MO.MultiScaleRegistrationOptimizer(modelName,sz,spacing,useMap,params)


we can simply instruct it to use our new similarity measure

.. code::
    
   mo.add_similarity_measure('mySSD', MySSD)
   

This will propagate through all the registration models. Hence, all of them will instantly be able to use the new similarity measure.

   
