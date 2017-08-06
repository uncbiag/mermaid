How to write your own registration model
========================================

This note explains how to write your own registration model.
For simplicity we assume that the stationary velocity field registration (SVF) does not exist. Here we explain, step-by-step, how to recreate it. We also assume that a new similarity measure should be added. We will start with the similarity measure as it is the easiest to implement.

Writing a similarity measure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similarity measures are derived from :class:`SimilarityMeasure`. To create a new similarity measure simply derive your own class from :class:`SimilarityMeasure` and implement the method `computeSimilarity`. Assume you want to rewrite the sum of squared difference SSD similarity measure then

.. code::

   class MySSD(SimilarityMeasure):
     def computeSimilarity(self,I0,I1):
       
   
