Parameters
==========

Parameters are generally handled via a customized dicitionary-like class :class:`ParameterDict`.
To use it simply import the package `module_parameters`

.. code:: python

   import module_parameters as MP

You can done create a new parameter instance via

.. code:: python
   
    p = MP.ParameterDict()

It is possilbe to use this parameter dictionary similarly to a typical python dictionary. However, it also can be used to take care of default values and to keep track of parameter documentation.

For example, we can create *categories* (these are hierarchical levels to group parameters). The following command creates a category *registration_model*. This is indicated by assigning it a tupe composed of an empty dictionary ({}) and a comment about the purpose of the category.

.. code:: python

   p['registration_model'] = ({},'general settings for registration models')   

This can of course be done hierarchically

.. code:: python
   
    p['registration_model']['similarity_measure'] = ({},'settings for the similarity measures')

And we can then assign actual values

.. code:: python
   
    p['registration_model']['similarity_measure']['type']=('ssd','similarity measure type')

Of course, this is only half the functionality, as we typically want to retrieve values and specify default values in case a given key does not exist. For example,

.. code:: python
    
    p['registration_model'][('nrOfIterations',10,'number of iterations')]

asks for the parameter *nrOfIterations* of the category *registration_model*. It uses *10* as the default value (in case the parameter *nrOfIterations* cannot be found) and specifies a description of the parameter.

We can also create a new category with default values if it does not exist yet

.. code:: python

   p[('new_category',{},'this is a new category')]
   p[('registration_model',{},'this category already existed')]

And we can of course print everything if desired

.. code:: python
   
    print(p)

We can write everything out as JSON:

.. code:: python
    
    p.write_JSON('test_pars.json')
    p.write_JSON_comments('test_pars_comments.json')


The former command just writes out the settings that were used (and hence are ideal to look up what the parameters that were used for a particular registration model were). The latter command writes out an annotated configuration file that explains all the settings.

Lastly, we can of course also read settings by

.. code:: python

   p.load_JSON('test_pars.json)




