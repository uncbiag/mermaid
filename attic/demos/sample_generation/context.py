import os
import numpy as np
import torch
from mermaid import module_parameters as pars
##############################################################################3
"""
Description:
the classes include:
Transfrom:  a series supported transform ( svf, lddmm)
Shape : a single shape object which has already linear transformed, include ( type, intensity,color, affine_param, area),   function related: generate/get_shape, get shape_area index
MovingShape : a single shape object inherit from Shape, should include momentum and affine information , function related: generate/get_momentum, get_momentum_area_index, get_momentum
CombinedShape: a object combined Moving Shape, function related: combined_shape, combined_momentum 
CombinedMovingShape: 1. moving combinedShape object by combined momentum
                    2. moving  MovingShape objects separately then combined their shape together ( prefer)

"""



#############################################################################