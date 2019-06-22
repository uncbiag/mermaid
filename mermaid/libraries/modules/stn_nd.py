"""
This package implements spatial transformations in 1D, 2D, and 3D.
This is needed for the map-based registrations for example.

.. todo::
  Implement CUDA version. There is already a 2D CUDA version available (in the source directory here).
  But it needs to be extended to 1D and 3D. We also make use of a different convention for images which needs
  to be accounted for, as we use the BxCxXxYxZ image format and BxdimxXxYxZ for the maps.
"""
#TODO

from torch.nn.modules.module import Module
###########TODO temporal comment for torch1 compatability
# from mermaid.libraries.functions.stn_nd import  STNFunction_ND_BCXYZ, STNFunction_ND_BCXYZ_Compile
# from mermaid.libraries.functions.nn_interpolation import get_nn_interpolationf
################################################################3
from ..functions.stn_nd import  STNFunction_ND_BCXYZ
from functools import partial
# class STN_ND(Module):
#     """
#     Legacy code for nD spatial transforms. Ignore for now. Implements spatial transforms, but in BXYZC format.
#     """
#     def __init__(self, dim):
#         super(STN_ND, self).__init__()
#         self.dim = dim
#         """spatial dimension"""
#         self.f = STNFunction_ND( self.dim )
#         """spatial transform function"""
#     def forward(self, input1, input2):
#         """
#         Simply returns the transformed input
#
#         :param input1: image in BCXYZ format
#         :param input2: map in BdimXYZ format
#         :return: returns the transformed image
#         """
#         return self.f(input1, input2)

class STN_ND_BCXYZ(Module):
    """
    Spatial transform code for nD spatial transoforms. Uses the BCXYZ image format.
    """
    def __init__(self, spacing, zero_boundary=False,use_bilinear=True,use_01_input=True,use_compile_version=False):
        super(STN_ND_BCXYZ, self).__init__()
        self.spacing = spacing
        """spatial dimension"""
        if use_compile_version:
            if use_bilinear:
                self.f = STNFunction_ND_BCXYZ_Compile(self.spacing,zero_boundary)
            else:
                self.f = partial(get_nn_interpolation,spacing = self.spacing)
        else:
            self.f = STNFunction_ND_BCXYZ( self.spacing,zero_boundary= zero_boundary,using_bilinear= use_bilinear,using_01_input = use_01_input)

        """spatial transform function"""
    def forward(self, input1, input2):
        """
       Simply returns the transformed input

       :param input1: image in BCXYZ format 
       :param input2: map in BdimXYZ format
       :return: returns the transformed image
       """
        return self.f(input1, input2)
