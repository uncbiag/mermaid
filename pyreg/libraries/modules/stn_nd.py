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
from pyreg.libraries.functions.stn_nd import  STNFunction_ND_BCXYZ

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
    def __init__(self, spacing, zero_boundary=True):
        super(STN_ND_BCXYZ, self).__init__()
        self.spacing = spacing
        """spatial dimension"""
        self.f = STNFunction_ND_BCXYZ( self.spacing,zero_boundary)
        """spatial transform function"""
    def forward(self, input1, input2):
        """
       Simply returns the transformed input

       :param input1: image in BCXYZ format 
       :param input2: map in BdimXYZ format
       :return: returns the transformed image
       """
        return self.f(input1, input2)
