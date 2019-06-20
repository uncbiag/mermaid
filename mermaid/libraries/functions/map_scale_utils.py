# from builtins import range
import torch

def scale_map(map,spacing):
    """
    Scales the map to the [-1,1]^d format

    :param map: map in BxCxXxYxZ format
    :param spacing: spacing in XxYxZ format
    :return: returns the scaled map
    """
    sz = map.size()
    map_scaled = torch.zeros_like(map)
    ndim = len(spacing)

    # This is to compensate to get back to the [-1,1] mapping of the following form
    # id[d]*=2./(sz[d]-1)
    # id[d]-=1.

    for d in range(ndim):
        if sz[d+2] >1:
            map_scaled[:, d, ...] = map[:, d, ...] * (2. / (sz[d + 2] - 1.) / spacing[d]) - 1.
        else:
            map_scaled[:, d, ...] = map[:,d,...]

    return map_scaled

def scale_map_grad(grad_map,spacing):
    """
    Scales the gradient back
    :param grad_map: gradient (computed based on map normalized to [-1,1]
    :param spacing: spacing in XxYxZ format
    :return: n/a (overwrites grad_map; results in gradient based on original spacing)
    """

    # need to compensate for the rescaling of the gradient in the backward direction
    sz = grad_map.size()
    ndim = len(spacing)
    for d in range(ndim):
        #grad_map[:, d, ...] *= spacing[d] * (sz[d + 2] - 1) / 2.
        #grad_map[:, d, ...] *= (sz[d + 2] - 1)/2.
        if sz[d + 2] > 1:
            grad_map[:, d, ...]  *= (2. / (sz[d + 2] - 1.) / spacing[d])
        else:
            grad_map[:, d, ...] = grad_map[:, d, ...]

