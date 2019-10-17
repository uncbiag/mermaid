from builtins import range
import set_pyreg_paths
import mermaid.utils as utils
import matplotlib.pyplot as plt
import numpy as np

import mermaid.custom_pytorch_extensions as ce

dim = 2
nr = 19
sz = np.array([nr]*dim)
spacing = [1./(nr-1.)]*dim

dtype = 'float64'

cid = utils.centered_identity_map(sz, spacing, dtype=dtype)

sigma = 0.1
stds = sigma * np.ones(dim,dtype=dtype)
means = 0.0 * np.zeros(dim,dtype=dtype)

spatial_filter = utils.compute_normalized_gaussian(cid, means, stds)

maxIndex = np.unravel_index(np.argmax(spatial_filter), spatial_filter.shape)
maxValue = spatial_filter[maxIndex]
loc = np.where(spatial_filter == maxValue)
nrOfMaxValues = len(loc[0])
if nrOfMaxValues > 1:
    raise ValueError('Cannot enforce max symmetry as maximum is not unique')

spatial_filter_max_at_zero = np.roll(spatial_filter, -np.array(maxIndex),list(range(len(spatial_filter.shape))))

ce.symmetrize_filter_center_at_zero(spatial_filter_max_at_zero,renormalize=True)


F = np.fft.fftn(spatial_filter_max_at_zero, s=sz)



if dim==1:
    plt.subplot(2,1,1)
    plt.plot(cid[0,...],spatial_filter)

    plt.subplot(2,1,2)
    plt.plot(cid[0,...],spatial_filter_max_at_zero)
elif dim==2:
    plt.subplot(2,1,1)
    plt.imshow(spatial_filter)

    plt.subplot(2,1,2)
    plt.imshow(spatial_filter_max_at_zero)
else:
    raise ValueError('cannot visualize this dimension')

plt.show()


