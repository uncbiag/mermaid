"""
Computes the inverse of a map

Contributors:
  Marc Niethammer: mn@cs.unc.edu
"""
from __future__ import print_function

from builtins import str
from builtins import range
import set_pyreg_paths
import torch
from torch.nn.parameter import Parameter
import pyreg.utils as utils
from pyreg.data_wrapper import AdaptVal
import pyreg.fileio as fileio
import pyreg.custom_optimizers as CO

def invert_map(map,spacing):
    """
    Inverts the map and returns its inverse. Assumes standard map parameterization [-1,1]^d
    :param map: Input map to be inverted
    :return: inverted map
    """
    # make pytorch arrays for subsequent processing
    map_t = AdaptVal(torch.from_numpy(map))

    # identity map
    id = utils.identity_map_multiN(map_t.data.shape,spacing)
    id_t = AdaptVal(torch.from_numpy(id))

    # parameter to store the inverse map
    invmap_t = Parameter(AdaptVal(torch.from_numpy(id.copy())))

    # some optimizer settings, probably too strict
    nr_of_iterations = 200
    rel_ftol = 1e-8
    optimizer = CO.LBFGS_LS([invmap_t],lr=1, max_iter=1, tolerance_grad=rel_ftol * 10, tolerance_change=rel_ftol, max_eval=10,history_size=30, line_search_fn='backtracking')
    # optimizer = torch.optim.SGD([invmap_t], lr=0.0001, momentum=0.9, dampening=0, weight_decay=0,nesterov=True)
    # optimizer = torch.optim.Adam([invmap_t], lr=0.00001, betas=(0.9, 0.999), eps=rel_ftol, weight_decay=0)

    def compute_loss():
        # warps map_t with inv_map, if it is the inverse should result in the identity map
        wmap = utils.compute_warped_image_multiNC(map_t, invmap_t, spacing,3)
        current_loss = ((wmap-id_t)**2).sum()
        return current_loss

    def _closure():
        optimizer.zero_grad()
        loss = compute_loss()
        loss.backward()
        return loss

    last_loss = utils.t2np( compute_loss() )

    for iter in range(nr_of_iterations):
        optimizer.step(_closure )
        current_loss = utils.t2np( compute_loss() )
        print( 'Iter = ' + str( iter ) + '; E = ' + str( current_loss ) )
        if ( current_loss >= last_loss ):
            break
        else:
            last_loss = current_loss

    return utils.t2np( invmap_t )

if __name__ == "__main__":
    # execute this as a script

    import argparse

    print('WARNING: TODO: need to add support for different spline orders for image warping!! (I.e., support for params for compute_warped_image_multiNC')

    parser = argparse.ArgumentParser(description='Invert map')

    required = parser.add_argument_group('required arguments')
    required.add_argument('--map', required=True, help='Map that should be inverted')
    required.add_argument('--imap', required=True, help='Resulting inverted map')

    args = parser.parse_args()
    map_filename = args.map
    imap_filename = args.imap

    map, map_hdr = fileio.MapIO().read(map_filename)
    spacing = map_hdr['spacing']
    #TODO: check that the spacing is correct here
    imap = invert_map(map,spacing)

    # now write it out
    print( 'Writing inverted map to file: ' + imap_filename )
    fileio.MapIO().write(imap_filename, imap, map_hdr)



