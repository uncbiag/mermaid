"""
Various utility functions
"""

import numpy as np
import math

# TODO: maybe reorganize them in a more meaningful way

def identityMap(nrOfVals):
    dim = len(nrOfVals)
    if dim==1:
        id = np.mgrid[0:nrOfVals[0]]
    elif dim==2:
        id = np.mgrid[0:nrOfVals[0],0:nrOfVals[1]]
    elif dim==3:
        id = np.mgrid[0:nrOfVals[0],0:nrOfVals[1],0:nrOfVals[2]]
    else:
        raise ValueError('Only dimensions 1-3 are currently supported for the identity map')

    # now get it into range [-1,1]^d
    id = np.array( id.astype('float32') )
    if dim==1:
        id*=2./(nrOfVals[0]-1)
        id-=1
    else:
        for d in range(dim):
            id[d]*=2./(nrOfVals[d]-1)
            id[d]-=1

    return id

def getpar( params, key, defaultval ):
    if params.has_key( key ):
        return params[ key ]
    else:
        print( 'Using default value: ' + str( defaultval) + ' for key = ' + key )
        return defaultval

def t2np( v ):
    return (v.data).cpu().numpy()

def showI( I ):
    plt.figure(2)
    plt.imshow(t2np(I))
    plt.colorbar()
    plt.show()

def showV( v ):
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(t2np(v[:, :, 0]))
    plt.colorbar()
    plt.title('v[0]')

    plt.subplot(122)
    plt.imshow(t2np(v[:, :, 1]))
    plt.colorbar()
    plt.title('v[1]')
    plt.show()


def showI(I):
    plt.figure(2)
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.imshow(t2np(I))
    plt.colorbar()
    plt.show()
    plt.title('I')

def showVandI(v,I):
    plt.figure(1)
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.subplot(131)
    plt.imshow(t2np(I))
    plt.colorbar()
    plt.title('I')

    plt.subplot(132)
    plt.imshow(t2np(v[:, :, 0]))
    plt.colorbar()
    plt.title('v[0]')

    plt.subplot(133)
    plt.imshow(t2np(v[:, :, 1]))
    plt.colorbar()
    plt.title('v[1]')
    plt.show()