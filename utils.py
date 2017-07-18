"""
Various utility functions
"""

# TODO: maybe reorganize them in a more meaningful way
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