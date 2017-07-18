import matplotlib.pyplot as plt
import utils
import sys

import finite_differences as fd

# some debugging output to show image gradients

def debugOutput_1d( I0, I1, spacing):
    # compute gradients
    fdnp = fd.FD_np(spacing)  # numpy finite differencing

    dx0 = fdnp.dXc(I0)
    dx1 = fdnp.dXc(I1)

    plt.figure(1)
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.subplot(221)
    plt.plot(I0)
    plt.title('I0')
    plt.subplot(222)
    plt.plot(dx0)
    plt.title('dI0/dx')

    plt.subplot(223)
    plt.plot(I1)
    plt.title('I1')
    plt.subplot(224)
    plt.plot(dx1)
    plt.title('dI1/dx')

    # plt.axis('tight')
    plt.show(block=False)

def debugOutput_2d( I0, I1, spacing):
    # compute gradients
    fdnp = fd.FD_np(spacing)  # numpy finite differencing

    dx0 = fdnp.dXc(I0)
    dy0 = fdnp.dYc(I0)

    dx1 = fdnp.dXc(I1)
    dy1 = fdnp.dYc(I1)

    plt.figure(1)
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.subplot(321)
    plt.imshow(I0)
    plt.title('I0')
    plt.subplot(323)
    plt.imshow(dx0)
    plt.subplot(325)
    plt.imshow(dy0)

    plt.subplot(322)
    plt.imshow(I1)
    plt.title('I1')
    plt.subplot(324)
    plt.imshow(dx1)
    plt.subplot(326)
    plt.imshow(dy1)
    # plt.axis('tight')
    plt.show(block=False)

def debugOutput_3d( I0, I1, spacing):
    # compute gradients
    fdnp = fd.FD_np(spacing)  # numpy finite differencing

    s = I0.shape
    c = s[2]/2

    dx0 = fdnp.dXc(I0)
    dy0 = fdnp.dYc(I0)
    dz0 = fdnp.dZc(I0)

    dx1 = fdnp.dXc(I1)
    dy1 = fdnp.dYc(I1)
    dz1 = fdnp.dZc(I1)

    plt.figure(1)
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.subplot(241)
    plt.imshow(I0[:,:,c])
    plt.title('I0')
    plt.subplot(242)
    plt.imshow(dx0[:,:,c])
    plt.title('dI0/dx')
    plt.subplot(243)
    plt.imshow(dy0[:,:,c])
    plt.title('dI0/dy')
    plt.subplot(244)
    plt.imshow(dz0[:, :, c])
    plt.title('dI0/dz')

    plt.subplot(245)
    plt.imshow(I1[:,:,c])
    plt.title('I1')
    plt.subplot(246)
    plt.imshow(dx1[:,:,c])
    plt.title('dI1/dx')
    plt.subplot(247)
    plt.imshow(dy1[:,:,c])
    plt.title('dI1/dy')
    plt.subplot(248)
    plt.imshow(dz1[:,:,c])
    plt.title('dI1/dz')

    # plt.axis('tight')
    plt.show(block=False)


def debugOutput( I0, I1, spacing ):

    dim = spacing.size

    if dim==1:
        debugOutput_1d( I0, I1, spacing )
    elif dim==2:
        debugOutput_2d( I0, I1, spacing )
    elif dim==3:
        debugOutput_3d( I0, I1, spacing )
    else:
        raise ValueError( 'Debug output only supported in 1D and 3D at the moment')


def showCurrentImages_1d( iS, iT, iW):
    plt.subplot(131)
    plt.plot(utils.t2np(iS))
    plt.title('source image')

    plt.subplot(132)
    plt.plot(utils.t2np(iT))
    plt.title('target image')

    plt.subplot(133)
    plt.plot(utils.t2np(iW))
    plt.plot(utils.t2np(iT),'g')
    plt.plot(utils.t2np(iS),'r')
    plt.title('warped image')

def showCurrentImages_2d( iS, iT, iW):

    plt.subplot(131)
    plt.imshow(utils.t2np(iS))
    plt.colorbar()
    plt.title('source image')

    plt.subplot(132)
    plt.imshow(utils.t2np(iT))
    plt.colorbar()
    plt.title('target image')

    plt.subplot(133)
    plt.imshow(utils.t2np(iW))
    plt.colorbar()
    plt.title('warped image')

def showCurrentImages_3d( iS, iT, iW):

    sz = iS.size()
    c = sz[2]/2

    plt.subplot(131)
    plt.imshow(utils.t2np(iS[:,:,c]))
    plt.colorbar()
    plt.title('source image')

    plt.subplot(132)
    plt.imshow(utils.t2np(iT[:,:,c]))
    plt.colorbar()
    plt.title('target image')

    plt.subplot(133)
    plt.imshow(utils.t2np(iW[:,:,c]))
    plt.colorbar()
    plt.title('warped image')


def showCurrentImages(iter,iS,iT,iW):
    """
    Show current 2D registration results in relation to the source and target images
    :param iter: iteration number
    :param iS: source image
    :param iT: target image
    :param iW: current warped image
    :return: no return arguments
    """

    plt.figure(1)
    plt.clf()

    plt.suptitle('Iteration = ' + str(iter))
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    dim = iS.ndimension()

    if dim==1:
        showCurrentImages_1d( iS, iT, iW )
    elif dim==2:
        showCurrentImages_2d( iS, iT, iW )
    elif dim==3:
        showCurrentImages_3d( iS, iT, iW )
    else:
        raise ValueError( 'Debug output only supported in 1D and 3D at the moment')

    plt.show(block=False)
    plt.draw_all(force=True)

    print( 'Click mouse to continue press any key to exit' )
    wasKeyPressed = plt.waitforbuttonpress()
    if wasKeyPressed:
        sys.exit()