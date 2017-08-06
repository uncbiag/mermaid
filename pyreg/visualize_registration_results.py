import matplotlib.pyplot as plt
import numpy as np

import utils
import viewers

import finite_differences as fd

# some debugging output to show image gradients

def debug_output_1d(I0, I1, spacing):
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

def debug_output_2d(I0, I1, spacing):
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

def debug_output_3d(I0, I1, spacing):
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


def debug_output(I0, I1, spacing):

    dim = spacing.size

    if dim==1:
        debug_output_1d(I0, I1, spacing)
    elif dim==2:
        debug_output_2d(I0, I1, spacing)
    elif dim==3:
        debug_output_3d(I0, I1, spacing)
    else:
        raise ValueError( 'Debug output only supported in 1D and 3D at the moment')

def show_current_images_1d(iS, iT, iW, iter, phiWarped):

    plt.suptitle('Iteration = ' + str(iter))
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.subplot(221)
    plt.plot(utils.t2np(iS))
    plt.title('source image')

    plt.subplot(222)
    plt.plot(utils.t2np(iT))
    plt.title('target image')

    plt.subplot(224)
    plt.plot(utils.t2np(iT),'g',linestyle=':')
    plt.plot(utils.t2np(iS),'r',linestyle='--')
    plt.plot(utils.t2np(iW),linestyle=':')
    plt.title('warped image')

    if phiWarped is not None:
        plt.subplot(223)
        plt.plot(utils.t2np(phiWarped[0,:]))
        plt.title('phi')

    plt.show()

def checkerboard_2d(I0,I1,nrOfTiles=8):
    sz = I0.shape
    tileSize = int( np.array(sz).min()/nrOfTiles )
    nrOfTileXH = int( np.ceil(sz[0]/tileSize)/2+1 )
    nrOfTileYH = int( np.ceil(sz[1]/tileSize)/2+1 )
    cb_grid = np.kron([[1, 0] * nrOfTileYH, [0, 1] * nrOfTileYH] *nrOfTileXH, np.ones((tileSize, tileSize)))
    # now cut it to the same size
    cb_grid=cb_grid[0:sz[0],0:sz[1]]
    cb_image = I0*cb_grid + I1*(1-cb_grid)
    return cb_image

def show_current_images_2d_no_map(iS, iT, iW, iter):
    plt.suptitle('Iteration = ' + str(iter))
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.subplot(221)
    plt.imshow(utils.t2np(iS),cmap='gray')
    plt.colorbar()
    plt.title('source image')

    plt.subplot(222)
    plt.imshow(utils.t2np(iT),cmap='gray')
    plt.colorbar()
    plt.title('target image')

    plt.subplot(223)
    plt.imshow(utils.t2np(iW),cmap='gray')

    plt.colorbar()
    plt.title('warped image')

    plt.subplot(224)
    plt.imshow(checkerboard_2d(utils.t2np(iW),utils.t2np(iT)),cmap='gray')
    plt.colorbar()
    plt.title('checkerboard')

    plt.show()

def show_current_images_2d_map(iS, iT, iW, iter, phiWarped):

    plt.suptitle('Iteration = ' + str(iter))
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.subplot(231)
    plt.imshow(utils.t2np(iS),cmap='gray')
    plt.colorbar()
    plt.title('source image')

    plt.subplot(232)
    plt.imshow(utils.t2np(iT),cmap='gray')
    plt.colorbar()
    plt.title('target image')

    plt.subplot(233)
    plt.imshow(utils.t2np(iW),cmap='gray')

    plt.colorbar()
    plt.title('warped image')

    plt.subplot(234)
    plt.imshow(checkerboard_2d(utils.t2np(iW), utils.t2np(iT)),cmap='gray')
    plt.colorbar()
    plt.title('checkerboard')

    if phiWarped is not None:
        plt.subplot(235)
        plt.imshow(utils.t2np(iW),cmap='gray')

        plt.contour(utils.t2np(phiWarped[0, :, :]), np.linspace(-1, 1, 20), colors='r', linestyles='solid')
        plt.contour(utils.t2np(phiWarped[1, :, :]), np.linspace(-1, 1, 20), colors='r', linestyles='solid')

        plt.colorbar()
        plt.title('warped image + grid')

    plt.show()


def show_current_images_2d(iS, iT, iW, iter, phiWarped):

    if phiWarped is not None:
        show_current_images_2d_map(iS, iT, iW, iter, phiWarped)
    else:
        show_current_images_2d_no_map(iS, iT, iW, iter)


def show_current_images_3d_simple(iS, iT, iW, iter, phiWarped):

    plt.suptitle('Iteration = ' + str(iter))
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    sz = iS.size()
    c = sz[2]/2

    plt.subplot(221)
    plt.imshow(utils.t2np(iS[:,:,c]))
    plt.colorbar()
    plt.title('source image')

    plt.subplot(222)
    plt.imshow(utils.t2np(iT[:,:,c]))
    plt.colorbar()
    plt.title('target image')

    plt.subplot(223)
    plt.imshow(utils.t2np(iW[:,:,c]))
    plt.colorbar()
    plt.title('warped image')

    if phiWarped is not None:

        plt.subplot(224)
        plt.imshow(utils.t2np(iW[:, :, c]))
        plt.contour(utils.t2np(phiWarped[0,:, :, c]), np.linspace(-1, 1, 20),colors='r',linestyles='solid')
        plt.contour(utils.t2np(phiWarped[1,:, :, c]), np.linspace(-1, 1, 20),colors='r',linestyles='solid')

        plt.colorbar()
        plt.title('warped image + grid')

    plt.show()


def show_current_images_3d(iS, iT, iW, iter, phiWarped):

    if phiWarped is not None:
        fig, ax = plt.subplots(4,3)
    else:
        fig, ax = plt.subplots(3,3)

    plt.suptitle('Iteration = ' + str(iter))
    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    ivsx = viewers.ImageViewer3D_Sliced(ax[0][0], utils.t2np(iS), 0, 'source X', True)
    ivsy = viewers.ImageViewer3D_Sliced(ax[0][1], utils.t2np(iS), 1, 'source Y', True)
    ivsz = viewers.ImageViewer3D_Sliced(ax[0][2], utils.t2np(iS), 2, 'source Z', True)

    ivtx = viewers.ImageViewer3D_Sliced(ax[1][0], utils.t2np(iT), 0, 'target X', True)
    ivty = viewers.ImageViewer3D_Sliced(ax[1][1], utils.t2np(iT), 1, 'target Y', True)
    ivtz = viewers.ImageViewer3D_Sliced(ax[1][2], utils.t2np(iT), 2, 'target Z', True)

    ivwx = viewers.ImageViewer3D_Sliced(ax[2][0], utils.t2np(iW), 0, 'warped X', True)
    ivwy = viewers.ImageViewer3D_Sliced(ax[2][1], utils.t2np(iW), 1, 'warped Y', True)
    ivwz = viewers.ImageViewer3D_Sliced(ax[2][2], utils.t2np(iW), 2, 'warped Z', True)

    if phiWarped is not None:
        ivwxc = viewers.ImageViewer3D_Sliced_Contour(ax[3][0], utils.t2np(iW), utils.t2np(phiWarped), 0, 'warped X', True)
        ivwyc = viewers.ImageViewer3D_Sliced_Contour(ax[3][1], utils.t2np(iW), utils.t2np(phiWarped), 1, 'warped Y', True)
        ivwzc = viewers.ImageViewer3D_Sliced_Contour(ax[3][2], utils.t2np(iW), utils.t2np(phiWarped), 2, 'warped Z', True)

    feh = viewers.FigureEventHandler(fig)

    feh.add_axes_event('button_press_event', ax[0][0], ivsx.on_mouse_release, ivsx.get_synchronize, ivsx.set_synchronize)
    feh.add_axes_event('button_press_event', ax[0][1], ivsy.on_mouse_release, ivsy.get_synchronize, ivsy.set_synchronize)
    feh.add_axes_event('button_press_event', ax[0][2], ivsz.on_mouse_release, ivsz.get_synchronize, ivsz.set_synchronize)

    feh.add_axes_event('button_press_event', ax[1][0], ivtx.on_mouse_release, ivtx.get_synchronize, ivtx.set_synchronize)
    feh.add_axes_event('button_press_event', ax[1][1], ivty.on_mouse_release, ivty.get_synchronize, ivty.set_synchronize)
    feh.add_axes_event('button_press_event', ax[1][2], ivtz.on_mouse_release, ivtz.get_synchronize, ivtz.set_synchronize)

    feh.add_axes_event('button_press_event', ax[2][0], ivwx.on_mouse_release, ivwx.get_synchronize, ivwx.set_synchronize)
    feh.add_axes_event('button_press_event', ax[2][1], ivwy.on_mouse_release, ivwy.get_synchronize, ivwy.set_synchronize)
    feh.add_axes_event('button_press_event', ax[2][2], ivwz.on_mouse_release, ivwz.get_synchronize, ivwz.set_synchronize)

    if phiWarped is not None:
        feh.add_axes_event('button_press_event', ax[3][0], ivwxc.on_mouse_release, ivwxc.get_synchronize, ivwxc.set_synchronize)
        feh.add_axes_event('button_press_event', ax[3][1], ivwyc.on_mouse_release, ivwyc.get_synchronize, ivwyc.set_synchronize)
        feh.add_axes_event('button_press_event', ax[3][2], ivwzc.on_mouse_release, ivwzc.get_synchronize, ivwzc.set_synchronize)

    if phiWarped is not None:
        feh.synchronize([ax[0][0], ax[1][0], ax[2][0], ax[3][0]])
        feh.synchronize([ax[0][1], ax[1][1], ax[2][1], ax[3][1]])
        feh.synchronize([ax[0][2], ax[1][2], ax[2][2], ax[3][2]])
    else:
        feh.synchronize([ax[0][0], ax[1][0], ax[2][0]])
        feh.synchronize([ax[0][1], ax[1][1], ax[2][1]])
        feh.synchronize([ax[0][2], ax[1][2], ax[2][2]])

    plt.show()


def show_current_images(iter, iS, iT, iW, phiWarped=None):
    """
    Show current 2D registration results in relation to the source and target images
    :param iter: iteration number
    :param iS: source image
    :param iT: target image
    :param iW: current warped image
    :return: no return arguments
    """

    dim = iS.ndimension()-2

    iSF = iS[0,0,...]
    iTF = iT[0,0,...]
    iWF = iW[0,0,...]

    if phiWarped is not None:
        pwF = phiWarped[0,...]
    else:
        pwF = None

    if dim==1:
        show_current_images_1d(iSF, iTF, iWF, iter, pwF)
    elif dim==2:
        show_current_images_2d(iSF, iTF, iWF, iter, pwF)
    elif dim==3:
        show_current_images_3d(iSF, iTF, iWF, iter, pwF)
    else:
        raise ValueError( 'Debug output only supported in 1D and 3D at the moment')

    '''
    plt.show(block=False)
    plt.draw_all(force=True)

    print( 'Click mouse to continue press any key to exit' )
    wasKeyPressed = plt.waitforbuttonpress()
    if wasKeyPressed:
        sys.exit()
    '''