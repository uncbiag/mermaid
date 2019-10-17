import matplotlib.pyplot as plt

def pd(analytical,numerical,clim):
    plt.subplot(131)
    plt.imshow(analytical[0].numpy(),clim=clim)
    plt.title('Analytical')
    plt.subplot(132)
    plt.imshow(numerical[0].numpy(),clim=clim)
    plt.title('Numerical')
    plt.subplot(133)
    plt.imshow(numerical[0].numpy()-analytical[0].numpy(),clim=clim)
    plt.title('Numerical-analytical')
    plt.show()

def pdnl(analytical,numerical):
    plt.subplot(131)
    plt.imshow(analytical[0].numpy())
    plt.colorbar()
    plt.title('Analytical')
    plt.subplot(132)
    plt.imshow(numerical[0].numpy())
    plt.colorbar()
    plt.title('Numerical')
    plt.subplot(133)
    plt.imshow(numerical[0].numpy()-analytical[0].numpy())
    plt.colorbar()
    plt.title('Numerical-analytical')
    plt.show()
    
    
