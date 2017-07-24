'''
Implements various viewers to visualize registration results in 1D, 2D, and 3D
'''

from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def create_some_test_data():
    a = np.sin(np.linspace(0, np.pi, 20))
    b = np.sin(np.linspace(0, np.pi*5, 20))
    data = np.outer(a, b)[..., np.newaxis] * a
    return data

def print_debug(str,flag=False):
    if flag:
        print(str)

class FigureEventHandler(object):

    def __init__(self, fig):
        self.fig = fig
        self.remove_keymap_conflicts({'j', 'k'})
        self.ax_events = dict()
        self.ax_events['button_press_event']=[]
        self.ax_events['button_release_event']=[]
        self.ax_events['key_press_event']=[]
        self.ax_events['key_release_event']=[]
        self.supported_events = ['button_press_event','button_release_event',
                                 'key_press_event','key_release_event']
        self.sync_d = dict()

        self.connect()

    def reset_synchronize(self):
        self.sync_d.clear()

    def synchronize(self,axes):
        for e in axes:
            self.sync_d[e]=axes

    def is_supported_event(self,eventname):
        if self.supported_events.count(eventname)>0:
            return True
        else:
            return False

    def remove_keymap_conflicts(self,new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def connect(self):
        'connect to all the events we need'
        self.cidbuttonpress = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_mouse_press)
        self.cidbuttonrelease = self.fig.canvas.mpl_connect(
            'button_release_event', self.on_mouse_release)
        self.cidkeypress = self.fig.canvas.mpl_connect(
            'key_press_event', self.on_key_press)
        self.cidkeyrelease = self.fig.canvas.mpl_connect(
            'key_release_event', self.on_key_release)

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.fig.canvas.mpl_disconnect(self.on_mouse_press)
        self.fig.canvas.mpl_disconnect(self.on_mouse_release)
        self.fig.canvas.mpl_disconnect(self.on_key_press)
        self.fig.canvas.mpl_disconnect(self.on_key_release)

    def eventIsRegistered(self,eventname,ax):
        if self.ax_events.has_key(eventname):
            registeredEvents = self.ax_events[eventname]
            for e in registeredEvents:
                if e[0] is ax:
                    return True
        return False

    def add_axes_event(self, eventname, ax, eventfcn, getsyncfcn=None,setsyncfcn=None):
        if self.is_supported_event(eventname):
            print_debug('Registering an event')
            if not self.eventIsRegistered(eventname,ax):
                self.ax_events[eventname].append((ax,eventfcn,getsyncfcn,setsyncfcn))
        else:
            print('Event ' + eventname + ' is not supported and cannot be registered')

    def remove_axes_event(self, eventname, ax):
        if self.is_supported_event(eventname):
            print_debug('Removing an event ... ')
            if self.ax_events.has_key(eventname):
                registeredEvents = self.ax_events[eventname]
                for e in registeredEvents:
                    if e[0] is ax:
                        registeredEvents.remove( e )
                        print_debug('Removed!')
                        return
        else:
            print('Event ' + eventname + ' is not supported and cannot be removed')

    def broadcast(self,broadCastTo,syncInfo,eventName):
        registeredEvents = self.ax_events[eventName]
        for e in registeredEvents:
            for b in broadCastTo:
                if b is e[0]: # found the axes
                    if e[3] is not None:
                        e[3](syncInfo) # now sync it

    def on_mouse_press(self, event):
        print_debug('Pressed mouse button')
        # get current axis
        if event.inaxes is not None:
            ax = event.inaxes
            registeredEvents = self.ax_events['button_press_event']
            # get corresponding event function and dispatch it
            for e in registeredEvents:
                if e[0] is ax:
                    print_debug( 'Dispatching event')
                    e[1](event)

                    if self.sync_d.has_key(e[0]) and e[2] is not None:
                        print_debug('Broadcasting')
                        syncInfo = e[2]()
                        self.broadcast( self.sync_d[e[0]], syncInfo, 'button_press_event')

                    # and now draw the canvas
                    self.fig.canvas.draw()

    def on_mouse_release(self, event):
        pass

    def on_key_press(self, event):
        pass

    def on_key_release(self, event):
        pass


class ImageViewer(object):
    __metaclass__ = ABCMeta

    def __init__(self, ax, data ):
        self.ax = ax
        self.data = data
        self.show()

    @abstractmethod
    def show(self):
        pass

    def displayTitle(self):
        pass

class ImageViewer3D(ImageViewer):

    __metaclass__ = ABCMeta

    def __init__(self, ax, data ):
        super(ImageViewer3D,self).__init__(ax,data)

    @abstractmethod
    def previous_slice(self):
        pass

    @abstractmethod
    def next_slice(self):
        pass

    @abstractmethod
    def set_synchronize(self,index):
        pass

    @abstractmethod
    def get_synchronize(self):
        pass

    def on_mouse_release(self,event):
        x = event.xdata
        y = event.ydata

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xrange = abs(xlim[0] - xlim[1])
        yrange = abs(ylim[0] - ylim[1])

        print_debug('X = ' + str(x))
        print_debug('Y = ' + str(y))
        print_debug('xlim = ' + str(xlim))
        print_debug('ylim = ' + str(ylim))

        if abs(y - ylim[0]) <= yrange:
            if abs(x - xlim[0]) <= 0.4 * xrange:
                print_debug('Previous slice')
                self.previous_slice()
            elif abs(x - xlim[1]) <= 0.4 * xrange:
                print_debug('Next slice')
                self.next_slice()

        self.displayTitle()

class ImageViewer3D_Sliced(ImageViewer3D):

    def __init__(self, ax, data, sliceDim, textStr='Slice', showColorbar=False):
        self.sliceDim = sliceDim
        self.textStr = textStr
        self.index = data.shape[self.sliceDim] // 2
        self.showColorbar = showColorbar
        super(ImageViewer3D_Sliced,self).__init__(ax,data)

    def get_slice_at_dimension(self,index):
        # slicing a along a given dimension at index, index
        slc = [slice(None)] * len(self.data.shape)
        slc[self.sliceDim] = slice(index, index+1)
        return (self.data[slc]).squeeze()

    def previous_slice(self):
        plt.sca(self.ax)
        plt.cla()
        self.index = (self.index - 1) % self.data.shape[self.sliceDim]  # wrap around using %
        self.ax.imshow(self.get_slice_at_dimension(self.index))
        #self.ax.images[0].set_array(self.get_slice_at_dimension(self.index))

    def next_slice(self):
        plt.sca(self.ax)
        plt.cla()
        self.index = (self.index + 1) % self.data.shape[self.sliceDim]
        self.ax.imshow(self.get_slice_at_dimension(self.index))
        #self.ax.images[0].set_array(self.get_slice_at_dimension(self.index))

    def set_synchronize(self,index):
        plt.sca(self.ax)
        plt.cla()
        self.index = (index) % self.data.shape[self.sliceDim]
        self.ax.imshow(self.get_slice_at_dimension(self.index))
        #self.ax.images[0].set_array(self.get_slice_at_dimension(self.index))
        self.displayTitle()

    def get_synchronize(self):
        return self.index

    def displayTitle(self):
        plt.sca(self.ax)
        plt.title( self.textStr + ' = ' + str(self.index) + '/' + str(self.data.shape[self.sliceDim]-1) )

    def show(self):
        plt.sca(self.ax)
        plt.cla()
        cim = self.ax.imshow(self.get_slice_at_dimension(self.index))
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.gcf().colorbar(cim, cax=cax, orientation='vertical')
        self.displayTitle()


class ImageViewer3D_Sliced_Contour(ImageViewer3D_Sliced):
    def __init__(self, ax, data, phi, sliceDim, textStr='Slice', showColorbar=False):
        self.phi = phi
        super(ImageViewer3D_Sliced_Contour,self).__init__(ax,data, sliceDim, textStr, showColorbar)

    def get_phi_slice_at_dimension(self,index):
        # slicing a along a given dimension at index, index
        slc = [slice(None)] * len(self.phi.shape)
        slc[self.sliceDim] = slice(index, index+1)
        return (self.phi[slc]).squeeze()

    def show_contours(self):
        plt.sca(self.ax)
        phiSliced = self.get_phi_slice_at_dimension(self.index)
        for d in range(0,self.sliceDim):
            plt.contour(phiSliced[:,:,d], np.linspace(-1,1,20))
        for d in range(self.sliceDim+1,3):
            plt.contour(phiSliced[:,:,d], np.linspace(-1,1,20))

    def previous_slice(self):
        super(ImageViewer3D_Sliced_Contour,self).previous_slice()
        self.show_contours()

    def next_slice(self):
        super(ImageViewer3D_Sliced_Contour,self).next_slice()
        self.show_contours()

    def set_synchronize(self,index):
        super(ImageViewer3D_Sliced_Contour,self).set_synchronize(index)
        self.show_contours()

    def show(self):
        super(ImageViewer3D_Sliced_Contour,self).show()
        self.show_contours()


def test_viewer():
    data = create_some_test_data()
    fig,ax = plt.subplots(1,3)

    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    ivx = ImageViewer3D_Sliced( ax[0], data, 0, 'X slice')
    ivy = ImageViewer3D_Sliced( ax[1], data, 1, 'Y slice')
    ivz = ImageViewer3D_Sliced( ax[2], data, 2, 'Z slice')

    feh = FigureEventHandler(fig)

    feh.add_axes_event('button_press_event',ax[0],ivx.on_mouse_release)
    feh.add_axes_event('button_press_event',ax[1],ivy.on_mouse_release)
    feh.add_axes_event('button_press_event',ax[2],ivz.on_mouse_release)

    feh.synchronize([ax[0], ax[1], ax[2]])

    plt.show()
