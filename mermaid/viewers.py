"""
Implements various viewers to display 3D data
"""
from __future__ import print_function

from builtins import str
from builtins import range
from builtins import object
from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from future.utils import with_metaclass


def _create_some_test_data():
    a = np.sin(np.linspace(0, np.pi, 20))
    b = np.sin(np.linspace(0, np.pi*5, 20))
    data = np.outer(a, b)[..., np.newaxis] * a
    return data


def _print_debug(str, flag=False):
    if flag:
        print(str)


class FigureEventHandler(object):
    """Class to implement general event handling for matplotlib figures.
    In particular this class allows for easy event handling within different subplots.
    """

    def __init__(self, fig):
        """
        Constructor
        
        :param fig: figure handle 
        """
        self.fig = fig
        """figure handle"""
        self._remove_keymap_conflicts({'j', 'k'})
        """conflicts with keys that should be removed"""
        self.ax_events = dict()
        """dictionary that keeps track of all the events"""
        self.ax_events['button_press_event']=[]
        self.ax_events['button_release_event']=[]
        self.ax_events['key_press_event']=[]
        self.ax_events['key_release_event']=[]
        self.supported_events = ['button_press_event','button_release_event',
                                 'key_press_event','key_release_event']
        """events that are currently supported by the figure event handler"""
        self.sync_d = dict()
        """dictionary to hold information about synchronization of subplots (for synchronized slicing)"""

        self.connect()

    def reset_synchronize(self):
        """
        Removes all the subplot synchronizations
        """
        self.sync_d.clear()

    def synchronize(self,axes):
        """
        Sets synchornization information (i.e., which axes need to be synchronized)
        
        :param axes: list of axes
        """
        for e in axes:
            self.sync_d[e]=axes

    def _is_supported_event(self, eventname):
        if self.supported_events.count(eventname)>0:
            return True
        else:
            return False

    def _remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def connect(self):
        """
        Connect to all the events
        """
        'connect to all the events we need'
        self.cidbuttonpress = self.fig.canvas.mpl_connect(
            'button_press_event', self._on_mouse_press)
        self.cidbuttonrelease = self.fig.canvas.mpl_connect(
            'button_release_event', self._on_mouse_release)
        self.cidkeypress = self.fig.canvas.mpl_connect(
            'key_press_event', self._on_key_press)
        self.cidkeyrelease = self.fig.canvas.mpl_connect(
            'key_release_event', self._on_key_release)

    def disconnect(self):
        """
        Disconnect from all events
        """
        'disconnect all the stored connection ids'
        self.fig.canvas.mpl_disconnect(self._on_mouse_press)
        self.fig.canvas.mpl_disconnect(self._on_mouse_release)
        self.fig.canvas.mpl_disconnect(self._on_key_press)
        self.fig.canvas.mpl_disconnect(self._on_key_release)

    def _event_is_registered(self, eventname, ax):
        if eventname in self.ax_events:
            registeredEvents = self.ax_events[eventname]
            for e in registeredEvents:
                if e[0] is ax:
                    return True
        return False

    def add_axes_event(self, eventname, ax, eventfcn, getsyncfcn=None,setsyncfcn=None):
        """
        Associates events with a particular axis
        
        :param eventname: event name: 'button_press_event', not yet supported: 'button_release_event', 'key_press_event', 'key_release_event'
        :param ax: axis handle
        :param eventfcn: function that should be called
        :param getsyncfcn: function that returns synchronization information
        :param setsyncfcn: function that takes synchronization information
        """
        if self._is_supported_event(eventname):
            _print_debug('Registering an event')
            if not self._event_is_registered(eventname, ax):
                self.ax_events[eventname].append((ax,eventfcn,getsyncfcn,setsyncfcn))
        else:
            print('Event ' + eventname + ' is not supported and cannot be registered')

    def remove_axes_event(self, eventname, ax):
        """
        Removes an event from an axis
        
        :param eventname: event name: 'button_press_event', not yet supported: 'button_release_event', 'key_press_event', 'key_release_event'
        :param ax: axis handle
        """
        if self._is_supported_event(eventname):
            _print_debug('Removing an event ... ')
            if eventname in self.ax_events:
                registeredEvents = self.ax_events[eventname]
                for e in registeredEvents:
                    if e[0] is ax:
                        registeredEvents.remove( e )
                        _print_debug('Removed!')
                        return
        else:
            print('Event ' + eventname + ' is not supported and cannot be removed')

    def _broadcast(self, broadCastTo, syncInfo, eventName):
        registeredEvents = self.ax_events[eventName]
        for e in registeredEvents:
            for b in broadCastTo:
                if b is e[0]: # found the axes
                    if e[3] is not None:
                        e[3](syncInfo) # now sync it

    def _on_mouse_press(self, event):
        _print_debug('Pressed mouse button')
        # get current axis
        if event.inaxes is not None:
            ax = event.inaxes
            registeredEvents = self.ax_events['button_press_event']
            # get corresponding event function and dispatch it
            for e in registeredEvents:
                if e[0] is ax:
                    _print_debug('Dispatching event')
                    e[1](event)

                    if e[0] in self.sync_d and e[2] is not None:
                        _print_debug('Broadcasting')
                        syncInfo = e[2]()
                        self._broadcast(self.sync_d[e[0]], syncInfo, 'button_press_event')

                    # and now draw the canvas
                    self.fig.canvas.draw()

    def _on_mouse_release(self, event):
        pass

    def _on_key_press(self, event):
        pass

    def _on_key_release(self, event):
        pass


class ImageViewer(with_metaclass(ABCMeta, object)):
    """
    Abstract class for an image viewer.
    """

    def __init__(self, ax, data ):
        """
        Constructor.
        
        :param ax: axis
        :param data: data to show
        """
        self.ax = ax
        self.data = data
        self.show()

    @abstractmethod
    def show(self):
        """
        Displays the image
        """
        pass

    def display_title(self):
        """
        Displays the title for a figure
        """
        pass


class ImageViewer3D(with_metaclass(ABCMeta, ImageViewer)):
    """
    Abstract class for a 3D image viewer.
    """

    def __init__(self, ax, data ):
        super(ImageViewer3D,self).__init__(ax,data)

    @abstractmethod
    def previous_slice(self):
        """
        display the previous slice
        """
        pass

    @abstractmethod
    def next_slice(self):
        """
        display the next slice
        """
        pass

    @abstractmethod
    def set_synchronize(self, index):
        """
        Synchronize to a particular slice
        
        :param index: slice index 
        """
        pass

    @abstractmethod
    def get_synchronize(self):
        """
        Get index to which should be synchronized
        
        :return: slice index 
        """
        pass

    def on_mouse_press(self, event):
        """
        Implements going forward and backward in slices depending based on clicking in the left or the right of an image
        
        :param event: event data
        """
        x = event.xdata
        y = event.ydata

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xrange = abs(xlim[0] - xlim[1])
        yrange = abs(ylim[0] - ylim[1])

        _print_debug('X = ' + str(x))
        _print_debug('Y = ' + str(y))
        _print_debug('xlim = ' + str(xlim))
        _print_debug('ylim = ' + str(ylim))

        if abs(y - ylim[0]) <= yrange:
            if abs(x - xlim[0]) <= 0.4 * xrange:
                _print_debug('Previous slice')
                self.previous_slice()
            elif abs(x - xlim[1]) <= 0.4 * xrange:
                _print_debug('Next slice')
                self.next_slice()

        self.display_title()


class ImageViewer3D_Sliced(ImageViewer3D):
    """
    3D image viewer specialization to 3D sliced viewing
    """

    def __init__(self, ax, data, sliceDim, textStr='Slice', showColorbar=False):
        """
        Constructor
        
        :param ax: axis 
        :param data: data to be displayed (3D image volume)
        :param sliceDim: dimension along which to slice
        :param textStr: text string that should be displayed
        :param showColorbar: (bool) should a colorbar be displayed? 
        """
        self.sliceDim = sliceDim
        """dimension to slice"""
        self.textStr = textStr
        """title string"""
        self.index = data.shape[self.sliceDim] // 2
        """slice index to display"""
        self.showColorbar = showColorbar
        """dispaly colorbar True/False"""
        super(ImageViewer3D_Sliced,self).__init__(ax,data)

    def _get_slice_at_dimension(self, index):
        # slicing a along a given dimension at index, index
        slc = [slice(None)] * len(self.data.shape)
        slc[self.sliceDim] = slice(index, index+1)
        return (self.data[slc]).squeeze()

    def previous_slice(self):
        """
        Display previous slice
        """
        plt.sca(self.ax)
        plt.cla()
        self.index = (self.index - 1) % self.data.shape[self.sliceDim]  # wrap around using %
        self.ax.imshow(self._get_slice_at_dimension(self.index))
        #self.ax.images[0].set_array(self.get_slice_at_dimension(self.index))

    def next_slice(self):
        """
        Display next slice
        """
        plt.sca(self.ax)
        plt.cla()
        self.index = (self.index + 1) % self.data.shape[self.sliceDim]
        self.ax.imshow(self._get_slice_at_dimension(self.index))
        #self.ax.images[0].set_array(self.get_slice_at_dimension(self.index))

    def set_synchronize(self, index):
        """
        Synchronize slice view to a particular slice
        
        :param index: slice index to synchronize to 
        """
        plt.sca(self.ax)
        plt.cla()
        self.index = (index) % self.data.shape[self.sliceDim]
        self.ax.imshow(self._get_slice_at_dimension(self.index))
        #self.ax.images[0].set_array(self.get_slice_at_dimension(self.index))
        self.display_title()

    def get_synchronize(self):
        """
        Get current slice index
        
        :return: current slice index
        """
        return self.index

    def display_title(self):
        """
        Display figure title
        """
        font = {'size': 10}
        plt.sca(self.ax)
        plt.title( self.textStr + ' = ' + str(self.index) + '/' + str(self.data.shape[self.sliceDim]-1),font )

    def show(self):
        """
        Show the current slice
        """

        plt.sca(self.ax)
        plt.cla()
        #print('debugging {}'.format(self.index), 'slice_dim{}'.format(self.sliceDim),'img_shape{}'.format(self.data.shape))
        cim = self.ax.imshow(self._get_slice_at_dimension(self.index))
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.gcf().colorbar(cim, cax=cax, orientation='vertical').ax.tick_params(labelsize=3)
        self.display_title()


class ImageViewer3D_Sliced_Contour(ImageViewer3D_Sliced):
    """
    Specialization of 3D sliced viewer to also display contours
    """
    def __init__(self, ax, data, phi, sliceDim, textStr='Slice', showColorbar=False):
        """
        Constructor
        :param ax: axis
        :param data: data (image array, XxYxZ)
        :param phi: map (dimxXxYxZ)
        :param sliceDim: slice dimension
        :param textStr: title string
        :param showColorbar: (bool) show colorbar
        """
        self.phi = phi
        """map"""
        super(ImageViewer3D_Sliced_Contour,self).__init__(ax,data, sliceDim, textStr, showColorbar)

    def get_phi_slice_at_dimension(self,index):
        """
        Get map (based on which we can draw contours) at a particular slice index
        
        :param index: slice index 
        :return: returns the map at this slice index
        """
        # slicing a along a given dimension at index, index
        slc = [slice(None)] * len(self.phi.shape)
        slc[self.sliceDim+1] = slice(index, index+1)
        return (self.phi[slc]).squeeze()

    def show_contours(self):
        """
        display the contours for a particular slice
        """
        plt.sca(self.ax)
        phiSliced = self.get_phi_slice_at_dimension(self.index)
        for d in range(0,self.sliceDim):
            plt.contour(phiSliced[d,:,:], np.linspace(-1,1,20),colors='r',linestyles='solid')
        for d in range(self.sliceDim+1,3):
            plt.contour(phiSliced[d,:,:], np.linspace(-1,1,20),colors='r',linestyles='solid')

    def previous_slice(self):
        """
        display previous slice
        """
        super(ImageViewer3D_Sliced_Contour,self).previous_slice()
        self.show_contours()

    def next_slice(self):
        """
        display next slice
        """
        super(ImageViewer3D_Sliced_Contour,self).next_slice()
        self.show_contours()

    def set_synchronize(self, index):
        """
        set slice to a particular index (to synchronize views)

        :param index: slice index 
        """
        super(ImageViewer3D_Sliced_Contour,self).set_synchronize(index)
        self.show_contours()

    def show(self):
        """
        Show the image with contours overlaid
        """
        super(ImageViewer3D_Sliced_Contour,self).show()
        self.show_contours()


def test_viewer():
    """
    simple test viewer
    """

    data = _create_some_test_data()
    fig,ax = plt.subplots(1,3)

    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    ivx = ImageViewer3D_Sliced( ax[0], data, 0, 'X slice')
    ivy = ImageViewer3D_Sliced( ax[1], data, 1, 'Y slice')
    ivz = ImageViewer3D_Sliced( ax[2], data, 2, 'Z slice')

    feh = FigureEventHandler(fig)

    feh.add_axes_event('button_press_event', ax[0], ivx.on_mouse_press)
    feh.add_axes_event('button_press_event', ax[1], ivy.on_mouse_press)
    feh.add_axes_event('button_press_event', ax[2], ivz.on_mouse_press)

    feh.synchronize([ax[0], ax[1], ax[2]])

    plt.show()
