from skimage.draw import polygon as draw_polygon, circle as draw_circle, ellipse as draw_ellipse
import numpy as np
class Shape(object):
    def __init__(self,name='unamed',type=None,img_sz=None):
        self.name = name
        self.type = type
        self.shape_info = {}
        self.color_info = {}
        self.img_sz = img_sz
        self.index = None
        self.overlay=False

    def get_img_sz(self):
        return self.img_sz
    def get_index(self):
        return self.index

    def get_standard_rotation(self,rotation):
        if not ( rotation >=-180 and rotation <=180):
            raise ValueError("the rotation should between -180 and 180")
        standard_rotation = rotation/180.*np.pi
        return standard_rotation

    def get_shape_info(self):
        return self.shape_info

    def get_color_info(self):
        return self.color_info

    def get_numpy(self,data):
        if isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)
    def get_standard_coord(self,pos,syn=True,standard=1.):
        assert all([p>=-1 and p<=1 for p in pos])
        pos = self.get_numpy(pos)
        if syn==True:
            assert len(self.img_sz) == len(pos)
            img_sz = self.get_numpy(self.img_sz)
            standard_pos = img_sz*(0.5*pos+0.5)
        else:
            standard_pos = standard*(0.5*pos+0.5)
        return standard_pos


    def get_standard_length(self,length,syn=True,standard=1.):
        length = self.get_numpy(length)
        if syn is True:
            assert len(self.img_sz) == len(length)
            img_sz = self.get_numpy(self.img_sz)
            standard_length = img_sz/2.*length
        else:
            standard_length = standard/2 * length
        return standard_length