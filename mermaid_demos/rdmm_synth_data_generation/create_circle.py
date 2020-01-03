from __future__ import print_function
from __future__ import absolute_import
from mermaid_demos.rdmm_synth_data_generation.shape import *
from mermaid_demos.rdmm_synth_data_generation.create_ellipse import Ellipse

class Circle(Ellipse):
    def __init__(self, setting,scale=1.):
        name, img_sz, center_pos, radius = setting['name'],setting['img_sz'],setting['center_pos'],setting['radius']
        radius = self.rescale(radius, scale)
        setting_for_ellipse = dict(name=name,img_sz=img_sz,center_pos=center_pos,radius=[radius,radius],rotation=0.)
        super(Circle,self).__init__(setting_for_ellipse)
        self.name = setting['name']
        self.type = 'circle'
        self.shape_info = {'radius':radius, 'center_pos':center_pos}


    def rescale(self,radius,scale):
        return radius*scale


