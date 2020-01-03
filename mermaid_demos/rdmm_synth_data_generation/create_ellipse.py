from mermaid_demos.rdmm_synth_data_generation.shape import *


class Ellipse(Shape):
    def __init__(self,setting,scale=1.):
        name,img_sz, center_pos, radius,rotation = setting['name'],setting['img_sz'],setting['center_pos'],setting['radius'],setting['rotation']
        super(Ellipse,self).__init__(type='ellipse',img_sz=img_sz)
        self.name = name
        radius = self.rescale(radius,scale)
        self.radius = radius
        self.center_pos = center_pos
        self.rotation =self.get_standard_rotation(rotation)
        self.shape_info = {'radius':radius, 'center_pos':center_pos,'rotation':rotation}
        self.index = None
        self.scale_ratio_min = 0.5
        self.radius_ratio_min = 0.5
        self.scale = -1
        self.boundary_bias= 0.1
        self.avail = self.check_avail(radius,center_pos)

    def rescale(self,radius,scale):
        return [r*scale for r in radius]

    def check_avail(self,radius,center_pos):
        avail =all([-radius[0]+center_pos[0]>(-1+self.boundary_bias),
                    -radius[1]+center_pos[1]>(-1+self.boundary_bias),
                    radius[0]+center_pos[0]<(1-self.boundary_bias),
                    radius[1]+center_pos[1]<(1-self.boundary_bias)])
        return avail

    def verify_radius_ratio(self):
        ratio = self.radius[0]/self.radius[1]
        ratio = min(ratio,1/ratio)
        return ratio>self.radius_ratio_min

    def get_current_scale(self):
        self.scale = self.radius[0]*self.radius[1]

    def verifty_scale_ratio(self,scale_to_compare):
        current_scale = self.get_current_scale()
        scale_ratio = current_scale/scale_to_compare
        scale_ratio = min(scale_ratio,1./scale_ratio)
        return scale_ratio>self.scale_ratio_min

    def get_center_pos(self):
        return self.center_pos
    def get_radius(self):
        return self.radius

    def create_shape(self):
        center_pos = self.get_center_pos()
        radius = self.get_radius()
        center_pos = self.get_standard_coord(center_pos)
        radius = self.get_standard_length(radius)
        index = draw_ellipse(center_pos[0], center_pos[1], radius[0], radius[1], shape=None, rotation=self.rotation)
        self.index = index
        return index