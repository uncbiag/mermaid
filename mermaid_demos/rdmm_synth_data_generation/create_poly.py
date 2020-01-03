from mermaid_demos.rdmm_synth_data_generation.shape import *
import numpy as np

class Poly(Shape):
    def __init__(self, setting,scale=1.):
        img_sz, vertices,rotation = setting['img_sz'],setting['vertices'],setting['rotation']
        super(Poly,self).__init__(type='poly',img_sz=img_sz)
        self.name = setting['name']
        self.vertices = self.get_numpy(vertices)
        self.shape_info = {'vertices':vertices}
        self.rotation =self.get_standard_rotation(rotation)
        self.__rotate_shape()
        self.scale_ratio_min = 0.5
        self.boundary_bias= 0.1
        self.avail = self.check_avail(self.vertices)



    def __rotate_shape(self):
        rotation = self.rotation
        if rotation!=0:
            #rotation %= np.pi
            rotation = -rotation
            sin_alpha, cos_alpha = np.sin(rotation), np.cos(rotation)
            centered_x = self.vertices[0]-self.vertices[0].mean()  # x refers to height here
            centered_y = self.vertices[1]-self.vertices[1].mean()  # x refers to height here
            self.vertices = [centered_y*sin_alpha+ centered_x*cos_alpha+ self.vertices[0].mean(),
                             centered_y* cos_alpha- centered_x*sin_alpha+ self.vertices[1].mean()]

    def verify_scale(self,index):
        self.create_shape()
        scale_ratio = len(self.scale)/len(index)
        scale_ratio = min(scale_ratio,1/scale_ratio)
        return scale_ratio>self.scale_ratio_min

    def check_avail(self,vertice):
        return (vertice[0]>(-1+self.boundary_bias)).all() and (vertice[0]<(1-self.boundary_bias)).all() and (vertice[1]>(-1+self.boundary_bias)).all() and (vertice[1]<(1-self.boundary_bias)).all()




    def get_point(self):
        return self.vertices

    def create_shape(self):
        coord = Poly.get_point(self)
        coord_x = self.get_standard_coord(coord[0],syn=False,standard=self.img_sz[0])
        coord_y = self.get_standard_coord(coord[1],syn=False,standard= self.img_sz[1])
        index = draw_polygon(coord_x, coord_y)
        self.index = index
        return index