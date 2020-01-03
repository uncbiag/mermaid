from mermaid_demos.rdmm_synth_data_generation.create_poly import Poly
import numpy as np
class Rectangle(Poly):
    def __init__(self,setting,scale=1.):
        name, img_sz, center_pos, height, width, rotation = setting['name'],setting['img_sz'], setting['center_pos'], setting['height'], setting['width'], setting['rotation']
        self.center_pos = center_pos
        height,width = self.rescale(height*2,width*2,scale)
        self.height = height
        self.width = width
        vertices = self.get_point()
        setting_for_poly = dict(name=name, img_sz=img_sz, vertices=vertices,rotation=rotation)
        super(Rectangle,self).__init__(setting_for_poly)
        self.name = setting['name']
        self.type='rect'
        self.shape_info = {'center_pos':center_pos, 'height':height,'width':width}


    def rescale(self,height,width,scale):
        return height*scale, width*scale



    def get_point(self):
        r= self.height
        c = self.width
        point = self.center_pos
        points = [[point[0]-r/2,point[0] + r/2,point[0] + r/2,point[0]-r/2],
                  [point[1]-c/2, point[1]-c/2,point[1] + c/2,point[1]+c/2]]
        points = np.array(points)
        return points