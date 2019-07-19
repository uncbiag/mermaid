import  numpy as np
from mermaid.data_wrapper import MyTensor
class MovingShape():
    def __init__(self,shape,multi_gaussian_weights,using_weight=True, weight_type='w_K_w'):
        self.shape = shape
        self.multi_gaussian_weights = multi_gaussian_weights if weight_type!='w_K_w' else np.sqrt(multi_gaussian_weights)
        self.using_weight =using_weight
        self.index = shape.index if shape.index is not None else shape.create_shape()


class MovingShapes():
    def __init__(self,img_sz,mov_shape_list, default_multi_gaussian_weights, local_smoother=None, weight_type='w_K_w'):
        self.mov_shapes = mov_shape_list
        self.img_sz = img_sz
        self.default_multi_gaussian_weights = default_multi_gaussian_weights if weight_type!='w_K_w' else np.sqrt(default_multi_gaussian_weights)
        self.local_smoother = local_smoother


    def create_weight_map(self):
        sz = self.img_sz
        default_multi_gaussian_weights = self.default_multi_gaussian_weights
        nr_of_mg_weights = len(default_multi_gaussian_weights)
        sh_weights = [1] + [nr_of_mg_weights] + list(sz)
        weights = np.zeros(sh_weights, dtype='float32')
        for g in range(nr_of_mg_weights):
            weights[:, g, ...] = default_multi_gaussian_weights[g]

        for shape in self.mov_shapes:
            if shape.using_weight:
                x_index = shape.index[0]
                y_index = shape.index[1]
                for g in range(nr_of_mg_weights):
                    weights[:,g,x_index,y_index] = shape.multi_gaussian_weights[g]
        weights = MyTensor(weights)
        sm_weight = self.local_smoother.smooth(weights)
        return sm_weight