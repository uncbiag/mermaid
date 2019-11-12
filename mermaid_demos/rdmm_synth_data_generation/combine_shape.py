from __future__ import print_function
from __future__ import absolute_import
import matplotlib as matplt
matplt.use('Agg')
import numpy as np
import random as rd
from skimage.draw._random_shapes import _generate_random_colors
from .create_circle import Circle
from .create_ellipse import Ellipse
from .create_triangle import Triangle
from .create_rect import Rectangle
from .create_poly import Poly
from .utils_for_general import convert_index_into_coord_set,normalize_image_into_standard_form
object_table = {'circle':Circle,'ellipse':Ellipse,'tri':Triangle,'rect':Rectangle,'poly':Poly}
default_deform_table={'ellipse':{'radius_x':[0.6,0.8],'radius_y':[0.6,0.8]},
                      'tri':{'height':[0.6,0.8],'width':[0.6,0.8],},'rect':{'height':[0.4,0.8],'width':[0.4,0.8]}}
import matplotlib.pyplot as plt






def generate_shape_objects(object_settings):
    objects = []
    for settings in object_settings:
        object_type = settings['type']
        object = object_table[object_type](settings)
        objects.append(object)
    return objects




def randomize_shape_setting(img_sz,shape_type,center_pos=None,scale=1.,var_center_pos=0.1,random=None):
    random = random if random else np.random
    rotation = random.uniform(-60,60)
    cp_rx, cp_ry = [random.uniform(-var_center_pos,var_center_pos) for _ in range(2)]
    setting=None
    if shape_type=='ellipse':
        radius = [random.uniform(*default_deform_table['ellipse']['radius_x']),random.uniform(*default_deform_table['ellipse']['radius_y'])]
        radius = [r*scale for r in radius]
        max_radius = max(radius)
        center_pos = [center_pos[0] +cp_rx*max_radius,center_pos[1] +cp_ry*max_radius ] if center_pos else [random.uniform(*[max_radius-1,1-max_radius]) for _ in range(2)]
        setting= dict(name='c',type='ellipse', img_sz= img_sz, center_pos=center_pos,radius=radius,rotation=rotation)
    if shape_type=='tri':
        height, width = random.uniform(*default_deform_table['tri']['height']),random.uniform(*default_deform_table['tri']['width'])
        height, width = height*scale, width*scale
        max_len = max([height,width])
        center_pos = [center_pos[0] +cp_rx*max_len,center_pos[1] +cp_ry*max_len ]if center_pos else [random.uniform(*[max_len/2-1,1-max_len/2]) for _ in range(2)]
        setting= dict(name='t',type='tri', img_sz= img_sz, center_pos=center_pos,height=height,width=width,rotation=rotation)
    if shape_type=='rect':
        height, width = random.uniform(*default_deform_table['rect']['height']),random.uniform(*default_deform_table['rect']['width'])
        height, width = height*scale, width*scale
        max_len = max([height,width])
        center_pos =[center_pos[0] +cp_rx*max_len,center_pos[1] +cp_ry*max_len ] if center_pos else [random.uniform(*[max_len/2-1,1-max_len/2]) for _ in range(2)]
        setting= dict(name='r',type='rect', img_sz= img_sz, center_pos=center_pos,height=height,width=width,rotation=rotation)
    return setting,center_pos


class Count():
    def __init__(self,limit=30):
        self.count=0
        self.limit=30
    def inc(self):
        self.count +=1
    def over_count(self):
        return self.count>self.limit








def get_image(img_sz,shape_setting_list,color_info=None,visual=False):
    shape_list = generate_shape_objects(shape_setting_list)
    img,color = generate_image(img_sz,shape_list,multichannel=False,num_channels=3,color_info=color_info, random_seed=32,overlay=True)
    if visual:
        plt.style.use('bmh')
        fig, ax = plt.subplots()
        plt.imshow(img,alpha =0.8)
        #fig.patch.set_visible(False)
        ax.axis('off')
        plt.show()
    return normalize_image_into_standard_form(np.array(img)),color









def randomize_pair(img_sz,fg_setting,bg_setting, overlay_num=3,bg_num=7, random_state=None):
    """
    :param scale_dict: the scale dict provide the scale settings for the source image
    {'type':[circle,rectangle'],''scale':[0.8,0.9],'t_var':0.1}
    {'type':[rectangle',eclipse,triangle],'scale':[0.3,0.4],'t_var':0.05}
    :return: shape_list, image
    """
    random_state = random_state if random_state else np.random

    complete_flg=False

    fg_scale_range = fg_setting['scale']
    fg_type_range = fg_setting['type']
    fg_t_var = fg_setting['t_var']
    bg_scale_range = bg_setting['scale']
    bg_type_range = bg_setting['type']
    bg_t_var = bg_setting['t_var']
    sfg = None
    tfg = None
    s_setting_list=[]
    t_setting_list=[]
    s_list=[]
    t_list =[]
    overall_suspect = Count(20)


    avail = False
    while not avail:
        sfg_scale = random_state.uniform(fg_scale_range[0], fg_scale_range[1])
        tfg_scale = random_state.uniform(fg_scale_range[0], fg_scale_range[1])
        sfg_type = rd.sample(fg_type_range,1)[0]
        tfg_type = rd.sample(fg_type_range,1)[0]
        sfg_setting,sfg_cp = randomize_shape_setting(img_sz,sfg_type,scale=sfg_scale,random=random_state)
        tfg_setting,tfg_cp = randomize_shape_setting(img_sz,tfg_type,center_pos=sfg_cp,scale=tfg_scale,var_center_pos=fg_t_var, random=random_state)
        sfg = object_table[sfg_type](sfg_setting)
        tfg = object_table[tfg_type](tfg_setting)
        avail = sfg.avail and tfg.avail
    sfg.create_shape()
    tfg.create_shape()
    sfg_set = convert_index_into_coord_set(sfg.index)
    sfg_inshape_set = set()
    tfg_set = convert_index_into_coord_set(tfg.index)
    tfg_inshape_set = set()
    sfg_setting['use_weight']=True
    tfg_setting['use_weight']=True
    s_setting_list.append(sfg_setting)
    t_setting_list.append(tfg_setting)
    s_list.append(sfg)
    t_list.append(tfg)
    count_bg = 0
    count_ibg = 0
    while count_bg<overlay_num or count_ibg<bg_num :
        sbg_avail = False
        tbg_avail = False
        count = Count(30)
        while not sbg_avail:
            sbg_scale = random_state.uniform(bg_scale_range[0], bg_scale_range[1])
            sbg_type = rd.sample(bg_type_range, 1)[0]
            sbg_setting, sbg_cp = randomize_shape_setting(img_sz,sbg_type,scale=sbg_scale,random=random_state)
            sbg = object_table[sbg_type](sbg_setting)
            sbg_avail = sbg.avail
            if sbg.avail:
                sbg.create_shape()
                #sbg_set = convert_index_into_coord_set(sbg.index,expand=10)
                sbg_set_f = convert_index_into_coord_set(sbg.index,expand=10)
                sbg_set_i = convert_index_into_coord_set(sbg.index,expand=5)
                sbg_set_b = convert_index_into_coord_set(sbg.index,expand=5)
                sbg_fg_avail_f = len(sbg_set_f - sfg_set) == 0
                sbg_fg_avail_b = len(sbg_set_b - sfg_set) == len(sbg_set_b)
                sbg_inter_avail = len(sbg_set_i - sfg_inshape_set) == len(sbg_set_i)
                sbg_avail = ((sbg_fg_avail_f and count_bg<overlay_num) or (sbg_fg_avail_b and count_ibg<bg_num)) and sbg_inter_avail
            count.inc()
            if count.over_count():
                print("warning, over count when find {} foreground elem in source".format(count_bg))
                break
        if count.over_count():
            overall_suspect.inc()
            if overall_suspect.over_count():
                print("regenerate the sample:")
                break
            continue
        count = Count(30)
        while not tbg_avail:
            tbg_scale = random_state.uniform(bg_scale_range[0], bg_scale_range[1])
            tbg_type = rd.sample(bg_type_range, 1)[0]
            tbg_setting,_ = randomize_shape_setting(img_sz,tbg_type,center_pos=sbg_cp,scale=tbg_scale,var_center_pos=bg_t_var,random=random_state)
            tbg = object_table[tbg_type](tbg_setting)
            tbg_avail = tbg.avail
            if tbg.avail:
                tbg.create_shape()
                #tbg_set = convert_index_into_coord_set(tbg.index,expand=10)
                tbg_set_f = convert_index_into_coord_set(tbg.index,expand=10)
                tbg_set_i = convert_index_into_coord_set(tbg.index,expand=5)
                tbg_set_b = convert_index_into_coord_set(tbg.index,expand=5)
                tbg_fg_avail_f = len(tbg_set_f - tfg_set) == 0
                tbg_fg_avail_b = len(tbg_set_b - tfg_set) == len(tbg_set_b)
                tbg_inter_avail = len(tbg_set_i - tfg_inshape_set) == len(tbg_set_i)
                tbg_avail = ((tbg_fg_avail_f and count_bg<overlay_num) or (tbg_fg_avail_b and count_ibg<bg_num)) and tbg_inter_avail

            count.inc()
            if count.over_count():
                print("warning, over count when find {} foreground elem in target".format(count_bg))
                break
        if count.over_count():
            overall_suspect.inc()
            if overall_suspect.over_count():
                print("regenerate the sample:")
                break
            continue

        sfg_inshape_set = sfg_inshape_set.union(sbg_set_i)
        tfg_inshape_set = tfg_inshape_set.union(tbg_set_i)
        if sbg_fg_avail_f and count_bg<overlay_num:
            count_bg += 1
            sbg_setting['use_weight']=True
            tbg_setting['use_weight']=True
        if sbg_fg_avail_b and count_ibg<bg_num:
            count_ibg +=1
            sbg_setting['use_weight'] = False
            tbg_setting['use_weight'] = False
        s_setting_list.append(sbg_setting)
        t_setting_list.append(tbg_setting)
        s_list.append(sbg)
        t_list.append(tbg)
        complete_flg = True
    if complete_flg:
        get_image(img_sz, s_setting_list, color_info=None, visual=True)
        get_image(img_sz, t_setting_list, color_info=None, visual=True)
    return s_setting_list,t_setting_list,complete_flg









def generate_image(image_shape,
                  image_objects,
                  multichannel=True,
                  num_channels=3,
                  intensity_range=None,
                  random_seed=None,
                   color_info= None,
                   overlay= False):
    """Generate an image with random shapes, labeled with bounding boxes.

    The image is populated with random shapes with random sizes, random
    locations, and random colors, with or without overlap.

    Shapes have random (row, col) starting coordinates and random sizes bounded
    by `min_size` and `max_size`.

    Parameters
    ----------
    image_shape : tuple
        The number of rows and columns of the image to generate.
    multichannel : bool, optional
        If True, the generated image has ``num_channels`` color channels,
        otherwise generates grayscale image.
    num_channels : int, optional
        Number of channels in the generated image. If 1, generate monochrome
        images, else color images with multiple channels. Ignored if
        ``multichannel`` is set to False.
    intensity_range : {tuple of tuples of uint8, tuple of uint8}, optional
        The range of values to sample pixel values from. For grayscale images
        the format is (min, max). For multichannel - ((min, max),) if the
        ranges are equal across the channels, and ((min_0, max_0), ... (min_N, max_N))
        if they differ. As the function supports generation of uint8 arrays only,
        the maximum range is (0, 255). If None, set to (0, 254) for each
        channel reserving color of intensity = 255 for background.
    num_trials : int, optional
        How often to attempt to fit a shape into the image before skipping it.
    seed : int, optional
        Seed to initialize the random number generator.
        If `None`, a random seed from the operating system is used.

    Returns
    -------
    image : uint8 array
        An image with the fitted shapes.
    labels : list
        A list of labels, one per shape in the image. Each label is a
        (category, ((r0, r1), (c0, c1))) tuple specifying the category and
        bounding box coordinates of the shape.

"""


    if not multichannel:
        num_channels = 1
    if intensity_range is None:
        intensity_range = (0, 254) if num_channels == 1 else ((0, 254), )
    else:
        tmp = (intensity_range, ) if num_channels == 1 else intensity_range
        for intensity_pair in tmp:
            for intensity in intensity_pair:
                if not (0 <= intensity <= 255):
                    msg = 'Intensity range must lie within (0, 255) interval'
                    raise ValueError(msg)

    random = np.random.RandomState(random_seed)
    image_shape = (image_shape[0], image_shape[1], num_channels)
    image = np.ones(image_shape, dtype=np.uint8) * 0
    filled = np.zeros(image_shape, dtype=bool)
    num_shapes = len(image_objects)
    if color_info is None:
        colors = _generate_random_colors(num_shapes, num_channels,
                                         intensity_range, random)
    else:
        colors = color_info
    for shape_idx in range(num_shapes):
        indices = image_objects[shape_idx].create_shape()

        if not filled[indices].any() or overlay:
            image[indices] = colors[shape_idx]
            filled[indices] = True
            continue
    if not multichannel:
        image = np.squeeze(image, axis=2)
    return image, colors






























