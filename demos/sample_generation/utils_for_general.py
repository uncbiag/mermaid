import torch
import numpy as np
import mermaid.pyreg.module_parameters as pars
import mermaid.pyreg.smoother_factory as sf
from mermaid.pyreg.data_wrapper import AdaptVal
def get_parameter_value(command_line_par,params, params_name, default_val, params_description):

    if command_line_par is None:
        ret = params[(params_name, default_val, params_description)]
    else:
        params[params_name]=command_line_par
        ret = command_line_par

    return ret

def get_parameter_value_flag(command_line_par,params, params_name, default_val, params_description):

    if command_line_par==default_val:
        ret = params[(params_name, default_val, params_description)]
    else:
        params[params_name]=command_line_par
        ret = command_line_par

    return ret

def normalize_image_into_standard_form(img):
    """the input image should in gray x,y"""
    img = (img-img.min())/(img.max()-img.min())
    sz = [1,1]+ list(img.shape)
    img_t = torch.Tensor(img)
    img_t = img_t.view(sz)
    return img_t


def convert_index_into_coord_set(index,expand=0):
    if expand==0:
        coord = [(int(index[0][i]),int(index[1][i])) for i in range(len(index[0]))]
    else:
        coord=[]
        for ex in range(-expand,expand):
            for ey in range(-expand,expand):
                coord += [(int(index[0][i]+ex), int(index[1][i]+ey)) for i in range(len(index[0]))]
    return set(coord)


def add_texture_on_img(im_orig,texture_gaussian_smoothness=0.1,texture_magnitude=0.3):

    # do this separately for each integer intensity level
    levels = np.unique((np.floor(im_orig)).astype('int'))

    im = np.zeros_like(im_orig)

    for current_level in levels:

        sz = im_orig.shape
        rand_noise = np.random.random(sz[2:]).astype('float32')-0.5
        rand_noise = rand_noise.view().reshape(sz)
        r_params = pars.ParameterDict()
        r_params['smoother']['type'] = 'gaussian'
        r_params['smoother']['gaussian_std'] = texture_gaussian_smoothness
        spacing = 1.0 / (np.array(sz[2:]).astype('float32') - 1)
        s_r = sf.SmootherFactory(sz[2::], spacing).create_smoother(r_params)

        rand_noise_smoothed = s_r.smooth(AdaptVal(torch.from_numpy(rand_noise))).detach().cpu().numpy()
        rand_noise_smoothed /= rand_noise_smoothed.max()
        rand_noise_smoothed *= texture_magnitude

        c_indx = (im_orig>=current_level-0.5)
        im[c_indx] = im_orig[c_indx] + rand_noise_smoothed[c_indx]

    return torch.Tensor(im)











