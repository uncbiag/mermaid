import torch
import numpy as np
import mermaid.module_parameters as pars
import mermaid.smoother_factory as sf
from mermaid.data_wrapper import AdaptVal
import matplotlib.pyplot as plt
import os

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


def t2np(v):
    """
    Takes a torch array and returns it as a numpy array on the cpu

    :param v: torch array
    :return: numpy array
    """

    if type(v) == torch.Tensor:
        return v.detach().cpu().numpy()
    else:
        try:
            return v.cpu().numpy()
        except:
            return v




def plot_2d_img(img,name,path=None):
    """
    :param img:  X x Y x Z
    :param name: title
    :param path: saving path
    :param show:
    :return:
    """
    sp=111
    img = torch.squeeze(img)

    font = {'size': 10}

    plt.setp(plt.gcf(), 'facecolor', 'white')
    plt.style.use('bmh')

    plt.subplot(sp).set_axis_off()
    plt.imshow(t2np(img)) #vmin=0.15, vmax=0.21
    plt.colorbar().ax.tick_params(labelsize=10)
    plt.title(name, font)
    if not path:
        plt.show()
    else:
        plt.savefig(path, dpi=300)
        plt.clf()




def save_smoother_map(adaptive_smoother_map,gaussian_stds,t,path=None,weighting_type=None):
    dim = len(adaptive_smoother_map.shape)-2
    adaptive_smoother_map = adaptive_smoother_map.detach()
    if weighting_type=='w_K_w':
        adaptive_smoother_map = adaptive_smoother_map**2
    gaussian_stds = gaussian_stds.detach()
    view_sz = [1] + [len(gaussian_stds)] + [1] * dim
    gaussian_stds = gaussian_stds.view(*view_sz)
    smoother_map = adaptive_smoother_map*(gaussian_stds**2)
    smoother_map = torch.sqrt(torch.sum(smoother_map,1,keepdim=True))
    print(t)
    fname = str(t)+"sm_map"
    if dim ==2:
        plot_2d_img(smoother_map[0,0],fname,path)
    elif dim==3:
        y_half = smoother_map.shape[3]//2
        plot_2d_img(smoother_map[0,0,:,y_half,:],fname,path)




def write_list_into_txt(file_path, list_to_write):
    with open(file_path, 'w') as f:
        if len(list_to_write):
            if isinstance(list_to_write[0],(float, int, str)):
                f.write("\n".join(list_to_write))
            elif isinstance(list_to_write[0],(list, tuple)):
                new_list = ["     ".join(sub_list) for sub_list in list_to_write]
                f.write("\n".join(new_list))
            else:
                raise(ValueError,"not implemented yet")

def read_txt_into_list(file_path):
    import re
    lists= []
    with open(file_path,'r') as f:
        content = f.read().splitlines()
        if len(content)>0:
            lists= [[x if x!='None'else None for x in re.compile('\s*[,|\s+]\s*').split(line)] for line in content]
            lists = [list(filter(lambda x: x is not None, items)) for items in lists]
        lists = [item[0] if len(item) == 1 else item for item in lists]
    return lists


def get_file_name(file_path,last_ocur=True):
    if not last_ocur:
        name= os.path.split(file_path)[1].split('.')[0]
    else:
        name = os.path.split(file_path)[1].rsplit('.',1)[0]
    name = name.replace('.nii','')
    name = name.replace('.','d')
    return name
