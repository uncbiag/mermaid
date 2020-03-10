"""
the first demo is to input a txt file
each line includes the moving image, target image
the output should be the warped image, the momentum
( we would remove the first demo into easyreg package)


the second demo is to input a txt file
each line include the moving image, label,  momentum1, momentum2,...
the output should be the the warped image ( by random momentum, by random time) and the corresponding warped label
"""
""""
demo on RDMM on 2d synthetic image registration
"""
import matplotlib as matplt
matplt.use('Agg')
import sys,os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
sys.path.insert(0,os.path.abspath('.'))
sys.path.insert(0,os.path.abspath('..'))
sys.path.insert(0,os.path.abspath('../mermaid'))
from mermaid.model_evaluation import evaluate_model
import random
from mermaid.utils import *
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import copy



def get_pair_list(txt_pth):
    moving_momentum_path_list = read_txt_into_list(txt_pth)
    return moving_momentum_path_list

def get_init_weight_list(folder_path):
    weight_path = os.path.join(folder_path,'pair_weight_path_list.txt')
    init_weight_path = read_txt_into_list(weight_path)
    return init_weight_path

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


def get_mermaid_setting(path,output_path):
    params = pars.ParameterDict()
    params.load_JSON(path)
    os.makedirs(output_path,exist_ok=True)
    output_path = os.path.join(output_path,'mermaid_setting.json')
    params.write_JSON(output_path,save_int=False)
    return params



def get_file_name(file_path,last_ocur=True):
    if not last_ocur:
        name= os.path.split(file_path)[1].split('.')[0]
    else:
        name = os.path.split(file_path)[1].rsplit('.',1)[0]
    name = name.replace('.nii','')
    name = name.replace('.','d')
    return name


def get_fluid_input(moving_momentum_path_list, init_weight_path_list,use_affine=True):

    fr_sitk = lambda x: torch.Tensor(sitk.GetArrayFromImage(sitk.ReadImage(x)))
    spacing = np.flipud(sitk.ReadImage(moving_momentum_path_list[0]).GetSpacing())

    moving = fr_sitk(moving_momentum_path_list[0])[None][None]
    l_moving = fr_sitk(moving_momentum_path_list[1])[None][None]
    use_random_m = not len(moving_momentum_path_list)>2
    if not use_random_m:
        if not use_affine:
            momentum_list =[np.transpose(fr_sitk(path))[None] for path in moving_momentum_path_list[2:]]
            affine_list = []
        else:
            num_m = int((len(moving_momentum_path_list)-2)/2)
            momentum_list =[fr_sitk(path).permute(3,2,1,0)[None] for path in moving_momentum_path_list[2:num_m+2]]
            affine_list =[fr_sitk(path).permute(3,2,1,0)[None] for path in moving_momentum_path_list[num_m+2:]]
    else:
        momentum_list = None

    if init_weight_path_list is not None:
        init_weight_list=[[fr_sitk(path) for path in init_weight_path_list]]
    else:
        init_weight_list=None
    if not use_random_m:
        fname_list =[get_file_name(path) for path in moving_momentum_path_list[2:]]
        fname_list = [fname.replace("_0000Momentum",'') for fname in fname_list]
        fname_list = [fname.replace("_0000_Momentum",'') for fname in fname_list]
    else:
        fname_list = [get_file_name(moving_momentum_path_list[0])]
    return moving, l_moving, momentum_list, init_weight_list, affine_list, fname_list, spacing



def get_bspline_input(moving_path_list):
    moving =[sitk.ReadImage(pth[0]) for pth in moving_path_list]
    l_moving =[sitk.ReadImage(pth[1]) for pth in moving_path_list]
    fname_list = [get_file_name(pth[0]) for pth in moving_path_list]
    return moving, l_moving, fname_list


class RandomBSplineTransform(object):
    """
    Apply random BSpline Transformation to a 3D image
    check https://itk.org/Doxygen/html/classitk_1_1BSplineTransform.html for details of BSpline Transform
    """

    def __init__(self, mesh_size=(3,3,3), bspline_order=2, deform_scale=1.0, ratio=0.5, interpolator=sitk.sitkLinear,
                 random_mode = 'Normal'):
        self.mesh_size = mesh_size
        self.bspline_order = bspline_order
        self.deform_scale = deform_scale
        self.ratio = ratio  # control the probability of conduct transform
        self.interpolator = interpolator
        self.random_mode = random_mode

    def resample(self,image, transform, interpolator=sitk.sitkBSpline, default_value=0.0):
        """Resample a transformed image"""
        reference_image = image
        return sitk.Resample(image, reference_image, transform,
                             interpolator, default_value)

    def __call__(self, sample):
        random_state = np.random.RandomState()

        if np.random.rand(1)[0] < self.ratio:
            img_tm, seg_tm = sample['image'], sample['label']
            img = sitk.GetImageFromArray(sitk.GetArrayFromImage(img_tm).copy())
            img.CopyInformation(img_tm)
            seg = sitk.GetImageFromArray(sitk.GetArrayFromImage(seg_tm).copy())
            seg.CopyInformation(seg_tm)

            # initialize a bspline transform
            bspline = sitk.BSplineTransformInitializer(img, self.mesh_size, self.bspline_order)

            # generate random displacement for control points, the deformation is scaled by deform_scale
            if self.random_mode == 'Normal':
                control_point_displacements = random_state.normal(0, self.deform_scale/2, len(bspline.GetParameters()))
            elif self.random_mode == 'Uniform':
                control_point_displacements = random_state.random(len(bspline.GetParameters())) * self.deform_scale

            #control_point_displacements[0:int(len(control_point_displacements) / 3)] = 0  # remove z displacement
            bspline.SetParameters(control_point_displacements)

            # deform and resample image
            img_trans = self.resample(img, bspline, interpolator=self.interpolator, default_value=0.01)
            seg_trans = self.resample(seg, bspline, interpolator=sitk.sitkNearestNeighbor, default_value=0)
            new_sample = {}

            new_sample['image'] = img_trans
            new_sample['label'] = seg_trans
        else:
            new_sample = sample

        return new_sample

def generate_aug_data(moving_momentum_path_list,init_weight_path_list,output_path, mermaid_setting_path,compute_inverse,use_random_m,fluid_aug=True):
    if fluid_aug:
        generate_fluid_aug_data(moving_momentum_path_list, init_weight_path_list, output_path, mermaid_setting_path,
                                compute_inverse, use_random_m)
    else:
        moving_path_list = moving_momentum_path_list
        generate_bspline_aug_data(moving_path_list, output_path)


def generate_bspline_aug_data(moving_path_list,output_path):
    num_pair = len(moving_path_list)
    num_aug = int(1500. / num_pair)
    moving_list, l_moving_list,fname_list = get_bspline_input(moving_path_list)
    bspline_func1 = RandomBSplineTransform(mesh_size=(10,10,10), bspline_order=2, deform_scale=3.0, ratio=0.95)
    bspline_func2 = RandomBSplineTransform(mesh_size=(20,20,20), bspline_order=2, deform_scale=2.0, ratio=0.95)
    bspline_func3 = RandomBSplineTransform(mesh_size=(10,10,10), bspline_order=2, deform_scale=4.0, ratio=0.95)
    for i in range(num_pair):
        sample = {'image': moving_list[i], 'label':l_moving_list[i]}
        for _ in range(num_aug):
            bspline_func =random.sample([bspline_func1,bspline_func2,bspline_func3],1)
            aug_sample = bspline_func[0](sample)
            fname = fname_list[i]+'_{:.4f}'.format(random.random())
            fname = fname.replace('.', 'd')
            sitk.WriteImage(aug_sample['image'],os.path.join(output_path,fname+'_image.nii.gz'))
            sitk.WriteImage(aug_sample['label'],os.path.join(output_path,fname+'_label.nii.gz'))







def generate_fluid_aug_data(moving_momentum_path_list,init_weight_path_list,output_path, mermaid_setting_path,compute_inverse,use_random_m):
    num_pair = len(moving_momentum_path_list)
    num_aug = int(1500./num_pair)
    for i in range(num_pair):
        #num_aug=20
        moving, l_moving, momentum_list, init_weight_list,affine_list, fname_list, spacing = get_fluid_input(moving_momentum_path_list[i], init_weight_path_list[i] if init_weight_path_list else None)
        for _ in range(num_aug):
            time_aug = random.random()*2-0.5#random.sample([-1,-0.75, -0.5, -0.25,0.25,0.5, 0.75, 1, 1.25,1.5,1.75, 2.0], 1)[0] # random.random()*3-1
            if not use_random_m:
                num_momentum = len(momentum_list)
                selected_index = random.sample(list(range(num_momentum)), 1)
                momentum  = momentum_list[0]
                affine = affine_list[0] if len(affine_list) else None
                fname = fname_list[selected_index[0]] +'_t_{:.2f}'.format(time_aug)
            else:
                momentum = None
                fname = fname_list[0]+'_{:.4f}_t_{:.2f}'.format(random.random(),time_aug)
                affine = None
            fname = fname.replace('.', 'd')
            init_weight = None
            if init_weight_list is not None:
                init_weight = random.sample(init_weight_list,1)

            generate_fluid_single_res(moving,l_moving, momentum,init_weight,affine, fname, time_aug, output_path, mermaid_setting_path, spacing,moving_momentum_path_list[i][0],compute_inverse)



def generate_fluid_single_res(moving,l_moving, momentum, init_weight, initial_map, fname, time_aug, output_path, mermaid_setting_path, spacing, moving_path, compute_inverse):
    params = pars.ParameterDict()
    params.load_JSON(mermaid_setting_path)
    params['model']['registration_model']['forward_model']['tTo'] = time_aug

    # here we assume the momentum is computed at low_resol_factor=0.5
    if momentum is not None:
        input_img_sz = [1,1]+ [int(sz*2) for sz in momentum.shape[2:]]
    else:
        input_img_sz = list(moving.shape)
        momentum_sz = [1,3] + [int(dim/2) for dim in input_img_sz[2:]]
        momentum =  (np.random.rand(*momentum_sz)*2-1)*1.5
        momentum = torch.Tensor(momentum)

    org_spacing = 1.0/(np.array(moving.shape[2:])-1)
    input_spacing = 1.0/(np.array(input_img_sz[2:])-1)
    if not  input_img_sz == list(moving.shape):
        input_img,_ = resample_image(moving,spacing,input_img_sz)
    else:
        input_img = moving
    low_initial_map = None
    if initial_map is not None:
        low_initial_map, _ = resample_image(initial_map,input_spacing,[1,3]+ list(momentum.shape[2:]))
    individual_parameters = dict(m=momentum,local_weights=init_weight)
    sz = np.array(input_img.shape)
    extra_info = None
    visual_param = None
    res= evaluate_model(input_img, input_img, sz, input_spacing,
                   use_map=True,
                   compute_inverse_map=compute_inverse,
                   map_low_res_factor=0.5,
                   compute_similarity_measure_at_low_res=False,
                   spline_order=1,
                   individual_parameters=individual_parameters,
                   shared_parameters=None, params=params, extra_info=extra_info,visualize=False,visual_param=visual_param, given_weight=False,
                        init_map=initial_map, lowres_init_map=low_initial_map)
    phi = res[1]
    phi_new = phi
    #phi_new,_ = resample_image(phi,spacing,[1,3]+list(moving.shape[2:]))
    warped = compute_warped_image_multiNC(moving,phi_new,org_spacing,spline_order=1,zero_boundary=True)
    l_warped = compute_warped_image_multiNC(l_moving,phi_new,org_spacing,spline_order=0,zero_boundary=True)
    save_image_with_given_reference(warped,[moving_path],output_path,[fname+'_image'])
    save_image_with_given_reference(l_warped,[moving_path], output_path,[fname+'_label'])
    save_deformation(phi, output_path, [fname + '_phi_map'])
    if compute_inverse:
        phi_inv = res[2]
        #inv_phi_new,_ = resample_image(phi_inv,spacing,[1,3]+list(moving.shape[2:]))
        save_deformation(phi_inv,output_path,[fname+'_inv_map'])

def save_deformation(phi,output_path,fname_list):
    phi_np = phi.detach().cpu().numpy()
    for i in range(phi_np.shape[0]):
        phi = nib.Nifti1Image(phi_np[i], np.eye(4))
        nib.save(phi, os.path.join(output_path,fname_list[i]+'.nii.gz'))


def resample_image(I,spacing,desiredSize, spline_order=1,zero_boundary=False,identity_map=None):
    """
    Resample an image to a given desired size

    :param I: Input image (expected to be of BxCxXxYxZ format)
    :param spacing: array describing the spatial spacing
    :param desiredSize: array for the desired size (excluding B and C, i.e, 1 entry for 1D, 2 for 2D, and 3 for 3D)
    :return: returns a tuple: the downsampled image, the new spacing after downsampling
    """
    desiredSize = desiredSize[2:]
    is_numpy = False
    if not isinstance(I,torch.Tensor):
        I = torch.Tensor(I)
        is_numpy = True
    sz = np.array(list(I.size()))

    # check that the batch size and the number of channels is the same
    nrOfI = sz[0]
    nrOfC = sz[1]

    desiredSizeNC = np.array([nrOfI,nrOfC]+list(desiredSize))

    newspacing = spacing*((sz[2::].astype('float')-1.)/(desiredSizeNC[2::].astype('float')-1.)) ###########################################
    if identity_map is not None:
        idDes= identity_map
    else:
        idDes = torch.from_numpy(identity_map_multiN(desiredSizeNC,newspacing))
    # now use this map for resampling
    ID = compute_warped_image_multiNC(I, idDes, newspacing, spline_order,zero_boundary)

    return ID if not is_numpy else ID.numpy(), newspacing


def save_image_with_given_reference(img=None,reference_list=None,path=None,fname=None):

    num_img = len(fname)
    os.makedirs(path,exist_ok=True)
    for i in range(num_img):
        img_ref = sitk.ReadImage(reference_list[i])
        if img is not None:
            if type(img) == torch.Tensor:
                img = img.detach().cpu().numpy()
            spacing_ref = img_ref.GetSpacing()
            direc_ref = img_ref.GetDirection()
            orig_ref = img_ref.GetOrigin()
            img_itk = sitk.GetImageFromArray(img[i,0])
            img_itk.SetSpacing(spacing_ref)
            img_itk.SetDirection(direc_ref)
            img_itk.SetOrigin(orig_ref)
        else:
            img_itk=img_ref
        fn =  '{}_batch_'.format(i)+fname if not type(fname)==list else fname[i]
        fpath = os.path.join(path,fn+'.nii.gz')
        sitk.WriteImage(img_itk,fpath)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Registration demo for data augmentation')
    parser.add_argument('--txt_path', required=False, default=None,
                        help='the file path of input txt, exclusive with random_m')
    parser.add_argument('--rdmm_preweight_txt_path', required=False, default=None,
                        help='the file path of rdmm preweight txt, only needed when use predefined rdmm model')
    parser.add_argument('--random_m',required=False,action='store_true',
                        help='data augmentation with random momentum, exclusive with txt_path')
    parser.add_argument('--bspline', required=False, action='store_true',
                        help='data augmentation with bspline, exclusive random_m, rdmm_preweight_txt_path,compute_inverse')
    parser.add_argument('--output_path', required=False, default='./rdmm_synth_data_generation/data_task',
                        help='the path of task folder')
    parser.add_argument('--use_predefined_weight_in_rdmm',required=False,action='store_true',
                        help='this flag is only for RDMM model, if set true, the predefined regularizer mask will be loaded and only the momentum will be optimized; if set false, both weight and momenutm will be jointly optimized')
    parser.add_argument('--mermaid_setting_path', required=False, default=None,
                        help='path of mermaid setting json')
    parser.add_argument('--compute_inverse', required=False, action='store_true',
                        help='compute the inverse map')
    num_of_pair_to_process = 4
    args = parser.parse_args()
    txt_path = args.txt_path
    rdmm_preweight_txt_path = args.rdmm_preweight_txt_path
    use_init_weight = rdmm_preweight_txt_path is not None # for now, the init weight has yet supported
    mermaid_setting_path = args.mermaid_setting_path
    compute_inverse = args.compute_inverse
    use_random_m = args.random_m
    use_bspline = args.bspline
    output_path = args.output_path
    do_affine = False
    os.makedirs(output_path,exist_ok=True)
    # if the use_random_m is false or use_bspline, then the list only include moving and label info
    moving_momentum_path_list = get_pair_list(txt_path)
    init_weight_path_list = None
    if use_init_weight:
        init_weight_path_list = get_init_weight_list(rdmm_preweight_txt_path)

    generate_aug_data(moving_momentum_path_list,init_weight_path_list,output_path, mermaid_setting_path,compute_inverse,use_random_m, fluid_aug= not use_bspline)


