"""
the input txt
each line include a series of label and a series of phi, and the first column refers to the gt label

"""

import sys,os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
import SimpleITK as sitk
from mermaid.utils import *
from mermaid.metrics import *
from mermaid_demos.gen_aug_samples import resample_image, save_image_with_given_reference, get_file_name, read_txt_into_list
from glob import glob

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

def get_input(label_phi_path, compose_transform=False):
    if not compose_transform:
        return get_one_step_input(label_phi_path)
    else:
        return get_two_step_input(label_phi_path)

def get_one_step_input(label_phi_path):
    num_aug = (len(label_phi_path)-1)/2
    num_aug = int(num_aug)
    fr_sitk = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
    #label_list =[fr_sitk(path.replace("/zyshen/data/lpba_seg_resize/label_filtered","/data/quicksilver_data/testdata/LPBA40/label_affine_icbm"))[None] for path in label_phi_path[1:num_aug+1]]
    label_list =[fr_sitk(path)[None] for path in label_phi_path[1:num_aug+1]]
    phi_list =[np.transpose(fr_sitk(path)) for path in label_phi_path[num_aug+1:]]
    gt= fr_sitk(label_phi_path[0])
    label = np.stack(label_list,0)
    phi = np.stack(phi_list,0)
    spacing = 1. / (np.array(label.shape[2:]) - 1)
    if phi.shape[2:] !=label.shape[2:]:
        phi_new,_ = resample_image(phi,spacing,[num_aug,3]+list(label.shape[2:]))
    else:
        phi_new = phi
    fname =get_file_name(label_phi_path[0])
    return label, gt, phi_new, fname, spacing

def get_two_step_input(label_phi_path):
    num_aug = (len(label_phi_path)-1)/3
    num_aug = int(num_aug)
    fr_sitk = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
    label_list =[fr_sitk(path)[None] for path in label_phi_path[1:num_aug+1]]
    phi1_list =[np.transpose(fr_sitk(path)) for path in label_phi_path[num_aug+1:num_aug*2+1]]
    phi2_list =[np.transpose(fr_sitk(path)) for path in label_phi_path[num_aug*2+1:]]
    gt= fr_sitk(label_phi_path[0])
    label = np.stack(label_list,0)
    phi1 = torch.Tensor(np.stack(phi1_list,0))
    phi2 = torch.Tensor(np.stack(phi2_list,0))
    phi_spacing = 1./(np.array(phi1.shape[2:])-1)
    phi  = compute_warped_image_multiNC(phi1,phi2,phi_spacing,spline_order=1,zero_boundary=False)
    spacing = 1. / (np.array(label.shape[2:]) - 1)
    phi_new,_ = resample_image(phi,spacing,[num_aug,3]+list(label.shape[2:]))
    fname =get_file_name(label_phi_path[0])
    return label, gt, phi_new, fname, spacing


def get_warped_res(label, phi,spacing):
    label = torch.Tensor(label)
    phi = torch.Tensor(phi)
    warped = compute_warped_image_multiNC(label,phi,spacing,spline_order=0)
    return warped.numpy()

def get_voting_res(label,gt):
    label_index = np.unique(gt)
    gt_sz = list(gt.shape)
    voting_hist = np.zeros([len(label_index)]+gt_sz)
    for i, index in enumerate(label_index):
        voting_hist[i] =np.sum(label[0]==index,axis=0)
    voting_res = np.argmax(voting_hist,axis=0)
    voting_res_cp = voting_res.copy()
    for i, index in enumerate(label_index):
        voting_res_cp[voting_res==i]=label_index[i]
    return voting_res_cp
def get_voting_res_linear(label, phi, spacing):
    def make_one_hot(labels, C=2):
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3), labels.size(4)).zero_()
        target = one_hot.scatter_(1, labels.data, 1)
        target = target
        return target

    l = torch.LongTensor(label)
    to_atlas = torch.Tensor(phi)
    num_c = len(torch.unique(l))
    l_onehot = make_one_hot(l, C=num_c)
    spacing = 1. / (np.array(l.shape[2:])-1)
    l_onehot = l_onehot.to(torch.float32)
    warped_one_hot = compute_warped_image_multiNC(l_onehot, to_atlas, spacing=spacing, spline_order=1,
                                                  zero_boundary=True)
    sum_one_hot = torch.sum(warped_one_hot, 0, keepdim=True)
    voting = torch.max(torch.Tensor(sum_one_hot), 1)[1].to(torch.float32)
    voting = voting.numpy()[0]
    return voting

def get_val_res(res):
    return np.mean(res['batch_avg_res']['dice'][0, 1:]), res['batch_avg_res'][
                'dice']


def extract_interest_loss(loss_detail_list,sample_num):
    """" multi_metric_res:{iou: Bx #label , dice: Bx#label...} ,"""
    assert len(loss_detail_list)>0
    if isinstance(loss_detail_list[0],dict):
        label_num =  loss_detail_list[0]['dice'].shape[1]
        records_detail_np = np.zeros([sample_num,label_num])
        sample_count = 0
        for multi_metric_res in loss_detail_list:
            batch_len = multi_metric_res['dice'].shape[0]
            records_detail_np[sample_count:sample_count+batch_len,:] = multi_metric_res['dice']
            sample_count += batch_len
    else:
        records_detail_np=np.array([-1])
    return records_detail_np




def compute(label_phi_txt_path,output_path,compose_transform=False,use_linear=True):
    label_phi_path_list = read_txt_into_list(label_phi_txt_path)
    num_image = len(label_phi_path_list)
    records_score_np = np.zeros(num_image)
    loss_detail_list = []

    for i, label_phi_path in enumerate(label_phi_path_list):
        label, gt, phi, fname, spacing = get_input(label_phi_path,compose_transform)
        if not use_linear:
            warped_label = get_warped_res(label,phi,spacing)
            voting_label = get_voting_res(warped_label,gt)
        else:
            voting_label = get_voting_res_linear(label,phi,spacing)
        res = get_multi_metric(voting_label[None],gt[None])
        dice, detailed_dice = get_val_res(res)
        records_score_np[i] = dice
        loss_detail_list.append(detailed_dice)
        print("the dice score for image {} is {}".format(fname,dice))
        print("the dice score for current average is {}".format(np.sum(records_score_np)/float(i+1)))
        print("the detailed dice score for image {} is {}".format(fname,detailed_dice))
        save_image_with_given_reference(voting_label[None][None],[label_phi_path[0]] , output_path, [fname + '_voting'])

    np.save(os.path.join(output_path, 'records'), records_score_np)
    records_detail_np = extract_interest_loss(loss_detail_list, num_image )
    np.save(os.path.join(output_path, 'records_detail'), records_detail_np)




def gen_label_phi_txt(test_txt,output_txt_pth, phi_saved_pth, l_saved_pth, phi_type, l_type, phi_name_switcher, label_name_switcher):
    file_pth_list = read_txt_into_list(test_txt)
    gt_pth_list = [pth[1] for pth in file_pth_list]
    fname_list = [get_file_name(l_pth).replace(*label_name_switcher) for l_pth in gt_pth_list]
    f_path = os.path.join(phi_saved_pth, '**', phi_type)
    phi_pth_list = glob(f_path, recursive=True)
    label_phi_dict = {fname:{"gt":None,"label_list":[],"phi_list":[]} for fname in fname_list}
    for i, fname in enumerate(fname_list):
        f = lambda x: fname in get_file_name(x)
        filtered_phi_pth_list = list(filter(f, phi_pth_list))
        filtered_phi_name_list = [get_file_name(pth).replace(*phi_name_switcher) for pth in filtered_phi_pth_list]
        filtered_label_list = [os.path.join(l_saved_pth,f.split("_")[0]+l_type.replace("*",'')) for f in filtered_phi_name_list]
        for pth in filtered_phi_pth_list:
            assert os.path.isfile(pth), "the file {} not exist".format(pth)
        for pth in filtered_label_list:
            assert os.path.isfile(pth), "the file {} not exist".format(pth)
        label_phi_dict[fname]['gt'] = gt_pth_list[i]
        label_phi_dict[fname]['label_list'] = filtered_label_list
        label_phi_dict[fname]['phi_list'] = filtered_phi_pth_list
    label_phi_list = []
    for fname, value in label_phi_dict.items():
        label_phi_list.append([label_phi_dict[fname]['gt']] +label_phi_dict[fname]['label_list']+label_phi_dict[fname]['phi_list'])
    write_list_into_txt(output_txt_pth,label_phi_list)
    return output_txt_pth



def gen_label_two_step_phi_txt(test_txt,output_txt_pth, phi1_saved_pth,phi2_saved_pth, l_saved_pth, phi2_type,phi2_to_phi1, l_type, label_name_switcher):
    """ here if we use two step phi composition, the line structure is : gt, label1, label2, ...., trans11, trans21,...  trans12,trans22..."""
    file_pth_list = read_txt_into_list(test_txt)
    # get gt path
    gt_pth_list = [pth[1] for pth in file_pth_list]
    fname_list = [get_file_name(l_pth).replace(*label_name_switcher) for l_pth in gt_pth_list]
    f_path = os.path.join(phi2_saved_pth, '**', phi2_type)
    phi2_pth_list = glob(f_path, recursive=True)
    label_phi_dict = {fname:{"gt":None,"label_list":[],"phi1_list":[],"phi2_list":[]} for fname in fname_list}
    for i, fname in enumerate(fname_list):
        f = lambda x: get_file_name(x).split('_')[-2]==fname
        filtered_phi2_pth_list = list(filter(f,phi2_pth_list))
        filtered_ph2_name_list = [get_file_name(pth) for pth in filtered_phi2_pth_list]
        filtered_phi1_pth_list = [pth.replace(phi2_saved_pth,phi1_saved_pth).replace('image_'+fname+'_','').replace(*phi2_to_phi1) for pth in filtered_phi2_pth_list]
        for pth in filtered_phi1_pth_list:
            assert os.path.isfile(pth), "the file {} not exist".format(pth)
        label_list = [os.path.join(l_saved_pth,fname.split("_")[0]+l_type.replace("*",'')) for fname in filtered_ph2_name_list]
        for pth in label_list:
            assert os.path.isfile(pth), "the file {} not exist".format(pth)
        label_phi_dict[fname]['gt'] = gt_pth_list[i]
        label_phi_dict[fname]['label_list'] = label_list
        label_phi_dict[fname]['phi1_list'] = filtered_phi1_pth_list
        label_phi_dict[fname]['phi2_list'] = filtered_phi2_pth_list
    label_phi_list = []
    for fname, value in label_phi_dict.items():
        label_phi_list.append([label_phi_dict[fname]['gt']] +label_phi_dict[fname]['label_list']+label_phi_dict[fname]['phi1_list']+label_phi_dict[fname]['phi2_list'])
    write_list_into_txt(output_txt_pth,label_phi_list)
    return output_txt_pth



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Registeration demo for test augmentation')
    parser.add_argument('--test_txt','-t', required=False, default=None,
                        help='the file path of test txt')
    parser.add_argument('--phi_saved_pth','-p', required=False, default=None,
                        help='the folder path of phi')
    parser.add_argument('--phi_addtional_saved_pth','-pa', required=False, default=None,
                        help='the folder path of the additional transformation phi')
    parser.add_argument('--l_saved_pth','-l', required=False,default=None,
                        help='the folder path of the label to be warped')
    parser.add_argument('--output_pth','-o', required=True, default=None,
                        help='the file path of output txt')

    args = parser.parse_args()
    test_txt = args.test_txt
    phi_saved_pth = args.phi_saved_pth
    phi_addtional_saved_pth = args.phi_addtional_saved_pth
    l_saved_pth  =args.l_saved_pth
    output_pth = args.output_pth
    os.makedirs(output_pth,exist_ok=True)
    output_txt_pth = os.path.join(output_pth,'label_phi.txt')
    two_step_transform = phi_addtional_saved_pth is not None

    if not two_step_transform:
        phi_type = '*_phi.nii.gz'
        l_type = "*.nii.gz"
        label_name_switcher = ("", '')
        phi_name_switcher = ("_phi", '')
        label_phi_txt_path = gen_label_phi_txt(test_txt, output_txt_pth, phi_saved_pth, l_saved_pth, phi_type, l_type, phi_name_switcher,label_name_switcher)
        compute(label_phi_txt_path, output_pth)

    else:
        phi_type = '*_phi.nii.gz'
        l_type = "*.nii.gz"
        phi1_saved_pth = phi_saved_pth
        phi2_saved_pth = phi_addtional_saved_pth
        phi2_type = phi_type
        phi2_to_phi1 = ("_phi","_lphi_map")
        label_name_switcher = ('','')
        label_phi_txt_path = gen_label_two_step_phi_txt(test_txt, output_txt_pth, phi1_saved_pth, phi2_saved_pth, l_saved_pth, phi2_type,
                                   phi2_to_phi1, l_type, label_name_switcher)
        compute(label_phi_txt_path, output_pth,compose_transform=True)


