"""
the input txt
each line include a series of label and a series of inv_phi, and the last column refers to the gt label

"""
import SimpleITK as sitk
from mermaid.utils import *
from mermaid.metrics import *
from .gen_aug_samples import resample_image, save_image_with_given_reference, get_file_name, read_txt_into_list
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

def get_input(label_phi_path):
    num_aug = (len(label_phi_path[0])-1)/2
    num_aug = int(num_aug)
    fr_sitk = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
    label_list =[fr_sitk(path)[None] for path in label_phi_path[:num_aug]]
    phi_list =[np.transpose(fr_sitk(path)) for path in label_phi_path[num_aug:2*num_aug]]
    gt= fr_sitk(label_phi_path[-1])
    label = np.stack(label_list,0)
    phi = np.stack(phi_list,0)
    spacing = 1. / (np.array(label[2:]) - 1)
    phi_new,_ = resample_image(phi,spacing,[num_aug,3]+list(label.shape[2:]))
    fname =get_file_name(label_phi_path[-1])
    return label, gt, phi, fname, spacing


def get_warped_res(label, phi,spacing):
    warped = compute_warped_image_multiNC(label,phi,spacing,spline_order=0)
    return warped

def get_voting_res(label,gt):
    label_index = np.unique(gt)
    gt_sz = list(gt.shape[2:])
    voting_hist = np.zeros([label.shape[0],len(label_index)]+gt_sz)
    for i, index in enumerate(label_index):
        voting_hist[:,i] =np.sum(label==index,axis=1)
    voting_res = np.argmax(voting_hist,axis=1)
    voting_res = voting_res[:,None]
    for i, index in enumerate(label_index):
        voting_res[voting_res==i]=label_index[i]
    return voting_res


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




def compute(label_phi_txt_path,output_path,save_voting_res):
    label_phi_path_list = read_txt_into_list(label_phi_txt_path)
    num_image = len(label_phi_path_list)
    records_score_np = np.zeros(num_image)
    loss_detail_list = []

    for i, label_phi_path in enumerate(label_phi_path_list):
        label, gt, phi, fname, spacing = get_input(label_phi_path)
        warped_label = get_warped_res(label,phi,spacing)
        voting_label = get_voting_res(warped_label,gt)
        res = get_multi_metric(voting_label,gt)
        dice, detailed_dice = get_val_res(res)
        records_score_np[i] = dice
        loss_detail_list.append(detailed_dice)
        print("the dice score for image {} is {}".format(fname,dice))
        print("the dice score for current average is {}".format(np.sum(records_score_np)/float(i)))
        print("the detailed dice score for image {} is {}".format(fname,detailed_dice))
        if save_voting_res:
            save_image_with_given_reference(voting_label,[label_phi_path[-1]] , output_path, [fname + '_voting'])

    np.save(os.path.join(output_path, 'records'), records_score_np)
    records_detail_np = extract_interest_loss(loss_detail_list, num_image )
    np.save(os.path.join(output_path, 'records_detail'), records_detail_np)




def gen_label_phi_txt(test_txt,output_txt_pth, phi_saved_pth, lwarped_saved_pth, phi_type, l_type, label_img_switcher, lwarped_img_switcher):
    file_pth_list = read_txt_into_list(test_txt)
    gt_pth_list = [pth[1] for pth in file_pth_list]
    fname_list = [get_file_name(l_pth).replace(*label_img_switcher) for l_pth in gt_pth_list]
    f_path = os.path.join(phi_saved_pth, '**', phi_type)
    phi_pth_list = glob(f_path, recursive=True)
    f_path = os.path.join(lwarped_saved_pth, '**', l_type)
    lwarped_saved_pth_list = glob(f_path, recursive=True)
    label_phi_dict = {fname:{"gt":None,"label_list":[],"phi_list":[]} for fname in fname_list}
    for i, fname in enumerate(fname_list):
        f = lambda x: get_file_name(x).find(fname)==0
        filtered_lpth_list = filter(f,lwarped_saved_pth_list)
        filtered_iname_list = [get_file_name(l_pth).replace(*lwarped_img_switcher) for l_pth in filtered_lpth_list]
        filtered_phi_pth_list = []
        for image_name in filtered_iname_list:
            f = lambda x: get_file_name(x).find(image_name) == 0
            filtered_phi_pth_list += filter(f,phi_pth_list)
        label_phi_dict[fname]['gt'] = gt_pth_list[i]
        label_phi_dict[fname]['label_list'] = filtered_lpth_list
        label_phi_dict[fname]['phi_list'] = filtered_phi_pth_list
    label_phi_list = []
    for fname, value in label_phi_dict.items():
        label_phi_list.append([label_phi_dict[fname]['gt']] +label_phi_dict[fname]['label_list']+label_phi_dict[fname]['phi_list'])
    write_list_into_txt(output_txt_pth,label_phi_list)
    return output_txt_pth















