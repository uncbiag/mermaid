from __future__ import print_function
from builtins import str
from builtins import range
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import torch
import os

def change_pt_to_nii(pt_file, save_path):
    """
    From the pt dictionary to extract phi.nii 
    """
    d = torch.load(pt_file)
    phi = d['phi'].detach().cpu().numpy().squeeze()
    print(phi.shape)
    phi[0,...] = phi[0,...] / d['spacing'][0]
    phi[1,...] = phi[1,...] / d['spacing'][1]
    phi[2,...] = phi[2,...] / d['spacing'][2]
    D = nib.Nifti1Image(phi, np.eye(4))
    D.set_data_dtype(np.dtype(np.float32))
    nib.save(D, save_path)


def warp_image_nn(moving, phi):
    """
    Warp labels according to deformation field
    :param moving: Moving image in numpy array format
    :param phi: Deformation field in numpy array format
    :return: Warped labels according to deformation field 
    """

    # get image demensions
    dim1 = moving.shape[0]
    dim2 = moving.shape[1]
    dim3 = moving.shape[2]
    
    # round the deformation map to integer
    phi_round = np.round(phi).astype('int')
    idx_x = np.reshape(phi_round[0,:,:,:], (dim1*dim2*dim3, 1))
    idx_y = np.reshape(phi_round[1,:,:,:], (dim1*dim2*dim3, 1))
    idx_z = np.reshape(phi_round[2,:,:,:], (dim1*dim2*dim3, 1))
    
    # deal with extreme cases
    idx_x[idx_x < 0] = 0
    idx_x[idx_x > dim1-1] = dim1-1
    idx_y[idx_y < 0] = 0
    idx_y[idx_y > dim2-1] = dim2-1
    idx_z[idx_z < 0] = 0
    idx_z[idx_z > dim3-1] = dim3-1
    
    # get the wrapped results 
    ind = np.ravel_multi_index([idx_x, idx_y, idx_z], [dim1, dim2, dim3])
    result = moving.flatten()[ind]
    result = np.reshape(result, (dim1, dim2, dim3))
    return result


def calculate_dataset_overlap(dataset_name, dataset_dir, output_name):
    """
    Calculate the overlapping rate of specified dataset
    :param dataset_name: 'LPBA', 'IBSR', 'CUMC' or 'MGH'
    :param directory for the dataset
    :param output_name: saved result name in .mat format
    :return: averaged overlapping rate among each labels, saved in .mat format file
    """

    if dataset_name == 'LPBA':
        label_name = './l_Labels.mat'
        label_files_dir = os.path.join(dataset_dir,'LPBA_label_affine/')
        dataset_size = 40
        label_prefix = 's'
    elif dataset_name == 'IBSR':
        label_name = './c_Labels.mat'
        label_files_dir = os.path.join(dataset_dir,'IBSR_label_affine/')
        dataset_size = 18
        label_prefix = 'c'
    elif dataset_name == 'CUMC':
        label_name = './m_Labels.mat'
        label_files_dir = os.path.join(dataset_dir,'label_affine_icbm/')
        dataset_size = 12
        label_prefix = 'm'
    elif dataset_name == 'MGH':
        label_name = './g_Labels.mat'
        label_files_dir = os.path.join(dataset_dir,'MGH_label_affine/')
        dataset_size = 10
        label_prefix = 'g'
    else:
        raise TypeError("Unknown Dataset Name: Dataset name must be 'LPBA', 'IBSR', 'CUMC' or 'MGH'")


    Labels = sio.loadmat(label_name)
    result = np.zeros((dataset_size*(dataset_size-1), len(Labels['Labels'])))
    result_mean = np.zeros((dataset_size*(dataset_size-1), 1))
    registration_results_dir = np.chararray((dataset_size, dataset_size), itemsize=200)

    # Change the directory if needed (one directory for one phiinv.nii.gz file)
    for i in range(dataset_size):
        for j in range(dataset_size):
            if (i == j):
                continue
            registration_results_dir[i][j] += './' + dataset_name + '/' + str(i+1) + '_to_' + str(j+1) + '/'

    label_images = [None]*dataset_size

    for i in range(dataset_size):
        label_images[i] = sitk.GetArrayFromImage(sitk.ReadImage(label_files_dir + label_prefix + str(i+1) + '.nii')).squeeze()

    base_idx = 0
    for L_from in range(dataset_size):
        for L_to in range(dataset_size):
            if L_from == L_to:
                continue
            label_from = label_images[L_from]
            label_to = label_images[L_to]
            registration_results_path = registration_results_dir[L_from][L_to] + 'phiinv.nii.gz'
            print(registration_results_path)
            phi = nib.load(registration_results_path).get_data().squeeze()
            warp_result = warp_image_nn(label_from, phi)
            for label_idx in range(len(Labels['Labels'])):
                warp_idx = np.reshape(warp_result == Labels['Labels'][label_idx], (warp_result.shape[0]*warp_result.shape[1]*warp_result.shape[2], 1))
                to_idx = np.reshape(label_to == Labels['Labels'][label_idx], (label_to.shape[0]*label_to.shape[1]*label_to.shape[2], 1))
                result[base_idx][label_idx] = float(np.sum(np.logical_and(warp_idx, to_idx)))/np.sum(to_idx)
            base_idx += 1
            print((base_idx, ' out of ', dataset_size*(dataset_size-1)))

    for i in range(dataset_size*(dataset_size-1)):
        single_result = result[i, :]
        single_result = single_result[~np.isnan(single_result)]
        result_mean[i] = np.mean(single_result)

    sio.savemat(output_name, {'result_mean':result_mean})


def calculate_image_overlap(dataset_name, dataset_dir, phi_path, moving_id, target_id):
    """
    Calculate the overlapping rate of a specified case
    :param dataset_name: 'LPBA', 'IBSR', 'CUMC' or 'MGH'
    :param dataset_dir: path to the label datasets
    :param phi_path: deformation field path
    :param moving_id: moving image id
    :param target_id: target image id
    :return:
    """

    if dataset_name == 'LPBA':
        label_name = './l_Labels.mat'
        label_files_dir = os.path.join(dataset_dir,'LPBA_label_affine/')
        dataset_size = 40
        label_prefix = 's'
    elif dataset_name == 'IBSR':
        label_name = './c_Labels.mat'
        label_files_dir = os.path.join(dataset_dir,'IBSR_label_affine/')
        dataset_size = 18
        label_prefix = 'c'
    elif dataset_name == 'CUMC':
        label_name = './m_Labels.mat'
        label_files_dir = os.path.join(dataset_dir,'label_affine_icbm/')
        dataset_size = 2
        label_prefix = 'm'
    elif dataset_name == 'MGH':
        label_name = './g_Labels.mat'
        label_files_dir = os.path.join(dataset_dir,'MGH_label_affine/')
        dataset_size = 10
        label_prefix = 'g'
    else:
        raise TypeError("Unknown Dataset Name: Dataset name must be 'LPBA', 'IBSR', 'CUMC' or 'MGH'")

    Labels = sio.loadmat(label_name)
    result = np.zeros((len(Labels['Labels'])))
    result_mean = np.zeros((1))

    label_images = [None]*dataset_size

    for i in range(dataset_size):
        label_images[i] = sitk.GetArrayFromImage(sitk.ReadImage(label_files_dir + label_prefix + str(i+1) + '.nii')).squeeze()

    label_from = label_images[moving_id-1]
    label_to = label_images[target_id-1]
    phi = nib.load(phi_path).get_data().squeeze()
    warp_result = warp_image_nn(label_from, phi)
    for label_idx in range(len(Labels['Labels'])):
        warp_idx = np.reshape(warp_result == Labels['Labels'][label_idx], (warp_result.shape[0]*warp_result.shape[1]*warp_result.shape[2], 1))
        to_idx = np.reshape(label_to == Labels['Labels'][label_idx], (label_to.shape[0]*label_to.shape[1]*label_to.shape[2], 1))
        result[label_idx] = float(np.sum(np.logical_and(warp_idx, to_idx)))/np.sum(to_idx)

    single_result = result
    single_result = single_result[~np.isnan(single_result)]
    result_mean = np.mean(single_result)

    print('overlapping rate of without unNaN value')
    print(single_result)
    print('Averaged overlapping rate')
    print(result_mean)


def overlapping_plot(old_results_path, new_result_path):
    """
    Plot the overlaping results of 14 old appraoch and the proposed appraoch
    :param old_results_path: Old results stored in .mat format file
    :param new_result_path: New result stored in .mat format file
    :return:
    """
    old_results = sio.loadmat(old_results_path)
    new_result = sio.loadmat(new_result_path)

    # combine old names with proposed method
    compound_names = []
    for item in old_results['direc_name']:
        compound_names.append(str(item[0])[3:-2])

    compound_names.append('Proposed')
    
    # combine data
    compound_results = np.concatenate((old_results['results'], new_result['result_mean'].reshape(-1, 1)), axis=1)
    
    # create a figure instance
    fig = plt.figure(1, figsize=(8, 6))

    # create an axes instance
    ax = fig.add_subplot(111)

    # set axis tick
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5) 
    ax.yaxis.set_tick_params(left=True, direction='in', width=1)
    ax.yaxis.set_tick_params(right=True, direction='in', width=1)
    ax.xaxis.set_tick_params(top=False, direction='in', width=1)
    ax.xaxis.set_tick_params(bottom=False, direction='in', width=1)

    # create the boxplot
    bp = plt.boxplot(compound_results, vert=True, whis=1.5, meanline=True, widths=0.16, showfliers=True,
                     showcaps=False, patch_artist=True, labels=compound_names)

    # rotate x labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    # set properties of boxes, medians, whiskers, fliers    
    plt.setp(bp['medians'], color='orange')
    plt.setp(bp['boxes'], color='blue')
    #plt.setp(bp['caps'], color='b')
    plt.setp(bp['whiskers'], linestyle='-', color='blue')
    plt.setp(bp['fliers'], marker='o', markersize=5, markeredgecolor='blue')

    #matplotlib.rcParams['ytick.direction'] = 'in'
    #matplotlib.rcParams['xtick.direction'] = 'inout'

    # setup font
    font = {'family' : 'normal', 'weight' : 'semibold', 'size' : 10}
    matplotlib.rc('font', **font)

    # set the line width of the figure
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    # set the range of the overlapping rate
    plt.ylim([0, 0.8])

    # add two lines to represent the lower quartile and upper quartile
    lower_quartile = np.percentile(compound_results[:,-1], 25)
    upper_quartile = np.percentile(compound_results[:,-1], 75)
    ax.axhline(lower_quartile, ls='-', color='r', linewidth=1)
    ax.axhline(upper_quartile, ls='-', color='r', linewidth=1)

    # set the target box to red color
    bp['boxes'][-1].set(color='red')
    bp['boxes'][-1].set(facecolor='red')
    bp['whiskers'][-1].set(color='red')
    bp['whiskers'][-2].set(color='red')
    bp['fliers'][-1].set(color='red', markeredgecolor='red') 

    # save figure
    #plt.savefig('myfig.pdf', dpi=1000, bbox_inches='tight')

    # show figure
    plt.show()


if __name__ == "__main__":

    ###########################################################################################
    # example one: if you want to see the overlapping rate of one registration case
    ###########################################################################################
    # print the overlapping rate of two images
    # dataset: 'LPBA'
    # moving image: s1.nii
    # target image: s2.nii
    # deformation map: D_id.nii.gz
    change_pt_to_nii('../demos/reg_results_3d.pt', '../demos/phi.nii')
    calculate_image_overlap('CUMC', '/Users/mn/data/testdata/CUMC12', '../demos/phi.nii', 1, 2)

    ###########################################################################################
    # example two: if you want to see the overlapping rate of one dataset
    ########################################################################################### 
    # dataset: 'LPBA'
    # output file: LPBA_new.mat
    # calculate_dataset_overlap('LPBA', './LPBA_new.mat')


    ###########################################################################################
    # example three: if you want to see the boxplot of overlapping rate in one dataset
    ########################################################################################### 
    # dataset: 'LPBA'
    # overlapping_plot('./Quicksilver_results/LPBA_results.mat', './LPBA_new.mat')

