import set_pyreg_paths

import pyreg.fileio as fileio
import os
import glob

#dataroot = '/Users/mn/data/testdata/CUMC12'
#filenames_intensity = glob.glob(os.path.join(dataroot,'brain_affine_icbm','*.nii'))
#filenames_labels = glob.glob(os.path.join(dataroot,'label_affine_icbm','*.nii'))

#outdataroot = 'cumc12_2d'
#
#outdir_intensity = os.path.join(outdataroot,'brain_affine_icbm')
#outdir_labels = os.path.join(outdataroot,'label_affine_icbm')

def write_sliced_file(filename_in,filename_out,slice_dim=2):
    print('Processing file: ' + filename_in)

    squeeze_image = True
    normalize_spacing = True

    I, hdr, spacing, normalized_spacing = \
        fileio.ImageIO().read_to_nc_format(filename_in,
                                           intensity_normalize=False,
                                           squeeze_image=squeeze_image,
                                           normalize_spacing=normalize_spacing)

    hdr['sizes'] = list(hdr['sizes'])

    if slice_dim==0:
        hdr['sizes'][0] = 1
        hdr['sizes'] = tuple(hdr['sizes'])
        sliceX = I.shape[-3] // 2

        I_sliced = I[:, :, sliceX, :, :]
        is_shape = list(I_sliced.shape)
        new_shape = is_shape[0:2] + [1] + is_shape[2:]
        I_sliced_t = I_sliced.view().reshape(new_shape)

    elif slice_dim==1:
        hdr['sizes'][1] = 1
        hdr['sizes'] = tuple(hdr['sizes'])
        sliceY = I.shape[-2] // 2

        I_sliced = I[:, :, :, sliceY, :]
        is_shape = list(I_sliced.shape)
        new_shape = is_shape[0:3] + [1] + [is_shape[-1]]
        I_sliced_t = I_sliced.view().reshape(new_shape)

    elif slice_dim==2:
        hdr['sizes'][-1] = 1
        hdr['sizes'] = tuple(hdr['sizes'])
        sliceZ = I.shape[-1] // 2

        I_sliced = I[:, :, :, :, sliceZ]
        new_shape = list(I_sliced.shape) + [1]
        I_sliced_t = I_sliced.view().reshape(new_shape)

    else:
        raise ValueError('Can only slice along dimensions 0, 1, or 2')

    fileio.ImageIO().write(filename_out, I_sliced_t[0, 0, ...], hdr)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Extracts slices from 3D volumes. Call for example, like: python extractSlicesFrom3DDataSet.py --dataset_directory ~/data/testdata/IBSR18/ --output_directory ./IBSR18_2d --slice_dim 2')

    parser.add_argument('--dataset_directory', required=True, help='Main directory where dataset is stored; this directory should contain the subdirectory label_affine_icbm')
    parser.add_argument('--output_directory', required=True, help='Where the output is stored')
    parser.add_argument('--slice_dim', required=False, default=2,  type=int, help='Slice dimension: 0, 1, or 2')

    args = parser.parse_args()

    slice_dim = args.slice_dim

    dataroot = os.path.expanduser(args.dataset_directory)
    outdataroot = os.path.expanduser(args.output_directory)

    outdir_intensity = os.path.join(outdataroot, 'brain_affine_icbm')
    outdir_labels = os.path.join(outdataroot, 'label_affine_icbm')

    if not os.path.isdir(outdataroot):
        print('Creating directory: ' + outdataroot)
        os.makedirs(outdataroot)

    if not os.path.isdir(outdir_intensity):
        print('Creating directory: ' + outdir_intensity)
        os.makedirs(outdir_intensity)

    if not os.path.isdir(outdir_labels):
        print('Creating directory: ' + outdir_labels)
        os.makedirs(outdir_labels)

    filenames_intensity = glob.glob(os.path.join(dataroot, 'brain_affine_icbm', '*.nii'))
    filenames_labels = glob.glob(os.path.join(dataroot, 'label_affine_icbm', '*.nii'))

    print('Found: ' + str(len(filenames_intensity)) + ' intensity files.')

    for filename in filenames_intensity:
        filename_out = os.path.join(outdir_intensity, os.path.split(filename)[1])
        write_sliced_file(filename, filename_out, slice_dim=slice_dim)

    print('Found: ' + str(len(filenames_labels)) + ' label files.')

    for filename in filenames_labels:
        filename_out = os.path.join(outdir_labels, os.path.split(filename)[1])
        write_sliced_file(filename, filename_out, slice_dim=slice_dim)