import set_pyreg_paths

import pyreg.fileio as fileio
import os
import glob

dataroot = '/Users/mn/data/testdata/CUMC12'
filenames_intensity = glob.glob(os.path.join(dataroot,'brain_affine_icbm','*.nii'))
filenames_labels = glob.glob(os.path.join(dataroot,'label_affine_icbm','*.nii'))

outdataroot = 'cumc12_2d'

outdir_intensity = os.path.join(outdataroot,'brain_affine_icbm')
outdir_labels = os.path.join(outdataroot,'label_affine_icbm')

if not os.path.isdir(outdataroot):
    print('Creating directory: ' + outdataroot)
    os.makedirs(outdataroot)

if not os.path.isdir(outdir_intensity):
    print('Creating directory: ' + outdir_intensity)
    os.makedirs(outdir_intensity)

if not os.path.isdir(outdir_labels):
    print('Creating directory: ' + outdir_labels)
    os.makedirs(outdir_labels)

def write_sliced_file(filename_in,filename_out):
    print('Processing file: ' + filename_in)

    squeeze_image = True
    normalize_spacing = True

    I, hdr, spacing, normalized_spacing = \
        fileio.ImageIO().read_to_nc_format(filename_in,
                                           intensity_normalize=False,
                                           squeeze_image=squeeze_image,
                                           normalize_spacing=normalize_spacing)

    hdr['sizes'] = list(hdr['sizes'])
    hdr['sizes'][-1] = 1
    hdr['sizes'] = tuple(hdr['sizes'])
    sliceZ = I.shape[-1] // 2

    I_sliced = I[:, :, :, :, sliceZ]
    I_sliced_t = I_sliced.view().reshape(list(I_sliced.shape) + [1])

    fileio.ImageIO().write(filename_out, I_sliced_t[0, 0, ...], hdr)

print('Found: ' + str( len(filenames_intensity)) + ' intensity files.')

for filename in filenames_intensity:
    filename_out = os.path.join(outdir_intensity,os.path.split(filename)[1])
    write_sliced_file(filename,filename_out)

print('Found: ' + str(len(filenames_labels)) + ' label files.')

for filename in filenames_labels:
    filename_out = os.path.join(outdir_labels,os.path.split(filename)[1])
    write_sliced_file(filename,filename_out)


#It, hdrt, spacingt, normalized_spacingt = \
#    fileio.ImageIO().read_to_nc_format('tst_out.nhdr',
#                                       intensity_normalize=False,
#                                       squeeze_image=squeeze_image,
#                                       normalize_spacing=normalize_spacing)


print('Hello')