from __future__ import print_function
from builtins import str
from builtins import range

import set_pyreg_paths

import torch

import numpy as np
import scipy.stats as sstats
import pyreg.fileio as fio

import glob
import os

import matplotlib.pyplot as plt

def quantile_to_cdf(qf,nr_of_bins):

    min_qf = qf.min()
    max_qf = qf.max()

    bins = np.linspace(start=min_qf,stop=max_qf,num=nr_of_bins+1)

    cdf = []

    for b in bins[1:]:
        cdf.append(sstats.percentileofscore(qf,b))

    return cdf,bins

def compute_quantile_function(vals,nr_of_quantiles):
    perc = np.linspace(start=1,stop=100,num=nr_of_quantiles)
    quant = np.percentile(vals,q=perc)
    return quant,perc

def compute_average_quantile_function(filenames,nr_of_bins,remove_background=True,background_value=0,plot_results=False):
    im_io = fio.ImageIO()

    perc = None
    all_quants = []

    for f in filenames:
        im_orig, hdr, spacing, squeezed_spacing = im_io.read(f, intensity_normalize=False, normalize_spacing=False)

        print('Image: {:s}: min={:4.0f}; max={:4.0f}'.format(f, im_orig.min(), im_orig.max()))

        if remove_background:
            indx_keep = (im_orig > background_value)
            im = im_orig[indx_keep]
        else:
            im = im_orig
        imquant, perc = compute_quantile_function(im.flatten(), nr_of_quantiles=nr_of_bins)
        all_quants.append(imquant)

    avg_quant = np.zeros_like(all_quants[0])
    for cquant in all_quants:
        avg_quant += 1.0 / len(all_quants) * cquant

    if plot_results:
        for cquant in all_quants:
            plt.plot(perc, cquant)
        plt.plot(perc, avg_quant, color='k', linewidth=3.0)
        plt.show()

    return avg_quant,perc

def remove_average_quantile_outliers(avg_quant,perc,perc_removal=1):

    indx = (perc>=100-perc_removal)
    min_val = (avg_quant[indx]).min()

    avg_quant_res = avg_quant.copy()
    avg_quant_res[indx] = min_val

    return avg_quant_res,perc

def histogram_match(imsrc,target_cdf,target_bins,remove_background=True,background_value=0):

    if remove_background:
        indx_keep = (imsrc > background_value)
    else:
        # keep all
        indx_keep = (imsrc >= imsrc.min())

    nr_of_bins = len(target_bins)
    imhist, source_bins = np.histogram(imsrc[indx_keep], bins=nr_of_bins, density=True)

    cdfsrc = imhist.cumsum()  # cumulative distribution function
    cdfsrc = cdfsrc / cdfsrc[-1]  # normalize

    cdftgt = target_cdf  # cumulative distribution function
    cdftgt = cdftgt / cdftgt[-1]  # normalize

    im2 = np.interp(imsrc[indx_keep], source_bins[:-1], cdfsrc)
    im3 = np.interp(im2, cdftgt, target_bins[:-1])

    imres = imsrc.copy()
    imres[indx_keep] = im3

    if remove_background:
        indx_remove = (imsrc <= background_value)
        imres[indx_remove] = background_value

    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(imsrc[:,:,100])
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(imres[:, :, 100])
    plt.colorbar()
    plt.show()

    return imres

dataroot = '/Users/mn/data/testdata/CUMC12/'
suffix = 'nii'
nr_of_bins = 500
remove_background = True
background_value = 0

orig_filenames = glob.glob(os.path.join(dataroot, 'brain_affine_icbm', '*.' + suffix))

avg_quant_orig,perc_orig = compute_average_quantile_function(orig_filenames,nr_of_bins=nr_of_bins,remove_background=remove_background,background_value=background_value,plot_results=True)
avg_quant,perc = remove_average_quantile_outliers(avg_quant_orig,perc_orig,perc_removal=1)

res = dict()
res['avg_quant'] = avg_quant
res['perc'] = perc
torch.save(res,'hist_res.pt')

plt.clf()
plt.plot(perc,avg_quant_orig,color='r')
plt.plot(perc,avg_quant,color='k')
plt.show()

cdf,cdf_bins_orig = quantile_to_cdf(avg_quant,nr_of_bins=nr_of_bins)

plt.clf()
plt.plot(cdf_bins_orig[0:-1],cdf)
plt.show()

standardize_cdf = True
if standardize_cdf:
    desired_perc = 99
    perc_b = np.percentile(cdf_bins_orig,desired_perc)
    cdf_bins = cdf_bins_orig/perc_b * desired_perc/100
else:
    cdf_bins = cdf_bins_orig

# now that we have the average we can do the histogram matching to the average
im_io = fio.ImageIO()

desired_sub_path = 'brain_affine_icbm_histmatch'

for f in orig_filenames:
    im_orig, hdr, spacing, squeezed_spacing = im_io.read(f, intensity_normalize=False, normalize_spacing=False)

    print('Image: {:s}: min={:4.0f}; max={:4.0f}'.format(f, im_orig.min(), im_orig.max()))

    hist_matched_im = histogram_match(im_orig,target_cdf=cdf,target_bins=cdf_bins,remove_background=remove_background,background_value=background_value)

    current_path, base_filename = os.path.split(f)
    current_main_path, current_sub_path = os.path.split(current_path)

    desired_output_path = os.path.join(current_main_path, desired_sub_path)
    if not os.path.exists(desired_output_path):
        print('Creating output path: {:s}'.format(desired_output_path))
        os.makedirs(desired_output_path)

    desired_output_filename = os.path.join(desired_output_path, base_filename)

    # and now write it out
    im_io = fio.ImageIO()
    im_io.write(desired_output_filename,hist_matched_im,hdr)


