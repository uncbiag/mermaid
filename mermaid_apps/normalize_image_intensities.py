from __future__ import print_function
from builtins import str
from builtins import range
from future.utils import native_str

import json

import torch

import numpy as np
import scipy.stats as sstats
import mermaid.fileio as fio

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

    cdf = np.array(cdf)

    return cdf,bins

def compute_quantile_function(vals,nr_of_quantiles):
    perc = np.linspace(start=1,stop=100,num=nr_of_quantiles)
    quant = np.percentile(vals,q=perc)
    return quant,perc

def compute_average_quantile_function(filenames,nr_of_bins,remove_background=True,background_value=0,save_results_to_pdf=False):
    im_io = fio.ImageIO()

    perc = None
    all_quants = []

    print('Computing the average quantile function (from the following files):')

    for f in filenames:
        im_orig, hdr, spacing, squeezed_spacing = im_io.read(native_str(f), intensity_normalize=False, normalize_spacing=False)

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

    if save_results_to_pdf:
        plt.clf()
        for cquant in all_quants:
            plt.plot(perc, cquant)
        plt.plot(perc, avg_quant, color='k', linewidth=3.0)
        plt.xlabel('P')
        plt.ylabel('I')
        print('Saving: quantile_averaging.pdf')
        plt.savefig('quantile_averaging.pdf')
        #plt.show()

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

    cdftgt = np.array(target_cdf)  # cumulative distribution function
    cdftgt = cdftgt / cdftgt[-1]  # normalize

    im2 = np.interp(imsrc[indx_keep], source_bins[:-1], cdfsrc)
    im3 = np.interp(im2, cdftgt, target_bins[:-1])

    imres = imsrc.copy()
    imres[indx_keep] = im3

    if remove_background:
        indx_remove = (imsrc <= background_value)
        imres[indx_remove] = background_value

    # plt.clf()
    # plt.subplot(1,2,1)
    # plt.imshow(imsrc[:,:,100])
    # plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.imshow(imres[:, :, 100])
    # plt.colorbar()
    # plt.show()

    return imres

def compute_average_cdf(filenames, nr_of_bins=500, remove_background=True, background_value=0, save_results_to_pdf=True):

    orig_filenames = filenames

    avg_quant_orig,perc_orig = compute_average_quantile_function(orig_filenames,nr_of_bins=nr_of_bins,remove_background=remove_background,background_value=background_value,save_results_to_pdf=save_results_to_pdf)
    avg_quant,perc = remove_average_quantile_outliers(avg_quant_orig,perc_orig,perc_removal=1)

    #res = dict()
    #res['avg_quant'] = avg_quant
    #res['perc'] = perc
    #torch.save(res,'hist_res.pt')

    cdf,cdf_bins_orig = quantile_to_cdf(avg_quant,nr_of_bins=nr_of_bins)

    if save_results_to_pdf:
        plt.clf()
        plt.plot(perc, avg_quant_orig, color='r')
        plt.plot(perc, avg_quant, color='k')
        plt.xlabel('P')
        plt.ylabel('I')
        # plt.show()
        print('Saving: quantile_function.pdf')
        plt.savefig('quantile_function.pdf')

        plt.clf()
        plt.plot(cdf_bins_orig[0:-1],cdf)
        plt.xlabel('I')
        plt.ylabel('P')
        print('Saving: average_cdf.pdf')
        plt.savefig('average_cdf.pdf')
        #plt.show()

    standardize_cdf = True
    if standardize_cdf:
        desired_perc = 99
        perc_b = np.percentile(cdf_bins_orig,desired_perc)
        cdf_bins = cdf_bins_orig/perc_b * desired_perc/100

        plt.clf()
        plt.plot(cdf_bins[0:-1], cdf)
        plt.xlabel('I')
        plt.ylabel('P')
        print('Saving: standardized_average_cdf.pdf')
        plt.savefig('standardized_average_cdf.pdf')

    else:
        cdf_bins = cdf_bins_orig

    return cdf,cdf_bins


def normalize_image_intensity(source_filename,target_filename,target_cdf,target_cdf_bins, remove_background=True, background_value=0):

    if os.path.exists(target_filename):
        if os.path.samefile(source_filename, target_filename):
            raise ValueError('ERROR: Source file {} is the same as target file {}. Refusing conversion.'.format(source_filename,target_filename))

    im_io = fio.ImageIO()

    im_orig, hdr, spacing, squeezed_spacing = im_io.read(native_str(source_filename), intensity_normalize=False, normalize_spacing=False)

    print('Histogram matching: {}'.format(source_filename))
    print('Image: {:s}: min={:4.0f}; max={:4.0f}'.format(source_filename, im_orig.min(), im_orig.max()))

    hist_matched_im = histogram_match(im_orig,target_cdf=target_cdf,target_bins=target_cdf_bins,remove_background=remove_background,background_value=background_value)

    # and now write it out
    print('Writing results to {}'.format(target_filename))
    im_io = fio.ImageIO()
    im_io.write(native_str(target_filename), hist_matched_im, hdr)

    return hist_matched_im,hdr


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Program to normalize image appearance. Histogram normalization to average histogram.')

    parser.add_argument('--desired_output_directory', required=False, type=str, default='brain_affine_icbm_histmatch', help='Subdirectory that the normalized images will be placed in.')

    parser.add_argument('--dataset_directory_to_compute_cdf', required=False, type=str, default=None, help='Directory that contains the files that will be used to compute an average CDF. Not needed if a stored CDF is used.')
    parser.add_argument('--suffix_to_compute_cdf', required=False, type=str, default='nii', help='Default suffix of the data files.')
    parser.add_argument('--files_to_compute_cdf_from_as_json', required=False, type=str, default=None, help='Allows specifying a list of files (can be in different directories) as a JSON file from which to compute the CDF from. Format: ["file1.ext",...,"file2.ext"]')
    parser.add_argument('--write_used_files_for_cdf_to_json', required=False, type=str, default=None, help='Filename to which used files for CDF computation will be written in JSON format')

    parser.add_argument('--directory_to_normalize', required=False, type=str, default=None, help='Directory that contains the files that will be normalized.')
    parser.add_argument('--image_to_normalize', required=False, type=str, default=None,help='Image to be normalized (if there is only one). Alternative to --directory_to_normalize, for one image at a time.')
    parser.add_argument('--suffix', required=False, type=str, default='nii', help='Default suffix of the data files to be normalized.')

    parser.add_argument('--load_average_cdf_from_file', required=False, type=str, default=None, help='File to load the average CDF from; previously created via --save_average_cdf_to_file.')
    parser.add_argument('--save_average_cdf_to_file', required=False, default='average_cdf.pt', help='File to save the compputed average CDF to (can then be used to normalize other images).')
    parser.add_argument('--do_not_remove_background', action='store_true', help='If specified than the entire image is used to compute normalization measures; otherwise background (default value 0) is ignored.')
    parser.add_argument('--nr_of_bins', required=False, type=int, default=500, help='Number of bins for the histogram; specify a sufficiently large number for good results.')
    parser.add_argument('--background_value', required=False, type=float, default=0.0, help='Value that indicates background (default is <=0.0).')

    # parse the arguments
    args = parser.parse_args()

    avg_cdf = None
    avg_cdf_bins = None

    if (args.dataset_directory_to_compute_cdf is not None) or (args.files_to_compute_cdf_from_as_json is not None):
        compute_average_cdf_files_specified = True
    else:
        compute_average_cdf_files_specified = False

    if compute_average_cdf_files_specified:

        if args.files_to_compute_cdf_from_as_json is not None:
            files_to_compute_average_cdf_from = []

            try:
                with open(args.files_to_compute_cdf_from_as_json) as data_file:
                    print('Loading {}'.format(args.files_to_compute_cdf_from_as_json))
                    files_to_compute_average_cdf_from_u = json.load(data_file)
            except IOError as e:
                print('Could not open file = {}; ignoring request.'.format(args.files_to_compute_cdf_from_as_json))

            for f in files_to_compute_average_cdf_from_u:

                current_files = glob.glob(f) # these entries support wildcards
                for c in current_files:
                    files_to_compute_average_cdf_from.append( str(c) )

        elif args.dataset_directory_to_compute_cdf is not None:
            files_to_compute_average_cdf_from = glob.glob(os.path.join(os.path.expanduser(args.dataset_directory_to_compute_cdf), '*.' + args.suffix_to_compute_cdf))
        else:
            parser.print_help()
            raise ValueError('--dataset_directory_to_compute_cdf or --files_to_compute_cdf_from_as_json needs to be specified')

        if args.write_used_files_for_cdf_to_json is not None:
            with open(args.write_used_filed_for_cdf_to_json, 'w') as outfile:
                print('Writing used files to {}'.format(args.write_used_files_for_cdf_to_json))
                json.dump(files_to_compute_average_cdf_from, outfile, indent=4, sort_keys=True)

        avg_cdf,avg_cdf_bins = compute_average_cdf(filenames=files_to_compute_average_cdf_from,
                                                   nr_of_bins=args.nr_of_bins,
                                                   remove_background=not args.do_not_remove_background,
                                                   background_value=args.background_value)

        a = dict()
        a['avg_cdf'] = avg_cdf
        a['avg_cdf_bins'] = avg_cdf_bins

        print('Saving the average CDF to {}'.format(args.save_average_cdf_to_file))
        torch.save(a,args.save_average_cdf_to_file)

    if args.load_average_cdf_from_file is not None:
        if compute_average_cdf_files_specified:
            print('WARNING: average cdf was computed, but now this result will be discarded by loading {}'.format(args.load_average_cdf_from_file))

        print('Loading average CDF from: {}'.format(args.load_average_cdf_from_file))
        a = torch.load(args.load_average_cdf_from_file)
        avg_cdf = a['avg_cdf']
        avg_cdf_bins = a['avg_cdf_bins']

    if (avg_cdf is None) or (avg_cdf_bins is None):
        parser.print_help()
        raise ValueError('ERROR: average CDF needs to be either computed by specifying --dataset_directory_to_compute_cdf or --files_to_compute_cdf_from_as_json; or can be loaded via --load_average_cdf')

    # now that we have the average we can do the histogram matching to the average

    if (args.directory_to_normalize is not None) or (args.image_to_normalize is not None):
        # now create the output directory
        if not os.path.isdir(args.desired_output_directory):
            print('Creating output directory {}'.format(args.desired_output_directory))
            os.makedirs(os.path.expanduser(args.desired_output_directory))
    else:
        print('Info: To create output, specify either --directory_to_normalize or --image_to_normalize')

    if args.directory_to_normalize is not None:

        input_filenames = glob.glob(os.path.join(os.path.expanduser(args.directory_to_normalize), '*.' + args.suffix))
        output_filenames = []

        for in_file in input_filenames:
            _,in_base_filename = os.path.split(in_file)
            target_filename = os.path.join(os.path.expanduser(args.desired_output_directory), in_base_filename)
            output_filenames.append(target_filename)

    elif args.image_to_normalize is not None:

        current_path, base_filename = os.path.split(args.image_to_normalize)
        target_filename = os.path.join(os.path.expanduser(args.desired_output_directory), base_filename)

        input_filenames = [os.path.expanduser(args.image_to_normalize)]
        output_filenames = [target_filename]

    else:

        input_filenames = []
        output_filenames = []

    # now we can do the normalization
    for in_file,out_file in zip(input_filenames,output_filenames):
        normalize_image_intensity(source_filename=in_file,
                                  target_filename=out_file,
                                  target_cdf=avg_cdf,
                                  target_cdf_bins=avg_cdf_bins,
                                  remove_background=not args.do_not_remove_background,
                                  background_value=args.background_value)




