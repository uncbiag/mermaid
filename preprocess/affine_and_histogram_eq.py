from __future__ import print_function
from builtins import str
from builtins import range
from subprocess import call
import argparse
import os.path
import sys

import itk
# needs to be imported after itk to overwrite itk's incorrect error handling
import mermaid.fixwarnings

import numpy as np
from skimage import exposure

parser = argparse.ArgumentParser(description='Perform optional preprocessing steps: affine transformation and histogram equalization or matching (optional)) to input images')

requiredNamed = parser.add_argument_group('required named arguments')

requiredNamed.add_argument('--input-images', nargs='+', required=True, metavar=('i1', 'i2, i3...'),
                           help='List of input images, seperated by space.')
requiredNamed.add_argument('--output-images', nargs='+', required=True, metavar=('o1', 'o2, o3...'),
                           help='List of output image directory and file names, seperated by space. The recommended file format to save is .mhd; .nii may create flippings later on.')
parser.add_argument('--atlas', default="~/data/atlas/icbm152.nii",
                    help="Atlas to use for (affine) pre-registration and histogram matching")

parser.add_argument('--input-labels', nargs='+', metavar=('l_i1', 'l_i2, l_i3...'),
                           help='List of input label maps for the input images, seperated by space.')

parser.add_argument('--output-labels', nargs='+', metavar=('l_o1', 'l_o2, l_o3...'),
                           help='List of output label maps, seperated by space.')

parser.add_argument('--affine', action='store_true', default=False,
                    help='Performs affine registration to the atlas image')

parser.add_argument('--histeq', action='store_true', default=False,
                    help='Perform histogram equalization to the moving and target images.')

parser.add_argument('--histmatch', action='store_true', default=False,
                    help='Perform histogram matching to the atlas image')

parser.add_argument('--histadapt', action='store_true', default=False,
                    help='Perform adaptive histogram equalization to the atlas image')

parser.add_argument('--perc', required=False, default=None,
                    help='Removes negative entries and scales image such that 0 stays zero and given percentile (e.g., 90)')

args = parser.parse_args()

def check_args(args):
    if (len(args.input_images) != len(args.output_images)):
        print('The number of input images is not consistent with the number of output images!')
        sys.exit(1)
    if ((args.input_labels is None) ^ (args.output_labels is None)):
        print('The input labels and output labels need to be both defined!')
        sys.exit(1)
    if ((args.input_labels is not None) and (len(args.input_labels) != len(args.output_labels))):
        print('The number of input labels is not consistent with the number of output labels!')

def affine_transformation(args):
    if args.affine:
        for i in range(0, len(args.input_images)):
            call(["reg_aladin",
                  "-noSym", "-speeeeed", "-ref", os.path.expanduser(args.atlas) ,
                  "-flo", os.path.expanduser(args.input_images[i]),
                  "-res", os.path.expanduser(args.output_images[i]),
                  "-aff", os.path.expanduser(args.output_images[i]+'_affine_transform.txt')])
            if (args.input_labels is not None):
                call(["reg_resample",
                      "-ref", os.path.expanduser(args.atlas),
                      "-flo", os.path.expanduser(args.input_labels[i]),
                      "-res", os.path.expanduser(args.output_labels[i]),
                      "-trans", os.path.expanduser(args.output_images[i]+'_affine_transform.txt'),
                      "-inter", str(0)]) 

def intensity_normalization_histmatch(args):

    # load the atlas image which will be used for histogram matching
    image_atlas = itk.imread(os.path.expanduser(args.atlas))

    for i in range(0, len(args.input_images)):
        if args.affine:
            # affine registration has happened before
            image = itk.imread(os.path.expanduser(args.output_images[i]))
        else:
            image = itk.imread(os.path.expanduser(args.input_images[i]))
            
        image_np = itk.GetArrayViewFromImage(image)

        nan_mask = np.isnan(image_np)
        image_np[nan_mask] = 0

        # perform histogram matching
        dim = image.GetImageDimension()
        ImageType = itk.Image[itk.F, dim]
        match_filter = itk.HistogramMatchingImageFilter[ImageType,ImageType].New()

        match_filter.SetReferenceImage(image_atlas)
        match_filter.SetInput(image)
        match_filter.Update()

        itk.imwrite(match_filter.GetOutput(),os.path.expanduser(args.output_images[i]),compression=True)

def intensity_normalization_histadapt(args):

    print('WARNING: Make sure the ouput is really want you want. Settings not well tested.')
    for i in range(0, len(args.input_images)):
        if args.affine:
            # affine registration has happened before
            image = itk.imread(os.path.expanduser(args.output_images[i]))
        else:
            image = itk.imread(os.path.expanduser(args.input_images[i]))

        image_np = itk.GetArrayViewFromImage(image)

        nan_mask = np.isnan(image_np)
        image_np[nan_mask] = 0

        # perform histogram matching
        dim = image.GetImageDimension()
        ImageType = itk.Image[itk.F, dim]
        adapt_filter = itk.AdaptiveHistogramEqualizationImageFilter[ImageType].New()
        adapt_filter.SetAlpha(0.1) # 0: classical histogram equalization; 1: unsharp mask
        adapt_filter.SetBeta(0.0)
        adapt_filter.SetRadius(5)

        adapt_filter.SetInput(image)
        adapt_filter.Update()

        itk.imwrite(adapt_filter.GetOutput(), os.path.expanduser(args.output_images[i]), compression=True)

def intensity_normalization_rescale(args):

    for i in range(0, len(args.input_images)):
        if args.affine:
            # affine registration has happened before
            filename = os.path.expanduser(args.output_images[i])
        else:
            filename = os.path.expanduser(args.input_images[i])

        print('Processing image ' + filename + ' ; percentile = ' + str(args.perc) + ' [0,100]')
        image = itk.imread(filename)
        image_np = itk.GetArrayViewFromImage(image)

        nan_mask = np.isnan(image_np)
        image_np[nan_mask] = 0

        # get the minumum, maximum and the given percentile
        current_min = image_np.min()
        current_max = image_np.max()
        ninety_perc = np.percentile(image_np,q=args.perc)

        dim = image.GetImageDimension()
        ImageType = itk.Image[itk.F, dim]

        # remove negative values
        clamp_filter = itk.ClampImageFilter[ImageType,ImageType].New()
        clamp_filter.SetInput(image)
        clamp_filter.SetBounds(0,float(current_max))
        clamp_filter.Update()

        # and now rescale it
        rescale_filter = itk.RescaleIntensityImageFilter[ImageType,ImageType].New()
        rescale_filter.SetOutputMinimum(0)
        new_max = 1.0/ninety_perc*current_max
        rescale_filter.SetOutputMaximum(float(new_max))

        rescale_filter.SetInput(clamp_filter.GetOutput())
        rescale_filter.Update()

        itk.imwrite(rescale_filter.GetOutput(), os.path.expanduser(args.output_images[i]), compression=True)


def intensity_normalization_histeq(args):
    print('Histogram equalization not currently supported.')
    sys.exit(1)

def intensity_normalization(args):
    nr_of_selected = 0
    if args.histadapt:
        nr_of_selected+=1
    if args.histeq:
        nr_of_selected+=1
    if args.histmatch:
        nr_of_selected+=1
    if args.perc is not None:
        nr_of_selected+=1

    if nr_of_selected>1:
        print('Can only choose one histogram normalization method at a time.')
        sys.exit(1)

    if args.histmatch:
        intensity_normalization_histmatch(args)
    elif args.histeq:
        intensity_normalization_histeq(args)
    elif args.histadapt:
        intensity_normalization_histadapt(args)
    elif args.perc is not None:
        intensity_normalization_rescale(args)
    
    
if __name__ == '__main__':
    #print((args.input_labels is None) and (args.output_labels is None))

    affine_transformation(args)
    intensity_normalization(args)
