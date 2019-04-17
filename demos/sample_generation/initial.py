from .context import *
import argparse
from .utils_for_general import *
import random

def get_argparse():
    parser = argparse.ArgumentParser(description='Creates a synthetic registration results')

    parser.add_argument('--config', required=False, default='synthetic_example_out_kernel_weighting_type_sqrt_w_K_sqrt_w/config.json', help='The main json configuration file that can be used to define the settings')

    parser.add_argument('--output_directory', required=False, default='synthetic_example_out', help='Where the output was stored (now this will be the input directory)')
    parser.add_argument('--nr_of_pairs_to_generate', required=False, default=10, type=int, help='number of image pairs to generate')
    parser.add_argument('--nr_of_circles_to_generate', required=False, default=None, type=int, help='number of circles to generate in an image') #2
    parser.add_argument('--circle_extent', required=False, default=None, type=float, help='Size of largest circle; image is [-0.5,0.5]^2') # 0.25

    parser.add_argument('--seed', required=False, type=int, default=2018, help='Sets the random seed which affects data shuffling')

    parser.add_argument('--create_publication_figures', action='store_true', help='If set writes out figures illustrating the generation approach of first example')

    parser.add_argument('--use_fixed_source', action='store_true', help='if set the source image is fixed; like a fixed atlas image')
    parser.add_argument('--use_random_source', action='store_true', help='if set then inital source is warped randomly, otherwise it is circular')

    parser.add_argument('--no_texture', action='store_true',help='if set then no texture is used, otherwise (default) texture is generated')
    parser.add_argument('--texture_gaussian_smoothness', required=False, type=float, default=None, help='Gaussian standard deviation used to smooth a random image to create texture.')
    parser.add_argument('--texture_magnitude', required=False, type=float, default=None, help='Magnitude of the texture')

    parser.add_argument('--do_not_randomize_momentum', action='store_true', help='if set, momentum is deterministic')
    parser.add_argument('--do_not_randomize_in_sectors', action='store_true', help='if set and randomize momentum is on, momentum is only randomized uniformly over circles')
    parser.add_argument('--put_weights_between_circles', action='store_true', help='if set, the weights will change in-between circles, otherwise they will be colocated with the circles')
    parser.add_argument('--start_with_fluid_weight', action='store_true', help='if set then the innermost circle is not fluid, otherwise it is fluid')

    parser.add_argument('--weight_smoothing_std',required=False,default=0.02,type=float,help='Standard deviation to smooth the weights with; to assure sufficient regularity')
    parser.add_argument('--stds', required=False,type=str, default=None, help='standard deviations for the multi-Gaussian; default=[0.01,0.05,0.1,0.2]')
    parser.add_argument('--weights_not_fluid', required=False,type=str, default=None, help='weights for a non fluid circle; default=[0,0,0,1]')
    parser.add_argument('--weights_fluid', required=False,type=str, default=None, help='weights for a fluid circle; default=[0.2,0.5,0.2,0.1]')
    parser.add_argument('--weights_background', required=False,type=str, default=None, help='weights for the background; default=[0,0,0,1]')

    parser.add_argument('--kernel_weighting_type', required=False, type=str, default=None, help='Which kernel weighting to use for integration. Specify as [w_K|w_K_w|sqrt_w_K_sqrt_w]; w_K is the default')

    parser.add_argument('--nr_of_angles', required=False, default=None, type=int, help='number of angles for randomize in sector') #10
    parser.add_argument('--multiplier_outer_factor', required=False, default=None, type=float, help='value the random momentum outward is multiplied by') #1.0
    parser.add_argument('--multiplier_inner_factor', required=False, default=None, type=float, help='value the random momentum innerward is multiplied by') #1.0
    parser.add_argument('--momentum_smoothing', required=False, default=None, type=int, help='how much the randomly generated momentum is smoothed') #0.05
    parser.add_argument('--visualize', required=False, default=True, type=int, help='visualize the samples') #0.05
    parser.add_argument('--print_image', required=False, default=False, type=int, help='print the image into pdf') #0.05

    parser.add_argument('--sz', required=False, type=str, default=None, help='Desired size of synthetic example; default=[128,128]')

    args = parser.parse_args()

    if args.seed is not None:
        print('Setting the random seed to {:}'.format(args.seed))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    mermaid_setting_path = None
    params = pars.ParameterDict()
    if args.config is not None:
        params.load_JSON(args.config)
        mermaid_setting_path = params['mermaid_setting_path']

    print_images = True

    nr_of_pairs_to_generate = args.nr_of_pairs_to_generate

    nr_of_circles_to_generate = get_parameter_value(args.nr_of_circles_to_generate, params, 'nr_of_circles_to_generate',
                                                    2, 'number of circles for the synthetic data')
    circle_extent = get_parameter_value(args.circle_extent, params, 'circle_extent', 0.2,
                                        'Size of largest circle; image is [-0.5,0.5]^2')

    randomize_momentum_on_circle = get_parameter_value_flag(not args.do_not_randomize_momentum, params=params,
                                                            params_name='randomize_momentum_on_circle',
                                                            default_val=True,
                                                            params_description='randomizes the momentum on the circles')

    randomize_in_sectors = get_parameter_value_flag(not args.do_not_randomize_in_sectors, params=params,
                                                    params_name='randomize_in_sectors',
                                                    default_val=True,
                                                    params_description='randomized the momentum sector by sector')

    put_weights_between_circles = get_parameter_value_flag(args.put_weights_between_circles, params=params,
                                                           params_name='put_weights_between_circles',
                                                           default_val=False,
                                                           params_description='if set, the weights will change in-between circles, otherwise they will be colocated with the circles')

    start_with_fluid_weight = get_parameter_value_flag(args.start_with_fluid_weight, params=params,
                                                       params_name='start_with_fluid_weight',
                                                       default_val=False,
                                                       params_description='if set then the innermost circle is not fluid, otherwise it is fluid')

    use_random_source = get_parameter_value_flag(args.use_random_source, params=params, params_name='use_random_source',
                                                 default_val=False,
                                                 params_description='if set then source image is already deformed (and no longer circular)')

    use_fixed_source = get_parameter_value_flag(args.use_fixed_source, params=params, params_name='use_fixed_source',
                                                default_val=False,
                                                params_description='if set then source image will be fixed; like a fixed atlas image)')

    add_texture_to_image = get_parameter_value_flag(not args.no_texture, params=params,
                                                    params_name='add_texture_to_image', default_val=True,
                                                    params_description='When set to true, texture is added to the images (based on texture_gaussian_smoothness)')

    texture_magnitude = get_parameter_value(args.texture_magnitude, params=params, params_name='texture_magnitude',
                                            default_val=0.3,
                                            params_description='Largest magnitude of the added texture')

    texture_gaussian_smoothness = get_parameter_value(args.texture_gaussian_smoothness, params=params,
                                                      params_name='texture_gaussian_smoothness',
                                                      default_val=0.02,
                                                      params_description='How much smoothing is used to create the texture image')

    kernel_weighting_type = get_parameter_value(args.kernel_weighting_type, params=params,
                                                params_name='kernel_weighting_type',
                                                default_val='sqrt_w_K_sqrt_w',
                                                params_description='Which kernel weighting to use for integration. Specify as [w_K|w_K_w|sqrt_w_K_sqrt_w]; w_K is the default')

    if use_random_source == True and use_fixed_source == True:
        raise ValueError('The source image cannot simultaneously be random and fixed. Aborting')

    nr_of_angles = get_parameter_value(args.nr_of_angles, params, 'nr_of_angles', 10,
                                       'number of angles for randomize in sector')
    multiplier_outer_factor = get_parameter_value(args.multiplier_outer_factor, params, 'multiplier_outer_factor', 0.5,
                                                  'value the random momentum is multiplied by')
    multiplier_inner_factor = get_parameter_value(args.multiplier_inner_factor, params, 'multiplier_inner_factor', 0.5,
                                                  'value the random momentum is multiplied by')
    momentum_smoothing = get_parameter_value(args.momentum_smoothing, params, 'momentum_smoothing', 0.05,
                                             'how much the randomly generated momentum is smoothed')

    if args.stds is None:
        multi_gaussian_stds_p = None
    else:
        mgsl = [float(item) for item in args.stds.split(',')]
        multi_gaussian_stds_p = list(np.array(mgsl))

    multi_gaussian_stds = get_parameter_value(multi_gaussian_stds_p, params, 'multi_gaussian_stds',
                                              list(np.array([0.01, 0.05, 0.1, 0.2])),
                                              'multi gaussian standard deviations')
    multi_gaussian_stds = np.array(multi_gaussian_stds).astype('float32')

    if args.weights_not_fluid is None:
        weights_not_fluid_p = None
    else:
        cw = [float(item) for item in args.weights_not_fluid.split(',')]
        weights_not_fluid_p = list(np.array(cw))

    weights_not_fluid = get_parameter_value(weights_not_fluid_p, params, 'weights_not_fluid',
                                            list(np.array([0, 0, 0, 1.0])), 'weights for the non-fluid regions')
    weights_not_fluid = np.array(weights_not_fluid).astype('float32')

    if len(weights_not_fluid) != len(multi_gaussian_stds):
        raise ValueError('Need as many weights as there are standard deviations')

    if args.weights_fluid is None:
        weights_fluid_p = None
    else:
        cw = [float(item) for item in args.weights_fluid.split(',')]
        weights_fluid_p = list(np.array(cw))

    weights_fluid = get_parameter_value(weights_fluid_p, params, 'weights_fluid', list(np.array([0.2, 0.5, 0.2, 0.1])),
                                        'weights for fluid regions')
    weights_fluid = np.array(weights_fluid).astype('float32')

    if len(weights_fluid) != len(multi_gaussian_stds):
        raise ValueError('Need as many weights as there are standard deviations')

    if args.weights_background is None:
        weights_neutral_p = None
    else:
        cw = [float(item) for item in args.weights_background.split(',')]
        weights_neutral_p = list(np.array(cw))

    weights_neutral = get_parameter_value(weights_neutral_p, params, 'weights_neutral', list(np.array([0, 0, 0, 1.0])),
                                          'weights in the neutral/background region')
    weights_neutral = np.array(weights_neutral).astype('float32')

    if kernel_weighting_type == 'w_K_w':
        print('INFO: converting weights to w_K_w format, i.e., taking their square root')
        # square of weights needs to sum up to one, so simply take the square root of the specified weights here
        weights_fluid = np.sqrt(weights_fluid)
        weights_neutral = np.sqrt(weights_neutral)
        weights_not_fluid = np.sqrt(weights_not_fluid)

    if len(weights_neutral) != len(multi_gaussian_stds):
        raise ValueError('Need as many weights as there are standard deviations')

    if args.sz is None:
        sz_p = None
    else:
        cw = [int(item) for item in args.sz.split(',')]
        sz_p = np.array(cw).astype('float32')

    sz = get_parameter_value(sz_p, params, 'sz', [128, 128], 'size of the synthetic example')
    if len(sz) != 2:
        raise ValueError('Only two dimensional synthetic examples are currently supported for sz parameter')

    sz = [1, 1, sz[0], sz[1]]
    spacing = 1.0 / (np.array(sz[2:]).astype('float32') - 1)

    output_dir = os.path.normpath(args.output_directory) + '_kernel_weighting_type_' + native_str(kernel_weighting_type)

    image_output_dir = os.path.join(output_dir, 'brain_affine_icbm')
    label_output_dir = os.path.join(output_dir, 'label_affine_icbm')
    misc_output_dir = os.path.join(output_dir, 'misc')
    pdf_output_dir = os.path.join(output_dir, 'pdf')
    publication_figs = os.path.join(output_dir, 'publication_figs')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if not os.path.isdir(image_output_dir):
        os.makedirs(image_output_dir)

    if not os.path.isdir(label_output_dir):
        os.makedirs(label_output_dir)

    if not os.path.isdir(misc_output_dir):
        os.makedirs(misc_output_dir)

    if not os.path.isdir(pdf_output_dir):
        os.makedirs(pdf_output_dir)

    if args.create_publication_figures:
        if not os.path.isdir(publication_figs):
            os.makedirs(publication_figs)

    pt = dict()
    pt['source_images'] = []
    pt['target_images'] = []
    pt['source_ids'] = []
    pt['target_ids'] = []

    im_io = fio.ImageIO()
    # image hdr
    hdr = dict()
    hdr['space origin'] = np.array([0, 0, 0])
    hdr['spacing'] = np.array(list(spacing) + [spacing[-1]])
    hdr['space directions'] = np.array([['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1']])
    hdr['dimension'] = 3
    hdr['space'] = 'left-posterior-superior'
    hdr['sizes'] = list(sz[2:]) + [1]

    for n in range(nr_of_pairs_to_generate):

        print('Writing file pair ' + str(n + 1) + '/' + str(nr_of_pairs_to_generate))

        if print_images:
            print_warped_name = os.path.join(pdf_output_dir, 'registration_image_pair_{:05d}.pdf'.format(2 * n + 1))
        else:
            print_warped_name = None

        publication_figures_directory = None
        if args.create_publication_figures and (n == 0):
            publication_figures_directory = publication_figs

    return args
