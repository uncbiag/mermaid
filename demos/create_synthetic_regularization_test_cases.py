import set_pyreg_paths

import matplotlib as matplt
from pyreg.config_parser import MATPLOTLIB_AGG
if MATPLOTLIB_AGG:
    matplt.use('Agg')

import matplotlib.pyplot as plt

import pyreg.utils as utils
import pyreg.finite_differences as fd
import pyreg.custom_pytorch_extensions as ce
import pyreg.smoother_factory as sf

from pyreg.data_wrapper import AdaptVal

import pyreg.fileio as fio

import pyreg.rungekutta_integrators as rk
import pyreg.forward_models as fm

import pyreg.module_parameters as pars

import torch
from torch.autograd import Variable

import numpy as np

import os

def create_rings(levels_in,multi_gaussian_weights,default_multi_gaussian_weights,
                 multi_gaussian_stds,randomize_momentum_on_circle,randomize_in_sectors,
                 put_weights_between_circles,
                 sz,spacing,visualize=False):

    if len(multi_gaussian_weights)+1!=len(levels_in):
        raise ValueError('There needs to be one more level than the number of weights, to define this example')

    id_c = utils.centered_identity_map_multiN(sz, spacing, dtype='float32')
    sh_id_c = id_c.shape

    # create the weights that will be used for the multi-Gaussian
    nr_of_mg_weights = len(default_multi_gaussian_weights)
    # format for weights: B x nr_gaussians x X x Y x Z
    sh_weights = [sh_id_c[0]] + [nr_of_mg_weights] + list(sh_id_c[2:])

    weights = np.zeros(sh_weights,dtype='float32')
    # set the default
    for g in range(nr_of_mg_weights):
        weights[:,g,...] = default_multi_gaussian_weights[g]

    # standard deviation image (only for visualization; this is the desired ground truth)
    sh_std_im = sh_weights[2:]
    std_im = np.zeros(sh_std_im,dtype='float32')

    # now get the memory for the ring data
    sh_ring_im = list(sh_id_c[2:])
    ring_im = np.zeros(sh_ring_im,dtype='float32')

    # just add one more level in case we put weights in between (otherwise add a dummy)
    levels = np.zeros(len(levels_in)+1)
    levels[0:-1] = levels_in
    if put_weights_between_circles:
        levels[-1] = levels_in[-1]+levels_in[-1]-levels_in[-2]
    else:
        levels[-1]=-1.

    for i in range(len(levels)-2):
        cl = levels[i]
        nl = levels[i+1]
        nnl = levels[i+2]
        cval = i+1

        indices_ring = (id_c[0,0,...]**2+id_c[0,1,...]**2>=cl**2) & (id_c[0,0,...]**2+id_c[0,1,...]**2<=nl**2)
        ring_im[indices_ring] = cval

        # as the momenta will be supported on the ring boundaries, we want the smoothing to be changing in the middle of the rings

        if put_weights_between_circles:
            indices_weight = (id_c[0,0,...]**2+id_c[0,1,...]**2>=((cl+nl)/2)**2) & (id_c[0,0,...]**2+id_c[0,1,...]**2<=((nl+nnl)/2)**2)
        else:
            indices_weight = (id_c[0, 0, ...] ** 2 + id_c[0, 1, ...] ** 2 >= cl ** 2) & (id_c[0, 0, ...] ** 2 + id_c[0, 1, ...] ** 2 <= nl ** 2)

        # set the desired weights in this ring
        current_desired_weights = multi_gaussian_weights[i]
        for g in range(nr_of_mg_weights):
            current_weights = weights[0,g,...]
            current_weights[indices_weight] = current_desired_weights[g]


    # now compute the resulting standard deviation image (based on the computed weights)
    for g in range(nr_of_mg_weights):
        std_im += weights[0,g,...]*multi_gaussian_stds[g]**2

    std_im = std_im**0.5

    ring_im = ring_im.view().reshape([1,1] + sh_ring_im)

    fd_np = fd.FD_np(spacing)

    # finite differences expect BxXxYxZ (without image channel)

    # gradients for the ring
    dxc = fd_np.dXc(ring_im[:,0,...])
    dyc = fd_np.dYc(ring_im[:,0,...])

    # gradients for the distance map
    dist = (id_c[:,0,...]**2 + id_c[:,1,...]**2)**0.5
    dxc_d = fd_np.dXc(dist)
    dyc_d = fd_np.dYc(dist)

    # zero them out everywhere, where the ring does not have a gradient
    ind_zero = (dxc**2+dyc**2==0)
    dxc_d[ind_zero] = 0
    dyc_d[ind_zero] = 0

    # and now randomly flip the sign ring by ring
    maxr = int(ring_im.max())

    for r in range(maxr):
        fromval = r
        toval = r+1

        #c_rand_val = np.random.randint(0,2)
        #if c_rand_val>0: # flip
        #    indx = ( (ring_im[:,0,...]>=fromval-0.1) & (ring_im[:,0,...]<=toval+0.1) & (dxc**2+dyc**2!=0) )
        #   dxc_d[indx] = -dxc_d[indx]
        #   dyc_d[indx] = -dyc_d[indx]


        if randomize_momentum_on_circle:

            randomize_over_angles = randomize_in_sectors

            if randomize_over_angles:
                nr_of_angles = 10
                angles = np.sort(2 * np.pi * np.random.rand(nr_of_angles))

                for a in range(nr_of_angles):
                    afrom = a
                    ato = (a+1)%nr_of_angles

                    nx_from = -np.sin(angles[afrom])
                    ny_from = np.cos(angles[afrom])

                    nx_to = -np.sin(angles[ato])
                    ny_to = np.cos(angles[ato])

                    indx = ((ring_im[:, 0, ...] >= fromval - 0.1) & (ring_im[:, 0, ...] <= toval + 0.1) & (dxc ** 2 + dyc ** 2 != 0)
                            & (id_c[:,0,...]*nx_from+id_c[:,1,...]*ny_from>=0)
                            & (id_c[:,0,...]*nx_to+id_c[:,1,...]*ny_to<0))

                    c_rand_choice = np.random.randint(0, 2)
                    if c_rand_choice==0:
                        multiplier = 1.0
                    else:
                        multiplier = -1.0

                    c_rand_val_field = multiplier*np.random.rand(*list(indx.shape))

                    dxc_d[indx] = dxc_d[indx] * c_rand_val_field[indx]
                    dyc_d[indx] = dyc_d[indx] * c_rand_val_field[indx]

            else:
                indx = ((ring_im[:, 0, ...] >= fromval - 0.1) & (ring_im[:, 0, ...] <= toval + 0.1) & (dxc ** 2 + dyc ** 2 != 0))
                c_rand_val_field = 2*2*(np.random.rand(*list(indx.shape))-0.5)

                dxc_d[indx] = dxc_d[indx] * c_rand_val_field[indx]
                dyc_d[indx] = dyc_d[indx] * c_rand_val_field[indx]

        else:
            indx = ((ring_im[:, 0, ...] >= fromval - 0.1) & (ring_im[:, 0, ...] <= toval + 0.1) & (dxc ** 2 + dyc ** 2 != 0))

            # multiply by a random number in [-1,1]
            c_rand_val = 2*(np.random.rand()-0.5)

            dxc_d[indx] = dxc_d[indx]*c_rand_val
            dyc_d[indx] = dyc_d[indx]*c_rand_val

    # now create desired initial momentum
    m = np.zeros_like(id_c)
    m[0,0,...] = dxc_d
    m[0,1,...] = dyc_d

    if visualize:
        plt.clf()
        plt.quiver(m[0,1,...],m[0,0,...])
        plt.axis('equal')
        plt.show()

        grad_norm_sqr = dxc**2 + dyc**2

        plt.clf()
        plt.subplot(221)
        plt.imshow(id_c[0, 0, ...])
        plt.colorbar()
        plt.subplot(222)
        plt.imshow(id_c[0, 1, ...])
        plt.colorbar()
        plt.subplot(223)
        plt.imshow(ring_im[0,0,...])
        plt.colorbar()
        plt.subplot(224)
        plt.imshow(grad_norm_sqr[0,...]>0)
        plt.colorbar()

        plt.show()

        plt.clf()

        nr_of_weights = weights.shape[1]

        for cw in range(nr_of_weights):
            plt.subplot(2,3,1+nr_of_weights)
            plt.imshow(weights[0,cw,...],vmin=0.0,vmax=1.0)
            plt.colorbar()

        plt.subplot(236)
        plt.imshow(std_im)
        plt.colorbar()
        plt.suptitle('weights')
        plt.show()


    return m,weights,ring_im,std_im

def create_random_image_pair(weights_not_fluid,weights_fluid,weights_neutral,multi_gaussian_stds,
                             randomize_momentum_on_circle,randomize_in_sectors,
                             put_weights_between_circles,
                             start_with_fluid_weight,
                             nr_of_circles_to_generate,
                             circle_extent,
                             sz,spacing,visualize=False,visualize_warped=False,print_warped_name=None):

    nr_of_rings = nr_of_circles_to_generate
    extent = circle_extent
    randomize_factor = 0.75
    randomize_radii = True
    smooth_initial_momentum = True
    add_texture_to_image = True

    nr_of_gaussians = len(multi_gaussian_stds)

    multi_gaussian_weights = []
    for r in range(nr_of_rings):
        if r%2==0:
            if start_with_fluid_weight:
                multi_gaussian_weights.append(weights_fluid)
            else:
                multi_gaussian_weights.append(weights_not_fluid)
        else:
            if start_with_fluid_weight:
                multi_gaussian_weights.append(weights_not_fluid)
            else:
                multi_gaussian_weights.append(weights_fluid)

    if randomize_radii:
        rings_at_default = np.linspace(0., extent, nr_of_rings + 1)
        diff_r = rings_at_default[1] - rings_at_default[0]
        rings_at = np.sort(rings_at_default + (np.random.random(nr_of_rings + 1) - 0.5) * diff_r * randomize_factor)
    else:
        rings_at = np.linspace(0., extent, nr_of_rings + 2)
    # first one needs to be zero:
    rings_at[0]=0

    m_orig,weights,ring_im_orig,std_im = create_rings(rings_at,multi_gaussian_weights=multi_gaussian_weights,
                    default_multi_gaussian_weights=weights_neutral,
                    multi_gaussian_stds=multi_gaussian_stds,
                    randomize_momentum_on_circle=randomize_momentum_on_circle,
                    randomize_in_sectors=randomize_in_sectors,
                    put_weights_between_circles=put_weights_between_circles,
                    sz=sz,spacing=spacing,
                    visualize=visualize)


    if smooth_initial_momentum:
        s_m_params = pars.ParameterDict()
        s_m_params['smoother']['type'] = 'gaussian'
        s_m_params['smoother']['gaussian_std'] = 0.025
        s_m = sf.SmootherFactory(sz[2::], spacing).create_smoother(s_m_params)

        m=s_m.smooth(AdaptVal(Variable(torch.from_numpy(m_orig),requires_grad=False))).data.cpu().numpy()

        if visualize:
            plt.clf()
            plt.subplot(121)
            plt.imshow(m_orig[0,0,...])
            plt.subplot(122)
            plt.imshow(m[0,0,...])
            plt.suptitle('smoothed mx')
            plt.show()
    else:
        m=m_orig

    # create a velocity field from this momentum using a multi-Gaussian kernel
    gaussian_fourier_filter_generator = ce.GaussianFourierFilterGenerator(sz[2:], spacing, nr_of_gaussians)

    vs = ce.fourier_set_of_gaussian_convolutions(AdaptVal(Variable(torch.from_numpy(m),requires_grad=False)),
                                                 gaussian_fourier_filter_generator,
                                                 sigma=AdaptVal(Variable(torch.from_numpy(multi_gaussian_stds),requires_grad=False)),
                                                 compute_std_gradients=False )


    # compute velocity based on default weights
    default_vs = np.zeros([1,2]+sz[2:],dtype='float32')
    for g in range(nr_of_gaussians):
        default_vs += weights_neutral[g]*vs[g,...].data.cpu().numpy()

    norm_d_vs = (default_vs[0,0,...]**2 + default_vs[0,1,...]**2)**0.5

    # compute velocity based on localized weights
    localized_v = np.zeros([1,2]+sz[2:],dtype='float32')
    dims = localized_v.shape[1]
    for g in range(nr_of_gaussians):
        for d in range(dims):
            localized_v[0,d,...] += weights[0,g,...]*vs[g,0,d,...].data.cpu().numpy()

    norm_localized_v = (localized_v[0,0,...]**2 + localized_v[0,1,...]**2)**0.5

    # now compute the deformation that belongs to this velocity field

    params = pars.ParameterDict()
    params['number_of_time_steps'] = 20

    advectionMap = fm.AdvectMap( sz[2:], spacing )
    pars_to_pass = utils.combine_dict({'v':AdaptVal(Variable(torch.from_numpy(localized_v),requires_grad=False))}, dict() )
    integrator = rk.RK4(advectionMap.f, advectionMap.u, pars_to_pass, params)

    tFrom = 0.
    tTo = 1.

    phi0 = AdaptVal(Variable(torch.from_numpy(utils.identity_map_multiN(sz,spacing)),requires_grad=False))
    phi1 = integrator.solve([phi0], tFrom, tTo )[0]

    if add_texture_to_image:

        rand_noise = np.random.random(sz[2:])
        rand_noise = rand_noise.view().reshape(sz)
        r_params = pars.ParameterDict()
        r_params['smoother']['type'] = 'gaussian'
        r_params['smoother']['gaussian_std'] = 0.015
        s_r = sf.SmootherFactory(sz[2::], spacing).create_smoother(r_params)

        rand_noise_smoothed = s_r.smooth(AdaptVal(Variable(torch.from_numpy(rand_noise), requires_grad=False))).data.cpu().numpy()
        rand_noise_smoothed /= rand_noise_smoothed.max()

        ring_im = ring_im_orig + rand_noise_smoothed
    else:
        ring_im = ring_im_orig

    # deform image based on this map
    I0_source = AdaptVal(Variable(torch.from_numpy(ring_im),requires_grad=False))
    I1_warped = utils.compute_warped_image_multiNC(I0_source, phi1, spacing, spline_order=1)

    # define the label images
    I0_label = AdaptVal(Variable(torch.from_numpy(ring_im_orig),requires_grad=False))
    I1_label = utils.get_warped_label_map(I0_label, phi1, spacing )

    if visualize_warped:
        plt.clf()
        # plot original image, warped image, and grids
        plt.subplot(3,4,1)
        plt.imshow(I0_source[0,0,...].data.cpu().numpy())
        plt.title('source')
        plt.subplot(3,4,2)
        plt.imshow(I1_warped[0,0,...].data.cpu().numpy())
        plt.title('warped = target')
        plt.subplot(3,4,3)
        plt.imshow(I0_source[0,0,...].data.cpu().numpy())
        plt.contour(phi0[0,0,...].data.cpu().numpy(), np.linspace(-1, 1, 40), colors='r', linestyles='solid')
        plt.contour(phi0[0,1,...].data.cpu().numpy(), np.linspace(-1, 1, 40), colors='r', linestyles='solid')
        plt.subplot(3,4,4)
        plt.imshow(I1_warped[0,0,...].data.cpu().numpy())
        plt.contour(phi1[0,0,...].data.cpu().numpy(), np.linspace(-1, 1, 40), colors='r', linestyles='solid')
        plt.contour(phi1[0,1,...].data.cpu().numpy(), np.linspace(-1, 1, 40), colors='r', linestyles='solid')

        nr_of_weights = weights.shape[1]
        for cw in range(nr_of_weights):
            plt.subplot(3,4,5+cw)
            plt.imshow(weights[0, cw, ...], vmin=0.0, vmax=1.0)
            plt.title('w: std' + str(multi_gaussian_stds[cw]))
            plt.colorbar()

        plt.subplot(3,4,12)
        plt.imshow(std_im)
        plt.title('std')
        plt.colorbar()

        if print_warped_name is not None:
            plt.savefig(print_warped_name)
        else:
            plt.show()

    if visualize:
        plt.clf()
        plt.subplot(121)
        plt.imshow(norm_d_vs)
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(norm_localized_v)
        plt.colorbar()
        plt.show()

        plt.subplot(121)
        plt.quiver(m[0,1,...],m[0,0,...])
        plt.axis('equal')
        plt.subplot(122)
        plt.quiver(default_vs[0,1,...],default_vs[0,0,...])
        plt.axis('equal')
        plt.show()

    return I0_source.data.cpu().numpy(), I1_warped.data.cpu().numpy(), weights, \
           I0_label.data.cpu().numpy(), I1_label.data.cpu().numpy()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Creates a synthetic registration results')

    parser.add_argument('--output_directory', required=False, default='synthetic_example_out', help='Where the output was stored (now this will be the input directory)')
    parser.add_argument('--nr_of_pairs_to_generate', required=False, default=10, type=int, help='number of image pairs to generate')
    parser.add_argument('--nr_of_circles_to_generate', required=False, default=2, type=int, help='number of circles to generate in an image')
    parser.add_argument('--circle_extent', required=False, default=0.25, type=float, help='Size of largest circle; image is [-0.5,0.5]^2')

    parser.add_argument('--do_not_randomize_momentum', action='store_true', help='if set, momentum is deterministic')
    parser.add_argument('--do_not_randomize_in_sectors', action='store_true', help='if set and randomize momentum is on, momentum is only randomized uniformly over circles')
    parser.add_argument('--put_weights_between_circles', action='store_true', help='if set, the weights will change in-between circles, otherwise they will be colocated with the circles')
    parser.add_argument('--start_with_fluid_weight', action='store_true', help='if set then the innermost circle is not fluid, otherwise it is fluid')

    parser.add_argument('--stds', required=False,type=str, default=None, help='standard deviations for the multi-Gaussian; default=[0.01,0.05,0.1,0.2]')
    parser.add_argument('--weights_not_fluid', required=False,type=str, default=None, help='weights for a non fluid circle; default=[0,0,0,1]')
    parser.add_argument('--weights_fluid', required=False,type=str, default=None, help='weights for a fluid circle; default=[0.1,0.3,0.3,0.3]')
    parser.add_argument('--weights_background', required=False,type=str, default=None, help='weights for the background; default=[0,0,0,1]')

    args = parser.parse_args()

    visualize = False
    visualize_warped = True
    print_images = True

    nr_of_pairs_to_generate = args.nr_of_pairs_to_generate
    nr_of_circles_to_generate = args.nr_of_circles_to_generate
    circle_extent = args.circle_extent
    randomize_momentum_on_circle = not args.do_not_randomize_momentum
    randomize_in_sectors = not args.do_not_randomize_in_sectors
    put_weights_between_circles = args.put_weights_between_circles
    start_with_fluid_weight = args.start_with_fluid_weight

    if args.stds is None:
        multi_gaussian_stds = np.array([0.01, 0.05, 0.1, 0.2])
    else:
        mgsl = [float(item) for item in args.stds.split(',')]
        multi_gaussian_stds = np.array(mgsl)

    if args.weights_not_fluid is None:
        weights_not_fluid = np.array([0,0,0,1.0])
    else:
        cw = [float(item) for item in args.stds.split(',')]
        weights_not_fluid = np.array(cw)

    if len(weights_not_fluid)!=len(multi_gaussian_stds):
        raise ValueError('Need as many weights as there are standard deviations')

    if args.weights_fluid is None:
        weights_fluid = np.array([0.1,0.3,0.3,0.3])
    else:
        cw = [float(item) for item in args.stds.split(',')]
        weights_fluid = np.array(cw)

    if len(weights_fluid)!=len(multi_gaussian_stds):
        raise ValueError('Need as many weights as there are standard deviations')

    if args.weights_background is None:
        weights_neutral = np.array([0,0,0,1.0])
    else:
        cw = [float(item) for item in args.stds.split(',')]
        weights_neutral = np.array(cw)

    if len(weights_neutral)!=len(multi_gaussian_stds):
        raise ValueError('Need as many weights as there are standard deviations')

    output_dir = args.output_directory

    image_output_dir = os.path.join(output_dir,'brain_affine_icbm')
    label_output_dir = os.path.join(output_dir,'label_affine_icbm')
    misc_output_dir = os.path.join(output_dir,'misc')
    pdf_output_dir = os.path.join(output_dir,'pdf')

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

    sz = [1,1,128,128]
    spacing = 1.0/(np.array(sz[2:])-1)


    pt = dict()
    pt['source_images'] = []
    pt['target_images'] = []
    pt['source_ids'] = []
    pt['target_ids'] = []

    im_io = fio.ImageIO()
    # image hdr
    hdr = dict()
    hdr['space origin'] = np.array([0,0,0])
    hdr['spacing'] = np.array(list(spacing) + [spacing[-1]])
    hdr['space directions'] = np.array([['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1']])
    hdr['dimension'] = 3
    hdr['space'] = 'left-posterior-superior'
    hdr['sizes'] = list(sz[2:])+[1]

    for n in range(nr_of_pairs_to_generate):

        print('Writing file pair ' + str(n+1) + '/' + str(nr_of_pairs_to_generate))

        if print_images:
            print_warped_name = os.path.join(pdf_output_dir,'registration_image_pair_{:05d}.pdf'.format(2*n+1))
        else:
            print_warped_name = None

        I0Source, I1Target, weights, I0Label, I1Label = create_random_image_pair(weights_not_fluid=weights_not_fluid,
                                                               weights_fluid=weights_fluid,
                                                               weights_neutral=weights_neutral,
                                                               multi_gaussian_stds=multi_gaussian_stds,
                                                               randomize_momentum_on_circle=randomize_momentum_on_circle,
                                                               randomize_in_sectors=randomize_in_sectors,
                                                               put_weights_between_circles=put_weights_between_circles,
                                                               start_with_fluid_weight=start_with_fluid_weight,
                                                               nr_of_circles_to_generate=nr_of_circles_to_generate,
                                                               circle_extent=circle_extent,
                                                               sz=sz,spacing=spacing,
                                                               visualize=visualize,
                                                               visualize_warped=visualize_warped,
                                                               print_warped_name=print_warped_name)

        source_filename = os.path.join(image_output_dir,'m{:d}.nii'.format(2*n+1))
        target_filename = os.path.join(image_output_dir,'m{:d}.nii'.format(2*n+1+1))
        source_label_filename = os.path.join(label_output_dir, 'm{:d}.nii'.format(2 * n + 1))
        target_label_filename = os.path.join(label_output_dir, 'm{:d}.nii'.format(2 * n + 1 + 1))

        weights_filename = os.path.join(misc_output_dir,'weights_{:05d}.pt'.format(2*n+1))

        reshape_size = list(sz[2:]) + [1]

        # save these files
        im_io.write(filename=source_filename,data=I0Source.view().reshape(reshape_size),hdr=hdr)
        im_io.write(filename=target_filename,data=I1Target.view().reshape(reshape_size),hdr=hdr)

        im_io.write(filename=source_label_filename, data=I0Label.view().reshape(reshape_size), hdr=hdr)
        im_io.write(filename=target_label_filename, data=I1Label.view().reshape(reshape_size), hdr=hdr)

        torch.save(weights,weights_filename)

        # create source/target configuration
        pt['source_images'].append(source_filename)
        pt['target_images'].append(target_filename)
        pt['source_ids'].append(2*n+1)
        pt['target_ids'].append(2*n+1+1)

    filename_pt = os.path.join(output_dir,'used_image_pairs.pt')
    torch.save(pt,filename_pt)

