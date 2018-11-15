import matplotlib.pyplot as plt
import matplotlib

import pyreg.finite_differences as FD


def compute_determinant_of_jacobian(phi,spacing):
    fdt = FD.FD_torch(spacing)
    dim = len(spacing)

    if dim==1:
        p0x = fdt.dXc(phi[0:1,0,...])
        det = p0x

    elif dim==2:
        p0x = fdt.dXc(phi[0:1, 0, ...])
        p0y = fdt.dYc(phi[0:1, 0, ...])
        p1x = fdt.dXc(phi[0:1, 1, ...])
        p1y = fdt.dYc(phi[0:1, 1, ...])

        det = p0x * p1y - p0y * p1x
    elif dim==3:
        p0x = fdt.dXc(phi[0:1, 0, ...])
        p0y = fdt.dYc(phi[0:1, 0, ...])
        p0z = fdt.dZc(phi[0:1, 0, ...])
        p1x = fdt.dXc(phi[0:1, 1, ...])
        p1y = fdt.dYc(phi[0:1, 1, ...])
        p1z = fdt.dZc(phi[0:1, 1, ...])
        p2x = fdt.dXc(phi[0:1, 2, ...])
        p2y = fdt.dYc(phi[0:1, 2, ...])
        p2z = fdt.dZc(phi[0:1, 2, ...])

        det = p0x*p1y*p2z + p0y*p1z*p2x + p0z*p1x*p2y -p0z*p1y*p2x -p0y*p1x*p2z -p0x*p1z*p2y
    else:
        raise ValueError('Can only compute the determinant of Jacobian for dimensions 1, 2 and 3')

    det = det.data[0, ...].detach().cpu().numpy()
    return det

def filter_names_for_boxplot(names,suppress_pattern,suppress_pattern_keep_first_as):
    idx = []
    eff_names = []
    found_first = False
    for i,n in enumerate(names):
        if n.endswith(suppress_pattern):
            if not found_first:
                found_first = True
                idx.append(i)
                eff_names.append(suppress_pattern_keep_first_as)
        else:
            idx.append(i)
            eff_names.append(n)

    return idx,eff_names


def plot_boxplot(compound_results_orig,compound_names_orig,semilogy=False, showfliers=True, suppress_pattern=None,suppress_pattern_keep_first_as=None):

    if suppress_pattern is not None:
        idx_to_keep,compound_names = filter_names_for_boxplot(names=compound_names_orig,
                                                              suppress_pattern=suppress_pattern,
                                                              suppress_pattern_keep_first_as=suppress_pattern_keep_first_as)
        compound_results = [compound_results_orig[i] for i in idx_to_keep]
    else:
        compound_results = compound_results_orig
        compound_names = compound_names_orig

    # create a figure instance
    fig = plt.figure(1, figsize=(8, 6))

    # create an axes instance
    ax = fig.add_subplot(111)

    # set axis tick
    if semilogy:
        ax.semilogy()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.yaxis.set_tick_params(left=True, direction='in', width=1)
    ax.yaxis.set_tick_params(right=True, direction='in', width=1)
    ax.xaxis.set_tick_params(top=False, direction='in', width=1)
    ax.xaxis.set_tick_params(bottom=False, direction='in', width=1)

    # create the boxplot
    bp = plt.boxplot(compound_results, vert=True, whis=1.5, meanline=True, widths=0.16, showfliers=showfliers,
                     showcaps=False, patch_artist=True, labels=compound_names)

    # rotate x labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    # set properties of boxes, medians, whiskers, fliers
    plt.setp(bp['medians'], color='orange')
    plt.setp(bp['boxes'], color='blue')
    # plt.setp(bp['caps'], color='b')
    plt.setp(bp['whiskers'], linestyle='-', color='blue')
    plt.setp(bp['fliers'], marker='o', markersize=5, markeredgecolor='blue')

    # matplotlib.rcParams['ytick.direction'] = 'in'
    # matplotlib.rcParams['xtick.direction'] = 'inout'

    # setup font
    font = {'family': 'normal', 'weight': 'semibold', 'size': 10}
    matplotlib.rc('font', **font)

    # set the line width of the figure
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    # set the range of the overlapping rate
    #plt.ylim([0, 1.0])

    # set the target box to red color
    #bp['boxes'][-1].set(color='red')
    #bp['boxes'][-1].set(facecolor='red')
    #bp['whiskers'][-1].set(color='red')
    #bp['whiskers'][-2].set(color='red')
    #bp['fliers'][-1].set(color='red', markeredgecolor='red')
