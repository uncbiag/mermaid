import matplotlib.pyplot as plt
import matplotlib

def plot_boxplot(compound_results,compound_names,semilogy=False, showfliers=True):
    # create a figure instance
    fig = plt.figure(1, figsize=(8, 6))

    # create an axes instance
    ax = fig.add_subplot(111)

    # set axis tick
    if semilogy:
        ax.semilogy()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.yaxis.set_tick_params(left='on', direction='in', width=1)
    ax.yaxis.set_tick_params(right='on', direction='in', width=1)
    ax.xaxis.set_tick_params(top='off', direction='in', width=1)
    ax.xaxis.set_tick_params(bottom='off', direction='in', width=1)

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
