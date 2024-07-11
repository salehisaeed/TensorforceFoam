import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def force_coeffs():

    font_size = 15
    params = {
        'image.origin': 'lower',
        'image.interpolation': 'nearest',
        'image.cmap': 'gray',
        'axes.grid': False,
        'savefig.dpi': 500,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'lines.markersize': 3,
        'lines.linewidth': 1,
        'font.size': font_size,
        'legend.fontsize': font_size*0.85,
        'xtick.labelsize': font_size*0.85,
        'text.usetex': True,
        'ytick.labelsize': font_size*0.85,
        'font.family': 'serif',
    }

    mpl.rcParams.update(params)
    mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'],
                      'monospace': ['Computer Modern Typewriter']})

    cases = ['baseline', 'controlled']
    LegNames = ['Baseline', 'Controlled with DRL']

    yAxisNames = ['$C_D$', '$C_L$']
    figNames = ['cd', 'cl']
    Colors = ['blue', 'red']
    lineStyle = ['-', '--', ':', '-.']
    yAxisLimits = [[1.34, 1.45], [-0.3, 0.4]]

    for i in range(len(yAxisNames)):

        fig, ax = plt.subplots(figsize=(5.8, 4))

        for j in range(len(cases)):

            fileName = 'results/test/history_' + cases[j]
            data = np.loadtxt(fileName, unpack=True, usecols=[0, 1, 2, 3], skiprows=1)

            ax.plot(data[0, :], data[i+2, :], lineStyle[j], color=Colors[j],
                    linewidth=1, label=LegNames[j])

        # ax.set_xlim((0, 15))
        # ax.set_ylim(yAxisLimits[i])
        ax.set_ylabel(yAxisNames[i], fontsize=12)
        ax.set_xlabel(r"$t$ (s)", fontsize=12)
        ax.legend(loc='best', fontsize=12)
        plt.savefig('results/test/' + figNames[i] + '.png', format='png', bbox_inches='tight')
        plt.show()
