import matplotlib as mpl
import pandas as pd
import argparse
import os
from approx.post.plot_settings import *

# Parameters for approximation rate lines
UNTRAINED_WEIGHT_PARAMS = {

    'g1f2': {'a1': 0.1, 'b1': -1. / 24, 'l1': r'$\alpha=\frac{1}{24}$',
             'a2': 0.001, 'b2': -1., 'l2': r'$\alpha=1$'},
    'g2f2': {'a1': 0.5, 'b1': -1. / 24, 'l1': r'$\alpha=\frac{1}{24}$',
             'a2': 0.001, 'b2': -1., 'l2': r'$\alpha=1$'},
    'g5f2': {'a1': 0.5, 'b1': -1. / 24, 'l1': r'$\alpha=\frac{1}{24}$',
             'a2': 0.001, 'b2': -1., 'l2': r'$\alpha=1$'},
    'cf2': {'a1': 0.2, 'b1': -1. / 24, 'l1': r'$\alpha=\frac{1}{24}$',
            'a2': 0.02, 'b2': -1. / 4, 'l2': r'$\alpha=\frac{1}{4}$'},
    'sf2': {'a1': 0.4, 'b1': -1. / 24, 'l1': r'$\alpha=\frac{1}{24}$',
            'a2': 0.2, 'b2': -1. / 4, 'l2': r'$\alpha=\frac{1}{4}$',
            'ymajor': 2, 'yminor': 2},

    'gb2': {'a1': 2, 'b1': -1. / 4, 'l1': r'$\alpha=\frac{1}{4}$',
            'a2': 0.2, 'b2': -1 / 2., 'l2': r'$\alpha=\frac{1}{2}$'},
    'cb2': {'a1': 2, 'b1': -1. / 4, 'l1': r'$\alpha=\frac{1}{4}$',
            'a2': 0.2, 'b2': -1. / 2, 'l2': r'$\alpha=\frac{1}{2}$'},
    'sb2': {'a1': 2, 'b1': -1. / 4, 'l1': r'$\alpha=\frac{1}{4}$',
            'a2': 0.2, 'b2': -1. / 2, 'l2': r'$\alpha=\frac{1}{2}$'},

    'gbinf': {'a1': 3, 'b1': -1. / 4, 'l1': r'$\alpha=\frac{1}{4}$',
              'a2': 0.3, 'b2': -1 / 2., 'l2': r'$\alpha=\frac{1}{2}$'},
    'cbinf': {'a1': 3, 'b1': -1. / 4, 'l1': r'$\alpha=\frac{1}{4}$',
              'a2': 0.3, 'b2': -1. / 2, 'l2': r'$\alpha=\frac{1}{2}$'},
    'sbinf': {'a1': 3, 'b1': -1. / 4, 'l1': r'$\alpha=\frac{1}{4}$',
              'a2': 0.3, 'b2': -1. / 2, 'l2': r'$\alpha=\frac{1}{2}$'}
}

TRAINED_AR_PARAMS = {

    'g1f2': {'a1': 0.05, 'b1': -1. / 24, 'l1': r'$\alpha=\frac{1}{24}$',
             'a2': 0.0005, 'b2': -1., 'l2': r'$\alpha=1$'},
    'g2f2': {'a1': 0.05, 'b1': -1. / 24, 'l1': r'$\alpha=\frac{1}{24}$',
             'a2': 0.0005, 'b2': -1., 'l2': r'$\alpha=1$'},
    'g5f2': {'a1': 0.05, 'b1': -1. / 24, 'l1': r'$\alpha=\frac{1}{24}$',
             'a2': 0.0005, 'b2': -1., 'l2': r'$\alpha=1$'},
    'cf2': {},
    'sf2': {'a1': 0.065, 'b1': -1. / 48, 'l1': r'$\alpha=\frac{1}{48}$',
            'a2': 0.052, 'b2': -1. / 24, 'l2': r'$\alpha=\frac{1}{24}$',
            'ymajor': 2, 'yminor': 2},

    'gb2': {'a1': 1, 'b1': -1. / 4, 'l1': r'$\alpha=\frac{1}{4}$',
            'a2': 0.1, 'b2': -1 / 2., 'l2': r'$\alpha=\frac{1}{2}$'},
    'cb2': {'ymajor': 1, 'yminor': 1},
    'sb2': {'a1': 0.8, 'b1': -1. / 24, 'l1': r'$\alpha=\frac{1}{24}$',
            'a2': 0.3, 'b2': -1. / 8, 'l2': r'$\alpha=\frac{1}{8}$',
            'ymajor': 1, 'yminor': 1},

    'gbinf': {'a1': 3, 'b1': -1. / 4, 'l1': r'$\alpha=\frac{1}{4}$',
              'a2': 0.3, 'b2': -1 / 2., 'l2': r'$\alpha=\frac{1}{2}$'},
    'cbinf': {'ymajor': 1, 'yminor': 1},
    'sbinf': {'ymajor': 1, 'yminor': 1},

    'ga2': {'a1': 2, 'b1': -1. / 4, 'l1': r'$\alpha=\frac{1}{4}$',
            'a2': 0.1, 'b2': -1 / 2., 'l2': r'$\alpha=\frac{1}{2}$'},
    'ca2': {'a1': 5, 'b1': -1. / 4, 'l1': r'$\alpha=\frac{1}{4}$',
            'a2': 0.4, 'b2': -1. / 2, 'l2': r'$\alpha=\frac{1}{2}$'},
    'sa2': {'a1': 25, 'b1': -1. / 4, 'l1': r'$\alpha=\frac{1}{4}$',
            'a2': 11, 'b2': -1. / 2, 'l2': r'$\alpha=\frac{1}{2}$'},

    'gainf': {'a1': 10, 'b1': -1. / 4, 'l1': r'$\alpha=\frac{1}{4}$',
              'a2': 0.2, 'b2': -1 / 2., 'l2': r'$\alpha=\frac{1}{2}$'},
    'cainf': {'a1': 20, 'b1': -1. / 4, 'l1': r'$\alpha=\frac{1}{4}$',
              'a2': 1, 'b2': -1. / 2, 'l2': r'$\alpha=\frac{1}{2}$'},
    'sainf': {'a1': 70, 'b1': -1. / 4, 'l1': r'$\alpha=\frac{1}{4}$',
              'a2': 15, 'b2': -1. / 2, 'l2': r'$\alpha=\frac{1}{2}$'}

}


def plot_rate(ax, results_file, func, norm, title, widths=None, params=None):

    df = pd.read_csv(results_file).dropna()

    if widths is None:
        widths = df['width'].unique()
        widths.sort()

    vals = []
    for i, w in enumerate(widths):
        val = df.loc[(df['func'] == func) & (df['width'] == w)][norm].values
        vals.append(val)

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.boxplot(vals, positions=widths, widths=widths * 0.25, manage_ticks=False)
    ax.set_title(title)
    if params is not None:

        try:
            ymajor = params['ymajor']
            ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter(f'{{x:,.{ymajor}f}}'))
        except KeyError:
            pass

        try:
            yminor = params['yminor']
            ax.yaxis.set_minor_formatter(mpl.ticker.StrMethodFormatter(f'{{x:,.{yminor}f}}'))
        except KeyError:
            pass

        try:
            b1 = params['b1']
            a1 = params['a1'] / 10 ** b1
            l1 = params['l1']
            ax.plot(widths, a1 * widths ** b1, '-.', c='black', label=l1)
            ax.legend(loc='lower left')
        except KeyError:
            pass

        try:
            b2 = params['b2']
            a2 = params['a2'] / 10 ** b2
            l2 = params['l2']
            ax.plot(widths, a2 * widths ** b2, ':', c='black', label=l2)
            ax.legend(loc='lower left')
        except KeyError:
            pass


def main(data_folder, exp_name):

    include_ar = False
    if 'trained_ar' in exp_name:
        params = TRAINED_AR_PARAMS
        include_ar = True
    elif 'untrained_weights' in exp_name:
        params = UNTRAINED_WEIGHT_PARAMS
    else:
        params = {}

    # Non-smooth function convergence rates
    fig, ax = fig_ax_1_by_3(sharey=False)
    mfile = os.path.join(data_folder, 'results.csv')
    plot_rate(ax[0], mfile, '2gaussian', 'res_l2', 'Gaussian (a = 0.25)', params=params['g2f2'])
    plot_rate(ax[1], mfile, 'cusp', 'res_l2', 'Cusp', params=params['cf2'])
    plot_rate(ax[2], mfile, 'step', 'res_l2', 'Step', params=params['sf2'])
    ax[0].set_ylabel(rf'$||{NN_FUNCTION} - {TARGET_FUNCTION}||_2$')
    ax[1].set_xlabel('Width (m)')
    ax[1].set_ylim((0.001, 0.5))
    plt.tight_layout()
    fig.savefig(os.path.join(data_folder, f'{exp_name}_convergence_in_L2.png'), dpi=1000)

    # Smooth function convergence rates
    fig, ax = fig_ax_1_by_3(sharey=False)
    plot_rate(ax[0], mfile, '1gaussian', 'res_l2', 'Gaussian (a = 0.1)', params=params['g1f2'])
    plot_rate(ax[1], mfile, '2gaussian', 'res_l2', 'Gaussian (a = 0.25)', params=params['g2f2'])
    plot_rate(ax[2], mfile, '5gaussian', 'res_l2', 'Gaussian (a = 0.5)', params=params['g5f2'])
    ax[0].set_ylabel(rf'$||{NN_FUNCTION} - {TARGET_FUNCTION}||_2$')
    ax[1].set_xlabel('Width (m)')
    plt.tight_layout()
    fig.savefig(os.path.join(data_folder, f'{exp_name}_convergence_in_L2_smooth.png'), dpi=1000)

    # Breakpoint convergence in ||.||_2
    fig, ax = fig_ax_1_by_3(sharey=False)
    plot_rate(ax[0], mfile, '2gaussian', 'br_l2', 'Gaussian (a = 0.25)', params=params['gb2'])
    plot_rate(ax[1], mfile, 'cusp', 'br_l2', 'Cusp', params=params['cb2'])
    plot_rate(ax[2], mfile, 'step', 'br_l2', 'Step', params=params['sb2'])
    ax[0].set_ylabel(r'$||\theta(t) - \theta(0)||_{2}$')
    ax[1].set_xlabel('Width (m)')
    plt.tight_layout()
    fig.savefig(os.path.join(data_folder, f'{exp_name}_br_l2.png'), dpi=1000)

    # Breakpoint convergence in ||.||_\infty
    fig, ax = fig_ax_1_by_3(sharey=False)
    plot_rate(ax[0], mfile, '2gaussian', 'br_inf', 'Gaussian (a = 0.25)', params=params['gbinf'])
    plot_rate(ax[1], mfile, 'cusp', 'br_inf', 'Cusp', params=params['cbinf'])
    plot_rate(ax[2], mfile, 'step', 'br_inf', 'Step', params=params['sbinf'])
    ax[0].set_ylabel(r'$||\theta(t) - \theta(0)||_{\infty}$')
    ax[1].set_xlabel('Width (m)')
    plt.tight_layout()
    fig.savefig(os.path.join(data_folder, f'{exp_name}_br_inf.png'), dpi=1000)

    if not include_ar:
        return

    # Outer weight convergence in ||.||_2
    fig, ax = fig_ax_1_by_3(sharey=False)
    plot_rate(ax[0], mfile, '2gaussian', 'ar_l2', 'Gaussian (a = 0.25)', params=params['ga2'])
    plot_rate(ax[1], mfile, 'cusp', 'ar_l2', 'Cusp', params=params['ca2'])
    plot_rate(ax[2], mfile, 'step', 'ar_l2', 'Step', params=params['sa2'])
    ax[0].set_ylabel(r'$||a(t) - a(0)||_{2}$')
    ax[1].set_xlabel('Width (m)')
    plt.tight_layout()
    fig.savefig(os.path.join(data_folder, f'{exp_name}_ar_l2.png'), dpi=1000)

    # Outer weight convergence in ||.||_\infty
    fig, ax = fig_ax_1_by_3(sharey=False)
    plot_rate(ax[0], mfile, '2gaussian', 'ar_inf', 'Gaussian (a = 0.25)', params=params['gainf'])
    plot_rate(ax[1], mfile, 'cusp', 'ar_inf', 'Cusp', params=params['cainf'])
    plot_rate(ax[2], mfile, 'step', 'ar_inf', 'Step', params=params['sainf'])
    ax[0].set_ylabel(r'$||a(t) - a(0)||_{\infty}$')
    ax[1].set_xlabel('Width (m)')
    plt.tight_layout()
    fig.savefig(os.path.join(data_folder, f'{exp_name}_ar_inf.png'), dpi=1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True, help='Data folder')
    parser.add_argument('-e', type=str, required=True, help='Experiment name')
    parser.add_argument('-s', action='store_true', help='Show')
    args = parser.parse_args()
    main(args.d, args.e)

    if args.s:
        plt.show()
