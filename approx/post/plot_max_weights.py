import os
import argparse
import pandas as pd
from approx.post.plot_settings import *


def plot_curves(ax, folder, files):
    for i, f in enumerate(files):
        df = pd.read_csv(folder + f)
        steps = df['step']
        y = df['ar_inf']
        ax.plot(steps, y)


def main(data_folder, show, exp_name):

    sds = list(range(10))

    # Plot all functions, m = 1000
    fig, ax = fig_ax_1_by_3(False, height=2)
    gaussian_files = [f'/tests/s{s}-w1000-f2gaussian/losses.csv' for s in sds]
    cusp_files = [f'/tests/s{s}-w1000-fcusp/losses.csv' for s in sds]
    step_files = [f'/tests/s{s}-w1000-fstep/losses.csv' for s in sds]

    plot_curves(ax[0], data_folder, gaussian_files)
    plot_curves(ax[1], data_folder, cusp_files)
    plot_curves(ax[2], data_folder, step_files)

    ax[0].set_ylabel(r'$||a(t) - a(0)||_{\infty}$')
    ax[0].set_title('Gaussian')
    ax[1].set_title('Cusp')
    ax[2].set_title('Step')
    [a.set_box_aspect(1) for a in ax]

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0, right=0.95, top=1)

    fig.savefig(os.path.join(data_folder, f'{exp_name}_max_weights_m1000.png'), dpi=1000)

    # Plot step function, m = 10, 100, 1000
    fig, ax = fig_ax_1_by_3(False, height=2)
    m10_files = [f'/tests/s{s}-w10-fstep/losses.csv' for s in sds]
    m100_files = [f'/tests/s{s}-w100-fstep/losses.csv' for s in sds]
    m1000_files = [f'/tests/s{s}-w1000-fstep/losses.csv' for s in sds]

    plot_curves(ax[0], data_folder, m10_files)
    plot_curves(ax[1], data_folder, m100_files)
    plot_curves(ax[2], data_folder, m1000_files)

    ax[0].set_ylabel(r'$||a(t) - a(0)||_{\infty}$')
    ax[0].set_title('m = 10')
    ax[1].set_title('m = 100')
    ax[2].set_title('m = 1000')
    [a.set_box_aspect(1) for a in ax]

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0, right=0.95, top=1)

    fig.savefig(os.path.join(data_folder, f'{exp_name}_max_weights_step.png'), dpi=1000)

    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True, help='Data folder')
    parser.add_argument('-e', type=str, required=True, help='Experiment name')
    parser.add_argument('-s', action='store_true', help='Show')
    args = parser.parse_args()
    main(args.d, args.s, args.e)
