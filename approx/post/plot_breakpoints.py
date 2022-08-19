import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from approx.post.plot_settings import fig_ax_1_by_3
from approx.models.single_layer_simple import SingleLayerSimple
from approx.models.helper import ApproximationHelper


def plot_nn_br(ax, m, model_file, func, title):

    model = SingleLayerSimple(m)
    model.load_state_dict(torch.load(model_file))

    npts = int(1e5)
    x = np.linspace(-1, 1, npts)
    y = ApproximationHelper(func.replace('5', ''), n_train=npts, n_fine=1).y
    xt = torch.from_numpy(x).float().reshape([-1, 1])
    phi = model(xt).detach().numpy()

    br_plus, br_minus = model.get_breakpoints()
    y_plus = model(br_plus)
    y_minus = model(br_minus)

    x3 = br_plus.detach().numpy()
    x4 = br_minus.detach().numpy()

    y3 = y_plus.detach().numpy()
    y4 = y_minus.detach().numpy()

    ax.plot(x, y, c='k', lw=0.75, label=r'$g$')
    ax.plot(x, phi, '--', c='k', lw=0.5, label=r'$f_{\theta}$')
    ax.plot(x3, y3, '+', markersize=5, label=r'$\theta(+)$')
    ax.plot(x4, y4, 'o', markersize=3, fillstyle='none', label=r'$\theta(-)$')
    ax.set_title(title)


def main(data_folder, func, seed, exp_name):

    fig, ax = fig_ax_1_by_3()
    model_file1 = data_folder + f'/tests/s{seed}-w10-f{func}/trained.pt'
    model_file2 = data_folder + f'/tests/s{seed}-w100-f{func}/trained.pt'
    model_file3 = data_folder + f'/tests/s{seed}-w1000-f{func}/trained.pt'

    plot_nn_br(ax[0], 10, model_file1, func, 'm = 10')
    plot_nn_br(ax[1], 100, model_file2, func, 'm = 100')
    plot_nn_br(ax[2], 1000, model_file3, func, 'm = 1000')

    handles, labels = ax[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=(0, 0.09, 1, 1))

    fig.savefig(data_folder + f'/{exp_name}_{func}_breakpoints.png', dpi=1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True, help='Data folder')
    parser.add_argument('-e', type=str, required=True, help='Experiment name')
    parser.add_argument('-f', type=str, required=True, help='Function')
    parser.add_argument('-s', action='store_true', help='Show')
    args = parser.parse_args()
    main(args.d, args.f, 0, args.e)
    if args.s:
        plt.show()
