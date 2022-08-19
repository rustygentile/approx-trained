import os
import argparse
import torch
import numpy as np
from approx.post.plot_settings import *
from approx.models.single_layer_simple import SingleLayerSimple


def get_br_data(folder, func, wid):

    plus_data = np.array([])
    minus_data = np.array([])

    for root, dirs, files in os.walk(folder):
        for f in files:
            if f == 'trained.pt' and f'-f{func}' in root and f'-w{wid}-' in root:

                model = SingleLayerSimple(wid)
                model.load_state_dict(torch.load(os.path.join(root, f)))

                br_plus, br_minus = model.get_breakpoints()
                bp = br_plus.detach().numpy()
                bm = br_minus.detach().numpy()
                plus_data = np.append(plus_data, bp)
                minus_data = np.append(minus_data, bm)

    return np.concatenate((plus_data, minus_data))


def main(data_folder, exp_name, wid):

    mcolor = '#6495ED'

    n_bins = 100
    fig, ax = fig_ax_1_by_3(sharey=False, height=2)
    br = get_br_data(data_folder, '5gaussian', wid)
    ax[0].hist(br, facecolor=mcolor, bins=n_bins)
    ax[0].set_title('Gaussian')

    br = get_br_data(data_folder, 'cusp', wid)
    ax[1].hist(br, facecolor=mcolor, bins=n_bins)
    ax[1].set_title('Cusp')

    br = get_br_data(data_folder, 'step', wid)
    ax[2].hist(br, facecolor=mcolor, bins=n_bins)
    ax[2].set_title('Step')

    plt.tight_layout()

    file_name = os.path.join(data_folder, f'{exp_name}_m{wid}_breakpoint_hist_single.png')
    fig.savefig(file_name, dpi=1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True, help='Data folder')
    parser.add_argument('-e', type=str, required=True, help='Experiment name')
    parser.add_argument('-w', type=int, required=True, help='NN width')
    parser.add_argument('-s', action='store_true', help='Show')
    args = parser.parse_args()
    main(args.d, args.e, args.w)
    if args.s:
        plt.show()
