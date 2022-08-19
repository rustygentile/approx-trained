import os
import argparse
import torch
import numpy as np
from approx.post.plot_settings import *
from approx.models.single_layer_simple import SingleLayerSimple
import matplotlib.patches as mpatches


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

    return plus_data, minus_data


def main(data_folder, exp_name, wid):

    p_color = '#6495ED'
    m_color = '#FFA343'

    n_bins = 100
    fig, ax = fig_ax_2_by_3()
    p, m = get_br_data(data_folder, '5gaussian', wid)
    ax[0][0].hist(p, facecolor=p_color, bins=n_bins)
    ax[1][0].hist(m, facecolor=m_color, bins=n_bins)
    ax[0][0].set_title('Gaussian')

    p, m = get_br_data(data_folder, 'cusp', wid)
    ax[0][1].hist(p, facecolor=p_color, bins=n_bins)
    ax[1][1].hist(m, facecolor=m_color, bins=n_bins)
    ax[0][1].set_title('Cusp')

    p, m = get_br_data(data_folder, 'step', wid)
    ax[0][2].hist(p, facecolor=p_color, bins=n_bins)
    ax[1][2].hist(m, facecolor=m_color, bins=n_bins)
    ax[0][2].set_title('Step')

    p_patch = mpatches.Patch(color=p_color, label=r'$\theta_r(+)$')
    m_patch = mpatches.Patch(color=m_color, label=r'$\theta_r(-)$')
    fig.legend(handles=[p_patch, m_patch], loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=(0, 0.09, 1, 1))

    file_name = os.path.join(data_folder, f'{exp_name}_m{wid}_breakpoint_hist.png')
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
