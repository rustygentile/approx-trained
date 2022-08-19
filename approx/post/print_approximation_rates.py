import numpy as np
import pandas as pd
import argparse
import os


def make_rate_table(results_file, norm, widths=None, functions=None, labels=None):

    if widths is None:
        widths = [10, 18, 32, 56, 100, 178, 316, 562, 1000]

    if functions is None:
        functions = ['1gaussian', '2gaussian', '5gaussian', 'cusp', 'step']
        labels = ['a = 0.1', 'a = 0.25', 'a = 0.5', 'Cusp', 'Step']

    df1 = pd.read_csv(results_file).dropna()
    df2 = pd.DataFrame(data={'m': np.array(widths)[1:]})

    for j, func in enumerate(functions):
        vals = [df1.loc[(df1['func'] == func) & (df1['width'] == w)][norm].values for w in widths]
        means = np.array([v.mean() for v in vals])

        res = []
        for i in range(1, len(widths)):
            log_w = np.log(widths[i - 1] / widths[i])
            log_l = np.log(means[i - 1] / means[i])
            res.append('{alpha:.2f}'.format(alpha=-log_l/log_w))

        df2[labels[j]] = res

    return df2.to_latex(index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True, help='Data folder')
    parser.add_argument('-n', type=str, required=True, help='Norm')
    parser.add_argument('-e', type=str, required=True, help='Experiment name')
    args = parser.parse_args()

    result_file = os.path.join(args.d, 'results.csv')
    result = make_rate_table(result_file, norm=args.n)

    with open(os.path.join(args.d, args.e + '.txt'), 'w') as f:
        f.write(result)
