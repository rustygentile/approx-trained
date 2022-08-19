import matplotlib.pyplot as plt
from matplotlib import rc

TARGET_FUNCTION = 'g'
NN_FUNCTION = r'f_{\theta}'


def set_font():
    rc('font', **{'family': 'stixgeneral', 'size': 9})
    rc('mathtext', **{'fontset': 'stix'})


def fig_ax_1_by_2():
    set_font()
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
    return fig, ax


def fig_ax_1_by_3(sharey=True, height=2.5):
    set_font()
    fig, ax = plt.subplots(1, 3, figsize=(6.5, height), sharey=sharey)
    return fig, ax


def fig_ax_2_by_3(height=3):
    set_font()
    fig, ax = plt.subplots(2, 3, figsize=(6.5, height))
    return fig, ax
