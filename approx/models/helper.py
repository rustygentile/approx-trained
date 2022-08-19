import numpy as np
import torch


class ApproximationHelper:
    """
    Utilities for 1D function approximation. Generates training data and
    evaluates function norms.

    Parameters
    ----------
    target_function : str
        'step', 'cusp', 'gaussian'
    n_train : int
        number of data samples used for training. Note: for 'step'
        functions this should be even. For 'cusp' functions, it should
        be odd.
    n_fine : int
        number of fine grid points used for evaluating function norms
    params : dict
        optional parameters for tuning the target function

    Returns
    -------
    Helper object
    """

    def __init__(self, target_function, n_train, n_fine, params=None):

        self.target_function = target_function
        self.n_train = n_train
        self.n_fine = n_fine
        self.params = params

        x = np.linspace(-1, 1, n_train)
        xf = np.linspace(-1, 1, n_fine)
        self.grid_size = 2 / n_fine

        if target_function == 'step':
            y = np.zeros_like(x)
            yf = np.zeros_like(xf)
            y[x > 0] += 1
            yf[xf > 0] += 1

        elif target_function == 'cusp':
            if params is None:
                offset = 0
            else:
                offset = params['offset']

            y = 1 - np.sqrt(np.abs(x - offset))
            yf = 1 - np.sqrt(np.abs(xf - offset))

        elif 'gaussian' in target_function:
            if params is None:
                s = 0.2
                gain = 0.5
            else:
                s = params['s']
                gain = params['g']

            y = gain * 1 / (s * np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / (2 * s ** 2))
            yf = gain * 1 / (s * np.sqrt(2 * np.pi)) * np.exp(-xf ** 2 / (2 * s ** 2))

        self.x = torch.from_numpy(x).float().reshape([-1, 1])
        self.xf = torch.from_numpy(xf).float().reshape([-1, 1])
        self.y = torch.from_numpy(y).float().reshape([-1, 1])
        self.yf = torch.from_numpy(yf).float().reshape([-1, 1])

        # Convolution element for calculating first derivatives
        self.first_d_conv = torch.nn.Conv1d(1, 1, kernel_size=2, bias=False)
        self.first_d_conv.weight.data = torch.tensor([[[-1., 1.]]])
        self.first_d_conv.weight.requires_grad = False

    def residual_lp_norm(self, f, p):
        """
        ||f - g||_p
        """
        diff = torch.abs(f(self.xf) - self.yf)
        norm = (torch.sum(diff ** p) * self.grid_size) ** (1 / p)
        return norm.item()

    def residual_H1_norm(self, f):
        """
        Sobolev H1 norm: ||f - g||_{H^1}
        """
        diff = f(self.xf) - self.yf
        d_diff = self.first_d_conv(diff.view((1, 1, -1))) / self.grid_size
        norm = (torch.sum(diff ** 2) * self.grid_size) ** (1 / 2) + \
               (torch.sum(d_diff ** 2) * self.grid_size) ** (1 / 2)
        return norm.item()

    def nn_H1_norm(self, f):
        """
        Sobolev H1 norm: ||f||_{H^1}
        """
        y = f(self.xf)
        dy = self.first_d_conv(y.view((1, 1, -1))) / self.grid_size
        norm = (torch.sum(y ** 2) * self.grid_size) ** (1 / 2) + \
               (torch.sum(dy ** 2) * self.grid_size) ** (1 / 2)
        return norm.item()
