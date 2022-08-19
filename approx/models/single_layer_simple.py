import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.distributions.bernoulli import Bernoulli
import numpy as np


class SingleLayerSimple(nn.Module):
    """
    f(x) = 1/sqrt(m) sum_{r=1}^m a_r \sigma(x - b_r)
    """
    def __init__(self, width, ar_trained=False):
        super().__init__()

        self.width = width
        self.gain = 1 / np.sqrt(width)
        self.l1 = nn.Linear(1, width)
        self.l2 = nn.Linear(width, 1, bias=False)

        self.l1.weight.data.fill_(1.)
        self.l1.weight.requires_grad = False

        a_r = Bernoulli(torch.tensor([0.5])).sample((width,)) * 2 - 1
        self.l2.weight.data = a_r.reshape([1, width])
        self.l2.weight.requires_grad = ar_trained

    def forward(self, x):
        return self.gain * self.l2(functional.relu(self.l1(x)))

    def get_breakpoints(self):
        a_r = self.l2.weight.data
        b_r = self.l1.bias.data
        plus_breakpoints = torch.where(a_r > 0, -1., np.nan) * b_r
        minus_breakpoints = torch.where(a_r < 0, -1., np.nan) * b_r
        return plus_breakpoints.reshape([-1, 1]), minus_breakpoints.reshape([-1, 1])
