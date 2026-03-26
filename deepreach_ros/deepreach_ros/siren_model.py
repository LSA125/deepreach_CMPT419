from collections import OrderedDict
import math

import numpy as np
import torch
from torch import nn


class BatchLinear(nn.Linear):
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(nn.Sequential(BatchLinear(in_features, hidden_features), Sine()))

        for _ in range(num_hidden_layers):
            self.net.append(nn.Sequential(BatchLinear(hidden_features, hidden_features), Sine()))

        if outermost_linear:
            self.net.append(nn.Sequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(BatchLinear(hidden_features, out_features), Sine()))

        self.net = nn.Sequential(*self.net)
        self.net.apply(sine_init)
        self.net[0].apply(first_layer_sine_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())
        return self.net(coords)


class SingleBVPNet(nn.Module):
    def __init__(self, out_features=1, in_features=4, hidden_features=512, num_hidden_layers=3):
        super().__init__()
        self.net = FCBlock(
            in_features=in_features,
            out_features=out_features,
            num_hidden_layers=num_hidden_layers,
            hidden_features=hidden_features,
            outermost_linear=True,
        )

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        output = self.net(coords_org)
        return {'model_in': coords_org, 'model_out': output}


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)
