import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import numpy as np


def layer_init(self):
    in_w = self.weight.data.size()[0]
    lim = 1. / (np.sqrt(in_w))
    return -lim, lim


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed=None, fc_units=None):
        super(Actor, self).__init__()
        if fc_units is None:
            fc_units = [128, 128, 128]
        layers = OrderedDict()
        layers[f'fc{1}'] = nn.Linear(state_size, fc_units[0])
        layers[f'bn{1}'] = nn.BatchNorm1d(fc_units[0])
        layers[f'relu{1}'] = nn.ReLU()
        for i in range(1, len(fc_units)):
            layers[f'fc{i + 1}'] = nn.Linear(fc_units[i - 1], fc_units[i])
            layers[f'bn{i + 1}'] = nn.BatchNorm1d(fc_units[i])
            layers[f'relu{i + 1}'] = nn.ReLU()
        layers[f'fc{len(fc_units) + 1}'] = nn.Linear(fc_units[-1], action_size)
        layers[f'relu{len(fc_units) + 1}'] = nn.Tanh()
        self.net = nn.Sequential(layers)
        self.seed = torch.manual_seed(seed)
        # self.init_weights()

    def forward(self, state):
        return self.net(state)

    def init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(*layer_init(layer))


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed=None, fc_units=None):
        super(Critic, self).__init__()
        if fc_units is None:
            fc_units = [128, 128, 128]
        first_part = OrderedDict()
        first_part[f'fc_{1}'] = nn.Linear(state_size, fc_units[0])
        first_part[f'bn{1}'] = nn.BatchNorm1d(fc_units[0])
        first_part[f'relu_{1}'] = nn.ReLU()
        layers = OrderedDict()
        layers[f'fc_{2}'] = nn.Linear(fc_units[0] + action_size, fc_units[1])
        layers[f'bn{2}'] = nn.BatchNorm1d(fc_units[1])
        layers[f'relu_{2}'] = nn.ReLU()

        for i in range(2, len(fc_units)):
            layers[f'fc_{i + 1}'] = nn.Linear(fc_units[i - 1], fc_units[i])
            layers[f'bn{i + 1}'] = nn.BatchNorm1d(fc_units[i])
            layers[f'relu_{i + 1}'] = nn.ReLU()
        layers[f'fc_{len(fc_units) + 1}'] = nn.Linear(fc_units[-1], 1)
        self.first = nn.Sequential(first_part)
        self.net = nn.Sequential(layers)
        self.seed = torch.manual_seed(seed)
        # self.init_weights()

    def forward(self, state, action):
        x = self.first(state)
        x = torch.cat((x, action), dim=1)
        return self.net(x)

    def init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(*layer_init(layer))


if __name__ == '__main__':
    actor = Actor(1, 4, 1)
    critic = Critic(1, 1, 1)
    print(actor(torch.Tensor([[2]])))
    print(critic(torch.Tensor([[2]]), torch.Tensor([[1]])))
