# import torch
# from torch import nn
# from torch.nn import functional as F
# import numpy as np
#
#
# class Actor(nn.Module):
#     def __init__(self, state_size, action_size, action_range=None):
#         super(Actor, self).__init__()
#         hidden_units = 256
#
#         if action_range is None:
#             action_range = [[-1, 1] for _ in range(action_size)]
#         action_range = np.array(action_range)
#         self.action_0 = torch.from_numpy(action_range[:, 0]).float()
#         self.action_range = torch.from_numpy(
#             np.diff(action_range, axis=1)[:, 0]).float()
#
#         self.dropout = nn.Dropout()
#         self.pi = nn.Sequential(
#             nn.Linear(in_features=state_size, out_features=hidden_units),
#             nn.ELU(),
#             self.dropout,
#             nn.Linear(in_features=hidden_units, out_features=hidden_units),
#             nn.ELU(),
#             nn.Linear(in_features=hidden_units, out_features=action_size),
#         )
#
#     def forward(self, state):
#         x = self.pi(state)
#         x = F.tanh(x)
#         return x
#
#     def to(self, *args, **kwargs):
#         self.action_0 = self.action_0.to(*args, **kwargs)
#         self.action_range = self.action_range.to(*args, **kwargs)
#         return super(Actor, self).to(*args, **kwargs)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_range=None, seed=32,
                 fc1_units=256,
                 fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)
        self.b2 = nn.BatchNorm1d(action_size)
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = self.b2(x)
        x = self.tanh(x)

        return x
