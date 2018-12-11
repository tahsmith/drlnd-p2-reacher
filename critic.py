import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

#
# class Critic(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(Critic, self).__init__()
#         hidden_units = 128
#         self.dropout = nn.Dropout()
#
#         self.state_layer = nn.Sequential(
#             nn.Linear(in_features=state_size, out_features=hidden_units),
#             nn.ELU()
#         )
#         self.qa = nn.Sequential(
#             nn.ELU(),
#             self.dropout,
#             nn.Linear(in_features=hidden_units + action_size,
#                       out_features=hidden_units),
#             nn.ELU(),
#             nn.Linear(in_features=hidden_units, out_features=hidden_units),
#             nn.ELU(),
#             nn.Linear(in_features=hidden_units, out_features=1),
#         )
#
#     def forward(self, state, action):
#         x = self.state_layer(state)
#         x = torch.cat([x, action], dim=1)
#         return self.qa(x)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed=32, fc1_units=128,
                 fc2_units=128, fc3_units=128, fc4_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimesion of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc2_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc3_units, fc4_units)
        self.fc4 = nn.Linear(fc4_units, 1)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        xs = F.leaky_relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        return x
