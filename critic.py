import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        hidden_units = 64
        input_size = state_size + action_size
        self.dropout = nn.Dropout()
        self.qa = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_units),
            nn.ELU(),
            self.dropout,
            nn.Linear(in_features=hidden_units,
                            out_features=hidden_units),
            nn.ELU(),
            self.dropout,
            nn.Linear(in_features=hidden_units, out_features=1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.qa(x)
