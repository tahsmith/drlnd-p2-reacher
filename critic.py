import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        hidden_units = 400
        self.dropout = nn.Dropout()

        self.state_layer = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=hidden_units),
            nn.ELU()
        )
        self.qa = nn.Sequential(
            nn.ELU(),
            self.dropout,
            nn.Linear(in_features=hidden_units + action_size,
                      out_features=hidden_units),
            nn.ELU(),
            nn.Linear(in_features=hidden_units, out_features=1),
        )

    def forward(self, state, action):
        x = self.state_layer(state)
        x = torch.cat([x, action], dim=1)
        return self.qa(x)
