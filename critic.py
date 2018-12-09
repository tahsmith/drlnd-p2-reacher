import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.qa = nn.Sequential(
          nn.Linear(in_features=state_size + action_size, out_features=64),
          nn.ELU(),
          nn.Linear(in_features=64, out_features=64),
          nn.ELU(),
          nn.Linear(in_features=64, out_features=1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.qa(x)
