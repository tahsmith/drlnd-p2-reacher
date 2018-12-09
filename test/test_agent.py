import pytest
import torch
import numpy as np

from agent import (
    Agent
)


@pytest.fixture
def agent(device):
    return Agent(device, 2, 3)


# def test_epsilon_greedy_policy(device, agent):
#     state = torch.randn((10, )).to(device)
#     assert agent.epsilon_greedy_policy(1, state) in range(10)


def test_learn(agent: Agent):
    for i in range(20):
        agent.replay_buffer.add(np.zeros(2), np.zeros(3), 0.0, np.zeros(2), 1)
    agent.learn()


def test_save_restore(agent, tmpdir):
    path = str(tmpdir + '/model')
    agent.save(path)
    agent.restore(path)
