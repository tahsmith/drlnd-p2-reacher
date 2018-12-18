from collections import deque
from itertools import product
from operator import itemgetter

import gym
import sys
import torch
import numpy as np

from agent import Agent

if len(sys.argv) > 1:
    game = sys.argv[1]
else:
    game = 'LunarLanderContinuous-v2'
env = gym.make(game)
env.seed(2)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

def run(config, max_eps):

    agent = Agent(device,
                  state_size=state_size,
                  action_size=action_size,
                  action_range=list(zip(env.action_space.low,
                                         env.action_space.high)),
                  **config
                  )

    scores = []
    scores_deque = deque()
    for i_episode in range(1, max_eps):
        state = env.reset()[np.newaxis, :]
        score = 0
        for t in range(1000):
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action[0, :])
            next_state = next_state[np.newaxis, :]
            reward = np.array([[reward]])
            done = np.array([[done]], dtype=np.uint8)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward[0][0]
            if done:
                break
        scores.append(score)
        scores_deque.append(score)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(
            scores_deque)), end="")
        if i_episode % 100 == 0 or i_episode == max_eps - 1:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode,
                                                               np.mean(
                                                                   scores_deque)))
        agent.end_of_episode(score)
    return scores


def product_in_dict(d):
    keys = list(d.keys())
    values = product(*d.values())
    for value in values:
        yield dict(zip(keys, value))


def param_sweep(configs, max_eps):
    scores = []
    trials = list(product_in_dict(configs))
    print('Trials: {}'.format(len(trials)))
    for config in trials:
        print(config)
        result = run(config, max_eps)[-100:]
        score = sum(result[-100:]) / 100.0
        scores.append((score, config))

    for score, config in sorted(scores, key=itemgetter(0)):
        print('{}: {!r}'.format(score, config))


configs = dict(
    buffer_size=[int(1e6)],
    batch_size=[64],
    actor_learning_rate=[1e-4],
    critic_learning_rate=[1e-3],
    discount_rate=[0.99],
    tau=[1e-3],
    steps_per_update=[1],
    weight_decay=[0.01],
    noise_decay=[1.0],
    noise_max=[0.0],
    dropout_p=[0.2]
)

param_sweep(configs, 1000)
