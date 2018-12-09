from collections import deque

import gym
import torch
import numpy as np

from agent import Agent

env = gym.make('MountainCarContinuous-v0')
env.seed(2)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

agent = Agent(device,
              state_size=env.observation_space.shape[0],
              action_size=env.action_space.shape[0],
              buffer_size=int(1e5),
              batch_size=64, learning_rate=1e-4, discount_rate=0.99,
              tau=1e-4, steps_per_update=4,
              action_range=list(zip(env.action_space.low,
                                     env.action_space.high)))

scores = []
scores_deque = deque()
for i_episode in range(1, int(1e5)):
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
        score += reward
        if done:
            break
    scores.append(score)
    scores_deque.append(score)

    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(
        scores_deque)), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode,
                                                           np.mean(
                                                               scores_deque)))
    agent.end_of_episode(score)
