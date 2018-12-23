from functools import partial

import numpy as np
from agent import Agent


def unity_episode(env, agent: Agent, brain_name, max_t=10000, train=True):
    env_info = env.reset(train_mode=train)[brain_name]
    state = np.array(env_info.vector_observations)
    score = 0
    for t in range(max_t):
        action = agent.policy(state, train)
        env_info = env.step(action)[brain_name]
        next_state = np.array(env_info.vector_observations)
        reward = np.array(env_info.rewards)
        done = np.array(env_info.local_done, dtype=np.uint8)
        if train:
            agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward.mean()

        if done.any():
            break

    agent.end_of_episode(score)
    return score


def wrap_env(env, brain_name):
    return partial(unity_episode, env, brain_name=brain_name)
