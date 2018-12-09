import platform
from contextlib import contextmanager

import numpy as np
from unityagents import UnityEnvironment

from agent import Agent

if platform.system() == 'Darwin':
    REACHER_APP = './Reacher.app'
else:
    REACHER_APP = './Reacher_Linux_NoVis/Reacher.x86_64'


@contextmanager
def make_reacher_env():
    env = UnityEnvironment(file_name=REACHER_APP)
    yield env
    env.close()


def reacher_episode(env, agent: Agent, brain_name, max_t=1000):
    env_info = env.reset(train_mode=True)[brain_name]
    state = np.array(env_info.vector_observations)
    score = 0
    for t in range(max_t):
        action = agent.policy(state)
        env_info = env.step(action)[brain_name]
        next_state = np.array(env_info.vector_observations)
        reward = np.array(env_info.rewards)
        done = np.array(env_info.local_done, dtype=np.uint8)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward

        if done.any():
            break

    agent.end_of_episode(score)
    return score
