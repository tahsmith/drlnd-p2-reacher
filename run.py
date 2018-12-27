import sys
from unityagents import UnityEnvironment

from unity_env import wrap_env

import torch
from agent import default_agent


def main(argv):
    env_path = argv[1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = UnityEnvironment(file_name=env_path)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]

    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    agent = default_agent(device, state_size, action_size)

    agent.restore('best')

    episode_fn = wrap_env(env, brain_name, train=False)

    return run(episode_fn, agent)


def run(episode_fn, agent):
    while True:
        episode_fn(agent)


if __name__ == '__main__':
    main(sys.argv)
