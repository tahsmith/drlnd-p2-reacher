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

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    episode_fn = wrap_env(env, brain_name)

    agent = default_agent(device, state_size, action_size)

    return train(episode_fn, agent)


def train(episode_fn, agent, window_size=100, max_eps=int(2e5),
          min_score=400.0):
    scores = []
    best_score = float('-inf')

    for i in range(max_eps):
        score = episode_fn(agent)
        scores.append(score)

        print('\r{i} - {score:.2f}'.format(i=i + 1, score=score), end="")

        if (i + 1) % window_size == 0:
            avg_score = sum(scores[-window_size:]) / window_size

            print('\r{i} - {score:.2f}'.format(i=i + 1, score=avg_score))

            if avg_score > best_score:
                best_score = avg_score
                agent.save('best')

            if avg_score > min_score:
                break
    return scores


if __name__ == '__main__':
    main(sys.argv)