from reacher_env import reacher_episode, make_reacher_env

import torch
from agent import Agent


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = make_reacher_env()
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    agent = Agent(
        device,
        state_size,
        action_size,
        buffer_size=int(1e6),
        batch_size=64,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        discount_rate=0.99,
        tau=1e-3,
        steps_per_update=5,
        weight_decay=0.00,
        noise_decay=0.999,
        noise_max=0.2,
        dropout_p=0.2,
        n_agents=20
    )

    return train(env, agent, brain_name)


def train(env, agent, brain_name, window_size=100, max_eps=int(2e5),
          min_score=30.0):
    scores = []
    best_score = float('-inf')

    for i in range(max_eps):
        score = reacher_episode(env, agent, brain_name)
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
    main()
