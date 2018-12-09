import copy
import random

import torch
import torch.optim
import numpy as np

from critic import Critic
from actor import Actor
from replay_buffer import ReplayBuffer


class Agent:
    def __init__(self, device, state_size, action_size, buffer_size=10,
                 batch_size=10,
                 learning_rate=1e-4,
                 discount_rate=0.99,
                 eps_decay=0.9,
                 tau=0.1,
                 steps_per_update=4
                 ):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size

        self.critic_control = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_control.parameters(),
            lr=learning_rate)

        self.actor_control = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_control.parameters(),
                                                lr=learning_rate)

        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(device, state_size, action_size,
                                          buffer_size)

        self.discount_rate = discount_rate

        self.eps = 1.0
        self.eps_decay = eps_decay

        self.tau = tau

        self.step_count = 0
        self.steps_per_update = steps_per_update

        self.noise = OUNoise(action_size, 15071988)

    def policy(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(self.device)
        action = self.actor_control(state).detach().cpu().numpy()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    #
    # def epsilon_greedy_policy(self, eps, state):
    #     self.critic_control.eval()
    #     with torch.no_grad():
    #         action_values = self.critic_control(state)
    #     self.critic_control.train()
    #
    #     if random.random() > eps:
    #         greedy_choice = np.argmax(action_values.cpu().data.numpy())
    #         return greedy_choice
    #     else:
    #         return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        p = self.calculate_p(state, action, reward, next_state, done)
        self.replay_buffer.add(state, action, reward, next_state, done, p)
        if self.step_count % self.steps_per_update == 0:
            self.learn()
        self.step_count += 1

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, p = \
            self.replay_buffer.sample(self.batch_size)

        error = self.bellman_eqn_error(
            states, actions, rewards, next_states, dones)
        importance_scaling = (self.replay_buffer.buffer_size * p) ** -1
        loss = (importance_scaling * (error ** 2)).sum() / self.batch_size
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        loss = -self.critic_control(states, self.actor_control(states)).mean()
        loss.backward()
        self.actor_optimizer.step()

        self.update_target(self.critic_control, self.critic_target)
        self.update_target(self.actor_control, self.actor_target)

    def bellman_eqn_error(self, states, actions, rewards, next_states, dones):
        """Double DQN error - use the control network to get the best action
        and apply the target network to it to get the target reward which is
        used for the bellman eqn error.
        """
        self.critic_control.eval()
        with torch.no_grad():
            next_actions = self.actor_control(next_states)

        target_action_values = self.critic_target(next_states, next_actions)

        target_rewards = rewards + self.discount_rate * (1 - dones) \
                         * target_action_values

        self.critic_control.train()
        current_rewards = self.critic_control(states, actions)
        error = current_rewards - target_rewards
        return error

    def calculate_p(self, state, action, reward, next_state, done):
        next_state = torch.from_numpy(next_state[np.newaxis, :]).float().to(
            self.device)
        state = torch.from_numpy(state[np.newaxis, :]).float().to(self.device)
        action = torch.from_numpy(action[np.newaxis, :]).float().to(self.device)
        reward = torch.from_numpy(np.array([reward])).float().to(self.device)
        done = torch.from_numpy(np.array([[done]], dtype=np.uint8)).float().to(
            self.device)

        return abs(self.bellman_eqn_error(state, action, reward, next_state,
                                          done)) + 1e-3

    def update_target(self, control, target):
        for target_param, control_param in zip(
                target.parameters(),
                control.parameters()):
            target_param.data.copy_(
                self.tau * control_param.data + (1.0 - self.tau) *
                target_param.data)

    def end_of_episode(self):
        self.eps *= self.eps_decay
        self.step_count = 0

    def save(self, path):
        torch.save(self.critic_control.state_dict(), path)

    def restore(self, path):
        self.critic_control.load_state_dict(torch.load(path))


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
