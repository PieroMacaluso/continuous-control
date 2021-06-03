import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy
from collections import namedtuple, deque

import numpy as np

import random

from models.ddpg.networks import Actor, Critic
from models.ddpg.replay_buffer import ReplayBuffer
from utils.noises import OUNoise


class Agent:

    def __init__(self, hp):

        self.hp = hp
        self.n_agents = hp['num_agents']
        self.action_min = hp['action_min']
        self.action_max = hp['action_max']
        self.batch_size = hp['batch_size']
        self.gamma = hp['gamma']
        self.tau = hp['tau']
        self.seed = hp['random_seed']
        self.device = torch.device(hp['device'])
        self.update_every = hp['update_every']
        self.target_update_every = hp['target_update_every']
        self.learning_steps = hp['learning_steps']
        self.gradient_clipping_critic = hp['gradient_clipping_critic']
        self.gradient_clipping_actor = hp['gradient_clipping_actor']

        # Actor networks local and target
        self.actor_local = Actor(hp['state_size'], hp['action_size'], self.seed, hp['actor_layers']).to(self.device)
        self.actor_target = Actor(hp['state_size'], hp['action_size'], self.seed, hp['actor_layers']).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=hp['actor_lr'])

        self.critic_local = Critic(hp['state_size'], hp['action_size'], self.seed, hp['critic_layers']).to(self.device)
        self.critic_target = Critic(hp['state_size'], hp['action_size'], self.seed, hp['critic_layers']).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=hp['critic_lr'])

        self.noise = OUNoise((self.n_agents, hp['action_size']), self.seed)

        self.memory = ReplayBuffer(hp['action_size'], hp['replay_mem_size'], self.batch_size, self.seed,
                                   self.device)

        # Create directory for experiment
        self.dir = f"{hp['results_path']}/{hp['env']}-{hp['model']}-{datetime.now():%Y-%m-%d_%H:%M:%S}/"
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.checkpoint = f"{hp['results_path']}/{hp['env']}-{hp['model']}-{datetime.now():%Y-%m-%d_%H:%M:%S}/checkpoints/"
        if not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint)
        # for learn every update step
        self.steps = 0
        self.t_steps = 0
        self.actor_loss = 0.0
        self.critic_loss = 0.0

    def act(self, state, add_noise=True, damping_noise=1.0):

        state = torch.from_numpy(state).float().to(self.device)  # convert state from numpy array to a tensor
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * damping_noise
            action = np.clip(action, self.action_min, self.action_max)
        return action

    def step(self, state, action, reward, next_state, done):
        for i in range(self.n_agents):
            self.memory.add(state[i, :], action[i, :], reward[i], next_state[i, :], done[i])
        self.steps += 1

        if len(self.memory) > self.batch_size * self.n_agents:
            if self.steps % self.update_every == 0:
                for _ in range(self.learning_steps):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma)

            if self.t_steps % self.target_update_every == 0:
                self.soft_update(self.actor_local, self.actor_target, self.tau)
                self.soft_update(self.critic_local, self.critic_target, self.tau)

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.gradient_clipping_critic:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.gradient_clipping_actor:
            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()
        self.t_steps += 1
        self.actor_loss = actor_loss.data
        self.critic_loss = critic_loss.data
        if self.t_steps % self.target_update_every == 0:
            self.soft_update(self.actor_local, self.actor_target, self.tau)
            self.soft_update(self.critic_local, self.critic_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_params, local_params in zip(target_model.parameters(), local_model.parameters()):
            target_params.data.copy_(tau * local_params.data + (1 - tau) * target_params.data)

    def reset(self):
        self.noise.reset()

    def load_weights(self, pth_path):
        self.actor_local.load_state_dict(torch.load(pth_path))
        self.actor_target.load_state_dict(torch.load(pth_path))

    def save_weights(self, i_episode):
        torch.save(self.actor_local.state_dict(), self.checkpoint + f'checkpoint_{i_episode}.pth')
