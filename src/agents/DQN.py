import gymnasium as gym
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
import itertools
from tqdm import tqdm
import collections
import random

from typing import List, Tuple, Deque, Optional, Callable

from agents.help_func import *




def train_naive_agent(env: gym.Env,
                      q_network: torch.nn.Module,
                      optimizer: torch.optim.Optimizer,
                      loss_fn: Callable,
                      epsilon_greedy: EpsilonGreedy,
                      device: torch.device,
                      lr_scheduler: _LRScheduler,
                      num_episodes: int,
                      gamma: float,
                      min_reward: int) -> List[float]:
    """
    Train the Q-network on the given environment.

    Parameters
    ----------
    env : gym.Env
        The environment to train on.
    q_network : torch.nn.Module
        The Q-network to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    loss_fn : callable
        The loss function to use for training.
    epsilon_greedy : EpsilonGreedy
        The epsilon-greedy policy to use for action selection.
    device : torch.device
        The device to use for PyTorch computations.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler to adjust the learning rate during training.
    num_episodes : int
        The number of episodes to train for.
    gamma : float
        The discount factor for future rewards.
    min_reward: int
        The min reward to stop the episode

    Returns
    -------
    List[float]
        A list of cumulated rewards per episode.
    """
    episode_reward_list = []
    q_network.to(device)
    q_network.train()

    for episode_index in tqdm(range(1, num_episodes)):
        state, info = env.reset()
        episode_reward = 0

        local_reward_list = []

        for t in itertools.count():
            # TODO...
            a = epsilon_greedy(state)

            #print("a = ", a, ", rewards = ", episode_reward)
            s_after, r , done = custom_step(env, a)[:3] # env.step(a)[:3]
            episode_reward += r
            s_after_tensor = torch.tensor(s_after, dtype=torch.float32, device=device).unsqueeze(0)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            max_qa_s_after = torch.max(q_network(s_after_tensor))
            loss = loss_fn(r + gamma * max_qa_s_after, q_network(state_tensor)[0, a])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            local_reward_list.append(episode_reward)

            state = s_after

            # test if reward is so low so done
            done = (episode_reward < min_reward)

            # if it is a final state
            if done: break
        
        print("reward at the end = ", episode_reward)
        print("reward max        = ", max(local_reward_list))

        episode_reward_list.append(episode_reward)
        epsilon_greedy.decay_epsilon()

    return episode_reward_list




    


def train_dqn1_agent(env: gym.Env,
                     q_network: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     loss_fn: Callable,
                     epsilon_greedy: EpsilonGreedy,
                     device: torch.device,
                     lr_scheduler: _LRScheduler,
                     num_episodes: int,
                     gamma: float,
                     batch_size: int,
                     replay_buffer: ReplayBuffer,
                     min_reward: int) -> List[float]:
    """
    Train the Q-network on the given environment.

    Parameters
    ----------
    env : gym.Env
        The environment to train on.
    q_network : torch.nn.Module
        The Q-network to train.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    loss_fn : callable
        The loss function to use for training.
    epsilon_greedy : EpsilonGreedy
        The epsilon-greedy policy to use for action selection.
    device : torch.device
        The device to use for PyTorch computations.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler to adjust the learning rate during training.
    num_episodes : int
        The number of episodes to train for.
    gamma : float
        The discount factor for future rewards.
    batch_size : int
        The size of the batch to use for training.
    replay_buffer : ReplayBuffer
        The replay buffer storing the experiences with their priorities.
    min_reward: int
        The min reward to stop the episode

    Returns
    -------
    List[float]
        A list of cumulated rewards per episode.
    """
    episode_reward_list = []

    for episode_index in tqdm(range(1, num_episodes)):
        state, info = env.reset()
        episode_reward = 0

        local_rewards = []

        for t in itertools.count():

            # Get action, next_state and reward

            action = epsilon_greedy(state)

            next_state, reward, terminated, truncated, info = custom_step(env, action) # env.step(action)
            done = terminated or truncated

            replay_buffer.add(state, action, reward, next_state, done)
            episode_reward += reward

            local_rewards.append(reward)

            if len(replay_buffer) > batch_size:
              batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)

              # Convert to PyTorch tensors
              batch_states_tensor = torch.tensor(batch_states, dtype=torch.float32, device=device)
              batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=device)
              batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
              batch_next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32, device=device)
              batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32, device=device)

              # TODO...
              batch_q_netword_pred_tensor = q_network(batch_next_states_tensor)
              batch_yj_tensor = batch_rewards_tensor + gamma * torch.max(batch_q_netword_pred_tensor, dim=1).values * (1 - batch_dones_tensor)

              second_part = q_network(batch_states_tensor).gather(1, batch_actions_tensor.unsqueeze(1)).squeeze()
              
              loss = loss_fn(batch_yj_tensor, second_part)

              # Optimize the model
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

              lr_scheduler.step()
              
            done = (episode_reward < min_reward)

            if done:
                break

            state = next_state

        print("reward at the end = ", episode_reward)
        print("reward max        = ", max(local_rewards))

        episode_reward_list.append(episode_reward)
        epsilon_greedy.decay_epsilon()

    return episode_reward_list





def train_dqn2_agent(env: gym.Env,
                     q_network: torch.nn.Module,
                     target_q_network: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     loss_fn: Callable,
                     epsilon_greedy: EpsilonGreedy,
                     device: torch.device,
                     lr_scheduler: _LRScheduler,
                     num_episodes: int,
                     gamma: float,
                     batch_size: int,
                     replay_buffer: ReplayBuffer,
                     target_q_network_sync_period: int,
                     min_reward: int) -> List[float]:
    """
    Train the Q-network on the given environment.

    Parameters
    ----------
    env : gym.Env
        The environment to train on.
    q_network : torch.nn.Module
        The Q-network to train.
    target_q_network : torch.nn.Module
        The target Q-network to use for estimating the target Q-values.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    loss_fn : callable
        The loss function to use for training.
    epsilon_greedy : EpsilonGreedy
        The epsilon-greedy policy to use for action selection.
    device : torch.device
        The device to use for PyTorch computations.
    lr_scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler to adjust the learning rate during training.
    num_episodes : int
        The number of episodes to train for.
    gamma : float
        The discount factor for future rewards.
    batch_size : int
        The size of the batch to use for training.
    replay_buffer : ReplayBuffer
        The replay buffer storing the experiences with their priorities.
    target_q_network_sync_period : int
        The number of episodes after which the target Q-network should be updated with the weights of the Q-network.
    min_reward: int
        The min reward to stop the episode

    Returns
    -------
    List[float]
        A list of cumulated rewards per episode.
    """
    iteration = 0
    episode_reward_list = []

    for episode_index in tqdm(range(1, num_episodes)):
        state, info = env.reset()
        episode_reward = 0

        local_reward_list = []

        for t in itertools.count():

            # Get action, next_state and reward

            action = epsilon_greedy(state)

            next_state, reward, terminated, truncated, info = custom_step(env, action)# env.step(action)
            done = terminated or truncated

            replay_buffer.add(state, action, reward, next_state, done)

            episode_reward += reward

            # Update the q_network weights with a batch of experiences from the buffer
            if len(replay_buffer) > batch_size:
              batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size)

              # Convert to PyTorch tensors
              batch_states_tensor = torch.tensor(batch_states, dtype=torch.float32, device=device)
              batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=device)
              batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
              batch_next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32, device=device)
              batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.float32, device=device)

              # TODO...
              batch_q_netword_pred_tensor = target_q_network(batch_next_states_tensor)
              batch_yj_tensor = batch_rewards_tensor + gamma * torch.max(batch_q_netword_pred_tensor, dim=1).values * (1 - batch_dones_tensor)

              loss = loss_fn(batch_yj_tensor, q_network(batch_states_tensor).gather(1, batch_actions_tensor.unsqueeze(1)).squeeze() )

              # Optimize the model
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

              lr_scheduler.step()

            # Update the target q-network

            # Every few training steps (e.g., every 100 steps), the weights of the target network are updated with the weights of the Q-network

            # TODO...
            if iteration % target_q_network_sync_period == 0:
              target_q_network.load_state_dict(q_network.state_dict())

            iteration += 1

            local_reward_list.append(episode_reward)

            done = (episode_reward < min_reward)
            
            if done:
                break

            state = next_state
        
        print("reward at the end = ", episode_reward)
        print("reward max        = ", max(local_reward_list))

        episode_reward_list.append(episode_reward)
        epsilon_greedy.decay_epsilon()

    return episode_reward_list