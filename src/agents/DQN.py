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
                      min_reward: int,
                      data_saver) -> List[float]:
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
        # some action before learning, defined in help_func.py
        after_reset_action(env)

        episode_reward = 0

        local_reward_list = []

        for t in itertools.count():
            # TODO...
            a = epsilon_greedy(state)

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
            if (episode_reward < min_reward):
                done = True

            # if it is a final state
            if done: break
        
        max_reward = max(local_reward_list)
        print("reward at the end = ", episode_reward)
        print("reward max        = ", max_reward)

        # save model and rewards 
        data_saver.save_model_data(q_network, episode_index,
                                   reward = max_reward)
        data_saver.save_rewards_data(
            {"final" : episode_reward, "max" : max_reward},
            episode_index)
        data_saver.save_gif(max_reward)

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
                     min_reward: int,
                     data_saver) -> List[float]:
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
        
        # some action before learning, defined in help_func.py
        after_reset_action(env)

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
              
            # test if reward is so low so done
            if (episode_reward < min_reward):
                done = True

            if done:
                break

            state = next_state

        max_reward = max(local_reward_list)
        print("reward at the end = ", episode_reward)
        print("reward max        = ", max_reward)

        # save model and rewards 
        data_saver.save_model_data(q_network, episode_index,
                                   reward = max_reward)
        data_saver.save_rewards_data(
            {"final" : episode_reward, "max" : max_reward},
            episode_index)
        data_saver.save_gif(max_reward)

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
                     min_reward: int,
                     data_saver) -> List[float]:
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
        
        # some action before learning, defined in help_func.py
        after_reset_action(env)

        episode_reward = 0

        local_reward_list = []

        for t in itertools.count():

            # Get action, next_state and reward

            action = epsilon_greedy(state)

            next_state, reward, terminated, truncated, info = custom_step(env, action) # env.step(action)
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

            if (episode_reward < min_reward):
                done = True
            
            if done:
                break

            state = next_state
        
        max_reward = max(local_reward_list)
        print("reward at the end = ", episode_reward)
        print("reward max        = ", max_reward)

        # save model and rewards 
        data_saver.save_model_data(q_network, episode_index,
                                   reward = max_reward)
        data_saver.save_rewards_data(
            {"final" : episode_reward, "max" : max_reward},
            episode_index)
        data_saver.save_gif(max_reward)

        episode_reward_list.append(episode_reward)
        epsilon_greedy.decay_epsilon()

    return episode_reward_list






def train_dqn2_strict_agent(env: gym.Env,
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
                     min_reward: int,
                     max_nb_negative: float,
                     data_saver) -> List[float]:
    """
    Train the Q-network on the given environment.
    This is a modified dqn2 environment.
    The episode finish when the agent get multiple successive negative rewards, 
    or the cumulated rewards is less then a given value

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
        
        # some action before learning, defined in help_func.py
        after_reset_action(env)

        episode_reward = 0
        nb_neg = 0

        local_reward_list = []

        for t in itertools.count():

            # Get action, next_state and reward

            action = epsilon_greedy(state)

            next_state, reward, terminated, truncated, info = custom_step(env, action)# env.step(action)
            done = terminated or truncated

            replay_buffer.add(state, action, reward, next_state, done)

            episode_reward += reward
            
            if reward < 0:
                nb_neg += 1
            else:
                nb_neg = 0

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

            if (episode_reward < min_reward) or nb_neg >= max_nb_negative:
                print("nb neg = ", nb_neg)
                done = True
            
            if done:
                break

            state = next_state
        
        max_reward = max(local_reward_list)
        print("reward at the end = ", episode_reward)
        print("reward max        = ", max_reward)

        # save model and rewards 
        data_saver.save_model_data(q_network, episode_index,
                                   reward = max_reward)
        data_saver.save_rewards_data(
            {"final" : episode_reward, "max" : max_reward},
            episode_index)
        data_saver.save_gif(max_reward)

        episode_reward_list.append(episode_reward)
        epsilon_greedy.decay_epsilon()

    return episode_reward_list



def train_dqn2_modified_agent(env: gym.Env,
                     q_network: torch.nn.Module,
                     target_q_network: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     loss_fn: Callable,
                     device: torch.device,
                     lr_scheduler: _LRScheduler,
                     num_episodes: int,
                     gamma: float,
                     batch_size: int,
                     replay_buffer: ReplayBuffer,
                     target_q_network_sync_period: int,
                     min_reward: int,
                     limit_nb_negative_before_exploration : float, 
                     max_nb_negative: float,
                     data_saver) -> List[float]:
    """
    Train the Q-network on the given environment.
    This is a modified dqn2 environment.
    The episode finish when the agent get multiple successive negative rewards, 
    or the cumulated rewards is less then a given value

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
        
        # some action before learning, defined in help_func.py
        after_reset_action(env)

        episode_reward = 0
        nb_neg = 0

        local_reward_list = []

        for t in itertools.count():

            # Get action, next_state and reward
            
            # choose q_network output when the car haven't done many error
            if (nb_neg <= limit_nb_negative_before_exploration):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = torch.argmax(q_network(state_tensor)).item()
            else:
                action = np.random.randint(NUMBER_ACTIONS)

            next_state, reward, terminated, truncated, info = custom_step(env, action)# env.step(action)
            done = terminated or truncated

            replay_buffer.add(state, action, reward, next_state, done)

            episode_reward += reward
            
            if reward < 0:
                nb_neg += 1
            else:
                nb_neg = 0

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

            if (episode_reward < min_reward) or nb_neg >= max_nb_negative:
                print("nb neg = ", nb_neg)
                done = True
            
            if done:
                break

            state = next_state
        
        max_reward = max(local_reward_list)
        print("reward at the end = ", episode_reward)
        print("reward max        = ", max_reward)

        # save model and rewards 
        data_saver.save_model_data(q_network, episode_index, 
                                   reward = max_reward)
        data_saver.save_rewards_data(
            {"final" : episode_reward, "max" : max_reward},
            episode_index)
        data_saver.save_gif(max_reward)


        episode_reward_list.append(episode_reward)

    return episode_reward_list


def train_reinforce_discrete(env: gym.Env,
                             num_train_episodes: int,
                             num_test_per_episode: int,
                             max_episode_duration: int,
                             learning_rate: float,
                             min_reward: float,
                             policy_nn: torch.nn.Module,
                             data_saver) -> Tuple[torch.nn.Module, List[float]]:
    """
    Train a policy using the REINFORCE algorithm.

    Parameters
    ----------
    env : gym.Env
        The environment to train in.
    num_train_episodes : int
        The number of training episodes.
    num_test_per_episode : int
        The number of tests to perform per episode.
    max_episode_duration : int
        The maximum length of an episode, by default EPISODE_DURATION.
    learning_rate : float
        The initial step size.

    Returns
    -------
    Tuple[PolicyNetwork, List[float]]
        The final trained policy and the average returns for each episode.
    """
    episode_avg_return_list = []

    optimizer = torch.optim.Adam(policy_nn.parameters(), lr=learning_rate)

    for episode_index in tqdm(range(num_train_episodes)):

        # TODO...
        episode_states, episode_actions, episode_rewards, episode_log_prob_actions = sample_one_episode(env, policy_nn, max_episode_duration, min_reward=min_reward)
        returns = []
        for i in range(len(episode_rewards)):
            returns.append(sum(episode_rewards[i:]))

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        log_prob_actions = torch.stack(episode_log_prob_actions)
        loss = -torch.sum(log_prob_actions * returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Test the current policy
        test_avg_return = avg_return_on_multiple_episodes(env=env,
                                                          policy_nn=policy_nn,
                                                          num_test_episode=num_test_per_episode,
                                                          max_episode_duration=max_episode_duration,
                                                          min_reward=min_reward,
                                                          render=False)
        
        # max reward
        max_reward = max(episode_rewards)
        print("reward at the end = ", test_avg_return)
        print("reward max        = ", max_reward)

        # save model and rewards 
        data_saver.save_model_data(policy_nn, episode_index,
                                   reward = max_reward)
        data_saver.save_rewards_data(
            {"final" : test_avg_return, "max" : max_reward},
            episode_index)
        data_saver.save_gif(max_reward)
        # Monitoring
        episode_avg_return_list.append(test_avg_return)
    return policy_nn, episode_avg_return_list