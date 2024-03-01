import gymnasium as gym
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
import itertools
from tqdm import tqdm
import collections
import random

from typing import List, Tuple, Deque, Optional, Callable


from agents.DQN import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def dqn_naive_agent_training(env, q_network, params):

    number_of_trainings = int(params["number_of_trainings"])
    epsilon_start = float(params["epsilon_start"])
    epsilon_decay = float(params["epsilon_decay"])
    num_episodes = int(params["num_episodes"])
    gamma = float(params["gamma"])
    min_reward = float(params["min_reward"])
    save_model_dir = params["save_model_dir"]

    naive_trains_result_list = [[], [], []]

    for train_index in range(number_of_trainings):

        # Instantiate required objects

        q_network = q_network.to(device)
        optimizer = torch.optim.AdamW(q_network.parameters(), lr=0.004, amsgrad=True)
        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        lr_scheduler = MinimumExponentialLR(optimizer, lr_decay=0.97, min_lr=0.0001)
        loss_fn = torch.nn.MSELoss()

        epsilon_greedy = EpsilonGreedy(epsilon_start=epsilon_start, epsilon_min=0.013, epsilon_decay=epsilon_decay, env=env, q_network=q_network)


        episode_reward_list = train_naive_agent(env,
                                                q_network,
                                                optimizer,
                                                loss_fn,
                                                epsilon_greedy,
                                                device,
                                                lr_scheduler,
                                                num_episodes=num_episodes,
                                                gamma=gamma,
                                                min_reward= min_reward)
        
        naive_trains_result_list[0].extend(range(len(episode_reward_list)))
        naive_trains_result_list[1].extend(episode_reward_list)
        naive_trains_result_list[2].extend([train_index for _ in episode_reward_list])
    
    torch.save(q_network, save_model_dir)
    env.close()

    return naive_trains_result_list



def dqn1_agent_training(env, q_network, params):

    number_of_trainings = int(params["number_of_trainings"])
    epsilon_start = float(params["epsilon_start"])
    epsilon_decay = float(params["epsilon_decay"])
    num_episodes = int(params["num_episodes"])
    gamma = float(params["gamma"])
    min_reward = float(params["min_reward"])
    save_model_dir = params["save_model_dir"]

    trains_result_list = [[], [], []]

    for train_index in range(number_of_trainings):

        # Instantiate required objects

        q_network = q_network.to(device)
        optimizer = torch.optim.AdamW(q_network.parameters(), lr=0.004, amsgrad=True)
        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        lr_scheduler = MinimumExponentialLR(optimizer, lr_decay=0.97, min_lr=0.0001)
        loss_fn = torch.nn.MSELoss()

        epsilon_greedy = EpsilonGreedy(epsilon_start=epsilon_start, epsilon_min=0.013, epsilon_decay=epsilon_decay, env=env, q_network=q_network)


        # Train the q-network
        replay_buffer = ReplayBuffer(2000)

        episode_reward_list = train_dqn1_agent(env,
                                                q_network,
                                                optimizer,
                                                loss_fn,
                                                epsilon_greedy,
                                                device,
                                                lr_scheduler,
                                                num_episodes=num_episodes,
                                                gamma=gamma,
                                                batch_size=128,
                                                replay_buffer=replay_buffer,
                                                min_reward= min_reward)
        
        trains_result_list[0].extend(range(len(episode_reward_list)))
        trains_result_list[1].extend(episode_reward_list)
        trains_result_list[2].extend([train_index for _ in episode_reward_list])

    torch.save(q_network, save_model_dir)
    env.close()

    return trains_result_list


def dqn2_agent_training(env, q_network, target_q_network, params):

    number_of_trainings = int(params["number_of_trainings"])
    epsilon_start = float(params["epsilon_start"])
    epsilon_decay = float(params["epsilon_decay"])
    num_episodes = int(params["num_episodes"])
    gamma = float(params["gamma"])
    min_reward = float(params["min_reward"])
    save_model_dir = params["save_model_dir"]
    
    dqn2_trains_result_list = [[], [], []]

    for train_index in range(number_of_trainings):

        # Instantiate required objects

        # TODO...
        q_network = q_network.to(device)
        target_q_network = target_q_network.to(device)
        
        target_q_network.load_state_dict(q_network.state_dict())

        optimizer = torch.optim.AdamW(q_network.parameters(), lr=0.004, amsgrad=True)
        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        lr_scheduler = MinimumExponentialLR(optimizer, lr_decay=0.97, min_lr=0.0001)
        loss_fn = torch.nn.MSELoss()

        epsilon_greedy = EpsilonGreedy(epsilon_start=epsilon_start, epsilon_min=0.013, epsilon_decay=epsilon_decay, env=env, q_network=q_network)

        replay_buffer = ReplayBuffer(2000)

        # Train the q-network

        episode_reward_list = train_dqn2_agent(env,
                                            q_network,
                                            target_q_network,
                                            optimizer,
                                            loss_fn,
                                            epsilon_greedy,
                                            device,
                                            lr_scheduler,
                                            num_episodes= num_episodes,
                                            gamma= gamma,
                                            batch_size=128,
                                            replay_buffer=replay_buffer,
                                            target_q_network_sync_period=30,
                                            min_reward = min_reward)

        dqn2_trains_result_list[0].extend(range(len(episode_reward_list)))
        dqn2_trains_result_list[1].extend(episode_reward_list)
        dqn2_trains_result_list[2].extend([train_index for _ in episode_reward_list])

    # Save the action-value estimation function

    torch.save(q_network, save_model_dir)

    env.close()

    return dqn2_trains_result_list