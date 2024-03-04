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

    epsilon_start = float(params["epsilon_start"])
    epsilon_decay = float(params["epsilon_decay"])
    num_episodes = int(params["num_episodes"])
    gamma = float(params["gamma"])
    min_reward = float(params["min_reward"])
    
    results_dir = params["results_dir"]
    model_file_name = params["model_file_name"]
    save_frequency = float(params["save_frequency"])

    
    data_saver = DataSaver(results_dir, model_file_name, save_frequency)

    # save_parameters
    data_saver.save_params_to_json(params)
    # link data saver to env
    env.data_saver = data_saver


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
                                            min_reward= min_reward,
                                            data_saver = data_saver)
    
    # saving finished model
    data_saver.save_model_data(q_network, num_episodes, True)
    env.close()

    return episode_reward_list



def dqn1_agent_training(env, q_network, params):

    epsilon_start = float(params["epsilon_start"])
    epsilon_decay = float(params["epsilon_decay"])
    num_episodes = int(params["num_episodes"])
    gamma = float(params["gamma"])
    min_reward = float(params["min_reward"])

    results_dir = params["results_dir"]
    model_file_name = params["model_file_name"]
    save_frequency = float(params["save_frequency"])


    data_saver = DataSaver(results_dir, model_file_name, save_frequency)

    # save_parameters
    data_saver.save_params_to_json(params)
    # link data saver to env
    env.data_saver = data_saver

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
                                            min_reward= min_reward,
                                            
                                            data_saver = data_saver)
    # saving finished model
    data_saver.save_model_data(q_network, num_episodes, True)
    env.close()

    return episode_reward_list


def dqn2_agent_training(env, q_network, target_q_network, params):

    epsilon_start = float(params["epsilon_start"])
    epsilon_decay = float(params["epsilon_decay"])
    num_episodes = int(params["num_episodes"])
    gamma = float(params["gamma"])
    min_reward = float(params["min_reward"])

    results_dir = params["results_dir"]
    model_file_name = params["model_file_name"]
    save_frequency = float(params["save_frequency"])

    data_saver = DataSaver(results_dir, model_file_name, save_frequency)

    # save_parameters
    data_saver.save_params_to_json(params)
    # link data saver to env
    env.data_saver = data_saver

    # Instantiate required objects
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
                                        min_reward = min_reward,

                                        data_saver = data_saver)

    # saving finished model
    data_saver.save_model_data(q_network, num_episodes, True)
    env.close()

    return episode_reward_list



def dqn2_strict_agent_training(env, q_network, target_q_network, params):

    epsilon_start = float(params["epsilon_start"])
    epsilon_decay = float(params["epsilon_decay"])
    num_episodes = int(params["num_episodes"])
    gamma = float(params["gamma"])
    min_reward = float(params["min_reward"])

    results_dir = params["results_dir"]
    model_file_name = params["model_file_name"]
    save_frequency = float(params["save_frequency"])

    max_nb_negative = float(params["max_nb_negative"])

    data_saver = DataSaver(results_dir, model_file_name, save_frequency)

    # save_parameters
    data_saver.save_params_to_json(params)
    # link data saver to env
    env.data_saver = data_saver

    # Instantiate required objects

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
    episode_reward_list = train_dqn2_strict_agent(env,
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
                                        min_reward = min_reward,
                                        max_nb_negative = max_nb_negative,
                                        data_saver = data_saver)

    # saving finished model
    data_saver.save_model_data(q_network, num_episodes, True)
    env.close()

    return episode_reward_list


def reinforce_discrete_agent_training(env, policy_network, params):

    num_episodes = int(params["num_episodes"])
    num_test_per_episode = int(params["num_test_per_episode"])
    max_episode_duration = int(params["max_episode_duration"])
    lr = float(params["lr"])
    min_reward = float(params["min_reward"])
    # max_nb_negative = float(params["max_nb_negative"])

    results_dir = params["results_dir"]
    model_file_name = params["model_file_name"]
    save_frequency = float(params["save_frequency"])

    data_saver = DataSaver(results_dir, model_file_name, save_frequency)

    # save_parameters
    data_saver.save_params_to_json(params)
    # link data saver to env
    env.data_saver = data_saver

    # Instantiate required objects
    policy_network = policy_network.to(device)
    # Train the policy-network
    reinforce_policy_nn, episode_reward_list = train_reinforce_discrete(env=env,
                                                                        num_train_episodes=num_episodes,
                                                                        num_test_per_episode=num_test_per_episode,
                                                                        max_episode_duration=max_episode_duration,
                                                                        learning_rate=lr,
                                                                        min_reward=min_reward,
                                                                        data_saver=data_saver,
                                                                        policy_nn=policy_network)
    # Save the action-value estimation function of the last train
    data_saver.save_model_data(reinforce_policy_nn, num_episodes, True)
    env.close()
    
    return episode_reward_list
