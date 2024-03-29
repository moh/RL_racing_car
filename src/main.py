from agents.DQN_train import *
from agents.PPO import *
from agents.TRPO import *

import car_racing
import car_racing_mod

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as trans
import numpy as np

import sys


class CNNModel(nn.Module):
    def __init__(self, len_action_space):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6,
                                kernel_size=7, stride=3)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(12 * 6 * 6, 216)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(216, len_action_space)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Change input format to (batch_size, channels, height, width)
        # to grayscale 
        x = trans.functional.rgb_to_grayscale(x)

        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x

class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=7, stride=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(12 * 6 * 6, 216) 
        #self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(216, 1)  # Output a single value representing the state value

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = trans.functional.rgb_to_grayscale(x)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        state_value = self.fc2(x)
        return state_value

class ActorCritic(nn.Module):
    def __init__(self, len_action_space):
        super(ActorCritic, self).__init__()
        
        self.critic = CriticNetwork()
        self.actor = CNNModel(len_action_space=len_action_space)
        
        
    def forward(self, x):
        return self.actor(x), self.critic(x)

## Application part



def test_saved_model():
    file = input("saved .pth file : ")
    model = torch.load(file, map_location = device)
    car_vesion = input("Original gym car env ? (y/n) : ")
    
    car_env = car_racing.CarRacing(render_mode="human")
    
    if car_vesion == "n":
        car_env = car_racing_mod.CarRacing(render_mode="human")
    
    model_is_ppo = input('Is the model PPO(y/n):')
    if model_is_ppo == 'y':
        test_ppo_agent(car_env, model, 10, -30)
    else:
        params = test_agent(car_env, model, 10, -30)
    title = "../results/testing_data/testing_data_" + file.split("/")[-1].split(".")[-2] # get file name from filepath
    with open(title+".json", "w") as outfile:
        json.dump(params, outfile)


def train_model():
    available_algos = ["naive_dqn", "dqn1", "dqn2", "dqn2_strict", 
                       "dqn2_modified", "reinforce", "ppo", "trpo"]

    common_params = ["num_episodes", "epsilon_start",
              "epsilon_decay", "gamma", "min_reward", 
              "results_dir", "model_file_name", "save_frequency"]

    params = {
        "naive_dqn" : common_params,
        "dqn1" : common_params, 
        "dqn2" : common_params,
        "dqn2_strict" : ["num_episodes", "epsilon_start",
              "epsilon_decay", "gamma", "min_reward", "max_nb_negative",  
              "results_dir", "model_file_name", "save_frequency"],

        "dqn2_modified" : ["num_episodes", "gamma", "min_reward", 
            "limit_nb_negative_before_exploration", "max_nb_negative",  
            "results_dir", "model_file_name", "save_frequency"],

        "reinforce" : ["num_episodes", "max_episode_duration",
                "lr", "temperature", "min_reward",
                "results_dir", "model_file_name", "save_frequency"],
        "ppo" : ["num_steps",
                 "learning_rate", "ppo_epochs", "mini_batch_size",
                 "max_frames", "threshold_reward",
                "results_dir", "model_file_name", "save_frequency"],

        "trpo" : ["num_steps",
                 "learning_rate", "max_frames", "threshold_reward",
                "results_dir", "model_file_name", "save_frequency",
                "damping", "cg_iters", "max_kl"]
    }
    
    params_value = dict()

    print("Available algos : ", available_algos)
    algo = input("choose one : ")

    if not(algo in available_algos):
        raise Exception("Algo : " + algo + " not supported")

    for p in params[algo]:
        params_value[p] = input(p + " : ")

    print("Render mode : human, nothing (press ENTER)")
    render_mode = input("Render mode : ")
    car_version = input("Original gym car env ? (y/n) : ")
    
    car_env = car_racing.CarRacing
    
    params_value["car_version"] = car_version

    if car_version == "n":
        car_env = car_racing_mod.CarRacing

    if render_mode == "":
        env = car_env()
    else:
        env = car_env(render_mode="human")


    if algo == "naive_dqn":
        q_network = CNNModel(NUMBER_ACTIONS)
        dqn_naive_agent_training(env, q_network, params_value)
    
    elif algo == "dqn1":
        q_network = CNNModel(NUMBER_ACTIONS)
        dqn1_agent_training(env, q_network, params_value)
    
    elif algo == "dqn2":
        q_network = CNNModel(NUMBER_ACTIONS)
        target_q_network = CNNModel(NUMBER_ACTIONS)
        dqn2_agent_training(env, q_network, target_q_network, params_value)

    elif algo == "dqn2_strict":
        q_network = CNNModel(NUMBER_ACTIONS)
        target_q_network = CNNModel(NUMBER_ACTIONS)
        dqn2_strict_agent_training(env, q_network, target_q_network, params_value)

    elif algo == "dqn2_modified":
        q_network = CNNModel(NUMBER_ACTIONS)
        target_q_network = CNNModel(NUMBER_ACTIONS)
        dqn2_modified_agent_training(env, q_network, target_q_network, params_value)

    elif algo == "reinforce":
        policy_network = CNNModel(NUMBER_ACTIONS)
        reinforce_discrete_agent_training(env, policy_network, params_value)
    
    elif algo == "ppo":
        actorCritic_network = ActorCritic(NUMBER_ACTIONS)
        ppo_agent_training(env, actorCritic_network, params_value)

    elif algo == "trpo":
        actorCritic_network = ActorCritic(NUMBER_ACTIONS)
        trpo_agent_training(env, actorCritic_network, params_value)


if __name__ == "__main__":
    # a simple arg parsing

    if len(sys.argv) < 2:
        print("You need to specify if train or test")
        sys.exit()

    mode = sys.argv[1]

    if mode == "train":
        train_model()
    
    elif mode == "test":
        test_saved_model()
    
    else:
        sys.exit()