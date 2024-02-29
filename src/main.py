from agents.DQN_train import *

import car_racing
import car_racing_mod

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as trans
import numpy as np

import sys


# NOT USED 

# perform result state frame preprocessing
def rgb_to_grayscale(screen):
    # create transcormation that transforms imag to grayscale
    transformation = trans.Compose([trans.ToTensor(), trans.ToPILImage(), trans.Grayscale(), trans.ToTensor()])

    # remove negative axis values from tensor
    # (they are just there somehow and this fixes them to start at zero)
    screen = np.flip(screen, axis = 0).copy()
    screen = np.flip(screen, axis=0).copy()
    # perform afforementioned transformation
    screen = transformation(screen).to(device)

    # cut track features from image
    track = screen[:, :66, 15:81]
    track[track < 0.5] = 0
    track[track > 0.5] = 1
    track = track.view(1, 1, 66, 66)


    return track


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
        x = self.fc2(x)

        return x



## Application part



def test_saved_model():
    file = input("saved .pth file : ")
    model = torch.load(file, map_location = device)
    env = CarRacing(render_mode="human", obstacle_prob = 0)

    test_agent(env, model, 10, -30)


def train_model():
    available_algos = ["naive_dqn", "dqn1", "dqn2"]
    params = ["number_of_trainings", "num_episodes", "epsilon_start", "gamma", "min_reward", "save_model_dir"]
    params_value = dict()

    print("Available algos : ", available_algos)
    algo = input("choose one : ")

    if not(algo in available_algos):
        raise Exception("Algo : " + algo + " not supported")

    for p in params:
        params_value[p] = input(p + " : ")

    print("\nrender mode : human, nothing (press ENTER)")
    render_mode = input("Render mode : ")
    car_vesion = input("Original gym car env ? (y/n) : ")
    
    car_env = car_racing.CarRacing
    
    if car_vesion == "n":
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