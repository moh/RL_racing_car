import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
import torch
from torch.optim.lr_scheduler import _LRScheduler
import itertools
from tqdm import tqdm
import collections
import random
import os
import json
import imageio

from typing import List, Tuple, Deque, Optional, Callable

device = "cuda" if torch.cuda.is_available() else "cpu"

NUMBER_ACTIONS = 5

class EpsilonGreedy:
    """
    An Epsilon-Greedy policy.

    Attributes
    ----------
    epsilon : float
        The initial probability of choosing a random action.
    epsilon_min : float
        The minimum probability of choosing a random action.
    epsilon_decay : float
        The decay rate for the epsilon value after each action.
    env : gym.Env
        The environment in which the agent is acting.
    q_network : torch.nn.Module
        The Q-Network used to estimate action values.

    Methods
    -------
    __call__(state: np.ndarray) -> np.int64
        Select an action for the given state using the epsilon-greedy policy.
    decay_epsilon()
        Decay the epsilon value after each action.
    """

    def __init__(self,
                 epsilon_start: float,
                 epsilon_min: float,
                 epsilon_decay:float,
                 env: gym.Env,
                 q_network: torch.nn.Module):
        """
        Initialize a new instance of EpsilonGreedy.

        Parameters
        ----------
        epsilon_start : float
            The initial probability of choosing a random action.
        epsilon_min : float
            The minimum probability of choosing a random action.
        epsilon_decay : float
            The decay rate for the epsilon value after each episode.
        env : gym.Env
            The environment in which the agent is acting.
        q_network : torch.nn.Module
            The Q-Network used to estimate action values.
        """
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.env = env
        self.q_network = q_network

    def __call__(self, state: np.ndarray) -> np.int64:
        """
        Select an action for the given state using the epsilon-greedy policy.

        If a randomly chosen number is less than epsilon, a random action is chosen.
        Otherwise, the action with the highest estimated action value is chosen.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        np.int64
            The chosen action.
        """

        # TODO...
        rand_nb = np.random.random()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        if (rand_nb < self.epsilon):
          action = np.random.randint(NUMBER_ACTIONS)#np.random.random()#self.env.action_space.sample()
        else:
          # In case of descrete action 
          action = torch.argmax(self.q_network(state_tensor)).item()

        return action

    def decay_epsilon(self):
        """
        Decay the epsilon value after each episode.

        The new epsilon value is the maximum of `epsilon_min` and the product of the current
        epsilon value and `epsilon_decay`.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)



    
class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer: torch.optim.Optimizer, lr_decay: float, last_epoch: int = -1, min_lr: float = 1e-6):
        """
        Initialize a new instance of MinimumExponentialLR.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate should be scheduled.
        lr_decay : float
            The multiplicative factor of learning rate decay.
        last_epoch : int, optional
            The index of the last epoch. Default is -1.
        min_lr : float, optional
            The minimum learning rate. Default is 1e-6.
        """
        self.min_lr = min_lr
        super().__init__(optimizer, lr_decay, last_epoch=-1)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler.

        Returns
        -------
        List[float]
            The learning rates of each parameter group.
        """
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]


class ReplayBuffer:
    """
    A Replay Buffer.

    Attributes
    ----------
    buffer : collections.deque
        A double-ended queue where the transitions are stored.

    Methods
    -------
    add(state: np.ndarray, action: np.int64, reward: float, next_state: np.ndarray, done: bool)
        Add a new transition to the buffer.
    sample(batch_size: int) -> Tuple[np.ndarray, float, float, np.ndarray, bool]
        Sample a batch of transitions from the buffer.
    __len__()
        Return the current size of the buffer.
    """

    def __init__(self, capacity: int):
        """
        Initializes a ReplayBuffer instance.

        Parameters
        ----------
        capacity : int
            The maximum number of transitions that can be stored in the buffer.
        """
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: np.int64, reward: float, next_state: np.ndarray, done: bool):
        """
        Add a new transition to the buffer.

        Parameters
        ----------
        state : np.ndarray
            The state vector of the added transition.
        action : np.int64
            The action of the added transition.
        reward : float
            The reward of the added transition.
        next_state : np.ndarray
            The next state vector of the added transition.
        done : bool
            The final state of the added transition.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, float, float, np.ndarray, bool]:
        """
        Sample a batch of transitions from the buffer.

        Parameters
        ----------
        batch_size : int
            The number of transitions to sample.

        Returns
        -------
        Tuple[np.ndarray, float, float, np.ndarray, bool]
            A batch of `batch_size` transitions.
        """
        # Here, `random.sample(self.buffer, batch_size)`
        # returns a list of tuples `(state, action, reward, next_state, done)`
        # where:
        # - `state`  and `next_state` are numpy arrays
        # - `action` and `reward` are floats
        # - `done` is a boolean
        #
        # `states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))`
        # generates 5 tuples `state`, `action`, `reward`, `next_state` and `done`, each having `batch_size` elements.
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        """
        Return the current size of the buffer.

        Returns
        -------
        int
            The current size of the buffer.
        """
        return len(self.buffer)



class DataSaver:
    """
    Class used to save model data and rewards during the training
    """

    def __init__(self, results_dir, model_file_name, save_frequency):
        self.results_dir = results_dir
        self.model_file_name = model_file_name
        
        self.save_frequency = save_frequency

        self.last_saved_file = ""
        self.__create_json_file()

        # this is for saving best model
        self.last_max_reward_for_model = 0

        # this is for recording the game, a list of rgb array representing the screen
        self.rgb_data = []
        self.last_max_reward = 0

    
    def __create_json_file(self):
        json_file_path = os.path.join(self.results_dir + "/rewards", self.model_file_name + "_rewards.json")
        
        if not os.path.exists(json_file_path):
            os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
            with open(json_file_path, 'w') as file:
                json.dump(
                    {'data' : []}, file, indent = 2)
        else:
            print("\n!!!! REWARDS JSON FILE ALREADY EXISTS !!!!\n")



    def save_model_data(self, q_network, episode, last = False, reward = None):
        
        model_file = os.path.join(self.results_dir, self.model_file_name + "_" + str(episode) + ".pth")

        if (episode % self.save_frequency == 0) or last:
            print("Saving model ..............")
            torch.save(q_network, model_file)

            # remove the last saved file if it exists
            if self.last_saved_file != "" and os.path.exists(self.last_saved_file):
                os.remove(self.last_saved_file)
            self.last_saved_file = model_file

        model_max_file = os.path.join(self.results_dir + "/models", self.model_file_name + "_max.pth")
        
        if ((reward != None) and (reward >= self.last_max_reward_for_model)):
            if not os.path.exists(self.results_dir + "/models"):
                os.makedirs(self.results_dir + "/models", exist_ok=True)

            self.last_max_reward_for_model = reward
            print("Saving max Model")
            torch.save(q_network, model_max_file)

    
    def save_rewards_data(self, rewards_dict, episode):
        json_file_path = os.path.join(self.results_dir + "/rewards", self.model_file_name + "_rewards.json")
        final_dict = {"episode" : episode, "rewards" : rewards_dict}

        with open(json_file_path, 'r') as file:
            existing_data = json.load(file)

        data_list = existing_data.get("data", [])
        data_list.append(final_dict)

        existing_data["data"] = data_list

        with open(json_file_path, 'w') as file:
            json.dump(existing_data, file, indent = 2)

    
    def save_params_to_json(self, params):
        # save parameters to params_info.json
        json_file = os.path.join(self.results_dir, "params_info.json")
        if not os.path.exists(json_file):
            with open(json_file, 'w') as file:
                json.dump({}, file, indent = 2)

        with open(json_file, 'r') as file:
            existing_data = json.load(file)

        existing_data.update({self.model_file_name : params})

        with open(json_file, 'w') as file:
            json.dump(existing_data, file, indent = 2)

    def update_rgb_data(self, rgb_array):
        self.rgb_data.append(rgb_array)

    def save_gif(self, reward):
        video_file_path = os.path.join(self.results_dir + "/records", self.model_file_name + "_max.gif")
        
        if not os.path.exists(video_file_path):
            os.makedirs(os.path.dirname(video_file_path), exist_ok=True)

        if (reward > self.last_max_reward):
            print("[INFO] SAVING GIF ...... ")
            imageio.mimsave(video_file_path, [np.array(img) for img in self.rgb_data], fps = 29)
            self.last_max_reward = reward

        self.rgb_data = []



def custom_step(env, action):
    """
    customized_actions = [
        [-1.0, 0.0, 0.0], # left
        [1.0, 0.0, 0.0], # right
        [0.0, 1.0, 0.0], # gas
        [0.0, 0.0, 0.5], # break
        [0.0, 0.0, 0.0], # do nothing
        
        # some soft actions
        [-1.0, 1.0, 0.0],
        [-0.5, 0.5, 0.0],
        [1.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],

        [-1.0, 1.0, 0.4],
        [-0.5, 0.5, 0.4],
        [1.0, 1.0, 0.4],
        [0.5, 0.5, 0.4],
    ]
    """

    customized_actions = [
        [-1.0, 0.0, 0.0], # left
        [1.0, 0.0, 0.0], # right
        [0.0, 1.0, 0.0], # gas
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0],

        # some soft actions
        #[-1.0, 1.0, 0.0],
        #[-0.5, 0.5, 0.0],
        #[1.0, 1.0, 0.0],
        #[0.5, 0.5, 0.0],
        #[0.0, 0.5, 0.0],
    ]

    # check that NUMBER_ACTIONS is the real number of actions
    assert NUMBER_ACTIONS == len(customized_actions)

    result = env.step(customized_actions[action])

    # save the rgb array, which is the state
    if hasattr(env, "data_saver"):
        env.data_saver.update_rgb_data(result[0])

    return result








# This function will be applied after env.reset and before learning loop
def after_reset_action(env):
    # env car should be in continous action mode
    return
    print("Waiting for zoom")
    for i in range(50):
        _  = env.step([0.0, 0.0, 0.0])
    print("Waiting for zoom FINISHED")


def test_agent(env: gym.Env,
                q_network: torch.nn.Module,
                num_episodes: int,
                min_rewad: int) -> List[float]:
    """
    Test a trained q_network
    """
    q_network.to(device)
    q_network.eval()

    test_parameters = {}     
    
    for episode_index in tqdm(range(1, num_episodes)):
        state, info = env.reset()
        
        # some action before learning, defined in help_func.py
        after_reset_action(env)

        total_reward = 0
        reward_list = []
        action_list = []

        for t in itertools.count():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits = torch.nn.Softmax()(q_network(state_tensor))
            
            dist=torch.distributions.categorical.Categorical(probs=logits)
            a = torch.argmax(logits).item() #dist.sample().item() 
            
            s_after, r, done = custom_step(env, a)[:3]
            total_reward += r
            reward_list.append(r)
            action_list.append(a)
            

            if (total_reward < min_rewad): done = True

            if done: break

            state = s_after

        episode_params = {"total_reward" : total_reward, "rewards": reward_list, "actions":action_list}
        test_parameters[episode_index] = episode_params

    return test_parameters

def test_ppo_agent(env: gym.Env,
                q_network: torch.nn.Module,
                num_steps: int,
                min_rewad: int) -> List[float]:
    """
    Test a trained q_network
    """
    q_network.to(device)
    q_network.eval()

    total_reward_list = []    
    
    for episode_index in tqdm(range(1, num_steps)):
        state, info = env.reset()
        
        # some action before learning, defined in help_func.py
        after_reset_action(env)

        total_reward = 0

        for t in itertools.count():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits = torch.nn.Softmax()(q_network(state_tensor)[0])
            
            dist=torch.distributions.categorical.Categorical(probs=logits)
            a = torch.argmax(logits).item() #dist.sample().item() 
            
            print(a)
            s_after, r, done = custom_step(env, a)[:3]
            total_reward += r
            

            if (total_reward < min_rewad): done = True

            if done: break

            state = s_after

        total_reward_list.append(total_reward)

    return total_reward_list

def sample_discrete_action(policy_nn: torch.nn.Module,
                           state: NDArray[np.float64],
                           temperature: float) -> Tuple[int, torch.Tensor]:
    """
    Sample a discrete action based on the given state and policy network.

    This function takes a state and a policy network, and returns a sampled action and its log probability.
    The action is sampled from a categorical distribution defined by the output of the policy network.

    Parameters
    ----------
    policy_nn : PolicyNetwork
        The policy network that defines the probability distribution of the actions.
    state : NDArray[np.float64]
        The state based on which an action needs to be sampled.
    temperature: float
        Temperature passed in softmax function

    Returns
    -------
    Tuple[int, torch.Tensor]
        The sampled action and its log probability.

    """

    # TODO...
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    
    logits = policy_nn(state_tensor)
    scaled_logits = logits / temperature
    action_probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)
    
    
    print("probas = ", action_probabilities.cpu().detach().numpy())
    
    
    action_distribution = torch.distributions.Categorical(action_probabilities)
    sampled_action = action_distribution.sample()
    sampled_action_log_probability = action_distribution.log_prob(sampled_action)
    sampled_action = sampled_action.item() 
    # Return the sampled action and its log probability.
    return sampled_action, sampled_action_log_probability


def sample_one_episode(env: gym.Env,
                       policy_nn: torch.nn.Module,
                       max_episode_duration: int,
                       min_reward: float,
                       temperature: float) -> Tuple[List[NDArray[np.float64]], List[int], List[float], List[torch.Tensor]]:
    """
    Execute one episode within the `env` environment utilizing the policy defined by the `policy_nn` parameter.

    Parameters
    ----------
    env : gym.Env
        The environment to play in.
    policy_nn : PolicyNetwork
        The policy neural network.
    max_episode_duration : int
        The maximum duration of the episode.
    
    Returns
    -------
    Tuple[List[NDArray[np.float64]], List[int], List[float], List[torch.Tensor]]
        The states, actions, rewards, and log probability of action for each time step in the episode.
    """
    state_t, info = env.reset()

    episode_states = []
    episode_actions = []
    episode_log_prob_actions = []
    episode_rewards = []
    episode_states.append(state_t)

    total_reward = 0

    nb_reward_neg = 0

    for t in range(max_episode_duration):

        action, log_prob_action = sample_discrete_action(policy_nn, state_t, temperature)
        next_state, reward, done, info, truncated = custom_step(env, action)
        episode_states.append(next_state)
        episode_actions.append(action)
        # test with reward clipping
        mod_reward = np.clip(reward, -1.0, 1.0)
        
        if mod_reward > 0: mod_reward = 1.0
        else: -1.0
        reward = mod_reward

        episode_rewards.append(reward)
        episode_log_prob_actions.append(log_prob_action)
        
        state_t = next_state

        total_reward += reward

        if reward < 0:
            nb_reward_neg += 1
        else:
            nb_reward_neg = 0

        if total_reward < min_reward:
            done = True
        
        #if nb_reward_neg >= 50:
        #    done = True
        #    print("STOPP nb neg reward >= 50")

        if done:
            break

    return episode_states, episode_actions, episode_rewards, episode_log_prob_actions


def avg_return_on_multiple_episodes(env: gym.Env,
                                    policy_nn: torch.nn.Module,
                                    num_test_episode: int,
                                    max_episode_duration: int,
                                    min_reward: float) -> float:
    """
    Play multiple episodes of the environment and calculate the average return.

    Parameters
    ----------
    env : gym.Env
        The environment to play in.
    policy_nn : PolicyNetwork
        The policy neural network.
    num_test_episode : int
        The number of episodes to play.
    max_episode_duration : int
        The maximum duration of an episode.

    Returns
    -------
    float
        The average return.
    """

    # TODO...
    returns = []
    for i in range(num_test_episode):
        episode_states, episode_actions, episode_rewards, episode_log_prob_actions = sample_one_episode(
            env, policy_nn, max_episode_duration, min_reward
            )
        returns.append(sum(episode_rewards))
    average_return = sum(returns) / num_test_episode
    return average_return