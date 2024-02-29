import gymnasium as gym
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
import itertools
from tqdm import tqdm
import collections
import random

from typing import List, Tuple, Deque, Optional, Callable

device = "cuda" if torch.cuda.is_available() else "cpu"

NUMBER_ACTIONS = 9

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

        # some soft actions
        [-1.0, 1.0, 0.0],
        [-0.5, 0.5, 0.0],
        [1.0, 1.0, 0.0],
        [0.5, 0.5, 0.0],

        [0.0, 0.5, 0.0],
        
    ]

    # check that NUMBER_ACTIONS is the real number of actions
    assert NUMBER_ACTIONS == len(customized_actions)

    return env.step(customized_actions[action])

def test_agent(env: gym.Env,
                q_network: torch.nn.Module,
                num_episodes: int,
                min_rewad: int) -> List[float]:
    """
    Test a trained q_network
    """
    q_network.to(device)
    q_network.eval()

    total_reward_list = []    
    
    for episode_index in tqdm(range(1, num_episodes)):
        state, info = env.reset()
        total_reward = 0

        for t in itertools.count():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            a = torch.argmax(q_network(state_tensor)).item()
            s_after, r, done = custom_step(env, a)[:3]
            total_reward += r
            

            if (total_reward < min_rewad): done = True

            if done: break

        total_reward_list.append(total_reward)

    return total_reward_list
