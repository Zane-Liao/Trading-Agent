import cv2
import copy
import numpy as np
import gymnasium as gym
from collections import OrderedDict
from typing import Dict, Tuple, List
from utils.networks.lstm_policy import PolicyLSTM
from utils.infrastructure import pytorch_util as ptu


def sample_trajectory(
    env: gym.Env, 
    policy: PolicyLSTM,
    max_length: int, 
    render: bool = False,
    init_hidden: bool = True,
) -> Dict[str, np.ndarray]:
    raise NotImplementedError

def sample_trajectories(
    env: gym.Env,
    policy: PolicyLSTM,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
    init_hidden: bool = True,
) -> Tuple[List[Dict[str, np.ndarray]], int]:
    raise NotImplementedError

def sample_n_trajectories(
    env: gym.Env,
    policy: PolicyLSTM,
    ntraj: int,
    max_length: int,
    render: bool = False,
    init_hidden: bool = True,
):
    raise NotImplementedError

def convert_listofrollouts(trajs):
    raise NotImplementedError

def get_traj_length(traj):
    raise NotImplementedError


######################################################################
######################################################################

def normalize_obs(obs):
    raise NotImplementedError

def build_state_sequence(obs, window_size):
    raise NotImplementedError

def add_to_buffer(traj):
    raise NotImplementedError

def sample_from_buffer(batch_size, seq_len):
    raise NotImplementedError

def compute_sharpe_ratio(trajs):
    raise NotImplementedError

def compute_max_drawdown(trajs):
    raise NotImplementedError

def compute_cumulative_returns(trajs):
    raise NotImplementedError