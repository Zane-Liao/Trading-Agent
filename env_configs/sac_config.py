import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Type, Callable, Dict, Any
from utils.networks.lstm_policy import PolicyLSTM
from utils.networks.state_action_value_critic import CriticLSTM

from dataclasses import dataclass
import gymnasium as gym
from gymnasium.wrappers import RescaleAction, ClipAction, RecordEpisodeStatistics


@dataclass
class SACHyperParams:
    env_name: str
    exp_name: Optional[str] = None
    hidden_size: int = 128
    num_layers: int = 5
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    total_steps: int = 500000
    random_steps: int = 5000
    training_starts: int = 10000
    batch_size: 256
    replay_buffer_capacity: 500000
    ep_len: Optional[int] = None
    discount: float = 0.99
    use_soft_target_update: bool = False
    target_update_period: Optional[int] = None
    soft_target_update_rate: Optional[float] = None

    # Actor-critic configuration
    actor_gradient_type: str = "reinforce"
    num_actor_samples: int = 1
    num_critic_updates: int = 1
    num_critic_networks: int = 1
    target_critic_backup_type: str = "mean"
    backup_entropy: bool = True
    use_entropy_bonus: bool = True
    temperature: float = 0.1
    actor_fixed_std: Optional[float] = None
    use_tanh: bool = True

    # Optimizer
    actor_optimizer_class: Type[torch.optim.AdamW] = torch.optim.AdamW
    critic_optimizer_class: Type[torch.optim.AdamW] = torch.optim.AdamW
    lr_scheduler_class: Type[torch.optim.lr_scheduler._LRScheduler] = torch.optim.lr_scheduler.ConstantLR
    
    # Networks
    actor_class: Callable[..., nn.Module] = PolicyLSTM
    critic_class: Callable[..., nn.Module] = CriticLSTM
    actor_kwargs: Dict[str, Any] = None
    critic_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.actor_kwargs is None:
            self.actor_kwargs = {}
        if self.critic_kwargs is None:
            self.critic_kwargs = {}

def make_actor(hp: SACHyperParams, observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
    raise NotImplementedError

def make_critic(hp: SACHyperParams, observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
    raise NotImplementedError

def make_actor_optimizer(hp: SACHyperParams, params) -> torch.optim.Optimizer:
    raise NotImplementedError

def make_critic_optimizer(hp: SACHyperParams, params) -> torch.optim.Optimizer:
    raise NotImplementedError

def make_lr_schedule(hp: SACHyperParams, params: torch.optim.Optimizer):
    raise NotImplementedError

def make_env(hp: SACHyperParams, render: bool = False):
    raise NotImplementedError

def make_log_string(hp: SACHyperParams) -> str:
    raise NotImplementedError

def sac_config(hp: SACHyperParams):
    raise NotImplementedError