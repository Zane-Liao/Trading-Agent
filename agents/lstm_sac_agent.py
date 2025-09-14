import os
import copy
import math
import numpy as np
import gc
import tempfile
from torchrl.data import ReplayBuffer
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, List, Tuple
from typing import Callable, Sequence
from utils.infrastructure.distributions import *
from utils.infrastructure import pytorch_utils as ptu
from utils.networks.lstm_policy import PocliyLSTM
from utils.networks.state_action_value_critic import CriticLSTM


class TradingAgent(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self):
        raise NotImplementedError
    
    def update(self):
        buffer = ReplayBuffer()
        raise NotImplementedError