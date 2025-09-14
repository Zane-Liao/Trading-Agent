from typing import Optional
import numpy as np

import torch
import torch.nn as nn
from torch import distributions
from utils.networks.rms_lstm import RMSNormLSTM
from infrastructure.distributions import *
from infrastructure import pytorch_utils as ptu


class PolicyLSTM(nn.Module):
    """
    LSTM-Policy
    
    Parameters:
        ac_dim: Action Dimension
        
        ob_dim: State Dimension
        
        discrete: Discrete and continuous action spaces
        
        n_layers: LSTM layers
        
        layer_size: LSTM hidden size
        
        use_tanh: tanh ==> range[-1, 1]
        
        state_dependent_std:
            Gaussion Distibutions a~N(u(s), o(s))
            if True ==> output(mean, variance)
            else Flase ==> Global Variance
        
        fixed_std: Fix standard deviation
    """
    def __init__(
        self,
        ac_dim: int, 
        ob_dim: int,
        hidden_size: int,
        n_layers: int,
        layer_size: int,
        discrete: bool,
        use_tanh: bool = False,
        state_dependent_std: bool = False,
        fixed_std: Optional[float] = None,
    ):
        super().__init__()
        self.use_tanh = use_tanh
        self.discrete = discrete
        self.state_dependent_std = state_dependent_std
        self.fixed_std = fixed_std
        
        self.encoder = RMSNormLSTM(
            input_size=ob_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False, # If Offline ==> True | else Online ==> False
        ).to(ptu.device)

        if discrete:
           self.logits = nn.Linear(hidden_size, ac_dim).to(ptu.device)
        else:
            if self.state_dependent_std:
                self.out = nn.Linear(hidden_size, 2*ac_dim).to(ptu.device)
            else:
                self.out = nn.Linear(hidden_size, ac_dim).to(ptu.device)

                if self.fixed_std:
                    self.std = 0.1
                else:
                    self.std = nn.Parameter(
                        torch.full((ac_dim,), 0.0, dtype=torch.float32, device=ptu.device)
                    )

    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        out, (h_n, c_n) = self.encoder(obs)
        features = out[:, -1, :]
        
        if self.discrete:
            logits = self.logits(features)
            # Categorical ==> Discrete Distribution
            action_distribution = distributions.Categorical(logits=logits)
        else:
            if self.state_dependent_std:
                mean, std = torch.chunk(self.net(out), 2, dim=-1)
                std = nn.functional.softplus(self.std) + 1e-2
            else:
                mean = self.out(features)
                if self.fixed_std:
                    self.std = std
                else:
                    std = nn.functional.softplus(self.std) + 1e-2

            if self.use_tanh:
                action_distribution = make_tanh_transformed(mean, std)
            else:
                action_distribution = make_multi_normal(mean, std)

        return action_distribution