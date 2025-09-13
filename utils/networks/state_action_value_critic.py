import numpy as np
import torch
import torch.nn as nn
from utils.networks.rms_lstm import RMSNormLSTM
from infrastructure.distributions import *
from infrastructure import pytorch_utils as ptu


class CriticLSTM(nn.Module):
    def __init__(
        self,
        ob_dim,
        ac_dim,
        hidden_size,
        n_layers,
    ):
        super().__init__()
        self.lstm = RMSNormLSTM(
            input_size=ob_dim + ac_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
        ).to(ptu.device)
        
        self.q_out = nn.Linear(hidden_size, 1).to(ptu.device)

    def forward(self, obs, acs):
        """
        obs: (batch, seq_len, ob_dim)
        acs: (batch, seq_len, ac_dim)
        """
        x = torch.cat([obs, acs], dim=-1)
        out, (h_n, c_n) = self.lstm(x)
        features = out[:, -1, :]
        q = self.q_out(features)

        return q.squeeze(-1)