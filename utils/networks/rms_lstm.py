import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from typing import List, Tuple

__all__ = [
    "swish",
    "hard_sigmoid",
    "RMSNorm",
    "RMSLSTMCell",
    "RMSNormLSTM",
]

def hard_sigmoid(x):
    return torch.clamp(0.2 * x + 0.5, 0, 1)

def swish(x):
    return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module.
        
        Parameters:
            d_model: int Hidden dimension of the model
        
            eps: float = 1e-5 Epsilon value for numerical stability
        
            device: torch.device | None = None Device to store the parameters on
        
            dtype: torch.dtype | None = None Data type of the parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        
    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.
        
        Parameter:
            x: torch.Tensor
        Return:
            torch.Tensor
        """
        # In torch, x.float() => float32
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
      

class RMSLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_x = nn.Linear(input_size, hidden_size*4, bias=True, **factory_kwargs)
        self.W_h = nn.Linear(hidden_size, hidden_size*4, bias=True, **factory_kwargs)

        self.i_t = RMSNorm(hidden_size, **factory_kwargs)
        self.f_t = RMSNorm(hidden_size, **factory_kwargs)
        self.g_t = RMSNorm(hidden_size, **factory_kwargs)
        self.o_t = RMSNorm(hidden_size, **factory_kwargs)
        self.c_t = RMSNorm(hidden_size, **factory_kwargs)

    def forward(self, x: Tensor, hidden: Optional[tuple[Tensor, Tensor]] = None) -> tuple[Tensor, Tensor]:
        if hidden is None:
            h_prev = torch.zeros(x.size(0), self.hidden_size, device = x.device, dtype = x.dtype)
            c_prev = torch.zeros(x.size(0), self.hidden_size, device = x.device, dtype = x.dtype)
        else:
            h_prev, c_prev = hidden

        gates = self.W_x(x) + self.W_h(h_prev)
        i, f, g, o = gates.chunk(4, dim=1)

        i = hard_sigmoid(self.i_t(i))
        f = hard_sigmoid(self.f_t(f))
        g = swish(self.g_t(g))
        o = hard_sigmoid(self.o_t(o))

        c = f * c_prev + i * g
        h = o * torch.tanh(self.c_t(c))

        return h, c


class RMSNormLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = False,
        bidirectional: bool = False,
        device = None,
        dtype = None,
        ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        layers: List[nn.Module] = []
        for layer_idx in range(num_layers):
            in_size = input_size if layer_idx == 0 else hidden_size * self.num_directions
            for direction in range(self.num_directions):
                layers.append(RMSLSTMCell(in_size, hidden_size, **factory_kwargs))
        self.layers = nn.ModuleList(layers)

    @torch.jit.export
    def forward(
        self,
        x: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if self.batch_first:
            x = x.transpose(0, 1)  # (seq_len, batch, input)

        seq_len, batch_size, _ = x.shape

        if hidden is None:
            h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c0 = torch.zeros_like(h0)
        else:
            h0, c0 = hidden

        out = x
        h_n: List[Tensor] = []
        c_n: List[Tensor] = []

        for layer_idx in range(self.num_layers):
            layer_outputs: List[Tensor] = []
            h_layer: List[Tensor] = []
            c_layer: List[Tensor] = []

            for direction in range(self.num_directions):
                cell_idx = layer_idx * self.num_directions + direction
                layer = self.layers[cell_idx]
                h, c = h0[cell_idx], c0[cell_idx]

                seq = torch.flip(out, [0]) if direction == 1 else out

                outputs_h = []
                for t in range(seq_len):
                    h, c = layer(seq[t], (h, c))
                    outputs_h.append(h.unsqueeze(0))
                outputs_h = torch.cat(outputs_h, dim=0)

                out_seq = torch.flip(outputs_h, [0]) if direction == 1 else outputs_h
                layer_outputs.append(out_seq)
                h_layer.append(h.unsqueeze(0))
                c_layer.append(c.unsqueeze(0))

            out = torch.cat(layer_outputs, dim=2)
            h_n.append(torch.cat(h_layer, dim=0))
            c_n.append(torch.cat(c_layer, dim=0))

        h_n = torch.cat(h_n, dim=0)
        c_n = torch.cat(c_n, dim=0)

        if self.batch_first:
            out = out.transpose(0, 1)

        return out, (h_n, c_n)


if __name__ == "__main__":
    batch_size = 4
    seq_len = 5
    input_size = 10
    hidden_size = 16
    num_layers = 5
    bidirectional = True

    # Gaussion distribution $\text{out}_{i} \sim \mathcal{N}(0, 1)$
    x = torch.randn(batch_size, seq_len, input_size)

    model_rms = RMSNormLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=bidirectional,
    )

    out_rms, (h_n_rms, c_n_rms) = model_rms(x)
    
    model = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=bidirectional,
    )

    out, (h_n, c_n) = model(x)

    print("Input shape:", x.shape)          # (batch, seq, input_size)
    print()
    print("Output shape:", out_rms.shape)        # (batch, seq, hidden * num_directions)
    print("h_n shape:", h_n_rms.shape)         # (num_layers * num_directions, batch, hidden)
    print("c_n shape:", c_n_rms.shape)
    print()
    print("Output shape:", out.shape)        # (batch, seq, hidden * num_directions)
    print("h_n shape:", h_n.shape)         # (num_layers * num_directions, batch, hidden)
    print("c_n shape:", c_n.shape)