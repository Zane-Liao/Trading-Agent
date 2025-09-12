import math
import torch
import torch.nn as nn
from torch import Tensor

def hard_sigmoid(x):
    return torch.clamp(0.2 * x + 0.5, 0, 1)

def swish(x):
    return x * torch.sigmoid(x)


class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    bias: Tensor = None

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        """
        linear transformation module.
    
        Parameters:
            in_features: int final dimension of the input
        
            out_features: int final dimension of the output
        
            device: torch.device | None = None Device to store the parameters on
        
            dtype: torch.dtype | None = None Data type of the parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        
        if bias:
          self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
          self.register_parameter('bias', None)

        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std
            )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the linear transformation to the input.
        
        Parameter:
            x: torch.Tensor
        Return:
            torch.Tensor
        """
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output


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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
      

class RMSLSTMCell(nn.modules):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_x = Linear(input_size, hidden_size*4, bias=True)
        self.W_h = Linear(hidden_size, hidden_size*4, bias=True)

        self.i_t = RMSNorm(hidden_size)
        self.f_t = RMSNorm(hidden_size)
        self.o_t = RMSNorm(hidden_size)
        self.g_t = RMSNorm(hidden_size)
        self.c_t = RMSNorm(hidden_size)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        gates = self.W_x(x) + self.W_h(h_prev)
        i, f, o, g = gates.chunk(4, dim=1)

        i = hard_sigmoid(self.i_t(i))
        f = hard_sigmoid(self.f_t(f))
        o = hard_sigmoid(self.o_t(o))
        g = swish(self.g_t(g))

        c = f * c_prev + i * g
        t_rms = self.c_t(c)
        h = o * torch.tanh(t_rms)

        return h, c


class RMSNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                RMSLSTMCell(
                input_size=input_size if layer_index == 0 else hidden_size,
                hidden_size=hidden_size
                )
                for layer_index in range(num_layers)
            ]
        )

    def forward(self, x):
        return NotImplementedError