import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikinggamma.dendrite.synapse.base import BaseSynapticFunction


class LinearPerNeuron(nn.Linear, BaseSynapticFunction):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = inp.transpose(1, 2)
        out = super().forward(inp) 
        out = out.transpose(1, 2)
        return out
    
class WSLinearPerNeuron(nn.Module, BaseSynapticFunction):
    def __init__(self, in_features, out_features, bias=False, eps: float = 1e-5):
        super(WSLinearPerNeuron, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.eps = eps
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=1.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def get_standardized_weight(self):
        w = self.weight
        mean = w.mean(dim=0, keepdim=True)
        var = w.var(dim=0, unbiased=False, keepdim=True)
        return (w - mean) / torch.sqrt(var + self.eps) 

    def forward(self, inp):
        weight = self.get_standardized_weight()
        inp = inp.transpose(-1, -2)
        out = F.linear(inp, weight, self.bias)
        return out.transpose(-1, -2)
