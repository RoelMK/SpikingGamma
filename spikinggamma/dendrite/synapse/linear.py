import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikinggamma.dendrite.synapse.base import BaseSynapticFunction
import math

class LinearPerNeuron(nn.Linear, BaseSynapticFunction):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = inp.transpose(1, 2)
        out = super().forward(inp) 
        out = out.transpose(1, 2)
        return out
    
class WSLinearPerNeuron(nn.Linear):

    def __init__(self, in_features, out_features, bias=False, gain=False, eps=1e-4):
        super(WSLinearPerNeuron, self).__init__(in_features, out_features, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_features, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = math.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1], keepdims=True)
        var = torch.var(self.weight, axis=[1], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        x = x.transpose(-1, -2)
        y = F.linear(x, self.get_weight(), self.bias)
        return y.transpose(-1, -2)

