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
    
