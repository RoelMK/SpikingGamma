import torch
import torch.nn as nn
import torch.nn.functional as F
from spikinggamma.dendrite.synapse.base import BaseSynapticFunction



class LinearPerNeuron(nn.Linear, BaseSynapticFunction):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
    ):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = inp.transpose(1, 2)
        weight = self.weight
        bias = self.bias if self.bias is not None else None
        out = F.linear(inp, weight, bias)
        return out.transpose(1, 2)

