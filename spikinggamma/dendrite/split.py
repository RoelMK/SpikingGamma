
from typing import Tuple
import torch
import torch.nn as nn
from spikinggamma.dendrite.base import BaseDendriticFunction
from spikinggamma.dendrite.synapse.base import BaseSynapticFunction
from spikinggamma.utils.enums import StepMode

# In the split model, each dendrite has a set of bucket weights that are applied after the synaptic function.
class SplitDendriticFunction(BaseDendriticFunction):
    def __init__(self, shape_out: Tuple[int, ...], n_buckets: int, synaptic_function: BaseSynapticFunction):
        super().__init__()
        self.synaptic_function = synaptic_function
        self.bucket_weights = nn.Parameter(
            torch.empty(*shape_out, n_buckets)
        )
        self.reset_parameters()

    def reset_parameters(self):
        w = torch.full_like(self.bucket_weights, fill_value=0.0) + torch.randn_like(self.bucket_weights) * 0.1
        with torch.no_grad():
            self.bucket_weights.copy_(w)

    def _forward(self, y_hat_buckets: torch.Tensor, s: torch.Tensor, step_mode: StepMode) -> torch.Tensor:
        xk = self.synaptic_function(y_hat_buckets)
        xkw = xk * self.bucket_weights.unsqueeze(0)
        x = xkw.sum(dim=-1)
        return x