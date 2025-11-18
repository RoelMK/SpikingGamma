import math
import torch
import torch.nn as nn
from spikinggamma.dendrite.base import BaseDendriticFunction
from spikinggamma.utils.enums import StepMode

# In the merged model, the synaptic function and bucket weights are combined into a single linear layer.
class MergedLinearDendriticFunction(BaseDendriticFunction):
    def __init__(self, in_features: int, out_features: int, n_buckets: int, bias: bool = True, add_noise_to_bucket_weights=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_buckets = n_buckets
        self.add_noise = add_noise_to_bucket_weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features, n_buckets))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, n_buckets))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        k = math.sqrt(1.0 / self.in_features) * 1/self.n_buckets
        base = torch.empty(self.out_features, self.in_features).uniform_(-k, k)
        noise = torch.randn(self.out_features, self.in_features, self.n_buckets) * self.add_noise
        w = base.unsqueeze(-1) + noise
        with torch.no_grad():
            self.weight.copy_(w)

        if self.bias is not None:
            nn.init.uniform_(self.bias, -k, k)

    def _forward(self, y_hat_buckets: torch.Tensor, s: torch.Tensor, step_mode: StepMode) -> torch.Tensor:
        x = torch.einsum('...ib,oib->...ob', y_hat_buckets, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x
