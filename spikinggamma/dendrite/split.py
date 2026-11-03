
from typing import Tuple
import torch
import torch.nn as nn
from spikinggamma.dendrite.base import BaseDendriticFunction
from spikinggamma.dendrite.synapse.base import BaseSynapticFunction
from spikinggamma.utils.enums import StepMode

class SplitDendriticFunction(BaseDendriticFunction):
    def __init__(self, shape_out: Tuple[int, ...], 
                 n_kernels: int, 
                 synaptic_function: BaseSynapticFunction, 
                 bias: bool = False,):
        super().__init__()
        self.synaptic_function = synaptic_function
        if bias:
            self.bias = nn.Parameter(torch.zeros(*shape_out))
        else:
            self.register_parameter("bias", None)
        self.kernel_weights = nn.Parameter(
            torch.empty(*shape_out, n_kernels)
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.kernel_weights is not None:
            raw_weights = torch.randn_like(self.kernel_weights) * 0.1
            with torch.no_grad():
                self.kernel_weights.copy_(raw_weights)
        if getattr(self, "bias", None) is not None:
            with torch.no_grad():
                self.bias.zero_()


    def get_kernel_weights(self) -> torch.Tensor:
        return self.kernel_weights
    
    def compute_xkw(self, xk: torch.Tensor) -> torch.Tensor:
        if self.kernel_weights is not None:
            xkw = xk * self.get_kernel_weights()
        else:
            xkw = xk
        return xkw

    def _forward(self, y_hat_k: torch.Tensor, step_mode: StepMode) -> torch.Tensor:
        xk = self.synaptic_function(y_hat_k)
        xkw = self.compute_xkw(xk)
        x = xkw.sum(dim=-1)
        if self.bias is not None:
            x = x + self.bias
        return x
