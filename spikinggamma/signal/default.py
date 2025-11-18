import torch
import torch.nn as nn
from typing import Optional
from spikinggamma.signal.base import BaseSignalFunction
from spikinggamma.utils.activation import ZeroActivatedReLU
from spikinggamma.utils.enums import StepMode

class DefaultSignalFunction(BaseSignalFunction):
    # act_fn is ZeroActivatedReLU by default
    def __init__(self, norm_fn: Optional[torch.nn.Module] = None, act_fn: Optional[torch.nn.Module] = None, dropout_fn: Optional[torch.nn.Module] = None):
        super().__init__()
        self.norm_fn = norm_fn if norm_fn is not None else nn.Identity()
        self.act_fn = act_fn if act_fn is not None else ZeroActivatedReLU()
        self.dropout_fn = dropout_fn if dropout_fn is not None else nn.Identity()

    def _forward(self, x: torch.Tensor, step_mode: StepMode) -> torch.Tensor:
        x = self.norm_fn(x)
        y_out = self.act_fn(x)
        y_out = self.dropout_fn(y_out)
        return y_out