import torch
import torch.nn as nn
from spikinggamma.signal.base import BaseSignalFunction
from spikinggamma.signal.activation import ZeroActivatedReLU
from spikinggamma.utils.enums import StepMode

class DefaultSignalFunction(BaseSignalFunction):
    # act_fn is ZeroActivatedReLU by default
    def __init__(self, 
                 pre_act_fn: nn.Module = nn.Identity(), 
                 act_fn: nn.Module = ZeroActivatedReLU(), 
                 post_act_fn: nn.Module = nn.Identity()):
        super().__init__()
        self.pre_act_fn = pre_act_fn
        self.act_fn = act_fn 
        self.post_act_fn = post_act_fn

    def reset(self):
        pass

    def _forward(self, y_hat_k: torch.Tensor, x: torch.Tensor, step_mode: StepMode) -> torch.Tensor:
        x = self.pre_act_fn(x)
        y = self.act_fn(x)
        y_out = (
            self.post_act_fn(y)
        )
        return y_out
