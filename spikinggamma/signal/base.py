
from spikinggamma.utils.enums import StepMode
import torch
import torch.nn as nn

class BaseSignalFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def is_step_mode_compatible(self, step_mode: StepMode) -> bool:
        return True
    
    def reset(self):
        pass

    def detach(self):
        pass

    def _forward(self, y_hat_k: torch.Tensor, x: torch.Tensor, step_mode: StepMode) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, y_hat_k: torch.Tensor, x: torch.Tensor, step_mode: StepMode = StepMode.SINGLE) -> torch.Tensor:
        if not self.is_step_mode_compatible(step_mode):
            raise ValueError(f"Signal function {self.__class__.__name__} is not compatible with step mode {step_mode}.")
        y = self._forward(y_hat_k, x, step_mode)
        return y