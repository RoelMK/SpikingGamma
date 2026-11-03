import torch.nn as nn
import torch

from spikinggamma.utils.enums import StepMode

class BaseDendriticFunction(nn.Module):
    def __init__(self, merge_TB=True):
        super().__init__()
        self.merge_TB = merge_TB

    def is_step_mode_compatible(self, step_mode: StepMode) -> bool:
        return True
    
    def reset(self):
        pass

    def detach(self):
        pass

    def forward(self, y_hat_k: torch.Tensor, step_mode: StepMode = StepMode.SINGLE) -> torch.Tensor:
        if not self.is_step_mode_compatible(step_mode):
            raise ValueError(f"Dendritic function {self.__class__.__name__} is not compatible with step mode {step_mode}.")

        if (step_mode == StepMode.MULTI_STATEFUL or step_mode == StepMode.MULTI_STATELESS) and self.merge_TB:
            T, B = y_hat_k.shape[:2]
            # Spike encoders may return non-contiguous views (for example after
            # broadcasting a threshold). ``reshape`` preserves the inexpensive
            # view when possible and makes a contiguous copy when necessary.
            y_hat_k = y_hat_k.reshape(-1, *y_hat_k.shape[2:])

        out = self._forward(y_hat_k, step_mode)

        if (step_mode == StepMode.MULTI_STATEFUL or step_mode == StepMode.MULTI_STATELESS) and self.merge_TB:
            out = out.reshape(T, B, *out.shape[1:])
        return out

    def _forward(self, y_hat_k: torch.Tensor, step_mode: StepMode) -> torch.Tensor:
        raise NotImplementedError
