import torch
import torch.nn as nn

from spikinggamma.utils.enums import StepMode


class LockedDropout(nn.Module):
    """Drop feature channels with one mask shared across time.

    Sequence inputs are expected to be shaped ``(T, B, ...)``. Each sample
    receives its own mask, but that mask is held fixed for all ``T`` steps.
    Explicit SINGLE or MULTI_STATEFUL calls reuse their mask until ``reset``.
    Without an explicit step mode, non-sequence inputs retain ordinary
    element-wise dropout semantics.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("dropout probability must be between 0 and 1")
        self.p = float(p)
        self.register_buffer("_mask", None, persistent=False)

    def reset(self) -> None:
        self._mask = None

    def forward(
        self, x: torch.Tensor, step_mode: StepMode = StepMode.MULTI_STATELESS
    ) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return torch.zeros_like(x)

        mask_shape = (
            x.shape[1:]
            if step_mode != StepMode.SINGLE and x.ndim >= 3
            else x.shape
        )
        mask = self._mask if step_mode != StepMode.MULTI_STATELESS else None
        if (
            mask is None
            or mask.shape != mask_shape
            or mask.device != x.device
            or mask.dtype != x.dtype
        ):
            mask = x.new_empty(mask_shape).bernoulli_(1.0 - self.p)
            mask.div_(1.0 - self.p)
            if step_mode != StepMode.MULTI_STATELESS:
                self._mask = mask
        return x * mask

    def extra_repr(self) -> str:
        return f"p={self.p}"
