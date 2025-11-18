from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn

from spikinggamma.utils.enums import StepMode


@dataclass
class SpikeConfig:
    spike_threshold: float = 0.3
    allow_multiple_spikes: bool = False
    use_adaptive_threshold: bool = True
    adaptation_mf: Optional[float] = None # will be set to spike_threshold if None
    spike_grad_multiplier: Optional[float] = None # will compute the multiplier during backward if None

    def __post_init__(self):
        if self.adaptation_mf is None:
            self.adaptation_mf = self.spike_threshold


def y_hat_backward(dL_dy_hat_buckets: torch.Tensor, n_buckets: int, spike_grad_multiplier: Optional[float] = None) -> torch.Tensor:
    if spike_grad_multiplier is None:
        spike_grad_multiplier = 1.0/n_buckets
    dL_dy = dL_dy_hat_buckets.sum(dim=-1) * spike_grad_multiplier
    return dL_dy

def compute_linspace_bucket_decay_rates(n_buckets: int, decay_factor: float, decay_start: float = 0.001, decay_end: float = 0.86) -> torch.Tensor:
    return torch.linspace(decay_start, decay_end, n_buckets)**(decay_factor)

class BaseSpikeFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def is_step_mode_compatible(self, step_mode: StepMode) -> bool:
        return True

    def reset(self):
        pass

    def detach(self):
        pass


def change_bucket_decay_rates_shape(bucket_decay_rates: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """
    Reshape bucket_decay_rates to match (*target_shape, n_buckets).

    Args:
        bucket_decay_rates: torch.Tensor of shape (n_buckets,), (1, n_buckets), or (*shape, n_buckets).
        target_shape: torch.Size of the desired leading shape (*shape,).

    Returns:
        torch.Tensor of shape (*target_shape, n_buckets).
    """
    n_buckets = bucket_decay_rates.shape[-1]

    # Case 1: shape is just (n_buckets,)
    if bucket_decay_rates.ndim == 1:
        bucket_decay_rates = bucket_decay_rates.view(1, -1)  # (1, n_buckets)

    # Case 2: shape is (1, n_buckets)
    if bucket_decay_rates.ndim == 2 and bucket_decay_rates.shape[0] == 1:
        bucket_decay_rates = bucket_decay_rates.expand(*target_shape, n_buckets)
    else:
        # Case 3: already (*shape, n_buckets) → just broadcast to target
        bucket_decay_rates = bucket_decay_rates.expand(*target_shape, n_buckets)

    return bucket_decay_rates


class BaseBucketSpikeFunction(BaseSpikeFunction):
    bucket_decay_rates: torch.Tensor

    def __init__(self, bucket_decay_rates: torch.Tensor, spike_config: SpikeConfig):
        super().__init__()
        self.spike_config = spike_config
        self.same_decay_for_all = bucket_decay_rates.ndim == 1
        self.register_buffer("bucket_decay_rates", bucket_decay_rates, False)

    def _forward(self, y: torch.Tensor, step_mode: StepMode) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, y: torch.Tensor, step_mode: StepMode = StepMode.SINGLE) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_step_mode_compatible(step_mode):
            raise ValueError(f"Step mode {step_mode} not compatible with {self.__class__.__name__}")
        if self.same_decay_for_all:
            if (step_mode == StepMode.SINGLE and y.shape[1:] != self.bucket_decay_rates.shape) or (
                (step_mode == StepMode.MULTI_STATELESS or step_mode == StepMode.MULTI_STATEFUL) and y.shape[2:] != self.bucket_decay_rates.shape):
                # Expand to match the shape of y excluding time and batch dimensions
                with torch.no_grad():
                    shape_to_expand = y.shape[1:] if step_mode == StepMode.SINGLE else y.shape[2:]
                    self.bucket_decay_rates = change_bucket_decay_rates_shape(self.bucket_decay_rates, shape_to_expand)
        # y can be (B, *shape) or (T, B, *shape)
        y_hat_buckets, s = self._forward(y, step_mode)
        return y_hat_buckets, s