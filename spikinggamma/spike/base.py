from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from spikinggamma.utils.enums import StepMode


@dataclass
class SpikeConfig:
    base_spike_threshold: float = 0.3
    use_dynamic_threshold: bool = False # Set during inference, stateless in between samples
    adaptation_mf: float = 0.3 
    spike_grad_multiplier: Optional[float] = None # will compute the multiplier during backward if None
    saturation_gate_softness: float = 0. # can_saturate, disabled by default
    saturation_gate_start_fraction: float = 0.9 # can_saturate
    enable_saturation_gate: bool = True # can_saturate

    @property
    def can_saturate(self) -> bool:
        return not (self.use_dynamic_threshold) and self.enable_saturation_gate



class BaseSpikeFunction(nn.Module):
    spike_threshold: torch.Tensor # may be shape () or neuron specific shape, note that this shape is of the input to the spike function, not the output!

    def __init__(self, spike_config: SpikeConfig):
        super().__init__()
        self.spike_config = spike_config
        self.register_buffer("spike_threshold", torch.tensor(spike_config.base_spike_threshold))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # A checkpoint may hold one threshold per neuron, while a freshly constructed module holds a
        # scalar. Torch silently copies only element 0 of a 1-d tensor into a 0-d buffer instead of
        # raising, so match the stored shape first.
        key = prefix + "spike_threshold"
        if key in state_dict and state_dict[key].shape != self.spike_threshold.shape:
            self.spike_threshold = self.spike_threshold.new_empty(state_dict[key].shape)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _forward(self, y: torch.Tensor, step_mode: StepMode) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, y: torch.Tensor, step_mode: StepMode = StepMode.SINGLE) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_step_mode_compatible(step_mode):
            raise ValueError(f"Step mode {step_mode} not compatible with {self.__class__.__name__}")
        # y can be (B, *shape) or (T, B, *shape)
        y_hat_k, s = self._forward(y, step_mode)
        return y_hat_k, s

    def is_step_mode_compatible(self, step_mode: StepMode) -> bool:
        return True

    def reset(self):
        pass

    def detach(self):
        pass