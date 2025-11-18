from typing import Optional, Tuple
import torch
import torch.nn as nn
from spikinggamma.dendrite.base import BaseDendriticFunction
from spikinggamma.signal.base import BaseSignalFunction
from spikinggamma.utils.enums import StepMode
from spikinggamma.spike.base import BaseSpikeFunction

class SpikingGammaNeuron(nn.Module):
    def __init__(self, spike_fn: BaseSpikeFunction, dendritic_fn: BaseDendriticFunction, signal_fn: BaseSignalFunction, return_all=False):
        super().__init__()
        self.spike_fn = spike_fn
        self.dendritic_fn = dendritic_fn
        self.signal_fn = signal_fn
        self.return_all = return_all

    def detach(self):
        self.spike_fn.detach()
        self.dendritic_fn.detach()
        self.signal_fn.detach()

    def reset(self):
        self.spike_fn.reset()
        self.dendritic_fn.reset()
        self.signal_fn.reset()
    
    def forward(self, y: torch.Tensor, step_mode: StepMode = StepMode.SINGLE) -> torch.Tensor:
        """Forward pass of the Spiking Gamma Neuron.
        Args:
            y (torch.Tensor): Any input, can be the analog neuron signal of upstream neuron, or network input,
                              shape (batch_size, *shape) if step_mode == 'single' or (time_steps, batch_size, *shape) if step_mode is 'multi'.
            step_mode (StepMode): Step mode, either 'single' or 'multi'.
        Returns:
            torch.Tensor: Analog neuron signal of current neuron, shape (batch_size, *shape) if step_mode == 'single' 
                          or (time_steps, batch_size, *shape) if step_mode is 'multi'.
        """
        y_hat_buckets, s = self.spike_fn(y, step_mode)
        x = self.dendritic_fn(y_hat_buckets, s, step_mode)
        y = self.signal_fn(x, step_mode)
        if self.return_all:
            return y, {'y_hat_buckets': y_hat_buckets, 's': s, 'x': x}
        else:
            return y


class SpikingGammaOutput(nn.Module):
    def __init__(self, spike_fn: BaseSpikeFunction):
        super().__init__()
        self.spike_fn = spike_fn

    def detach(self):
        self.spike_fn.detach()

    def reset(self):
        self.spike_fn.reset()

    def forward(self, y: torch.Tensor, step_mode: StepMode = StepMode.SINGLE) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.spike_fn(y, step_mode)
        