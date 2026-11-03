from typing import Optional
import torch
import torch.nn as nn
from spikinggamma.dendrite.base import BaseDendriticFunction
from spikinggamma.signal.base import BaseSignalFunction
from spikinggamma.utils.enums import StepMode
from spikinggamma.spike.base import BaseSpikeFunction

class SpikingGammaNeuron(nn.Module):
    def __init__(self, dendritic_fn: BaseDendriticFunction, signal_fn: Optional[BaseSignalFunction] = None, spike_fn:  Optional[BaseSpikeFunction] = None, return_all=False):
        super().__init__()
        self.dendritic_fn = dendritic_fn
        self.signal_fn = signal_fn
        self.spike_fn = spike_fn
        self.return_all = return_all

    def detach(self):
        self.dendritic_fn.detach()
        if self.signal_fn is not None:
            self.signal_fn.detach()
        if self.spike_fn is not None:
            self.spike_fn.detach()

    def reset(self):
        self.dendritic_fn.reset()
        if self.signal_fn is not None:
            self.signal_fn.reset()
        if self.spike_fn is not None:
            self.spike_fn.reset()
    
    def forward(self, y_hat_k_in: torch.Tensor, step_mode: StepMode = StepMode.SINGLE) -> torch.Tensor:
        x = self.dendritic_fn(y_hat_k_in, step_mode)
        if self.signal_fn is None and self.spike_fn is None:
            return x
        y = x if self.signal_fn is None else self.signal_fn(y_hat_k_in, x, step_mode)
        if self.spike_fn is None:
            if self.return_all:
                return y, {'x': x}
            return y
        y_hat_k_out, s_out = self.spike_fn(y, step_mode)
        if self.return_all:
            return y_hat_k_out, s_out, {'x': x, 'y': y}
        return y_hat_k_out, s_out
