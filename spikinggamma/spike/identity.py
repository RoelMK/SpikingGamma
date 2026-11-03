from typing import Optional
import torch

from spikinggamma.utils.enums import StepMode
from spikinggamma.spike.base import BaseSpikeFunction, SpikeConfig
from spikinggamma.spike.kernel.base import BaseKernelFunction, BaseKernelState, prototype_y_hat_k

# Takes input as spikes and passes them through the kernels.

class IdentitySpikeFunction(BaseSpikeFunction):
    init_from_kernel_state: Optional[BaseKernelState] = None

    def __init__(
        self,
        kernel_function: BaseKernelFunction,
        spike_config: SpikeConfig,
    ):
        super().__init__(spike_config)
        self.kernel_function = kernel_function
        self.kernel_state = self.kernel_function.construct_kernel_state()
        self.reset()

    def reset(self):
        self.kernel_state.reset()
        self.init_from_kernel_state = None

    def detach(self):
        self.kernel_state.detach()
        if self.init_from_kernel_state is not None:
            self.init_from_kernel_state.detach()

    def _forward(self, y: torch.Tensor, step_mode: StepMode) -> tuple[torch.Tensor, torch.Tensor]:
        if step_mode == StepMode.SINGLE:
            self.kernel_function.step(self.kernel_state)
            self.kernel_function.add(self.kernel_state, y * 2 * self.spike_threshold)
            y_hat_k = prototype_y_hat_k(self.kernel_state.compute_y_hat_k(), y.shape, self.kernel_function.n_kernels, y.device, y.dtype)
            return y_hat_k, y
        else:
            init_state = self.init_from_kernel_state
            if init_state is not None and not init_state.is_empty():
                init_state.detach()
            else:
                init_state = None

            kernel_state = self.kernel_function.construct_kernel_state()
            kernel_state.initialize_like(y[0], self.kernel_function.n_kernels)
            if init_state is not None:
                kernel_state.init_from(init_state)

            y_hat_k = torch.empty(
                (*y.shape, self.kernel_function.n_kernels),
                device=y.device,
                dtype=y.dtype
            )
            for t in range(y.shape[0]):
                self.kernel_function.step(kernel_state)
                self.kernel_function.add(kernel_state, y[t] * 2 * self.spike_threshold)
                y_hat_k[t] = prototype_y_hat_k(kernel_state.compute_y_hat_k(), y[t].shape, self.kernel_function.n_kernels, y.device, y.dtype)

            if step_mode == StepMode.MULTI_STATEFUL:
                self.init_from_kernel_state = kernel_state
            return y_hat_k, y

    
