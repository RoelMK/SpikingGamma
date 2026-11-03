from typing import Optional, Union
import torch

from spikinggamma.spike.saturation import y_hat_backward_saturated
from spikinggamma.utils.enums import StepMode
from spikinggamma.spike.base import BaseSpikeFunction, SpikeConfig
from spikinggamma.spike.kernel.base import BaseKernelFunction, BaseKernelState, prototype_y_hat_k

@torch.no_grad()
def compute_spikes_from_signal_sequential(
    y: torch.Tensor,
    spike_config: SpikeConfig,
    kernel_function: BaseKernelFunction,
    spike_threshold: Union[float, torch.Tensor],
    init_from_kernel_state: Optional[BaseKernelState]
):
    T, B, *S = y.shape
    n_kernels = kernel_function.n_kernels
    dtype = y.dtype
    kernel_state = kernel_function.construct_kernel_state()
    kernel_state.initialize_like(y[0], n_kernels)

    if init_from_kernel_state is not None and not init_from_kernel_state.is_empty():
        init_y_hat_k = init_from_kernel_state.compute_y_hat_k()
        if init_y_hat_k.shape != (B, *S, n_kernels):
            raise ValueError(f"init_y_hat_k must have shape {(B, *S, n_kernels)}, but got {init_y_hat_k.shape}")
        kernel_state.init_from(init_from_kernel_state)

    y_hat_k = torch.empty((T, B, *S, n_kernels), device=y.device, dtype=dtype)
    s_out = torch.empty((T, B, *S), device=y.device, dtype=dtype)
    for t in range(T):
        kernel_function.step(kernel_state)
        y_hat_k_t = prototype_y_hat_k(kernel_state.compute_y_hat_k(), y[t].shape, kernel_function.n_kernels, y.device, y.dtype)
        y_hat_t = y_hat_k_t.sum(dim=-1)
        z = y[t] - y_hat_t
        if spike_config.use_dynamic_threshold:
            threshold = spike_threshold + y_hat_t * (spike_config.adaptation_mf)
        else:
            threshold = spike_threshold
        s_t = (z > threshold).to(dtype)
        kernel_function.add(kernel_state, s_t * threshold * 2)
        y_hat_k[t] = prototype_y_hat_k(kernel_state.compute_y_hat_k(), y[t].shape, kernel_function.n_kernels, y.device, y.dtype)
        s_out[t] = s_t

    return y_hat_k, s_out, kernel_state

class MultiStepSigmaDeltaYHatFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        y: torch.Tensor,
        spike_config: SpikeConfig,
        kernel_function: BaseKernelFunction,
        spike_threshold: Union[float, torch.Tensor],
        init_from_kernel_state: Optional[BaseKernelState],
        saturation_cutoff: Optional[torch.Tensor],
    ):
        if spike_config.can_saturate and saturation_cutoff is None:
            raise ValueError("saturation_cutoff must be provided if spike_config.can_saturate is True")
        if spike_config.can_saturate:
            ctx.save_for_backward(y, saturation_cutoff)
        ctx.n_kernels = kernel_function.n_kernels
        ctx.spike_grad_multiplier = spike_config.spike_grad_multiplier
        ctx.saturation_gate_softness = spike_config.saturation_gate_softness
        ctx.saturation_gate_start_fraction = spike_config.saturation_gate_start_fraction
        ctx.can_saturate = spike_config.can_saturate
        return compute_spikes_from_signal_sequential(
            y,
            spike_config,
            kernel_function,
            spike_threshold,
            init_from_kernel_state
        )

    @staticmethod
    def backward(ctx, dL_dy_hat_k: torch.Tensor, *_):
        if ctx.can_saturate:
            y, saturation_cutoff = ctx.saved_tensors
        else:
            y = None
            saturation_cutoff = None
        dL_dy = y_hat_backward_saturated(dL_dy_hat_k, y, saturation_cutoff, ctx.n_kernels, 
                               ctx.saturation_gate_softness, ctx.saturation_gate_start_fraction, 
                               ctx.can_saturate, ctx.spike_grad_multiplier)
        return dL_dy, None, None, None, None, None


class SingleStepSigmaDeltaYHatFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        y: torch.Tensor,
        spike_config: SpikeConfig,
        kernel_function: BaseKernelFunction,
        kernel_state: BaseKernelState,
        spike_threshold: Union[float, torch.Tensor],
        saturation_cutoff: Optional[torch.Tensor],
    ):
        if spike_config.can_saturate and saturation_cutoff is None:
            raise ValueError("saturation_cutoff must be provided if spike_config.can_saturate is True")
        if spike_config.can_saturate:
            ctx.save_for_backward(y, saturation_cutoff)
        ctx.n_kernels = kernel_function.n_kernels
        ctx.spike_grad_multiplier = spike_config.spike_grad_multiplier
        ctx.saturation_gate_softness = spike_config.saturation_gate_softness
        ctx.saturation_gate_start_fraction = spike_config.saturation_gate_start_fraction
        ctx.can_saturate = spike_config.can_saturate
        
        kernel_function.step(kernel_state)
        y_hat_k = prototype_y_hat_k(kernel_state.compute_y_hat_k(), y.shape, kernel_function.n_kernels, y.device, y.dtype)
        y_hat = y_hat_k.sum(dim=-1)
        z = y - y_hat
        if spike_config.use_dynamic_threshold:
            threshold = spike_threshold + y_hat * (spike_config.adaptation_mf)
        else:
            threshold = spike_threshold
        s_out = (z > threshold).to(y.dtype)
        kernel_function.add(kernel_state, s_out * threshold * 2)
        y_hat_k = prototype_y_hat_k(kernel_state.compute_y_hat_k(), y.shape, kernel_function.n_kernels, y.device, y.dtype)
        return y_hat_k, s_out

    @staticmethod
    def backward(ctx, dL_dy_hat_k: torch.Tensor, *_):
        if ctx.can_saturate:
            y, saturation_cutoff = ctx.saved_tensors
        else:
            y = None
            saturation_cutoff = None
        dL_dy = y_hat_backward_saturated(
            dL_dy_hat_k,
            y,
            saturation_cutoff,
            ctx.n_kernels,
            ctx.saturation_gate_softness,
            ctx.saturation_gate_start_fraction,
            ctx.can_saturate,
            ctx.spike_grad_multiplier
        )
        return dL_dy, None, None, None, None, None


class SigmaDeltaSpikeFunction(BaseSpikeFunction):

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

    def compute_saturation_cutoff(self, device: torch.device):
        normal_cutoff = self.kernel_function.normal_saturation_cutoff.to(
            device=device,
            dtype=self.spike_threshold.dtype,
        )
        return normal_cutoff * self.spike_threshold.to(device) * 2.0

    def _forward(self, y: torch.Tensor, step_mode: StepMode) -> tuple[torch.Tensor, torch.Tensor]:
        saturation_cutoff = self.compute_saturation_cutoff(y.device)
        if step_mode == StepMode.SINGLE:
            y_hat_k, s_out = SingleStepSigmaDeltaYHatFunction.apply(
                y,
                self.spike_config,
                self.kernel_function,
                self.kernel_state,
                self.spike_threshold,
                saturation_cutoff
            )
            return y_hat_k, s_out
        else:
            if self.init_from_kernel_state is not None:
                self.init_from_kernel_state.detach()
            y_hat_k, s_out, kernel_state = MultiStepSigmaDeltaYHatFunction.apply(
                y,
                self.spike_config,
                self.kernel_function,
                self.spike_threshold,
                self.init_from_kernel_state,
                saturation_cutoff
            )
            if step_mode == StepMode.MULTI_STATEFUL:
                self.init_from_kernel_state = kernel_state
            return y_hat_k, s_out
