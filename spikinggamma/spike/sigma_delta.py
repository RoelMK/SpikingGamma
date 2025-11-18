from typing import Literal, Optional
import torch

from spikinggamma.spike.sigma_delta_graph import SigmaDeltaCudaGraphRunner
from spikinggamma.utils.enums import StepMode
from spikinggamma.spike.base import BaseBucketSpikeFunction, SpikeConfig, y_hat_backward

def update_buckets(current_buckets: torch.Tensor, x: torch.Tensor, bucket_decay_rates: torch.Tensor):
    feature_shape = current_buckets.shape[:-1]
    num_buckets = current_buckets.shape[-1]  
    
    bucket_input = torch.zeros_like(current_buckets)
    bucket_input[..., 0] = x
    bucket_input[..., 1:] = current_buckets[..., :-1] * (1 - bucket_decay_rates.unsqueeze(0).expand(feature_shape + (num_buckets,))[..., 1:])

    return current_buckets * bucket_decay_rates.unsqueeze(0).expand(feature_shape + (num_buckets,)) + bucket_input

@torch.no_grad()
def compute_spikes_from_signal_sequential(y: torch.Tensor, bucket_decay_rates: torch.Tensor, spike_config: SpikeConfig, init_y_hat_buckets: Optional[torch.Tensor] = None):
    T, B, *S = y.shape
    n_buckets = bucket_decay_rates.shape[-1]
    dtype  = y.dtype

    if init_y_hat_buckets is not None:
        if init_y_hat_buckets.shape != (B, *S, n_buckets):
            raise ValueError(f"init_y_hat_buckets must have shape {(B, *S, n_buckets)}, but got {init_y_hat_buckets.shape}")
        buckets = init_y_hat_buckets.clone()
    else:
        buckets = torch.zeros((B, *S, n_buckets), device=y.device)

    y_hat_buckets = torch.empty((T, B, *S, n_buckets), device=y.device)
    s_out = torch.empty((T, B, *S), device=y.device)
    for t in range(T):
        y_hat_t = buckets.sum(dim=-1)
        z = y[t] - y_hat_t
        if spike_config.use_adaptive_threshold:
            threshold = spike_config.spike_threshold + y_hat_t * (spike_config.adaptation_mf)
        else:
            threshold = spike_config.spike_threshold
        if spike_config.allow_multiple_spikes:
            s_t = torch.floor(z.clamp(min=0) / threshold)
        else:
            s_t = (z > threshold).to(dtype)
        buckets = update_buckets(buckets, s_t * 2 * threshold, bucket_decay_rates)
        y_hat_buckets[t] = buckets
        s_out[t] = s_t

    return y_hat_buckets, s_out

class MultiStepSigmaDeltaYHatFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, bucket_decay_rates: torch.Tensor, spike_config: SpikeConfig, 
                graph_runner: Optional[SigmaDeltaCudaGraphRunner] = None, init_y_hat_buckets: Optional[torch.Tensor] = None):
        ctx.n_buckets = bucket_decay_rates.shape[-1]
        ctx.spike_grad_multiplier = spike_config.spike_grad_multiplier
        if graph_runner is not None:
            return graph_runner.run(y, init_y_hat_buckets)
        else:
            return compute_spikes_from_signal_sequential(y, bucket_decay_rates, spike_config)

    @staticmethod
    def backward(ctx, dL_dy_hat_buckets: torch.Tensor, *_):
        dL_dy = y_hat_backward(dL_dy_hat_buckets, ctx.n_buckets, ctx.spike_grad_multiplier)
        return dL_dy, None, None, None, None


class SingleStepSigmaDeltaYHatFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, 
                y: torch.Tensor, 
                y_hat_buckets: torch.Tensor, # of last timestep or zeros if first step
                bucket_decay_rates: torch.Tensor,
                config: SpikeConfig):
        ctx.n_buckets = y_hat_buckets.shape[-1]
        ctx.spike_grad_multiplier = config.spike_grad_multiplier
        y_hat = y_hat_buckets.sum(dim=-1) 
        z = y - y_hat
        if config.use_adaptive_threshold:
            threshold = config.spike_threshold + y_hat * (config.adaptation_mf)
        else:
            threshold = config.spike_threshold
        if config.allow_multiple_spikes:
            s_out = torch.floor(z.clamp(min=0) / threshold)
        else:
            s_out = (z > threshold).float()
        y_hat_buckets = update_buckets(y_hat_buckets, s_out * 2 * threshold, bucket_decay_rates)
        return y_hat_buckets, s_out

    @staticmethod
    def backward(ctx, dL_dy_hat_buckets: torch.Tensor, *_):
        dL_dy = y_hat_backward(dL_dy_hat_buckets, ctx.n_buckets, ctx.spike_grad_multiplier)
        return dL_dy, None, None, None


class SigmaDeltaSpikeFunction(BaseBucketSpikeFunction):
    graph_mode: Literal["full", "1step", "none"] = "1step"
    allow_T_change_on_graph: bool = False
    allow_BS_change_on_graph: bool = True

    def __init__(self, bucket_decay_rates: torch.Tensor, spike_config: SpikeConfig):
        super().__init__(bucket_decay_rates, spike_config)
        self.y_hat_buckets = None
        self.graph_runner = None

    def reset(self):
        self.y_hat_buckets = None

    def detach(self):
        if self.y_hat_buckets is not None:
            self.y_hat_buckets = self.y_hat_buckets.detach()

    def construct_graph_runner(self, y: torch.Tensor):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot use CUDA graphs. Set SigmaDeltaSpikeFunction.graph_mode to 'none'.")
        if not SigmaDeltaSpikeFunction.allow_T_change_on_graph and self.graph_runner is not None and self.graph_runner.T != y.shape[0]:
            raise ValueError(f"T has changed from {self.graph_runner.T} to {y.shape[0]}, but allow_T_change_on_graph is False. It is recommended to keep T constant when using CUDA graphs.")
        if not SigmaDeltaSpikeFunction.allow_BS_change_on_graph and self.graph_runner is not None and self.graph_runner.BS != (y.shape[1], *y.shape[2:]):
            raise ValueError(f"Batch/Spatial shape has changed from {self.graph_runner.BS} to {(y.shape[1], *y.shape[2:])}, but allow_BS_change_on_graph is False. It is recommended to keep Batch/Spatial shape constant when using CUDA graphs.")
        if self.graph_runner is None or self.graph_runner.T != y.shape[0] or self.graph_runner.BS != (y.shape[1], *y.shape[2:]) or SigmaDeltaSpikeFunction.graph_mode != self.graph_runner.mode:
            self.graph_runner = SigmaDeltaCudaGraphRunner(
                y_shape=y.shape,
                bucket_decay_rates=self.bucket_decay_rates,
                dtype=y.dtype,
                device=y.device,
                spike_config=self.spike_config,
                mode=SigmaDeltaSpikeFunction.graph_mode,
            )

    def _forward(self, y: torch.Tensor, step_mode: StepMode) -> tuple[torch.Tensor, torch.Tensor]:
        if step_mode == StepMode.SINGLE:
            self.graph_runner = None
            if self.y_hat_buckets is None:
                self.y_hat_buckets = torch.zeros((*y.shape, self.bucket_decay_rates.shape[-1]), device=y.device, dtype=y.dtype)
            elif self.y_hat_buckets.shape[:-1] != y.shape:
                raise ValueError(f"Shape mismatch: y has shape {y.shape}, but y_hat_buckets has shape {self.y_hat_buckets.shape[:-1]}. Did you forget to call reset()?")
            self.y_hat_buckets, s_out = SingleStepSigmaDeltaYHatFunction.apply(y, self.y_hat_buckets.detach(), self.bucket_decay_rates, self.spike_config)
            return self.y_hat_buckets, s_out
        else:
            if SigmaDeltaSpikeFunction.graph_mode in ["full", "1step"]:
                self.construct_graph_runner(y)
            else:
                self.graph_runner = None
            init_y_hat_buckets = self.y_hat_buckets.detach() if self.y_hat_buckets is not None else None
            y_hat_buckets, s_out = MultiStepSigmaDeltaYHatFunction.apply(y, self.bucket_decay_rates, self.spike_config, self.graph_runner, init_y_hat_buckets)
            if step_mode == StepMode.MULTI_STATEFUL:
                self.y_hat_buckets = y_hat_buckets[-1].detach()
            return y_hat_buckets, s_out
        