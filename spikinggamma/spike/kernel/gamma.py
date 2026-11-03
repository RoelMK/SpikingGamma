import torch

from spikinggamma.spike.kernel.base import BaseKernelFunction, YHatKernelState


def compute_linspace_bucket_decay_rates(
    n_kernels: int,
    decay_factor: float,
    decay_start: float = 0.001,
    decay_end: float = 0.86,
) -> torch.Tensor:
    return torch.linspace(decay_start, decay_end, n_kernels) ** (decay_factor)


class GammaKernelFunction(BaseKernelFunction):
    def __init__(
        self,
        bucket_decay_rates: torch.Tensor,
        sequential_buckets: bool = False,
    ):
        super().__init__(n_kernels=bucket_decay_rates.shape[-1])
        self.bucket_decay_rates = bucket_decay_rates
        self.sequential_buckets = sequential_buckets
        self.bucket_input_rates = 1.0 - self.bucket_decay_rates.clone()
        if self.sequential_buckets and self.n_kernels > 1:
            # Correct for already decayed input (one step)
            self.bucket_input_rates[..., 1] = (
                self.bucket_input_rates[..., 1] / self.bucket_decay_rates[..., 0]
            )
        self._normal_saturation_cutoff = self.n_kernels / (
            1 - self.bucket_decay_rates[..., 0]
        )

    def construct_kernel_state(self) -> YHatKernelState:
        return YHatKernelState()

    @property
    def normal_saturation_cutoff(self) -> torch.Tensor:
        return self._normal_saturation_cutoff

    def step(
        self,
        state: YHatKernelState,
    ):
        if state.is_empty():
            return state
        if self.bucket_decay_rates.device != state.y_hat_k.device:
            self.bucket_decay_rates = self.bucket_decay_rates.to(state.y_hat_k.device)
            self.bucket_input_rates = self.bucket_input_rates.to(state.y_hat_k.device)

        feature_shape = state.y_hat_k.shape[:-1]
        num_buckets = state.y_hat_k.shape[-1]
        expanded_decay_rates = self.bucket_decay_rates.unsqueeze(0).expand(
            feature_shape + (num_buckets,)
        )
        expanded_input_rates = self.bucket_input_rates.unsqueeze(0).expand(
            feature_shape + (num_buckets,)
        )

        if self.sequential_buckets:
            # In-place writes otherwise retain the previous tensor's autograd graph.
            state.detach()
            state.y_hat_k[..., 0] = state.y_hat_k[..., 0].detach() * expanded_decay_rates[..., 0]

            for k in range(1, state.y_hat_k.shape[-1]):
                state.y_hat_k[..., k] = (
                    state.y_hat_k[..., k].detach() * expanded_decay_rates[..., k]
                    + state.y_hat_k[..., k - 1].detach() * expanded_input_rates[..., k]
                )
        else:
            bucket_input = torch.zeros_like(state.y_hat_k)
            bucket_input[..., 1:] = state.y_hat_k[..., :-1].detach() * expanded_input_rates[..., 1:]

            state.y_hat_k = state.y_hat_k.detach() * expanded_decay_rates + bucket_input

    def add(self, state: YHatKernelState, x: torch.Tensor):
        if state.is_empty():
            state.initialize_like(x, self.n_kernels)
        if not state.is_matching_shape(x):
            raise ValueError(
                f"Shape mismatch: x has shape {x.shape}, but y_hat_k has shape {state.y_hat_k.shape[:-1]}."
            )

        state.y_hat_k[..., 0] = state.y_hat_k[..., 0] + x
