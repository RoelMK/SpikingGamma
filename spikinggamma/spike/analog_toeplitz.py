
from typing import Optional
import torch
from spikinggamma.utils.enums import StepMode
from spikinggamma.spike.base import BaseBucketSpikeFunction, SpikeConfig

def make_A_B_C(decays: torch.Tensor):
    """
    decays: shape (n_buckets,) with entries in (0,1)
    Returns A (n_b,n_b), B (n_b,1), C (1,n_b)
    """
    n_b = decays.numel()
    A = torch.zeros(n_b, n_b, device=decays.device, dtype=decays.dtype)
    # diagonal
    A[range(n_b), range(n_b)] = decays
    # subdiagonal (i>0, i-1)
    A[1:, :-1] += (1 - decays[1:]).unsqueeze(-1) * torch.eye(n_b-1, device=decays.device, dtype=decays.dtype)
    # B injects input into bucket 0 only
    B = torch.zeros(n_b, 1, device=decays.device, dtype=decays.dtype)
    B[0, 0] = 1.0
    # C sums all buckets
    C = torch.ones(1, n_b, device=decays.device, dtype=decays.dtype)
    return A, B, C


def build_closed_loop_embedding(decays: torch.Tensor, T: int):
    """
    Build Toeplitz embedding K_cl so that:
      y_hat(t, neuron) = sum_{k=0}^t h_out[k] * y(t-k, neuron)
    and, if you also want per-bucket states, you can build the 3D embedding from H_states.
    """
    A, B, C = make_A_B_C(decays) # shapes: (n_b,n_b), (n_b,1), (1,n_b)
    F = A - (B @ C)  # closed-loop transition
    n_b = decays.numel()
    H_states = torch.zeros(T, n_b, 1, device=decays.device, dtype=decays.dtype)
    FkB = B.clone() 
    for k in range(T):
        H_states[k] = FkB
        FkB = F @ FkB  # next power

    # Output embedding (Toeplitz, upper triangular when time-major):
    K_states = torch.zeros(T, T, H_states.size(1), device=decays.device, dtype=decays.dtype)
    for i in range(T):
        K_states[i:, i, :] = H_states[:T-i, :, 0]  # (T-i, n_b)
    return K_states 

def solve(
    y: torch.Tensor,        # (T, B, n_neurons)
    K_states: torch.Tensor  # (T, T, n_buckets)
):
    """
    Returns:
      y_hat:         (T, B, n_neurons)
      y_hat_buckets: (T, B, n_neurons, n_buckets)
    """
    y_hat_buckets = torch.einsum('tsk,sbn->tbnk', K_states, y) 
    return y_hat_buckets.sum(dim=-1), y_hat_buckets



@torch.no_grad()
@torch.compile
def compute_analog_from_signal_kernel(y: torch.Tensor, K_states: torch.Tensor):
    T, B, *S = y.shape
    y = y.view(T, B, -1)  # (T, B, N)
    _, y_hat_buckets = solve(y, K_states)
    y_hat_buckets = y_hat_buckets.contiguous().view(T, B, *S, K_states.shape[-1])
    s_out = torch.zeros((T, B, *S), device=y.device)
    return y_hat_buckets, s_out

    
class AnalogToeplitzYHatFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                y: torch.Tensor, 
                bucket_decay_rates: torch.Tensor, 
                K_states: torch.Tensor,
                spike_grad_multiplier: Optional[float] = None):
        ctx.n_buckets = bucket_decay_rates.shape[-1]
        ctx.spike_grad_multiplier = spike_grad_multiplier
        return compute_analog_from_signal_kernel(y, K_states)

    @staticmethod
    def backward(ctx, dL_dy_hat_buckets: torch.Tensor, *_):
        if ctx.spike_grad_multiplier is None:
            spike_grad_multiplier = 1/ctx.n_buckets
        else:
            spike_grad_multiplier = ctx.spike_grad_multiplier
        dL_dy = dL_dy_hat_buckets.sum(dim=-1) * spike_grad_multiplier
        return dL_dy, None, None, None



class AnalogToeplitzSpikeFunction(BaseBucketSpikeFunction):
    K_states: torch.Tensor = None

    def __init__(self, bucket_decay_rates: torch.Tensor, spike_config: SpikeConfig):
        # It is necessary that K_states is precomputed and passed in here (and shared), because it is sometimes large.
        super().__init__(bucket_decay_rates, spike_config)

    def is_step_mode_compatible(self, step_mode: StepMode) -> bool:
        return step_mode == StepMode.MULTI_STATELESS

    def _forward(self, y: torch.Tensor, step_mode: StepMode) -> tuple[torch.Tensor, torch.Tensor]:
        T = y.shape[0]
        if AnalogToeplitzSpikeFunction.K_states is None:
            AnalogToeplitzSpikeFunction.K_states = build_closed_loop_embedding(self.bucket_decay_rates[0, ...], T)
        elif AnalogToeplitzSpikeFunction.K_states.shape[0] != T:
            raise RuntimeError("It is not allowed to change the length T of the input y after the first call.")
        return AnalogToeplitzYHatFunction.apply(y, self.bucket_decay_rates, AnalogToeplitzSpikeFunction.K_states, self.spike_config.spike_grad_multiplier)
        