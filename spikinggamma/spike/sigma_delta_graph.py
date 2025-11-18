import torch
from spikinggamma.spike.base import SpikeConfig
from typing import Literal, Optional, Tuple


def update_buckets_inplace(
    current_buckets: torch.Tensor,   # (..., n_buckets)
    x: torch.Tensor,                 # (...,)
    decay: torch.Tensor,             # (..., n_buckets)
    decay_next: Optional[torch.Tensor], # (..., n_buckets-1) or None if n_buckets==1
    scratch_prev: torch.Tensor,      # same shape as current_buckets
) -> torch.Tensor:
    """
    Semantics exactly match the reference:
      out = current_buckets * decay
      out[..., 0] += x
      out[..., 1:] += current_buckets_prev[..., :-1] * (1 - decay[..., 1:])
    No allocations; relies on preallocated scratch_prev.
    """
    n_buckets = current_buckets.shape[-1]
    scratch_prev.copy_(current_buckets)
    current_buckets.mul_(decay)
    current_buckets[..., 0].add_(x)
    if decay_next is not None and n_buckets > 1:
        current_buckets[..., 1:].add_(scratch_prev[..., :-1] * decay_next)
    return current_buckets


@torch.no_grad()
def compute_spikes_from_signal_sequential_into(
    y: torch.Tensor,                          # (T, B, *S) on CUDA
    decay: torch.Tensor,                      # (*S, n_buckets) on CUDA, constant
    decay_next: Optional[torch.Tensor],       # (*S, n_buckets-1) on CUDA, constant
    spike_config: SpikeConfig,
    buckets: torch.Tensor,                    # (B, *S, n_buckets)
    y_hat_buckets_out: torch.Tensor,          # (T, B, *S, n_buckets)
    s_out_out: torch.Tensor,                  # (T, B, *S)
    scratch: dict
):
    T, B, *S = y.shape
    dtype = y.dtype

    y_hat_t = scratch["y_hat_t"] # (B, *S)
    z = scratch["z"] # (B, *S)
    s_t = scratch["s_t"] # (B, *S)
    delta = scratch["delta"] # (B, *S)
    prev = scratch["prev_buckets"] # (B, *S, n_buckets)
    thr = scratch.get("thr", None)  # (B, *S) iff adaptive

    for t in range(T):
        # y_hat_t = buckets.sum(-1)
        torch.sum(buckets, dim=-1, out=y_hat_t)

        # z = y[t] - y_hat_t
        torch.sub(y[t], y_hat_t, out=z)

        # threshold
        if spike_config.use_adaptive_threshold:
            torch.mul(y_hat_t, spike_config.adaptation_mf, out=thr) 
            thr.add_(spike_config.spike_threshold)
            eff_thr = thr
        else:
            eff_thr = spike_config.spike_threshold 

        # spikes
        if spike_config.allow_multiple_spikes:
            z.clamp_min_(0)
            if spike_config.use_adaptive_threshold:
                torch.true_divide(z, eff_thr, out=s_t)
            else:
                s_t.copy_(z).div_(eff_thr)
            torch.floor_(s_t)
        else:
            s_t.copy_((z > eff_thr).to(dtype))

        # delta = s_t * 2 * threshold
        delta.copy_(s_t)
        if spike_config.use_adaptive_threshold:
            delta.mul_(2.0).mul_(thr)
        else:
            delta.mul_(2.0 * spike_config.spike_threshold)

        # buckets update (zero-alloc, in-place)
        update_buckets_inplace(buckets, delta, decay, decay_next, prev)

        # outputs
        y_hat_buckets_out[t].copy_(buckets)
        s_out_out[t].copy_(s_t)


class SigmaDeltaCudaGraphRunner:
    """
    CUDA Graph runner for spiking simulation with two modes:

    - mode="full": Capture once over the entire (T, B, *S) input buffer and replay once per batch.
                   (Higher device memory use; lowest per-run overhead.)
    - mode="1step": Capture exactly one time step and replay T times, staging y[t] each iteration.
                    (Lower device memory use; tiny per-step copy/replay.)
    """

    def __init__(
        self,
        y_shape: Tuple[int, ...],                  # (T, B, *S)
        bucket_decay_rates: torch.Tensor,          # (*S, n_buckets) on CUDA (or will be moved)
        device: torch.device,
        dtype: torch.dtype,
        spike_config: SpikeConfig,
        mode: Literal["full", "1step"] = "1step",   # choose the behavior
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CUDA Graphs.")

        if mode not in ("full", "1step"):
            raise ValueError("mode must be 'full' or '1step'.")

        self.mode = mode

        # ---- Shapes & meta ----
        T, B, *S = y_shape
        n_buckets = bucket_decay_rates.shape[-1]

        self.device, self.dtype = device, dtype
        self.T, self.BS, self.S, self.n_buckets = T, (B, *S), S, n_buckets

        # ---- Persistent device state & constants (shared) ----
        self.buckets = torch.empty((B, *S, n_buckets), device=device, dtype=dtype)

        self.decay = bucket_decay_rates.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        # next decay (1 - decay[..., 1:]) for all but the first bucket
        self.decay_next = (1 - self.decay[..., 1:]).contiguous() if n_buckets > 1 else None

        # ---- Outputs (shared; stored as histories on device) ----
        self.y_hat_buckets_out = torch.empty((T, B, *S, n_buckets), device=device, dtype=dtype)
        self.s_out = torch.empty((T, B, *S), device=device, dtype=dtype)

        # ---- Scratch (shared) ----
        self.scratch = {
            "y_hat_t": torch.empty((B, *S), device=device, dtype=dtype),
            "z": torch.empty((B, *S), device=device, dtype=dtype),
            "s_t": torch.empty((B, *S), device=device, dtype=dtype),
            "delta": torch.empty((B, *S), device=device, dtype=dtype),
            "prev_buckets": torch.empty((B, *S, n_buckets), device=device, dtype=dtype),
        }
        if spike_config.use_adaptive_threshold:
            self.scratch["thr"] = torch.empty((B, *S), device=device, dtype=dtype)

        # ---- Mode-specific staging & capture ----
        self.graph = torch.cuda.CUDAGraph()

        if self.mode == "full":
            # Full-mode keeps a device buffer for the whole (T, B, *S) input
            self.y_buf = torch.empty((T, B, *S), device=device, dtype=dtype)

            # Warm up static input buffer to a known state before capture
            self.y_buf.zero_()
            self.buckets.zero_()
            torch.cuda.synchronize()

            # Capture the full sequence compute once
            with torch.cuda.graph(self.graph):
                compute_spikes_from_signal_sequential_into(
                    self.y_buf,                          # (T, B, *S)
                    self.decay,
                    self.decay_next,
                    spike_config,
                    buckets=self.buckets,
                    y_hat_buckets_out=self.y_hat_buckets_out,
                    s_out_out=self.s_out,
                    scratch=self.scratch,
                )

        else:  # mode == "1step"
            # 1-step mode keeps only per-step staging
            self.y_t = torch.empty((B, *S), device=device, dtype=dtype)
            self.yhat_step = torch.empty((B, *S, n_buckets), device=device, dtype=dtype)
            self.s_step = torch.empty((B, *S), device=device, dtype=dtype)

            # Ensure state is initialized before capture
            self.buckets.zero_()
            torch.cuda.synchronize()

            # Capture exactly one "T=1" step by viewing the per-step buffers with a fake time dim
            with torch.cuda.graph(self.graph):
                compute_spikes_from_signal_sequential_into(
                    self.y_t.view(1, *self.BS),          # (1, B, *S)
                    self.decay,
                    self.decay_next,
                    spike_config,
                    buckets=self.buckets,
                    y_hat_buckets_out=self.yhat_step.view(1, *self.BS, n_buckets),
                    s_out_out=self.s_step.view(1, *self.BS),
                    scratch=self.scratch,
                )

    @torch.no_grad()
    def run(self, y: torch.Tensor, init_y_hat_buckets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        y: CUDA tensor, shape (T, B, *S), dtype == self.dtype
        Returns:
            (y_hat_buckets_out: (T, B, *S, n_buckets),
             s_out: (T, B, *S))
        """
        assert list(y.shape) == [self.T, *self.BS], f"Expected {self.T, *self.BS}, got {tuple(y.shape)}"
        assert y.device.type == "cuda", "Input must be on CUDA"
        assert y.dtype == self.dtype, f"Expected dtype {self.dtype}, got {y.dtype}"

        if init_y_hat_buckets is not None:
            if init_y_hat_buckets.shape != (self.BS + (self.n_buckets,)):
                raise ValueError(f"init_y_hat_buckets must have shape {(self.BS + (self.n_buckets,))}, but got {init_y_hat_buckets.shape}")
            self.buckets.copy_(init_y_hat_buckets, non_blocking=True)
        else:
            self.buckets.zero_()

        if self.mode == "full":
            # Single copy + single replay
            self.y_buf.copy_(y, non_blocking=True)
            self.graph.replay()

        else:  # mode == "1step"
            # Per-time-step: stage y[t], replay, store step outputs
            for t in range(self.T):
                self.y_t.copy_(y[t], non_blocking=True)
                self.graph.replay()
                # Save step results into the histories
                self.y_hat_buckets_out[t].copy_(self.yhat_step, non_blocking=True)
                self.s_out[t].copy_(self.s_step, non_blocking=True)

        return self.y_hat_buckets_out, self.s_out

