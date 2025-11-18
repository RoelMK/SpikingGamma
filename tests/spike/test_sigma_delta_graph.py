import torch
from spikinggamma.spike.base import SpikeConfig, compute_linspace_bucket_decay_rates
from spikinggamma.spike.sigma_delta_graph import update_buckets_inplace, compute_spikes_from_signal_sequential_into, SigmaDeltaCudaGraphRunner
from spikinggamma.spike.sigma_delta import update_buckets, compute_spikes_from_signal_sequential

# These tests assume that the sequential (non-graph) versions are correct.

def test_update_buckets_inplace_equivalence():
    # Compare inplace vs not inplace version
    g = torch.Generator().manual_seed(55) 
    shape = (5, 4, 3)
    n_buckets = 7
    batch_size = 10
    current_buckets = torch.rand(batch_size, *shape, n_buckets, generator=g)
    x = torch.rand(*shape, generator=g)
    decay = compute_linspace_bucket_decay_rates(n_buckets, 0.6).expand(*shape, n_buckets)
    decay_next = 1 - decay[..., 1:]
    scratch_prev = torch.empty_like(current_buckets)
    out_inplace = update_buckets_inplace(
        current_buckets.clone(), x, decay, decay_next, scratch_prev
    )
    out = update_buckets(current_buckets.clone(), x, decay)
    assert torch.allclose(out_inplace, out)
    assert not torch.isnan(out_inplace).any()
    assert out.sum().abs() > 0


def test_compute_spikes_from_signal_sequential_into_equivalence():
    T = 80
    B = 4
    S = (3, 2)
    n_buckets = 5
    g = torch.Generator().manual_seed(66) 
    y = torch.randn(T, B, *S, generator=g).cumsum(dim=0)
    decay = compute_linspace_bucket_decay_rates(n_buckets, 0.6).expand(*S, n_buckets)
    decay_next = 1 - decay[..., 1:]
    spike_threshold = 0.3
    allow_multiple_spikes = False
    use_adaptive_threshold = True
    adaptation_mf = 0.3
    spike_config = SpikeConfig(
        spike_threshold=spike_threshold,
        allow_multiple_spikes=allow_multiple_spikes,
        use_adaptive_threshold=use_adaptive_threshold,
        adaptation_mf=adaptation_mf
    )

    buckets_inplace = torch.zeros(B, *S, n_buckets)
    y_hat_buckets_inplace = torch.empty(T, B, *S, n_buckets)
    s_out_inplace = torch.empty(T, B, *S)
    scratch_inplace = {
        "y_hat_t": torch.empty(B, *S),
        "z": torch.empty(B, *S),
        "s_t": torch.empty(B, *S),
        "delta": torch.empty(B, *S),
        "prev_buckets": torch.empty(B, *S, n_buckets),
        "thr": torch.zeros(B, *S) if use_adaptive_threshold else None,
    }

    compute_spikes_from_signal_sequential_into(
        y, decay, decay_next, spike_config,
        buckets=buckets_inplace,
        y_hat_buckets_out=y_hat_buckets_inplace,
        s_out_out=s_out_inplace,
        scratch=scratch_inplace
    )

    y_hat_buckets, s_out = compute_spikes_from_signal_sequential(
        y, decay, spike_config
    )

    assert torch.allclose(y_hat_buckets_inplace, y_hat_buckets)
    assert torch.allclose(s_out_inplace, s_out)
    assert not torch.isnan(y_hat_buckets_inplace).any()
    assert not torch.isnan(s_out_inplace).any()
    assert y_hat_buckets_inplace.sum().abs() > 0
    assert s_out_inplace.sum().abs() > 0


def test_spike_graph_cuda_runner_equivalence():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    device = torch.device('cuda')
    T = 80
    B = 4
    S = (3, 2)
    n_buckets = 5
    g = torch.Generator(device=device).manual_seed(77)
    y = torch.randn(T, B, *S, device=device, generator=g).cumsum(dim=0)
    bucket_decay_rates = compute_linspace_bucket_decay_rates(n_buckets, 0.6).expand(*S, n_buckets).to(device)
    spike_config = SpikeConfig(
        spike_threshold=0.3,
        allow_multiple_spikes=False,
        use_adaptive_threshold=True,
        adaptation_mf=0.3
    )

    def test_mode(mode):
        runner = SigmaDeltaCudaGraphRunner(
            y_shape=y.shape,
            bucket_decay_rates=bucket_decay_rates,
            device=device,
            dtype=y.dtype,
            spike_config=spike_config,
            mode=mode
        )

        y_hat_buckets_graph, s_out_graph =runner.run(y)

        y_hat_buckets_seq, s_out_seq = compute_spikes_from_signal_sequential(
            y, bucket_decay_rates, spike_config
        )

        assert torch.allclose(y_hat_buckets_graph, y_hat_buckets_seq), f"y_hat_buckets mismatch in mode {mode}"
        assert torch.allclose(s_out_graph, s_out_seq), f"s_out mismatch in mode {mode}"
        assert not torch.isnan(y_hat_buckets_graph).any(), f"NaNs in y_hat_buckets in mode {mode}"
        assert not torch.isnan(s_out_graph).any(), f"NaNs in s_out in mode {mode}"
        assert y_hat_buckets_graph.sum().abs() > 0, f"All zero y_hat_buckets in mode {mode}"
        assert s_out_graph.sum().abs() > 0, f"All zero s_out in mode {mode}"

    test_mode('full')
    test_mode('1step')
