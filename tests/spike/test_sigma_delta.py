import torch

from spikinggamma.spike.base import SpikeConfig, compute_linspace_bucket_decay_rates
from spikinggamma.spike.sigma_delta import MultiStepSigmaDeltaYHatFunction, SingleStepSigmaDeltaYHatFunction

def test_compute_spikes_from_signal_sequential_multistep_equivalence():
    T = 8
    B = 4
    S = (3, 2)
    n_buckets = 5
    g = torch.Generator().manual_seed(324) 
    x = torch.randn(T, B, *S, generator=g).cumsum(dim=0)
    w = torch.randn(*S, requires_grad=True, generator=g)
    bucket_decay_rates = compute_linspace_bucket_decay_rates(n_buckets, 0.6).expand(*S, n_buckets)
    spike_config = SpikeConfig(
        spike_threshold=0.3,
        allow_multiple_spikes=False,
        use_adaptive_threshold=True,
        adaptation_mf=0.3
    )

    # Single-step sequential
    y_hat_buckets = torch.zeros((B, *S, n_buckets))
    y_hat_buckets_single = torch.empty((T, B, *S, n_buckets))
    s_single = torch.empty((T, B, *S))
    y = x * w.unsqueeze(0).unsqueeze(0)
    for t in range(T):
        y_t = y[t]
        y_hat_buckets, s_t = SingleStepSigmaDeltaYHatFunction.apply(y_t, y_hat_buckets.detach(), bucket_decay_rates, spike_config)
        y_hat_buckets_single[t] = y_hat_buckets
        s_single[t] = s_t
    L_single = y_hat_buckets_single.sum()
    L_single.backward()
    grad_w_single = w.grad.clone()
    w.grad.zero_()
    
    # Multi-step sequential
    y = x * w.unsqueeze(0).unsqueeze(0)
    y_hat_buckets_multi, s_out_multi = MultiStepSigmaDeltaYHatFunction.apply(y, bucket_decay_rates, spike_config, None)
    L_multi = y_hat_buckets_multi.sum()
    L_multi.backward()
    grad_w_multi = w.grad.clone()
    w.grad.zero_()

    assert torch.allclose(y_hat_buckets_single, y_hat_buckets_multi)
    assert torch.allclose(s_single, s_out_multi)
    assert torch.allclose(grad_w_single, grad_w_multi)
    assert not torch.isnan(grad_w_single).any()
    assert not torch.isnan(grad_w_multi).any()
    assert not torch.isnan(y_hat_buckets_single).any()
    assert not torch.isnan(y_hat_buckets_multi).any()
    assert not torch.isnan(s_single).any()
    assert not torch.isnan(s_out_multi).any()
    assert y_hat_buckets_single.sum().abs() > 0
    assert y_hat_buckets_multi.sum().abs() > 0