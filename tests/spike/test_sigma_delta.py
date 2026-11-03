import pytest
import torch

from spikinggamma.spike.base import SpikeConfig
from spikinggamma.spike.saturation import compute_saturation_gate
from spikinggamma.spike.sigma_delta import MultiStepSigmaDeltaYHatFunction, SigmaDeltaSpikeFunction, SingleStepSigmaDeltaYHatFunction
from spikinggamma.spike.kernel.gamma import GammaKernelFunction, compute_linspace_bucket_decay_rates
from spikinggamma.utils.enums import StepMode


def _simulate_all_spikes_cutoff(kernel_fn, spike_config, *, T=3000, y_max=100):
    spike_fn = SigmaDeltaSpikeFunction(kernel_fn, spike_config)
    spike_fn.reset()

    y = torch.linspace(0, y_max, T).unsqueeze(-1).unsqueeze(-1).requires_grad_(True)
    y_hat_k = torch.zeros(T, 1, 1, kernel_fn.n_kernels)

    for t in range(T):
        y_hat_k = y_hat_k.detach()
        y_hat_k[t], _ = spike_fn(y[t])

    return y_hat_k.sum(dim=-1).squeeze()[-1]


def _simulate_all_spikes_y_hat(kernel_fn, spike_config, *, T=3000):
    feedback_gain = 2.0 * spike_config.base_spike_threshold
    kernel_state = kernel_fn.construct_kernel_state()

    y_hat_post_spike = None
    for _ in range(T):
        kernel_fn.step(kernel_state)
        kernel_fn.add(kernel_state, torch.ones(1, 1) * feedback_gain)
        y_hat_post_spike = kernel_state.compute_y_hat_k().sum(dim=-1)

    return y_hat_post_spike.squeeze()


def test_saturation_gate_accepts_feature_cutoff_in_multi_step():
    y = torch.tensor(
        [
            [[0.5, 1.5, 2.5], [0.1, 2.1, 2.9]],
            [[1.1, 1.9, 3.1], [0.9, 1.8, 3.2]],
        ]
    )
    cutoff = torch.tensor([1.0, 2.0, 3.0])

    gate = compute_saturation_gate(y, cutoff, softness=0, start_fraction=0.9)

    assert gate.shape == y.shape
    assert torch.equal(gate, (y < cutoff.reshape(1, 1, 3)).to(y.dtype))


def test_saturation_gate_accepts_batch_feature_cutoff_in_multi_step():
    y = torch.tensor(
        [
            [[0.5, 1.5, 2.5], [0.1, 2.1, 2.9]],
            [[1.1, 1.9, 3.1], [0.9, 1.8, 3.2]],
        ]
    )
    cutoff = torch.tensor([[1.0, 2.0, 3.0], [0.5, 2.5, 3.0]])

    gate = compute_saturation_gate(y, cutoff, softness=0, start_fraction=0.9)

    assert gate.shape == y.shape
    assert torch.equal(gate, (y < cutoff.reshape(1, 2, 3)).to(y.dtype))


@pytest.mark.parametrize(
    "kernel_fn",
    [
        GammaKernelFunction(compute_linspace_bucket_decay_rates(4, 0.1), sequential_buckets=False),
        GammaKernelFunction(compute_linspace_bucket_decay_rates(4, 0.1), sequential_buckets=True),
    ],
)
def test_saturation_cutoff_matches_all_spikes_capacity(kernel_fn):
    spike_config = SpikeConfig(
        base_spike_threshold=0.3,
        use_dynamic_threshold=False,
        saturation_gate_softness=0,
    )
    spike_fn = SigmaDeltaSpikeFunction(kernel_fn, spike_config)

    simulated_cutoff = _simulate_all_spikes_cutoff(kernel_fn, spike_config)

    assert torch.allclose(
        spike_fn.compute_saturation_cutoff(torch.device("cpu")).squeeze(),
        simulated_cutoff,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "kernel_fn",
    [
        GammaKernelFunction(compute_linspace_bucket_decay_rates(4, 0.1), sequential_buckets=False),
        GammaKernelFunction(compute_linspace_bucket_decay_rates(4, 0.1), sequential_buckets=True),
    ],
)
def test_saturation_cutoff_gates_gradients_around_capacity(kernel_fn):
    spike_config = SpikeConfig(
        base_spike_threshold=0.2,
        use_dynamic_threshold=False,
        saturation_gate_softness=0,
    )
    device = torch.device("cpu")
    spike_fn = SigmaDeltaSpikeFunction(kernel_fn, spike_config)
    cutoff = spike_fn.compute_saturation_cutoff(device).detach().reshape(())
    y = torch.stack([cutoff - 1e-3, cutoff + 1e-3]).reshape(2, 1, 1).requires_grad_(True)

    y_hat_k, _ = spike_fn(y)
    y_hat_k.sum().backward()

    assert torch.allclose(y.grad[0], torch.ones_like(y.grad[0]))
    assert torch.allclose(y.grad[1], torch.zeros_like(y.grad[1]))


@pytest.mark.parametrize(
    "kernel_fn",
    [
        GammaKernelFunction(compute_linspace_bucket_decay_rates(4, 0.1), sequential_buckets=False),
        GammaKernelFunction(compute_linspace_bucket_decay_rates(4, 0.1), sequential_buckets=True),
    ],
)
def test_y_hat_stops_tracking_y_above_saturation_cutoff(kernel_fn):
    spike_config = SpikeConfig(
        base_spike_threshold=0.2,
        use_dynamic_threshold=False,
        saturation_gate_softness=0,
    )
    spike_fn = SigmaDeltaSpikeFunction(kernel_fn, spike_config)
    expected_y_hat = _simulate_all_spikes_y_hat(kernel_fn, spike_config)
    y_high = (5.0 * spike_fn.compute_saturation_cutoff(torch.device("cpu")).detach().reshape(()) + 1.0)
    y = torch.full((3000, 1, 1), y_high)

    y_hat_k, s_out = spike_fn(y, step_mode=StepMode.MULTI_STATELESS)
    y_hat = y_hat_k.sum(dim=-1)

    assert torch.all(s_out[-100:] == 1)
    assert torch.allclose(y_hat[-1].squeeze(), expected_y_hat, atol=1e-4, rtol=1e-4)
    assert y_hat[-1].squeeze() < y[-1].squeeze() - spike_config.base_spike_threshold


def test_compute_spikes_from_signal_sequential_multistep_equivalence():
    T = 8
    B = 4
    S = (3, 2)
    n_kernels = 5
    g = torch.Generator().manual_seed(324) 
    x = torch.randn(T, B, *S, generator=g).cumsum(dim=0)
    w = torch.randn(*S, requires_grad=True, generator=g)
    bucket_decay_rates = compute_linspace_bucket_decay_rates(n_kernels, 0.6).expand(*S, n_kernels)
    spike_config = SpikeConfig(
        base_spike_threshold=0.3,
        use_dynamic_threshold=True,
        adaptation_mf=0.3
    )
    feedback_gain = 2.0 * spike_config.base_spike_threshold
    kernel_fn = GammaKernelFunction(bucket_decay_rates)

    # Single-step sequential
    kernel_state = kernel_fn.construct_kernel_state()
    y_hat_k_single = torch.empty((T, B, *S, n_kernels))
    s_single = torch.empty((T, B, *S))
    y = x * w.unsqueeze(0).unsqueeze(0)
    for t in range(T):
        y_t = y[t]
        y_hat_k, s_t = SingleStepSigmaDeltaYHatFunction.apply(
            y_t,
            spike_config,
            kernel_fn,
            kernel_state,
            feedback_gain,
            None
        )
        y_hat_k_single[t] = y_hat_k
        s_single[t] = s_t
    L_single = y_hat_k_single.sum()
    L_single.backward()
    grad_w_single = w.grad.clone()
    w.grad.zero_()
    
    # Multi-step sequential
    y = x * w.unsqueeze(0).unsqueeze(0)
    y_hat_k_multi, s_out_multi, _ = MultiStepSigmaDeltaYHatFunction.apply(y, spike_config, kernel_fn, feedback_gain, None, None)
    L_multi = y_hat_k_multi.sum()
    L_multi.backward()
    grad_w_multi = w.grad.clone()
    w.grad.zero_()

    assert torch.allclose(y_hat_k_single, y_hat_k_multi)
    assert torch.allclose(s_single, s_out_multi)
    assert torch.allclose(grad_w_single, grad_w_multi)
    assert not torch.isnan(grad_w_single).any()
    assert not torch.isnan(grad_w_multi).any()
    assert not torch.isnan(y_hat_k_single).any()
    assert not torch.isnan(y_hat_k_multi).any()
    assert not torch.isnan(s_single).any()
    assert not torch.isnan(s_out_multi).any()
    assert y_hat_k_single.sum().abs() > 0
    assert y_hat_k_multi.sum().abs() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_bucket_decay_rates_follow_cuda_input_device():
    B = 4
    S = (3, 2)
    n_kernels = 5
    bucket_decay_rates = compute_linspace_bucket_decay_rates(n_kernels, 0.6)
    kernel_fn = GammaKernelFunction(bucket_decay_rates)
    kernel_state = kernel_fn.construct_kernel_state()
    x = torch.ones((B, *S), device="cuda")

    kernel_fn.step(kernel_state)
    kernel_fn.add(kernel_state, x)
    y_hat_k = kernel_state.compute_y_hat_k()

    assert y_hat_k.device == x.device
    assert y_hat_k.shape == (B, *S, n_kernels)
