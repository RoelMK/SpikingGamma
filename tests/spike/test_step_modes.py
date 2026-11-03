import pytest
import torch

from spikinggamma.spike.base import SpikeConfig
from spikinggamma.spike.identity import IdentitySpikeFunction
from spikinggamma.spike.kernel.gamma import GammaKernelFunction, compute_linspace_bucket_decay_rates
from spikinggamma.spike.sigma_delta import SigmaDeltaSpikeFunction
from spikinggamma.utils.enums import StepMode


def run_single_steps(encoder, y):
    steps = [encoder(y_t, StepMode.SINGLE) for y_t in y]
    return tuple(torch.stack(outputs) for outputs in zip(*steps))


def run_stateful_chunks(encoder, y):
    chunks = [encoder(chunk, StepMode.MULTI_STATEFUL) for chunk in y.split(5)]
    return tuple(torch.cat(outputs) for outputs in zip(*chunks))


@pytest.mark.parametrize("run", [run_single_steps, run_stateful_chunks], ids=["single", "stateful-chunks"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize(
    "encoder_cls, use_dynamic_threshold",
    [
        pytest.param(SigmaDeltaSpikeFunction, False, id="sigma-delta"),
        pytest.param(SigmaDeltaSpikeFunction, True, id="sigma-delta-dynamic"),
        pytest.param(IdentitySpikeFunction, False, id="identity"),
    ],
)
def test_step_modes_match_a_full_sequence(encoder_cls, use_dynamic_threshold, dtype, run):
    torch.manual_seed(0)
    y = torch.randn(12, 3, 5, dtype=dtype).cumsum(dim=0).requires_grad_()

    def make_encoder():
        kernel_fn = GammaKernelFunction(compute_linspace_bucket_decay_rates(4, 0.3))
        return encoder_cls(kernel_fn, SpikeConfig(use_dynamic_threshold=use_dynamic_threshold))

    expected = make_encoder()(y, StepMode.MULTI_STATELESS)
    actual = run(make_encoder(), y)

    assert torch.count_nonzero(expected[0]) > 0
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want)
    # Gradients do not flow through time, so cutting the sequence into chunks keeps them equal.
    weights = torch.randn_like(expected[0])
    grads = [torch.autograd.grad((y_hat_k * weights).sum(), y)[0] for y_hat_k, _ in (actual, expected)]
    torch.testing.assert_close(*grads)
