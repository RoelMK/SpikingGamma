import torch

from spikinggamma.spike.base import SpikeConfig
from spikinggamma.spike.identity import IdentitySpikeFunction
from spikinggamma.spike.kernel.gamma import GammaKernelFunction, compute_linspace_bucket_decay_rates
from spikinggamma.spike.sigma_delta import SigmaDeltaSpikeFunction
from spikinggamma.utils.enums import StepMode


def test_identity_reconstructs_the_sigma_delta_kernel_states_from_its_spikes():
    torch.manual_seed(0)
    spike_config = SpikeConfig(base_spike_threshold=0.3)
    decay_rates = compute_linspace_bucket_decay_rates(4, 0.3)
    y = torch.randn(20, 3, 5).cumsum(dim=0)

    encoder = SigmaDeltaSpikeFunction(GammaKernelFunction(decay_rates), spike_config)
    decoder = IdentitySpikeFunction(GammaKernelFunction(decay_rates), spike_config)
    encoded, spikes = encoder(y, StepMode.MULTI_STATELESS)
    decoded, _ = decoder(spikes, StepMode.MULTI_STATELESS)

    # With a static threshold every spike adds 2 * threshold to the first kernel, so the
    # receiving side can rebuild the encoder's kernel states from the spikes alone.
    assert torch.any(spikes == 1) and torch.any(spikes == 0)
    torch.testing.assert_close(decoded, encoded)
