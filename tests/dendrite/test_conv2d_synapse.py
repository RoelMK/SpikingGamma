import torch
import torch.nn.functional as F

from spikinggamma.dendrite.synapse.conv2d import Conv2dPerNeuron


def test_conv2d_per_neuron_applies_the_same_filters_to_every_kernel():
    torch.manual_seed(0)
    synapse = Conv2dPerNeuron(3, 4, kernel_size=3, stride=2, padding=1, bias=True)
    x = torch.randn(2, 3, 9, 9, 5)  # (batch, channels, height, width, kernels)

    out = synapse(x)

    expected = torch.stack(
        [F.conv2d(x[..., k], synapse.weight, synapse.bias, stride=2, padding=1) for k in range(5)],
        dim=-1,
    )
    torch.testing.assert_close(out, expected)
