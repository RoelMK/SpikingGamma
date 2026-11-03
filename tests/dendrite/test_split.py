import torch
import torch.nn as nn

from spikinggamma.dendrite.split import SplitDendriticFunction


def make_biased_dendritic_function(bias: bool):
    return SplitDendriticFunction(
        shape_out=(3,),
        n_kernels=4,
        synaptic_function=nn.Identity(),
        bias=bias,
    )


def test_bias_is_optional_and_initialised_to_zero():
    assert make_biased_dendritic_function(False).bias is None
    biased = make_biased_dendritic_function(True)
    assert biased.bias.shape == (3,)
    assert torch.all(biased.bias == 0)


def test_bias_is_added_once_after_the_kernel_sum():
    torch.manual_seed(0)
    without = make_biased_dendritic_function(False)
    torch.manual_seed(0)
    with_bias = make_biased_dendritic_function(True)
    with torch.no_grad():
        with_bias.bias.copy_(torch.tensor([1.0, -2.0, 0.5]))
    y_hat_k = torch.randn(5, 3, 4)  # (batch, neurons, kernels); Identity synapse

    x_without = without(y_hat_k)
    x_with = with_bias(y_hat_k)

    assert torch.allclose(x_with - x_without, torch.tensor([1.0, -2.0, 0.5]).expand(5, 3))
