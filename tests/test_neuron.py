import torch
from torch import nn

from spikinggamma.dendrite.split import SplitDendriticFunction
from spikinggamma.dendrite.synapse.linear import LinearPerNeuron
from spikinggamma.neuron import SpikingGammaNeuron
from spikinggamma.signal.default import DefaultSignalFunction
from spikinggamma.spike.base import SpikeConfig
from spikinggamma.spike.kernel.gamma import GammaKernelFunction
from spikinggamma.spike.sigma_delta import SigmaDeltaSpikeFunction
from spikinggamma.utils.enums import StepMode


def test_neuron_single_matches_multi_stateless():
    torch.manual_seed(42)
    time_steps, batch_size, in_features, out_features, n_kernels = 13, 3, 5, 4, 3
    # The two modes sum the parameter gradients over time in a different order. In float32 that
    # rounding can exceed the default tolerance on some CPUs, so compare in float64.
    neuron = SpikingGammaNeuron(
        dendritic_fn=SplitDendriticFunction(
            shape_out=(out_features,),
            n_kernels=n_kernels,
            synaptic_function=LinearPerNeuron(in_features, out_features),
            bias=True,
        ),
        signal_fn=DefaultSignalFunction(pre_act_fn=nn.LayerNorm(out_features)),
        spike_fn=SigmaDeltaSpikeFunction(
            GammaKernelFunction(torch.tensor([0.15, 0.4, 0.65])),
            SpikeConfig(base_spike_threshold=0.3),
        ),
        return_all=True,
    ).double().eval()
    inputs = torch.rand(
        time_steps, batch_size, in_features, n_kernels, dtype=torch.float64, requires_grad=True
    )

    single_outputs = [neuron(input_t, StepMode.SINGLE) for input_t in inputs]
    single_y_hat = torch.stack([output[0] for output in single_outputs])
    neuron.reset()
    y_hat, spikes, state = neuron(inputs, StepMode.MULTI_STATELESS)

    torch.testing.assert_close(single_y_hat, y_hat)
    torch.testing.assert_close(
        torch.stack([output[1] for output in single_outputs]), spikes, rtol=0, atol=0
    )
    for name in ("x", "y"):
        torch.testing.assert_close(
            torch.stack([output[2][name] for output in single_outputs]), state[name]
        )
    # Ensure the comparison exercises both spiking and silent outputs.
    assert torch.any(spikes == 1)
    assert torch.any(spikes == 0)

    # Use the same weighted loss through the spike reconstruction in both modes.
    weights = torch.randn_like(y_hat)
    grad_targets = (inputs, *neuron.parameters())
    single_grads = torch.autograd.grad((single_y_hat * weights).sum(), grad_targets)
    multi_grads = torch.autograd.grad((y_hat * weights).sum(), grad_targets)
    for single_grad, multi_grad in zip(single_grads, multi_grads):
        torch.testing.assert_close(single_grad, multi_grad)
        assert torch.isfinite(single_grad).all()
        assert torch.count_nonzero(single_grad) > 0


def test_neuron_without_signal_function_encodes_the_dendritic_output():
    neuron = SpikingGammaNeuron(
        dendritic_fn=SplitDendriticFunction(
            shape_out=(4,), n_kernels=3, synaptic_function=LinearPerNeuron(5, 4)
        ),
        spike_fn=SigmaDeltaSpikeFunction(
            GammaKernelFunction(torch.tensor([0.15, 0.4, 0.65])), SpikeConfig()
        ),
        return_all=True,
    )

    _, _, state = neuron(torch.rand(2, 5, 3))

    assert torch.equal(state["y"], state["x"])
