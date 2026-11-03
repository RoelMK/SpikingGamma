import pytest
import torch

from spikinggamma.spike.kernel.gamma import GammaKernelFunction


@pytest.mark.parametrize(
    "kernel_fn",
    [
        pytest.param(
            GammaKernelFunction(torch.tensor([0.25, 0.5, 0.75])), id="gamma"
        ),
        pytest.param(
            GammaKernelFunction(
                torch.tensor([0.25, 0.5, 0.75]), sequential_buckets=True
            ),
            id="gamma-sequential",
        ),
    ],
)
def test_kernel_step_detaches_previous_timestep(kernel_fn):
    state = kernel_fn.construct_kernel_state()
    previous_input = torch.ones(2, 3, requires_grad=True)
    kernel_fn.add(state, previous_input)

    # Call the kernel directly so a spike function's no_grad cannot hide leaks.
    kernel_fn.step(state)
    for name, value in vars(state).items():
        assert not value.requires_grad, name
        assert value.grad_fn is None, name
        assert torch.count_nonzero(value) > 0, name

    current_input = torch.full((2, 3), 2.0, requires_grad=True)
    kernel_fn.add(state, current_input)
    loss =sum(value.sum() for value in vars(state).values())
    previous_grad, current_grad = torch.autograd.grad(
        loss, (previous_input, current_input), allow_unused=True
    )
    assert previous_grad is None
    assert current_grad is not None
    assert torch.isfinite(current_grad).all()
    assert torch.all(current_grad != 0)
