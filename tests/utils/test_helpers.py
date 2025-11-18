import torch
import torch.nn as nn
from spikinggamma.utils.enums import StepMode
from spikinggamma.utils.helpers import MultiStepLayerWrapper, MultiStepCrossEntropyLoss, multi_split_step_forward

def test_multi_step_layer_wrapper():
    batch_size = 4
    time_steps = 10
    n_inputs = 5
    n_outputs = 3
    g = torch.Generator().manual_seed(42) 

    x = torch.randn(time_steps, batch_size, n_inputs, generator=g)
    layer = nn.Linear(n_inputs, n_outputs)
    wrapper = MultiStepLayerWrapper(layer)

    # Test multi-step mode
    y_multi = wrapper(x, step_mode=StepMode.MULTI_STATELESS)
    assert y_multi.shape == (time_steps, batch_size, n_outputs), f"Unexpected shape: {y_multi.shape}"

    # Test single-step mode
    y_single = []
    for t in range(time_steps):
        y_t = wrapper(x[t:t+1], step_mode=StepMode.SINGLE)
        y_single.append(y_t)
    y_single = torch.cat(y_single, dim=0)
    assert torch.allclose(y_multi, y_single), "Outputs do not match between multi and single step modes"


def test_multi_step_cross_entropy_loss():
    batch_size = 4
    time_steps = 10
    n_outputs = 3
    g = torch.Generator().manual_seed(23) 

    x = torch.randn(time_steps, batch_size, n_outputs, generator=g)
    target = torch.randint(0, n_outputs, (batch_size,), generator=g)
    weight = torch.randn(n_outputs, requires_grad=True, generator=g)
    loss_fn = nn.CrossEntropyLoss()
    
    loss_sum = 0
    for t in range(time_steps):
        y_t = x[t] * weight.unsqueeze(0)
        loss = loss_fn(y_t, target) * 1/time_steps
        loss_sum += loss.item()
        loss.backward()
    grad1 = weight.grad.clone()
    weight.grad.zero_()

    loss_fn_multi = MultiStepCrossEntropyLoss()
    y = x * weight.unsqueeze(0).unsqueeze(0)
    loss_multi = loss_fn_multi(y, target)
    loss_multi.backward()
    grad2 = weight.grad.clone()

    assert abs(loss_sum - loss_multi.item()) < 1e-6, f"Losses do not match: {loss_sum} vs {loss_multi.item()}"
    assert torch.allclose(grad1, grad2), f"Gradients do not match: {grad1} vs {grad2}"
    assert grad1.abs().sum() > 0, "Gradient is zero"


def test_multi_split_step_forward_simple():
    g = torch.Generator().manual_seed(780) 
    batch_size = 4
    time_steps = 10
    n_features = 3
    n_hidden = 2
    n_outputs = 2
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = MultiStepLayerWrapper(nn.Linear(n_features, n_hidden))
            self.relu = MultiStepLayerWrapper(nn.ReLU())
            self.layer2 = MultiStepLayerWrapper(nn.Linear(n_hidden, n_outputs))

        def forward(self, x, step_mode):
            x = self.layer1(x, step_mode=step_mode)
            x = self.relu(x)
            x = self.layer2(x, step_mode=step_mode)
            return x, x, x
    simple_model = SimpleModel()

    x = torch.randn(time_steps, batch_size, n_features, generator=g)
    target = torch.randint(0, n_outputs, (batch_size,), generator=g)
    loss_fn = MultiStepCrossEntropyLoss()

    T_per_split = 2
    y_split, _, _, loss_split = multi_split_step_forward(x, target, simple_model, loss_fn, T_per_split)
    y_full, _, _ = simple_model(x, step_mode=StepMode.MULTI_STATEFUL)
    loss_full = loss_fn(y_full, target)
    assert y_split.shape == (time_steps, batch_size, n_outputs), f"Unexpected shape: {y_split.shape}"
    assert torch.allclose(y_split, y_full), "Outputs do not match between split and full forward"
    assert abs(loss_split - loss_full.item()) < 1e-6, f"Losses do not match: {loss_split} vs {loss_full.item()}"
    