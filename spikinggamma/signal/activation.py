import torch
import torch.nn as nn

class ZeroActivatedReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
    
class ZeroActivatedReLU(nn.Module):
    def forward(self, x):
        return ZeroActivatedReLUFunction.apply(x)
