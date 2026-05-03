from spikinggamma.dendrite.synapse.base import BaseSynapticFunction
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Conv2dPerNeuron(nn.Conv2d, BaseSynapticFunction): # same filters for all buckets
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W, n_buckets = x.shape
        x = x.permute(0, 4, 1, 2, 3).reshape(B * n_buckets, C_in, H, W)
        y = super().forward(x)
        _, C_out, H_out, W_out = y.shape
        y = y.reshape(B, n_buckets, C_out, H_out, W_out)
        y = y.permute(0, 2, 3, 4, 1) 
        return y.contiguous()
    

class Conv2dPerBucket(nn.Conv2d, BaseSynapticFunction): # unique filters for each bucket
    def __init__(self, in_channels, out_channels, n_buckets, *args, groups=1, **kwargs):
        in_channels = in_channels * n_buckets
        out_channels = out_channels * n_buckets
        groups = n_buckets
        super().__init__(in_channels, out_channels, *args, groups=groups, **kwargs)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.groups != inp.shape[4]:
            raise ValueError("Incorrect number of buckets")
        inp = inp.permute(0, 1, 4, 2, 3)
        inp = inp.reshape(inp.shape[0], inp.shape[1] * inp.shape[2], inp.shape[3], inp.shape[4])
        out = super().forward(inp)
        batch_size, combined_c_out, x_out, y_out = out.shape
        c_out = combined_c_out // self.groups
        n_buckets = self.groups
        out = out.reshape(batch_size, c_out, n_buckets, x_out, y_out)
        out = out.permute(0, 1, 3, 4, 2)
        return out


class WSConv2dPerNeuron(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, gain=False, eps=1e-4):
        super(WSConv2dPerNeuron, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = math.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        B, C_in, H, W, n_buckets = x.shape
        x = x.permute(0, 4, 1, 2, 3).reshape(B * n_buckets, C_in, H, W)
        y = F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)
        _, C_out, H_out, W_out = y.shape
        y = y.reshape(B, n_buckets, C_out, H_out, W_out)
        y = y.permute(0, 2, 3, 4, 1) 
        return y.contiguous()

