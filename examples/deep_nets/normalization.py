
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ported, TODO: test and verify
class WSConv2dPerNeuron(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, gain=False, eps=1e-4):
        super(WSConv2dPerNeuron, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
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

# Ported, TODO: test and verify
class WSLinearPerNeuron(nn.Linear):

    def __init__(self, in_features, out_features, bias=False, gain=False, eps=1e-4):
        super(WSLinearPerNeuron, self).__init__(in_features, out_features, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_features, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1], keepdims=True)
        var = torch.var(self.weight, axis=[1], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        x = x.transpose(-1, -2)
        y = F.linear(x, self.get_weight(), self.bias)
        return y.transpose(-1, -2)

@torch.jit.script
def calc_bn(x, mean, var, eps, gamma, beta):
    return (x - mean) / torch.sqrt(var + eps) * gamma + beta
    
class BatchNorm1dThroughTime(nn.Module):
    """Batch Normalization through time.
    """
    def __init__(self, num_features, n_timesteps, **kwargs):
        super(BatchNorm1dThroughTime, self).__init__()
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_features, **kwargs) for _ in range(n_timesteps)
        ])
        self.t = 0

    def reset_state(self):
        if self.t != 0 and self.t < len(self.batch_norms) - 1:
            print("Warning: reset_state called before all timesteps have been processed.")
        self.t = 0

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, num_features, num_timesteps)
        Returns:
            torch.Tensor: normalized tensor
        """
        if self.t >= len(self.batch_norms):
            raise ValueError("Time step exceeds the number of batch norms.")
        x = self.batch_norms[self.t](x)
        self.t += 1
        return x

class BatchNorm2dThroughTime(nn.Module):
    """Batch Normalization through time.
    """
    def __init__(self, num_features, n_timesteps, **kwargs):
        super(BatchNorm2dThroughTime, self).__init__()
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(num_features, **kwargs) for _ in range(n_timesteps)
        ])
        self.t = 0

    def reset_state(self):
        if self.t != 0 and self.t < len(self.batch_norms) - 1:
            print("Warning: reset_state called before all timesteps have been processed.")
        self.t = 0

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, num_features, num_timesteps)
        Returns:
            torch.Tensor: normalized tensor
        """
        if self.t >= len(self.batch_norms):
            raise ValueError("Time step exceeds the number of batch norms.")
        x = self.batch_norms[self.t](x)
        self.t += 1
        return x