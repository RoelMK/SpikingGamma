from spikinggamma.dendrite.synapse.base import BaseSynapticFunction
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class WSConv2dPerNeuron(nn.Conv2d): # same filters for all buckets, with weight standardization
    def __init__(self, *args, eps: float = 1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def get_standardized_weight(self):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + self.eps
        return weight / std.expand_as(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W, n_buckets = x.shape
        x = x.permute(0, 4, 1, 2, 3).reshape(B * n_buckets, C_in, H, W)
        weight = self.get_standardized_weight()

        if self.padding_mode != 'zeros':
            y = F.conv2d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                self.bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        else:
            y = F.conv2d(
                x,
                weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        _, C_out, H_out, W_out = y.shape
        y = y.reshape(B, n_buckets, C_out, H_out, W_out)
        y = y.permute(0, 2, 3, 4, 1)
        return y.contiguous()
    
