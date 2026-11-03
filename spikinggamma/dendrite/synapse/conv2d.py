from spikinggamma.dendrite.synapse.base import BaseSynapticFunction
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dPerNeuron(nn.Conv2d, BaseSynapticFunction):  # same filters for all buckets
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W, n_kernels = x.shape
        x = x.permute(0, 4, 1, 2, 3).reshape(B * n_kernels, C_in, H, W)
        weight = self.weight
        bias = self.bias if self.bias is not None else None
        y = F.conv2d(
            x, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )
        _, C_out, H_out, W_out = y.shape
        y = y.reshape(B, n_kernels, C_out, H_out, W_out)
        y = y.permute(0, 2, 3, 4, 1)
        return y.contiguous()
