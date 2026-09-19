from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

TRUNC_NORMAL_CORRECTION = 1.1368


def trunc_normal_in_(tensor: Tensor, fan_in: int, outscale: float = 1.0) -> Tensor:
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)
        tensor.mul_(TRUNC_NORMAL_CORRECTION * math.sqrt(1.0 / fan_in) * outscale)
    return tensor


class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.features = int(features)
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(self.features, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        w = x.float()
        mean2 = w.pow(2).mean(-1, keepdim=True)
        y = w * torch.rsqrt(mean2 + self.eps) * self.scale
        return y.to(x.dtype)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, outscale: float = 1.0) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        trunc_normal_in_(self.weight, self.in_features, outscale)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


class BlockLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        blocks: int,
        outscale: float = 1.0,
    ) -> None:
        super().__init__()

        if in_features % blocks or out_features % blocks:
            raise ValueError("in_features and out_features must divide blocks")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.blocks = int(blocks)
        self.block_in = self.in_features // self.blocks
        self.block_out = self.out_features // self.blocks

        self.kernel = nn.Parameter(
            torch.empty(self.blocks, self.block_in, self.block_out)
        )
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        # fan_in is the FULL input width, not block_in: compute_fans takes the rank>=3 branch
        trunc_normal_in_(self.kernel, self.in_features, outscale)

    def forward(self, x: Tensor) -> Tensor:
        lead = x.shape[:-1]
        x = x.reshape(*lead, self.blocks, self.block_in)
        x = torch.einsum("...ki,kio->...ko", x, self.kernel)
        return x.reshape(*lead, self.out_features) + self.bias


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 5,
        outscale: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel = int(kernel)

        if self.kernel % 2 == 0:
            raise ValueError("kernel must be odd for symmetric 'same' padding")

        self.weight = nn.Parameter(
            torch.empty(self.out_channels, self.in_channels, self.kernel, self.kernel)
        )
        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        trunc_normal_in_(self.weight, self.kernel * self.kernel * self.in_channels, outscale)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight, self.bias, stride=1, padding=self.kernel // 2)


def flat2group(x: Tensor, blocks: int) -> Tensor:
    return x.reshape(*x.shape[:-1], blocks, x.shape[-1] // blocks)


def group2flat(x: Tensor) -> Tensor:
    return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
