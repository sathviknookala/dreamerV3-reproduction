from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .nets import BlockLinear, Conv2d, Linear, RMSNorm
from .twohot import BINS, make_bins, twohot_loss, twohot_readout


class Decoder(nn.Module):
    def __init__(
        self,
        deter: int = 512,
        stoch_flat: int = 128,
        depths: tuple[int, ...] = (8, 12, 16, 16),
        kernel: int = 5,
        minres: int = 4,
        blocks: int = 8,
        units: int = 64,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.depths = tuple(depths)
        self.minres = int(minres)
        self.blocks = int(blocks)
        self.seed_channels = self.depths[-1]
        self.tokens = self.minres * self.minres * self.seed_channels

        self.sp0 = BlockLinear(deter, self.tokens, self.blocks)
        self.sp1 = Linear(stoch_flat, 2 * units)
        self.sp1norm = RMSNorm(2 * units)
        self.sp2 = Linear(2 * units, self.tokens)
        self.spnorm = RMSNorm(self.seed_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_channels = self.seed_channels
        for i in reversed(range(len(self.depths) - 1)):
            self.convs.append(Conv2d(in_channels, self.depths[i], kernel))
            self.norms.append(RMSNorm(self.depths[i]))
            in_channels = self.depths[i]

        self.imgout = Conv2d(in_channels, out_channels, kernel, outscale=1.0)

    def forward(self, deter: Tensor, stoch: Tensor) -> Tensor:
        lead = deter.shape[:-1]
        deter = deter.reshape(-1, deter.shape[-1])
        stoch = stoch.reshape(deter.shape[0], -1)

        g, r = self.blocks, self.minres
        # '(g h w c) -> h w (g c)': deter block b owns channels [2b, 2b+1] everywhere
        x0 = self.sp0(deter).reshape(-1, g, r, r, self.seed_channels // g)
        x0 = x0.permute(0, 2, 3, 1, 4).reshape(-1, r, r, self.seed_channels)

        x1 = F.silu(self.sp1norm(self.sp1(stoch)))
        x1 = self.sp2(x1).reshape(-1, r, r, self.seed_channels)

        x = F.silu(self.spnorm(x0 + x1)).permute(0, 3, 1, 2)

        for conv, norm in zip(self.convs, self.norms):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = conv(x)
            x = F.silu(norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.sigmoid(self.imgout(x)).permute(0, 2, 3, 1)
        return x.reshape(*lead, *x.shape[1:])


class MLPHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layers: int = 1,
        units: int = 64,
        outscale: float = 1.0,
    ) -> None:
        super().__init__()
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        width = in_features
        for _ in range(layers):
            self.linears.append(Linear(width, units))
            self.norms.append(RMSNorm(units))
            width = units

        self.out = Linear(width, out_features, outscale=outscale)

    def forward(self, x: Tensor) -> Tensor:
        for linear, norm in zip(self.linears, self.norms):
            x = F.silu(norm(linear(x)))
        return self.out(x)


class RewardHead(nn.Module):
    def __init__(self, in_features: int = 640, layers: int = 1, units: int = 64) -> None:
        super().__init__()
        # outscale 0.0 makes the logits uniform at init, which with the mirror
        # readout gives a prediction of exactly 0
        self.mlp = MLPHead(in_features, BINS, layers, units, outscale=0.0)
        self.register_buffer("bins", make_bins(), persistent=False)

    def forward(self, feat: Tensor) -> Tensor:
        return self.mlp(feat)

    def predict(self, feat: Tensor) -> Tensor:
        return twohot_readout(F.softmax(self(feat), dim=-1), self.bins)

    def loss(self, feat: Tensor, target: Tensor) -> Tensor:
        return twohot_loss(self(feat), target, self.bins)


class ContinuationHead(nn.Module):
    def __init__(self, in_features: int = 640, layers: int = 1, units: int = 64) -> None:
        super().__init__()
        self.mlp = MLPHead(in_features, 1, layers, units, outscale=1.0)

    def forward(self, feat: Tensor) -> Tensor:
        return self.mlp(feat).squeeze(-1)

    def predict(self, feat: Tensor) -> Tensor:
        # the mean, sigma(logit) -- never the Binary mode
        return torch.sigmoid(self(feat))

    def loss(self, feat: Tensor, target: Tensor) -> Tensor:
        # accepts the fractional soft label 0.996997, not just 0/1
        return F.binary_cross_entropy_with_logits(self(feat), target, reduction="none")
