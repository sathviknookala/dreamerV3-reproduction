from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

BINS = 255
LIM = 20.0


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


def make_bins(bins: int = BINS, lim: float = LIM, dtype=torch.float32) -> Tensor:
    if bins % 2 == 0:
        raise ValueError("bins must be odd so that an exact zero bin exists")

    half = symexp(torch.linspace(-lim, 0.0, (bins - 1) // 2 + 1, dtype=torch.float64))
    full = torch.cat([half, -half[:-1].flip(0)])
    return full.to(dtype)


def twohot_encode(value: Tensor, bins: Tensor) -> Tensor:
    value = value.clamp(bins[0], bins[-1]).unsqueeze(-1)

    hi = torch.searchsorted(bins, value.contiguous()).clamp(1, bins.numel() - 1)
    lo = hi - 1

    lower, upper = bins[lo], bins[hi]
    weight = ((value - lower) / (upper - lower)).clamp(0.0, 1.0)

    target = torch.zeros(*value.shape[:-1], bins.numel(), device=value.device, dtype=value.dtype)
    target.scatter_(-1, lo, 1.0 - weight)
    target.scatter_add_(-1, hi, weight)
    return target


def twohot_readout(probs: Tensor, bins: Tensor) -> Tensor:
    m = (bins.numel() - 1) // 2
    terms = probs * bins
    # mirror form: reverse the negative half onto the positive half so each pair
    # cancels in one op. a plain sum over +-4.85e8 bins returns -1.0, not 0
    centre = terms[..., m : m + 1].sum(-1)
    return centre + (terms[..., :m].flip(-1) + terms[..., m + 1 :]).sum(-1)


def twohot_loss(logits: Tensor, target_value: Tensor, bins: Tensor) -> Tensor:
    target = twohot_encode(target_value, bins)
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1)
