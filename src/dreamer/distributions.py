from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def unimix_probs(logits: Tensor, unimix: float = 0.01) -> Tensor:
    probs = F.softmax(logits, dim=-1)

    if unimix:
        classes = logits.shape[-1]
        probs = (1.0 - unimix) * probs + unimix / classes

    return probs


def unimix_logits(logits: Tensor, unimix: float = 0.01) -> Tensor:
    return torch.log(unimix_probs(logits, unimix))


def sample_onehot(
    probs: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    classes = probs.shape[-1]
    flat = probs.reshape(-1, classes)
    index = torch.multinomial(flat, 1, replacement=True, generator=generator)
    onehot = F.one_hot(index.squeeze(-1), classes).to(probs.dtype)
    return onehot.reshape(probs.shape)


def straight_through_sample(
    logits: Tensor,
    unimix: float = 0.01,
    generator: torch.Generator | None = None,
) -> Tensor:
    probs = unimix_probs(logits, unimix)
    onehot = sample_onehot(probs.detach(), generator)
    # parenthesization is load-bearing: (probs - sg(probs)) is exactly 0, so the
    # forward value is an exact one-hot; associating left rounds it off
    return onehot.detach() + (probs - probs.detach())


def categorical_entropy(logits: Tensor, unimix: float = 0.01) -> Tensor:
    probs = unimix_probs(logits, unimix)
    log_probs = torch.log(probs)
    # sum over classes, then sum over the 32 factors -- never a mean
    return -(probs * log_probs).sum(-1).sum(-1)
