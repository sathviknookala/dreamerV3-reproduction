from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .heads import RewardHead
from .imagine import Imagination

HORIZON = 333


class ValueHead(RewardHead):
    """spec 4.7: structurally the reward head with layers=3. Same bins, same mirror readout."""

    def __init__(self, in_features: int = 640, layers: int = 3, units: int = 64) -> None:
        super().__init__(in_features, layers, units)


class SlowCritic(nn.Module):
    """EMA mirror. A regularizer target only -- never a bootstrap (slowtar: False, spec 5.7)."""

    def __init__(self, value: ValueHead, rate: float = 0.02) -> None:
        super().__init__()
        self.mirror = copy.deepcopy(value).requires_grad_(False)
        self.rate = float(rate)

    @torch.no_grad()
    def update(self, value: ValueHead) -> None:
        for slow, fast in zip(self.mirror.parameters(), value.parameters()):
            slow.lerp_(fast.detach(), self.rate)

    def predict(self, feat: Tensor) -> Tensor:
        return self.mirror.predict(feat)

    def forward(self, feat: Tensor) -> Tensor:
        return self.mirror(feat)


def lambda_return(
    last: Tensor,
    term: Tensor,
    rew: Tensor,
    boot: Tensor,
    disc: float,
    lam: float,
) -> Tensor:
    """spec 6.1: the bootstrap is boot[:, 1:], seeded at boot[:, -1]. Output L-1, R_t aligns with t."""
    if not (last.shape == term.shape == rew.shape == boot.shape):
        raise ValueError(
            f"lambda_return needs matching shapes, got last {tuple(last.shape)}, "
            f"term {tuple(term.shape)}, rew {tuple(rew.shape)}, boot {tuple(boot.shape)}"
        )

    rets = [boot[:, -1]]
    live = (1.0 - term.float())[:, 1:] * disc
    # the reference calls this `cont`; it is the lambda trace, NOT the continuation
    trace = (1.0 - last.float())[:, 1:] * lam
    interm = rew[:, 1:] + (1.0 - trace) * live * boot[:, 1:]

    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * trace[:, t] * rets[-1])

    return torch.stack(list(reversed(rets))[:-1], dim=1)


def scatter_imagined_return(
    ret: Tensor,
    index: Tensor,
    batch_size: int,
    length: int,
) -> tuple[Tensor, Tensor]:
    """ret[:, 0] back into the (B, P+T) grid that imagine.select_start_states' index addresses."""
    if ret.shape[0] != index.shape[0]:
        raise ValueError(
            f"{ret.shape[0]} returns do not match {index.shape[0]} start indices"
        )

    grid = torch.zeros(batch_size * length, dtype=ret.dtype, device=ret.device)
    filled = torch.zeros(batch_size * length, dtype=torch.bool, device=ret.device)
    grid[index] = ret[:, 0]
    filled[index] = True
    return grid.reshape(batch_size, length), filled.reshape(batch_size, length)


def check_bootstrap_holes(filled: Tensor, is_terminal: Tensor) -> None:
    """A hole corrupts every EARLIER target; only a terminal zeroes live[:, t-1] and contains it."""
    if bool((~filled.bool() & ~is_terminal.bool()).any()):
        raise ValueError(
            "bootstrap hole at a non-terminal position: the backward recursion would "
            "propagate it into every earlier target, which the position weight does not mask"
        )


@dataclass
class CriticOutput:
    loss: Tensor
    ret: Tensor
    weight: Tensor
    metrics: dict[str, float] = field(default_factory=dict)


class Critic(nn.Module):
    def __init__(
        self,
        in_features: int = 640,
        lam: float = 0.95,
        horizon: int = HORIZON,
        slowreg: float = 1.0,
        slow_rate: float = 0.02,
        imag_scale: float = 1.0,
        repl_scale: float = 0.3,
    ) -> None:
        super().__init__()
        self.value = ValueHead(in_features)
        self.slow = SlowCritic(self.value, slow_rate)

        self.lam = float(lam)
        self.horizon = int(horizon)
        self.slowreg = float(slowreg)
        self.scales = {"value": float(imag_scale), "repval": float(repl_scale)}

    def trainable_parameters(self):
        """slowval is excluded from the optimizer's module list (spec 5.7)."""
        return self.value.parameters()

    @torch.no_grad()
    def update_slow(self) -> None:
        self.slow.update(self.value)

    def _fit(self, feat: Tensor, target: Tensor, weight: Tensor, tag: str) -> CriticOutput:
        # sg on the slow prediction: without it gradient reaches the mirror and it stops being an EMA
        slow_target = self.slow.predict(feat).detach()
        per_position = self.value.loss(feat, target) + self.slowreg * self.value.loss(
            feat, slow_target
        )
        loss = (per_position * weight).mean()

        with torch.no_grad():
            predicted = self.value.predict(feat)
            metrics = {
                f"{tag}": float(loss),
                f"{tag}_value_mean": float(predicted.mean()),
                f"{tag}_target_mean": float(target.mean()),
                f"{tag}_value_mae": float((predicted - target).abs().mean()),
                f"{tag}_slow_gap": float((predicted - slow_target).abs().mean()),
                f"{tag}_weight_mean": float(weight.mean()),
                f"{tag}_positions": float(weight.numel()),
            }

        return CriticOutput(loss, target, weight, metrics)

    def imagined_loss(self, imagination: Imagination) -> CriticOutput:
        feat = imagination.feat
        # the FAST critic bootstraps the imagined path -- slowtar: False (spec 5.7)
        boot = self.value.predict(feat)

        zero = torch.zeros_like(imagination.reward[:, :1])
        rew = torch.cat([zero, imagination.reward], dim=1)
        # cont_start is not read by the kernel -- term[:, 1:] is exactly imagination.cont
        con = torch.cat([imagination.cont_start, imagination.cont], dim=1)
        last = torch.zeros_like(con)

        # disc = 1: the per-step discount IS the continuation head under contdisc (spec 5.4)
        ret = lambda_return(last, 1.0 - con, rew, boot, 1.0, self.lam)

        return self._fit(
            feat[:, :-1],
            ret.detach(),
            imagination.weight[:, :-1].detach(),
            "value",
        )

    def replay_loss(
        self,
        feat: Tensor,
        rewards: Tensor,
        is_last: Tensor,
        is_terminal: Tensor,
        boot: Tensor,
    ) -> CriticOutput:
        """feat is NOT detached: repval gradients reach the encoder and RSSM (spec 6.3)."""
        # contdisc does not apply here; the replay path hard-codes the discount (spec 5.4)
        disc = 1.0 - 1.0 / self.horizon
        ret = lambda_return(is_last, is_terminal, rewards, boot, disc, self.lam)
        weight = (~is_last.bool()).to(feat.dtype)[:, :-1]

        return self._fit(feat[:, :-1], ret.detach(), weight, "repval")

    def total(self, imagined: CriticOutput, replay: CriticOutput | None = None) -> Tensor:
        loss = self.scales["value"] * imagined.loss

        if replay is not None:
            loss = loss + self.scales["repval"] * replay.loss

        return loss
