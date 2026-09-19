from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .critic import Critic, CriticOutput
from .imagine import Imagination
from .nets import Linear, RMSNorm
from .rssm import State

ACTENT = 3e-4
LOG2PI = math.log(2.0 * math.pi)


class BoundedNormal:
    """spec 4.6: diagonal Gaussian with tanh on the MEAN only.

    The sample is unsquashed, so there is no change-of-variables correction anywhere. A tanh-Normal
    with a log-det-Jacobian term would be a different policy class.
    """

    __slots__ = ("mean", "stddev")

    def __init__(self, mean: Tensor, stddev: Tensor) -> None:
        if mean.shape != stddev.shape:
            raise ValueError(
                f"mean {tuple(mean.shape)} and stddev {tuple(stddev.shape)} must match"
            )

        self.mean = mean
        self.stddev = stddev

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.stddev * noise

    def log_prob(self, action: Tensor) -> Tensor:
        z = (action - self.mean) / self.stddev
        # summed over the action dim: space.shape is truthy, so the reference wraps this in Agg(sum)
        return (-0.5 * z.pow(2) - self.stddev.log() - 0.5 * LOG2PI).sum(-1)

    def entropy(self) -> Tensor:
        return (0.5 * (LOG2PI + 1.0) + self.stddev.log()).sum(-1)

    @property
    def mode(self) -> Tensor:
        return self.mean


class Actor(nn.Module):
    """spec 4.6: a 640->64->64->64 trunk, then SEPARATE mean and stddev heads at outscale 0.01."""

    def __init__(
        self,
        in_features: int = 640,
        action_dim: int = 6,
        layers: int = 3,
        units: int = 64,
        minstd: float = 0.1,
        maxstd: float = 1.0,
        outscale: float = 0.01,
    ) -> None:
        super().__init__()

        if not 0.0 < minstd <= maxstd:
            raise ValueError(f"need 0 < minstd <= maxstd, got {minstd} and {maxstd}")

        self.action_dim = int(action_dim)
        self.minstd = float(minstd)
        self.maxstd = float(maxstd)

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        width = int(in_features)
        for _ in range(layers):
            self.linears.append(Linear(width, units))
            self.norms.append(RMSNorm(units))
            width = units

        # two heads, not one 2A kernel: the counts agree but the init draws and biases do not
        self.mean = Linear(width, self.action_dim, outscale=outscale)
        self.stddev = Linear(width, self.action_dim, outscale=outscale)

    def forward(self, feat: Tensor) -> BoundedNormal:
        x = feat
        for linear, norm in zip(self.linears, self.norms):
            x = F.silu(norm(linear(x)))

        # the +2.0 offset is hard-coded in the reference: ~0.893 at zero pre-activation
        spread = (self.maxstd - self.minstd) * torch.sigmoid(self.stddev(x) + 2.0) + self.minstd
        return BoundedNormal(torch.tanh(self.mean(x)), spread)


class ActorActionProvider:
    """The actor as an imagination policy.

    imagine_trajectory samples inside its no_grad block, so nothing produced here is differentiable;
    logpi and entropy are recomputed afterwards on the detached features, which is what makes the
    estimator REINFORCE rather than pathwise (spec 5.9).
    """

    def __init__(self, actor: Actor, seed: int = 0) -> None:
        self.actor = actor
        self.seed = int(seed)
        self._generator: torch.Generator | None = None

    def reseed(self, seed: int | None = None) -> "ActorActionProvider":
        if seed is not None:
            self.seed = int(seed)

        self._generator = None
        return self

    def _stream(self, device) -> torch.Generator:
        # a Generator must live on the model's device or CUDA sampling raises
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.seed)

        return self._generator

    def __call__(self, state: State) -> Tensor:
        return self.actor(state.feat).sample(self._stream(state.deter.device))


class ReturnNormalizer(nn.Module):
    """spec 5.6 `retnorm`: percentile EMAs of the imagined return, S = max(1, hi - lo).

    `debias` is **False** at the pin -- `configs.yaml#L111` overrides the class default of True --
    so the EMAs start at zero and S sits at the floor of 1.0 until they climb. The corrected branch
    is implemented because it is the class default and is one flag away from being the live one.
    """

    lo: Tensor
    hi: Tensor
    corr: Tensor

    def __init__(
        self,
        rate: float = 0.01,
        limit: float = 1.0,
        perclo: float = 5.0,
        perchi: float = 95.0,
        debias: bool = False,
    ) -> None:
        super().__init__()

        if not 0.0 < rate <= 1.0:
            raise ValueError(f"rate must be in (0, 1], got {rate}")
        if not 0.0 <= perclo < perchi <= 100.0:
            raise ValueError(f"need 0 <= perclo < perchi <= 100, got {perclo} and {perchi}")

        self.rate = float(rate)
        self.limit = float(limit)
        self.perclo = float(perclo)
        self.perchi = float(perchi)
        self.debias = bool(debias)

        self.register_buffer("lo", torch.zeros((), dtype=torch.float32))
        self.register_buffer("hi", torch.zeros((), dtype=torch.float32))
        self.register_buffer("corr", torch.zeros((), dtype=torch.float32))

    @torch.no_grad()
    def update(self, ret: Tensor) -> None:
        x = ret.detach().float().reshape(-1)

        if x.numel() == 0:
            raise ValueError("cannot take percentiles of an empty return batch")

        lo = torch.quantile(x, self.perclo / 100.0)
        hi = torch.quantile(x, self.perchi / 100.0)

        # lerp_(end, w) is (1 - w) * self + w * end, the reference's _update exactly
        self.lo.lerp_(lo.to(self.lo.dtype), self.rate)
        self.hi.lerp_(hi.to(self.hi.dtype), self.rate)

        if self.debias:
            self.corr.lerp_(torch.ones_like(self.corr), self.rate)

    def stats(self) -> tuple[Tensor, Tensor]:
        lo, hi = self.lo, self.hi

        if self.debias:
            factor = 1.0 / self.corr.clamp(min=self.rate)
            lo, hi = lo * factor, hi * factor

        return lo, (hi - lo).clamp(min=self.limit)

    def forward(self, ret: Tensor, update: bool = True) -> tuple[Tensor, Tensor]:
        if update:
            self.update(ret)

        return self.stats()

    @property
    def scale(self) -> Tensor:
        return self.stats()[1]


@dataclass
class ActorOutput:
    loss: Tensor
    advantage: Tensor
    log_prob: Tensor
    entropy: Tensor
    weight: Tensor
    scale: Tensor
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class BehaviorOutput:
    actor: ActorOutput
    imagined: CriticOutput
    replay: CriticOutput | None
    loss: Tensor
    metrics: dict[str, float] = field(default_factory=dict)


def imagined_actor_loss(
    actor: Actor,
    critic: Critic,
    imagination: Imagination,
    ret: Tensor,
    normalizer: ReturnNormalizer,
    actent: float = ACTENT,
    update: bool = True,
) -> ActorOutput:
    """REINFORCE on H imagined positions: L = -w(logpi * A + eta * H), entropy SUBTRACTED (spec 6.2)."""
    feat = imagination.feat

    if feat.requires_grad:
        raise ValueError(
            "imagined features carry a graph: imagine_trajectory must build them under no_grad so "
            "the actor gets no pathwise gradient through the world model (spec 5.9)"
        )

    horizon = imagination.horizon
    if ret.shape != (imagination.rollouts, horizon):
        raise ValueError(
            f"ret must be (N, H) = {(imagination.rollouts, horizon)}, got {tuple(ret.shape)}"
        )

    # H actions against H+1 states, so the policy runs on feat[:, :-1] and nothing is sliced twice
    policy = actor(feat[:, :-1])
    # sg(act): the sample is a constant, which is precisely what makes this REINFORCE (spec 5.6)
    log_prob = policy.log_prob(imagination.actions.detach())
    entropy = policy.entropy()

    with torch.no_grad():
        # tarval = val when slowtar is False: the FAST critic is both bootstrap and baseline
        baseline = critic.value.predict(feat)[:, :-1]

    _, scale = normalizer(ret.detach(), update)
    # sg(adv): a gradient through the baseline or the normalizer would not be REINFORCE
    advantage = ((ret.detach() - baseline) / scale).detach()
    # sg(weight): the continuation weight is a cumprod of con, which is a live head output
    weight = imagination.weight[:, :-1].detach()

    # ordinary position mean, not a division by the summed weight
    loss = (-weight * (log_prob * advantage + actent * entropy)).mean()

    with torch.no_grad():
        metrics = {
            "policy": float(loss),
            "policy_adv_mean": float(advantage.mean()),
            "policy_adv_std": float(advantage.std()),
            "policy_adv_mag": float(advantage.abs().mean()),
            "policy_logp_mean": float(log_prob.mean()),
            "policy_entropy": float(entropy.mean()),
            "policy_retnorm_scale": float(scale),
            "policy_baseline_mean": float(baseline.mean()),
            "policy_weight_mean": float(weight.mean()),
            "policy_positions": float(weight.numel()),
        }

    return ActorOutput(loss, advantage, log_prob, entropy, weight, scale, metrics)


def behavior_losses(
    actor: Actor,
    critic: Critic,
    imagination: Imagination,
    normalizer: ReturnNormalizer,
    replay: CriticOutput | None = None,
    actent: float = ACTENT,
    update: bool = True,
    actor_scale: float = 1.0,
) -> BehaviorOutput:
    """One imagination batch feeds both heads; the return and the baseline see the same critic."""
    imagined = critic.imagined_loss(imagination)
    policy = imagined_actor_loss(actor, critic, imagination, imagined.ret, normalizer, actent, update)

    loss = actor_scale * policy.loss + critic.total(imagined, replay)
    metrics = dict(policy.metrics)
    metrics.update(imagined.metrics)

    if replay is not None:
        metrics.update(replay.metrics)

    metrics["behavior"] = float(loss.detach())
    return BehaviorOutput(policy, imagined, replay, loss, metrics)
