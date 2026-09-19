from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .heads import ContinuationHead, Decoder, RewardHead
from .rssm import RSSM, Encoder, State, observe_sequence
from .types import SequenceBatch

HORIZON = 333


@dataclass
class WorldModelOutput:
    losses: dict[str, Tensor]
    metrics: dict[str, float]
    total: Tensor
    post: State
    prior: State


def categorical_kl(q_probs: Tensor, p_probs: Tensor) -> Tensor:
    # reduce classes, then SUM the 32 factors -- free nats apply after this
    per_class = q_probs * (torch.log(q_probs) - torch.log(p_probs))
    return per_class.sum(-1).sum(-1)


class WorldModel(nn.Module):
    def __init__(
        self,
        action_dim: int,
        free_nats: float = 1.0,
        unimix: float = 0.01,
        dyn_scale: float = 1.0,
        rep_scale: float = 0.1,
        rec_scale: float = 1.0,
        rew_scale: float = 1.0,
        con_scale: float = 1.0,
        horizon: int = HORIZON,
        contdisc: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.rssm = RSSM(action_dim=action_dim, tokens=self.encoder.tokens, unimix=unimix)
        self.decoder = Decoder(deter=self.rssm.deter, stoch_flat=self.rssm.stoch_flat)
        self.reward = RewardHead(in_features=self.rssm.feat_size)
        self.cont = ContinuationHead(in_features=self.rssm.feat_size)

        self.free_nats = float(free_nats)
        self.scales = {
            "rec": float(rec_scale),
            "rew": float(rew_scale),
            "con": float(con_scale),
            "dyn": float(dyn_scale),
            "rep": float(rep_scale),
        }
        self.horizon = int(horizon)
        self.contdisc = bool(contdisc)

    def continuation_target(self, is_terminal: Tensor) -> Tensor:
        target = (~is_terminal.to(torch.bool)).to(torch.float32)

        if self.contdisc:
            target = target * (1.0 - 1.0 / self.horizon)

        return target

    def kl_losses(self, post: State, prior: State) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        q = self.rssm.probs(post.logit)
        p = self.rssm.probs(prior.logit)

        # dyn trains the prior toward the posterior; rep the posterior toward the prior
        dyn = categorical_kl(q.detach(), p)
        rep = categorical_kl(q, p.detach())

        return (
            dyn.clamp(min=self.free_nats),
            rep.clamp(min=self.free_nats),
            dyn.detach(),
            rep.detach(),
        )

    def loss(
        self,
        observations: Tensor,
        actions: Tensor,
        rewards: Tensor,
        is_terminal: Tensor,
        loss_mask: Tensor,
        generator: torch.Generator | None = None,
    ) -> WorldModelOutput:
        post, prior = observe_sequence(
            self.encoder, self.rssm, observations, actions, generator=generator
        )

        # transition j is supervised from the state resulting from it, s_{j+1};
        # this drops s_0, the reset observation with no preceding reward target
        post_t = post[:, 1:]
        prior_t = prior[:, 1:]
        feat = post_t.feat

        mask = loss_mask.to(torch.float32)
        denom = mask.sum().clamp(min=1.0)

        target_image = observations[:, 1:].to(torch.float32) / 255.0
        recon = self.decoder(post_t.deter, post_t.stoch)
        # summed over the 3 image axes, NOT meaned -- see spec 5.3
        rec = (recon - target_image.detach()).square().sum((-1, -2, -3))

        rew = self.reward.loss(feat, rewards)
        con = self.cont.loss(feat, self.continuation_target(is_terminal))
        dyn, rep, dyn_raw, rep_raw = self.kl_losses(post_t, prior_t)

        losses = {"rec": rec, "rew": rew, "con": con, "dyn": dyn, "rep": rep}
        reduced = {k: (v * mask).sum() / denom for k, v in losses.items()}
        total = sum(self.scales[k] * v for k, v in reduced.items())

        with torch.no_grad():
            predicted_reward = self.reward.predict(feat)
            metrics = {k: float(v) for k, v in reduced.items()}
            metrics.update(
                total=float(total),
                dyn_raw=float((dyn_raw * mask).sum() / denom),
                rep_raw=float((rep_raw * mask).sum() / denom),
                post_entropy=float((self._entropy(post_t.logit) * mask).sum() / denom),
                prior_entropy=float((self._entropy(prior_t.logit) * mask).sum() / denom),
                reward_mae=float(((predicted_reward - rewards).abs() * mask).sum() / denom),
                cont_mae=float(
                    (
                        (self.cont.predict(feat) - self.continuation_target(is_terminal)).abs()
                        * mask
                    ).sum()
                    / denom
                ),
                train_positions=float(denom),
            )

        return WorldModelOutput(reduced, metrics, total, post, prior)

    def _entropy(self, logit: Tensor) -> Tensor:
        probs = self.rssm.probs(logit)
        return -(probs * torch.log(probs)).sum(-1).sum(-1)

    def loss_from_batch(
        self,
        batch: SequenceBatch,
        device=None,
        generator: torch.Generator | None = None,
    ) -> WorldModelOutput:
        return self.loss(*batch_to_tensors(batch, device or self.device), generator=generator)

    @property
    def device(self):
        return next(self.parameters()).device


def batch_to_tensors(batch: SequenceBatch, device) -> tuple[Tensor, ...]:
    def to(array, dtype=None):
        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device)
        return tensor if dtype is None else tensor.to(dtype)

    return (
        to(batch.observations),
        to(batch.actions, torch.float32),
        to(batch.rewards, torch.float32),
        to(batch.is_terminal, torch.bool),
        to(batch.loss_mask, torch.float32),
    )
