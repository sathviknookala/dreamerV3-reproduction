from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .distributions import straight_through_sample, unimix_probs
from .nets import BlockLinear, Conv2d, Linear, RMSNorm, flat2group, group2flat
from .types import SequenceBatch


@dataclass
class State:
    deter: Tensor
    stoch: Tensor
    logit: Tensor

    @property
    def feat(self) -> Tensor:
        lead = self.stoch.shape[:-2]
        return torch.cat(
            [self.deter, self.stoch.reshape(*lead, -1)],
            dim=-1,
        )

    def detach(self) -> "State":
        return State(self.deter.detach(), self.stoch.detach(), self.logit.detach())

    def __getitem__(self, index) -> "State":
        return State(self.deter[index], self.stoch[index], self.logit[index])


def stack_states(states: list[State], dim: int) -> State:
    return State(
        deter=torch.stack([s.deter for s in states], dim=dim),
        stoch=torch.stack([s.stoch for s in states], dim=dim),
        logit=torch.stack([s.logit for s in states], dim=dim),
    )


class Encoder(nn.Module):
    def __init__(
        self,
        depths: tuple[int, ...] = (8, 12, 16, 16),
        kernel: int = 5,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.depths = tuple(depths)

        channels = (in_channels,) + self.depths
        self.convs = nn.ModuleList(
            [
                Conv2d(channels[i], channels[i + 1], kernel)
                for i in range(len(self.depths))
            ]
        )
        self.norms = nn.ModuleList([RMSNorm(d) for d in self.depths])
        self.minres = 4
        self.tokens = self.minres * self.minres * self.depths[-1]

    def forward(self, image: Tensor) -> Tensor:
        if image.dtype != torch.uint8:
            raise TypeError(f"Encoder expects uint8 NHWC images, got {image.dtype}")

        lead = image.shape[:-3]
        x = image.reshape(-1, *image.shape[-3:]).float()
        # encoder shift is /255 - 0.5; the decoder target is /255 with NO shift
        x = x / 255.0 - 0.5
        x = x.permute(0, 3, 1, 2)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = F.max_pool2d(x, 2)
            x = norm(x.permute(0, 2, 3, 1))
            x = F.silu(x).permute(0, 3, 1, 2)

        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        return x.reshape(*lead, self.tokens)


class RSSM(nn.Module):
    def __init__(
        self,
        action_dim: int,
        deter: int = 512,
        stoch: int = 32,
        classes: int = 4,
        hidden: int = 64,
        blocks: int = 8,
        tokens: int = 256,
        unimix: float = 0.01,
        imglayers: int = 2,
        obslayers: int = 1,
        dynlayers: int = 1,
    ) -> None:
        super().__init__()

        if dynlayers != 1:
            raise NotImplementedError("dynlayers > 1 is outside the pinned configuration")

        self.action_dim = int(action_dim)
        self.deter = int(deter)
        self.stoch = int(stoch)
        self.classes = int(classes)
        self.hidden = int(hidden)
        self.blocks = int(blocks)
        self.tokens = int(tokens)
        self.unimix = float(unimix)
        self.stoch_flat = self.stoch * self.classes
        self.feat_size = self.deter + self.stoch_flat

        self.dynin0 = Linear(self.deter, self.hidden)
        self.dynin1 = Linear(self.stoch_flat, self.hidden)
        self.dynin2 = Linear(self.action_dim, self.hidden)
        self.dynin0norm = RMSNorm(self.hidden)
        self.dynin1norm = RMSNorm(self.hidden)
        self.dynin2norm = RMSNorm(self.hidden)

        core_in = self.deter + 3 * self.hidden * self.blocks
        self.dynhid0 = BlockLinear(core_in, self.deter, self.blocks)
        self.dynhid0norm = RMSNorm(self.deter)
        self.dyngru = BlockLinear(self.deter, 3 * self.deter, self.blocks)

        self.obs0 = Linear(self.deter + self.tokens, self.hidden)
        self.obs0norm = RMSNorm(self.hidden)
        self.obslogit = Linear(self.hidden, self.stoch_flat)

        self.prior0 = Linear(self.deter, self.hidden)
        self.prior0norm = RMSNorm(self.hidden)
        self.prior1 = Linear(self.hidden, self.hidden)
        self.prior1norm = RMSNorm(self.hidden)
        self.priorlogit = Linear(self.hidden, self.stoch_flat)

        if obslayers != 1 or imglayers != 2:
            raise NotImplementedError("layer counts are fixed at obslayers=1, imglayers=2")

    def initial(self, batch_size: int, device=None, dtype=torch.float32) -> State:
        device = device or next(self.parameters()).device
        return State(
            deter=torch.zeros(batch_size, self.deter, device=device, dtype=dtype),
            stoch=torch.zeros(
                batch_size, self.stoch, self.classes, device=device, dtype=dtype
            ),
            logit=torch.zeros(
                batch_size, self.stoch, self.classes, device=device, dtype=dtype
            ),
        )

    def _core(self, deter: Tensor, stoch: Tensor, action: Tensor) -> Tensor:
        stoch = stoch.reshape(stoch.shape[0], -1)
        # stop-gradient divisor: for |a|>1 the gradient is scaled by 1/|a|, not zeroed
        action = action / action.abs().clamp(min=1.0).detach()

        x0 = F.silu(self.dynin0norm(self.dynin0(deter)))
        x1 = F.silu(self.dynin1norm(self.dynin1(stoch)))
        x2 = F.silu(self.dynin2norm(self.dynin2(action)))

        x = torch.cat([x0, x1, x2], dim=-1)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)
        x = group2flat(torch.cat([flat2group(deter, self.blocks), x], dim=-1))

        x = F.silu(self.dynhid0norm(self.dynhid0(x)))
        x = self.dyngru(x)

        reset, cand, update = self._split_gates(x)

        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)
        return update * cand + (1.0 - update) * deter

    def _split_gates(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # group BEFORE splitting: index b*192 + gate*64 + u -> gate index b*64 + u.
        # a flat split(x, 3, -1) misassigns whole blocks and still trains
        gates = x.reshape(*x.shape[:-1], self.blocks, 3, self.deter // self.blocks)
        return tuple(group2flat(gates[..., i, :]) for i in range(3))

    def _posterior_logits(self, deter: Tensor, embed: Tensor) -> Tensor:
        x = torch.cat([deter, embed], dim=-1)
        x = F.silu(self.obs0norm(self.obs0(x)))
        return self.obslogit(x).reshape(-1, self.stoch, self.classes)

    def _prior_logits(self, deter: Tensor) -> Tensor:
        x = F.silu(self.prior0norm(self.prior0(deter)))
        x = F.silu(self.prior1norm(self.prior1(x)))
        return self.priorlogit(x).reshape(-1, self.stoch, self.classes)

    def _sample(self, logits: Tensor, generator: torch.Generator | None) -> Tensor:
        return straight_through_sample(logits, self.unimix, generator)

    @staticmethod
    def _mask(x: Tensor, keep: Tensor) -> Tensor:
        return x * keep.reshape(-1, *([1] * (x.dim() - 1))).to(x.dtype)

    def _masked_inputs(
        self,
        state: State,
        action: Tensor,
        is_first: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        deter, stoch = state.deter, state.stoch

        if is_first is not None:
            keep = ~is_first.to(torch.bool)
            deter = self._mask(deter, keep)
            stoch = self._mask(stoch, keep)
            action = self._mask(action, keep)

        return deter, stoch, action

    def observe_step(
        self,
        state: State,
        action: Tensor,
        embed: Tensor,
        is_first: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[State, State]:
        deter, stoch, action = self._masked_inputs(state, action, is_first)
        deter = self._core(deter, stoch, action)

        post_logit = self._posterior_logits(deter, embed)
        prior_logit = self._prior_logits(deter)

        post = State(deter, self._sample(post_logit, generator), post_logit)
        prior = State(deter, self._sample(prior_logit, generator), prior_logit)
        return post, prior

    def imagine_step(
        self,
        state: State,
        action: Tensor,
        is_first: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> State:
        deter, stoch, action = self._masked_inputs(state, action, is_first)
        deter = self._core(deter, stoch, action)
        prior_logit = self._prior_logits(deter)
        return State(deter, self._sample(prior_logit, generator), prior_logit)

    def observe(
        self,
        embeds: Tensor,
        prev_actions: Tensor,
        is_first: Tensor | None = None,
        state: State | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[State, State]:
        batch, length = embeds.shape[0], embeds.shape[1]

        if prev_actions.shape[:2] != (batch, length):
            raise ValueError(
                f"prev_actions must be (B, T, A) aligned with embeds, got "
                f"{tuple(prev_actions.shape)} against {(batch, length)}"
            )

        carry = state or self.initial(batch, embeds.device, embeds.dtype)
        posts, priors = [], []

        for t in range(length):
            first = None if is_first is None else is_first[:, t]
            post, prior = self.observe_step(
                carry, prev_actions[:, t], embeds[:, t], first, generator
            )
            posts.append(post)
            priors.append(prior)
            carry = post

        return stack_states(posts, dim=1), stack_states(priors, dim=1)

    def imagine(
        self,
        state: State,
        actions: Tensor,
        generator: torch.Generator | None = None,
    ) -> State:
        states = []
        carry = state

        for t in range(actions.shape[1]):
            carry = self.imagine_step(carry, actions[:, t], None, generator)
            states.append(carry)

        return stack_states(states, dim=1)

    def probs(self, logits: Tensor) -> Tensor:
        return unimix_probs(logits, self.unimix)


def observe_sequence(
    encoder: Encoder,
    rssm: RSSM,
    observations: Tensor,
    actions: Tensor,
    is_first: Tensor | None = None,
    state: State | None = None,
    generator: torch.Generator | None = None,
) -> tuple[State, State]:
    batch, length = observations.shape[0], observations.shape[1]

    if actions.shape[1] != length - 1:
        raise ValueError(
            f"expected {length - 1} actions for {length} observations, "
            f"got {actions.shape[1]}"
        )

    device = actions.device
    zero = torch.zeros(batch, 1, actions.shape[-1], device=device, dtype=actions.dtype)
    # position t consumes action t-1; position 0 has none and is a reset
    prev_actions = torch.cat([zero, actions], dim=1)

    if is_first is None:
        is_first = torch.zeros(batch, length, dtype=torch.bool, device=device)
        is_first[:, 0] = True

    embeds = encoder(observations)
    return rssm.observe(embeds, prev_actions, is_first, state, generator)


def observe_replay_batch(
    encoder: Encoder,
    rssm: RSSM,
    batch: SequenceBatch,
    device=None,
    generator: torch.Generator | None = None,
) -> tuple[State, State]:
    device = device or next(rssm.parameters()).device
    observations = torch.from_numpy(np.ascontiguousarray(batch.observations)).to(device)
    actions = torch.from_numpy(np.ascontiguousarray(batch.actions)).to(
        device=device, dtype=torch.float32
    )
    return observe_sequence(encoder, rssm, observations, actions, generator=generator)
