from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from .rssm import State, stack_states


class ActionProvider(Protocol):
    """Chooses an action from a latent state alone. No observation is reachable from here."""

    def __call__(self, state: State) -> Tensor:
        ...


class RandomActionProvider:
    def __init__(
        self,
        action_dim: int,
        seed: int = 0,
        low: float = -1.0,
        high: float = 1.0,
    ) -> None:
        if high <= low:
            raise ValueError(f"action bounds must satisfy low < high, got [{low}, {high}]")

        self.action_dim = int(action_dim)
        self.low = float(low)
        self.high = float(high)
        self.seed = int(seed)
        self._generator: torch.Generator | None = None

    def reseed(self, seed: int | None = None) -> "RandomActionProvider":
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
        deter = state.deter
        unit = torch.rand(
            deter.shape[0],
            self.action_dim,
            generator=self._stream(deter.device),
            device=deter.device,
            dtype=deter.dtype,
        )
        return self.low + (self.high - self.low) * unit


class SequenceActionProvider:
    """Replays a fixed (N, H, A) action tensor, which makes the M5 open-loop rollout a provider."""

    def __init__(self, actions: Tensor) -> None:
        if actions.dim() != 3:
            raise ValueError(f"actions must be (N, H, A), got {tuple(actions.shape)}")

        self.actions = actions
        self.step = 0

    def reset(self) -> "SequenceActionProvider":
        self.step = 0
        return self

    def __call__(self, state: State) -> Tensor:
        if self.step >= self.actions.shape[1]:
            raise IndexError(
                f"provider exhausted after {self.actions.shape[1]} actions"
            )

        action = self.actions[:, self.step]
        self.step += 1
        return action


@dataclass(slots=True)
class Imagination:
    states: State
    actions: Tensor
    reward: Tensor
    cont: Tensor
    cont_start: Tensor
    weight: Tensor
    image: Tensor | None = None

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])

    @property
    def rollouts(self) -> int:
        return int(self.actions.shape[0])

    @property
    def feat(self) -> Tensor:
        return self.states.feat


def valid_start_mask(loss_mask: Tensor, is_terminal: Tensor) -> Tensor:
    """Loss-bearing and non-terminal: burn-in positions and terminal states are not rollout starts."""
    return loss_mask.to(torch.bool) & ~is_terminal.to(torch.bool)


def select_start_states(
    post: State,
    loss_mask: Tensor,
    is_terminal: Tensor,
) -> tuple[State, Tensor]:
    length = post.deter.shape[1]

    if loss_mask.shape[1] != length - 1 or is_terminal.shape[1] != length - 1:
        raise ValueError(
            f"expected {length - 1} transition masks for {length} posterior states, got "
            f"{loss_mask.shape[1]} and {is_terminal.shape[1]}"
        )

    # state j+1 is the one transition j produced; state 0 is the reset, which has no reward target
    keep = valid_start_mask(loss_mask, is_terminal)
    return post[:, 1:][keep], keep.reshape(-1).nonzero(as_tuple=True)[0]


def trajectory_weight(cont: Tensor, disc: float) -> Tensor:
    """spec 5.4: cumprod(disc*con)/disc over the H+1 positions, so weight[:, 0] == con[:, 0]."""
    return torch.cumprod(disc * cont, dim=1) / disc


def imagine_trajectory(
    model,
    start: State,
    policy: ActionProvider,
    horizon: int,
    generator: torch.Generator | None = None,
    decode: bool = False,
) -> Imagination:
    """Prior-only rollout from a replay posterior start. No observation is an argument."""
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")

    action_dim = model.rssm.action_dim
    rollouts = start.deter.shape[0]

    if start.deter.dim() != 2:
        raise ValueError(
            f"start must be a flat batch of states, got deter {tuple(start.deter.shape)}"
        )

    # stop-gradient over the start state and the whole imagined trajectory: the actor gets no
    # pathwise gradient through the dynamics, which is why REINFORCE is required (spec 5.9)
    with torch.no_grad():
        carry = start.detach()
        states: list[State] = [carry]
        actions: list[Tensor] = []

        for _ in range(horizon):
            action = policy(carry)

            if tuple(action.shape) != (rollouts, action_dim):
                raise ValueError(
                    f"action provider returned {tuple(action.shape)}, expected "
                    f"{(rollouts, action_dim)}"
                )

            action = action.to(dtype=carry.deter.dtype)
            carry = model.rssm.imagine_step(carry, action, None, generator)
            states.append(carry)
            actions.append(action)

        trajectory = stack_states(states, dim=1)
        # inspection only: the decoder is never part of the behaviour-training path
        image = model.decoder(trajectory.deter, trajectory.stoch) if decode else None

    feat = trajectory.feat

    # the heads run outside the no-grad block so their own parameters stay differentiable
    reward = model.reward.predict(feat[:, 1:])
    cont = model.cont.predict(feat)
    disc = 1.0 if model.contdisc else 1.0 - 1.0 / model.horizon

    return Imagination(
        states=trajectory,
        actions=torch.stack(actions, dim=1),
        reward=reward,
        cont=cont[:, 1:],
        cont_start=cont[:, :1],
        weight=trajectory_weight(cont, disc),
        image=image,
    )
