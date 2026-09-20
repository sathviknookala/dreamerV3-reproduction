from __future__ import annotations

from typing import Protocol

import numpy as np

from .replay import ReplayBuffer
from .types import EnvStep, StepCounters, Transition


class Environment(Protocol):
    physics_substeps: int

    def reset(self) -> np.ndarray:
        ...

    def step(self, action: np.ndarray) -> EnvStep:
        ...


class Policy(Protocol):
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        ...


class StatefulPolicy(Policy, Protocol):
    """A policy carrying recurrent state needs the boundary and the EXECUTED action back."""

    def reset(self) -> None:
        ...

    def observe_executed(self, action: np.ndarray) -> None:
        ...


class UniformRandomPolicy:
    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        seed: int,
    ) -> None:
        self._low = np.asarray(low, dtype=np.float32)
        self._high = np.asarray(high, dtype=np.float32)

        if self._low.shape != self._high.shape:
            raise ValueError("Action bounds must have matching shapes")

        if not np.isfinite(self._low).all():
            raise ValueError("Action lower bounds must be finite")

        if not np.isfinite(self._high).all():
            raise ValueError("Action upper bounds must be finite")

        self._rng = np.random.default_rng(seed)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        del observation

        return self._rng.uniform(
            self._low,
            self._high,
        ).astype(np.float32)


class Collector:
    def __init__(
        self,
        env: Environment,
        replay: ReplayBuffer,
        policy: Policy,
    ) -> None:
        self.env = env
        self.replay = replay
        self.policy = policy

        self.counters = StepCounters()
        self._observation: np.ndarray | None = None
        # the random and sequence policies of M1-M8 are stateless and define neither hook
        self._reset_policy = getattr(policy, "reset", None)
        self._feed_policy = getattr(policy, "observe_executed", None)

    def set_policy(self, policy: Policy) -> None:
        """Swapping the warm-up policy for the learned one re-resolves the lifecycle hooks."""
        self.policy = policy
        self._reset_policy = getattr(policy, "reset", None)
        self._feed_policy = getattr(policy, "observe_executed", None)

        if self._reset_policy is not None:
            self._reset_policy()

    def reset(self) -> np.ndarray:
        if self._reset_policy is not None:
            self._reset_policy()

        self._observation = self.env.reset()
        return self._observation

    def collect_step(self) -> Transition:
        if self._observation is None:
            self.reset()

        if self._observation is None:
            raise RuntimeError("Environment reset returned no observation")

        observation = self._observation
        requested_action = self.policy(observation)
        step = self.env.step(requested_action)

        transition = Transition(
            observation=observation.copy(),
            action=step.action.copy(),
            reward=step.reward,
            next_observation=step.next_observation.copy(),
            is_last=step.is_last,
            is_terminal=step.is_terminal,
            discount=step.discount,
        )

        self.replay.add(transition)
        self.counters.record_transition(
            self.env.physics_substeps
        )

        # the EXECUTED action, not the requested one: the recurrence must see what the
        # simulator actually integrated or the latent state diverges from the replayed batch
        if self._feed_policy is not None:
            self._feed_policy(step.action)

        if step.is_last:
            self.counters.episode += 1
            # a time limit resets the recurrence exactly as a termination does
            self._observation = self.reset()
        else:
            self._observation = step.next_observation

        return transition

    def collect(self, num_steps: int) -> StepCounters:
        if num_steps < 0:
            raise ValueError("num_steps cannot be negative")

        for _ in range(num_steps):
            self.collect_step()

        return self.counters
