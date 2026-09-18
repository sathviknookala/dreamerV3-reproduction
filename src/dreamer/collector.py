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

    def reset(self) -> np.ndarray:
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

        if step.is_last:
            self._observation = self.env.reset()
        else:
            self._observation = step.next_observation

        return transition

    def collect(self, num_steps: int) -> StepCounters:
        if num_steps < 0:
            raise ValueError("num_steps cannot be negative")

        for _ in range(num_steps):
            self.collect_step()

        return self.counters
