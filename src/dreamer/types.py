from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Image = NDArray[np.uint8]
FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class EnvStep:
    next_observation: Image
    action: FloatArray
    reward: np.float32
    is_last: bool
    is_terminal: bool
    discount: np.float32


@dataclass(frozen=True, slots=True)
class Transition:
    observation: Image
    action: FloatArray
    reward: np.float32
    next_observation: Image
    is_last: bool
    is_terminal: bool
    discount: np.float32


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    observations: Image
    actions: FloatArray
    rewards: FloatArray
    is_last: BoolArray
    is_terminal: BoolArray
    discounts: FloatArray
    loss_mask: BoolArray

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])

    @property
    def transition_length(self) -> int:
        return int(self.actions.shape[1])

    @property
    def train_positions(self) -> int:
        return int(self.loss_mask.sum())


@dataclass(slots=True)
class StepCounters:
    env_step: int = 0
    agent_step: int = 0
    physics_substep: int = 0
    replay_transition: int = 0
    gradient_step: int = 0
    train_position: int = 0
    episode: int = 0
    imagination_start: int = 0
    imagined_transition: int = 0
    eval_step: int = 0
    eval_seconds: float = 0.0

    def record_transition(self, physics_substeps: int) -> None:
        self.env_step += 1
        self.agent_step += 1
        self.physics_substep += physics_substeps
        self.replay_transition += 1

    def record_update(self, train_positions: int, starts: int, horizon: int) -> None:
        self.gradient_step += 1
        self.train_position += int(train_positions)
        self.imagination_start += int(starts)
        self.imagined_transition += int(starts) * int(horizon)

    def record_evaluation(self, steps: int, seconds: float) -> None:
        self.eval_step += int(steps)
        self.eval_seconds += float(seconds)

    def state_dict(self) -> dict[str, float]:
        return {f: getattr(self, f) for f in self.__slots__}

    def load_state_dict(self, state: dict[str, float]) -> None:
        for field_name in self.__slots__:
            if field_name in state:
                setattr(self, field_name, state[field_name])
