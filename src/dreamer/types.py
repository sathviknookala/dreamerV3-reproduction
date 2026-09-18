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

    def record_transition(self, physics_substeps: int) -> None:
        self.env_step += 1
        self.agent_step += 1
        self.physics_substep += physics_substeps
        self.replay_transition += 1
