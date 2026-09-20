from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .types import SequenceBatch, Transition


@dataclass(frozen=True, slots=True)
class _Episode:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    is_last: np.ndarray
    is_terminal: np.ndarray
    discounts: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


@dataclass(slots=True)
class _EpisodeBuilder:
    observations: list[np.ndarray]
    actions: list[np.ndarray]
    rewards: list[np.float32]
    is_last: list[bool]
    is_terminal: list[bool]
    discounts: list[np.float32]

    @property
    def length(self) -> int:
        return len(self.actions)


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        seed: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)

        self._episodes: deque[_Episode] = deque()
        self._current: _EpisodeBuilder | None = None

        self._completed_transitions = 0
        self._observation_shape: tuple[int, ...] | None = None
        self._action_shape: tuple[int, ...] | None = None

    def __len__(self) -> int:
        current = 0 if self._current is None else self._current.length
        return self._completed_transitions + current

    @property
    def num_complete_episodes(self) -> int:
        return len(self._episodes)

    @property
    def complete_episodes(self) -> tuple[_Episode, ...]:
        return tuple(self._episodes)

    def add(self, transition: Transition) -> None:
        observation = self._validate_image(
            transition.observation,
            "observation",
        )
        next_observation = self._validate_image(
            transition.next_observation,
            "next_observation",
        )
        action = self._validate_action(transition.action)

        reward = float(transition.reward)
        discount = float(transition.discount)

        if not np.isfinite(reward):
            raise ValueError("Reward must be finite")

        if discount not in (0.0, 1.0):
            raise ValueError(
                f"Expected binary discount, got {discount}"
            )

        if transition.is_terminal and not transition.is_last:
            raise ValueError("Terminal transition must also be last")

        if transition.is_terminal != (discount == 0.0):
            raise ValueError(
                "is_terminal must exactly match discount == 0"
            )

        self._validate_schema(observation, action)

        if self._current is None:
            self._current = _EpisodeBuilder(
                observations=[observation.copy()],
                actions=[],
                rewards=[],
                is_last=[],
                is_terminal=[],
                discounts=[],
            )
        else:
            expected = self._current.observations[-1]

            if not np.array_equal(expected, observation):
                raise ValueError(
                    "Transition observation does not match the "
                    "previous transition's next_observation"
                )

        self._current.actions.append(action.copy())
        self._current.rewards.append(np.float32(reward))
        self._current.is_last.append(bool(transition.is_last))
        self._current.is_terminal.append(bool(transition.is_terminal))
        self._current.discounts.append(np.float32(discount))
        self._current.observations.append(next_observation.copy())

        if transition.is_last:
            self._finish_episode()

        self._enforce_capacity()

    def sample_sequences(
        self,
        batch_size: int,
        train_length: int = 64,
        burn_in: int = 5,
    ) -> SequenceBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if train_length <= 0:
            raise ValueError("train_length must be positive")

        if burn_in < 0:
            raise ValueError("burn_in cannot be negative")

        sequence_length = burn_in + train_length

        eligible: list[_Episode] = []
        window_counts: list[int] = []

        for episode in self._episodes:
            windows = episode.length - sequence_length + 1

            if windows > 0:
                eligible.append(episode)
                window_counts.append(windows)

        if not eligible:
            raise RuntimeError(
                f"No complete episode contains {sequence_length} "
                "contiguous transitions"
            )

        cumulative = np.cumsum(
            np.asarray(window_counts, dtype=np.int64)
        )
        total_windows = int(cumulative[-1])

        observations = []
        actions = []
        rewards = []
        is_last = []
        is_terminal = []
        discounts = []

        draws = self._rng.integers(
            0,
            total_windows,
            size=batch_size,
        )

        for draw in draws:
            episode_index = int(
                np.searchsorted(cumulative, draw, side="right")
            )

            previous = (
                0
                if episode_index == 0
                else int(cumulative[episode_index - 1])
            )

            start = int(draw) - previous
            stop = start + sequence_length
            episode = eligible[episode_index]

            episode_is_last = episode.is_last[start:stop]

            if episode_is_last[:-1].any():
                raise RuntimeError(
                    "Sample crossed an episode boundary"
                )

            observations.append(
                episode.observations[start : stop + 1]
            )
            actions.append(
                episode.actions[start:stop]
            )
            rewards.append(
                episode.rewards[start:stop]
            )
            is_last.append(
                episode_is_last
            )
            is_terminal.append(
                episode.is_terminal[start:stop]
            )
            discounts.append(
                episode.discounts[start:stop]
            )

        loss_mask = np.ones(
            (batch_size, sequence_length),
            dtype=np.bool_,
        )

        if burn_in:
            loss_mask[:, :burn_in] = False

        return SequenceBatch(
            observations=np.stack(observations, axis=0),
            actions=np.stack(actions, axis=0),
            rewards=np.stack(rewards, axis=0),
            is_last=np.stack(is_last, axis=0),
            is_terminal=np.stack(is_terminal, axis=0),
            discounts=np.stack(discounts, axis=0),
            loss_mask=loss_mask,
        )

    def can_sample(self, sequence_length: int) -> bool:
        """No complete episode is long enough until warm-up has closed one (spec 7.5)."""
        return any(e.length >= sequence_length for e in self._episodes)

    def state_dict(self) -> dict:
        """Episode arrays, the partial episode, and the sampler stream -- resume must not redraw."""
        current = self._current

        return {
            "capacity": self.capacity,
            "rng": self._rng.bit_generator.state,
            "observation_shape": self._observation_shape,
            "action_shape": self._action_shape,
            "episodes": [
                {
                    "observations": episode.observations,
                    "actions": episode.actions,
                    "rewards": episode.rewards,
                    "is_last": episode.is_last,
                    "is_terminal": episode.is_terminal,
                    "discounts": episode.discounts,
                }
                for episode in self._episodes
            ],
            "current": None
            if current is None
            else {
                "observations": list(current.observations),
                "actions": list(current.actions),
                "rewards": list(current.rewards),
                "is_last": list(current.is_last),
                "is_terminal": list(current.is_terminal),
                "discounts": list(current.discounts),
            },
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError(
                f"checkpoint capacity {state['capacity']} does not match {self.capacity}"
            )

        self._rng.bit_generator.state = state["rng"]
        self._observation_shape = (
            None if state["observation_shape"] is None else tuple(state["observation_shape"])
        )
        self._action_shape = (
            None if state["action_shape"] is None else tuple(state["action_shape"])
        )

        self._episodes = deque(
            _Episode(
                observations=np.ascontiguousarray(entry["observations"]),
                actions=np.ascontiguousarray(entry["actions"]),
                rewards=np.ascontiguousarray(entry["rewards"]),
                is_last=np.ascontiguousarray(entry["is_last"]),
                is_terminal=np.ascontiguousarray(entry["is_terminal"]),
                discounts=np.ascontiguousarray(entry["discounts"]),
            )
            for entry in state["episodes"]
        )
        self._completed_transitions = sum(e.length for e in self._episodes)

        current = state["current"]
        self._current = (
            None
            if current is None
            else _EpisodeBuilder(
                observations=[np.ascontiguousarray(o) for o in current["observations"]],
                actions=[np.ascontiguousarray(a) for a in current["actions"]],
                rewards=[np.float32(r) for r in current["rewards"]],
                is_last=[bool(v) for v in current["is_last"]],
                is_terminal=[bool(v) for v in current["is_terminal"]],
                discounts=[np.float32(d) for d in current["discounts"]],
            )
        )

    def stats(self) -> dict[str, int]:
        current = 0 if self._current is None else self._current.length

        return {
            "stored_transitions": len(self),
            "complete_episodes": len(self._episodes),
            "current_episode_transitions": current,
            "capacity": self.capacity,
        }

    def _finish_episode(self) -> None:
        if self._current is None:
            raise RuntimeError("No episode is being built")

        builder = self._current

        if not builder.is_last[-1]:
            raise RuntimeError(
                "Cannot finalize an episode without a final boundary"
            )

        if any(builder.is_last[:-1]):
            raise RuntimeError(
                "Episode contains an internal boundary"
            )

        episode = _Episode(
            observations=np.stack(
                builder.observations,
                axis=0,
            ),
            actions=np.stack(
                builder.actions,
                axis=0,
            ),
            rewards=np.asarray(
                builder.rewards,
                dtype=np.float32,
            ),
            is_last=np.asarray(
                builder.is_last,
                dtype=np.bool_,
            ),
            is_terminal=np.asarray(
                builder.is_terminal,
                dtype=np.bool_,
            ),
            discounts=np.asarray(
                builder.discounts,
                dtype=np.float32,
            ),
        )

        if episode.length > self.capacity:
            raise RuntimeError(
                "Single episode exceeds replay capacity"
            )

        self._episodes.append(episode)
        self._completed_transitions += episode.length
        self._current = None

    def _enforce_capacity(self) -> None:
        while len(self) > self.capacity and self._episodes:
            episode = self._episodes.popleft()
            self._completed_transitions -= episode.length

        if len(self) > self.capacity:
            raise RuntimeError(
                "Current incomplete episode exceeds replay capacity"
            )

    def _validate_schema(
        self,
        observation: np.ndarray,
        action: np.ndarray,
    ) -> None:
        if self._observation_shape is None:
            self._observation_shape = observation.shape
        elif observation.shape != self._observation_shape:
            raise ValueError(
                f"Observation shape changed from "
                f"{self._observation_shape} to {observation.shape}"
            )

        if self._action_shape is None:
            self._action_shape = action.shape
        elif action.shape != self._action_shape:
            raise ValueError(
                f"Action shape changed from "
                f"{self._action_shape} to {action.shape}"
            )

    @staticmethod
    def _validate_image(
        image: np.ndarray,
        name: str,
    ) -> np.ndarray:
        image = np.asarray(image)

        if image.dtype != np.uint8:
            raise TypeError(
                f"{name} must be uint8, got {image.dtype}"
            )

        if image.ndim != 3:
            raise ValueError(
                f"{name} must be HWC rank 3, got {image.shape}"
            )

        return np.ascontiguousarray(image)

    @staticmethod
    def _validate_action(action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)

        if action.ndim != 1:
            raise ValueError(
                f"Action must be rank 1, got {action.shape}"
            )

        if not np.isfinite(action).all():
            raise ValueError("Action must be finite")

        return np.ascontiguousarray(action)
