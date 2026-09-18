from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from dm_control import suite

from .types import EnvStep, Image


class DMCEnv:
    SUPPORTED_TASKS = {
        ("walker", "walk"),
        ("cartpole", "swingup"),
    }

    def __init__(
        self,
        domain: str,
        task: str,
        seed: int,
        size: tuple[int, int] = (64, 64),
        camera_id: int = 0,
    ) -> None:
        if (domain, task) not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {domain}/{task}")

        self.domain = domain
        self.task = task
        self.seed = int(seed)
        self.height, self.width = size
        self.camera_id = int(camera_id)

        task_random = np.random.RandomState(self.seed)

        self._env = suite.load(
            domain_name=domain,
            task_name=task,
            task_kwargs={"random": task_random},
        )

        self._action_spec = self._env.action_spec()
        self._action_low = np.asarray(
            self._action_spec.minimum,
            dtype=np.float32,
        )
        self._action_high = np.asarray(
            self._action_spec.maximum,
            dtype=np.float32,
        )

        if not np.isfinite(self._action_low).all():
            raise ValueError("Action lower bound must be finite")

        if not np.isfinite(self._action_high).all():
            raise ValueError("Action upper bound must be finite")

        control_dt = float(self._env.control_timestep())
        physics_dt = float(self._env.physics.timestep())
        ratio = control_dt / physics_dt
        rounded = round(ratio)

        if not np.isclose(ratio, rounded):
            raise ValueError(
                f"Non-integral physics/control timestep ratio: {ratio}"
            )

        self.control_timestep = control_dt
        self.physics_timestep = physics_dt
        self.physics_substeps = int(rounded)

        self._needs_reset = True

    @property
    def action_shape(self) -> tuple[int, ...]:
        return tuple(self._action_spec.shape)

    @property
    def action_low(self) -> np.ndarray:
        return self._action_low.copy()

    @property
    def action_high(self) -> np.ndarray:
        return self._action_high.copy()

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return self.height, self.width, 3

    def reset(self) -> Image:
        self._env.reset()
        self._needs_reset = False
        return self._render()

    def step(self, action: np.ndarray) -> EnvStep:
        if self._needs_reset:
            raise RuntimeError("Environment must be reset before stepping")

        action = np.asarray(action, dtype=np.float32)

        if action.shape != self.action_shape:
            raise ValueError(
                f"Expected action shape {self.action_shape}, got {action.shape}"
            )

        if not np.isfinite(action).all():
            raise ValueError("Action contains non-finite values")

        clipped = np.clip(
            action,
            self._action_low,
            self._action_high,
        ).astype(np.float32, copy=False)

        env_action = clipped.astype(
            self._action_spec.dtype,
            copy=False,
        )

        time_step = self._env.step(env_action)

        if time_step.discount is None:
            raise RuntimeError("Non-reset timestep has no discount")

        discount = float(time_step.discount)

        if discount not in (0.0, 1.0):
            raise ValueError(
                f"Expected binary DMControl discount, got {discount}"
            )

        is_last = bool(time_step.last())
        is_terminal = bool(discount == 0.0)

        reward = 0.0 if time_step.reward is None else float(time_step.reward)
        next_observation = self._render()

        if is_last:
            self._needs_reset = True

        return EnvStep(
            next_observation=next_observation,
            action=clipped.copy(),
            reward=np.float32(reward),
            is_last=is_last,
            is_terminal=is_terminal,
            discount=np.float32(discount),
        )

    def _render(self) -> Image:
        frame = self._env.physics.render(
            self.height,
            self.width,
            camera_id=self.camera_id,
        )

        frame = np.asarray(frame)

        if frame.shape != self.observation_shape:
            raise ValueError(
                f"Expected frame shape {self.observation_shape}, "
                f"got {frame.shape}"
            )

        if frame.dtype != np.uint8:
            raise TypeError(
                f"Expected uint8 frame, got {frame.dtype}"
            )

        return np.ascontiguousarray(frame)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if close is not None:
            close()
