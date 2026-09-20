from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

TASKS = {
    "walker": ("walker", "walk"),
    "cartpole": ("cartpole", "swingup"),
}

TASK_BUDGETS = {
    "walker": 1_000_000,
    "cartpole": 500_000,
}

# one offset per stream; the mix is coprime with the offsets so two streams never collide
STREAM_OFFSETS = {
    "init": 0,
    "env": 1,
    "collect": 2,
    "replay": 3,
    "imagine": 4,
    "provider": 5,
    "eval": 6,
    "eval_env": 7,
}


def stream_seed(base: int, name: str) -> int:
    if name not in STREAM_OFFSETS:
        raise KeyError(f"unknown stream {name!r}; known streams are {sorted(STREAM_OFFSETS)}")

    return (int(base) * 1_000_003 + STREAM_OFFSETS[name] * 7_919) % (2**31 - 1)


def episode_seed(base: int, episode_index: int) -> int:
    """The simulator seed for one episode. Resume recreates the simulator from this."""
    return (stream_seed(base, "env") + 10_007 * int(episode_index)) % (2**31 - 1)


@dataclass
class RunConfig:
    task: str = "walker"
    seed: int = 100
    budget: int = 0  # 0 resolves to TASK_BUDGETS[task]

    batch_size: int = 16
    train_length: int = 64
    burn_in: int = 5
    horizon: int = 15

    replay_capacity: int = 500_000
    action_repeat: int = 1
    warmup_transitions: int = 5_000
    train_ratio: int = 64

    lr: float = 4e-5
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-20
    agc: float = 0.3
    lr_warmup: int = 1000

    eval_every: int = 25_000
    eval_episodes: int = 5
    eval_seed_base: int = 2000
    eval_video: bool = True

    checkpoint_every: int = 25_000
    keep_resume_checkpoints: int = 2
    log_every: int = 50

    recreate_env_per_episode: bool = True
    image_size: int = 64

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}; known tasks are {sorted(TASKS)}")

        if self.budget <= 0:
            self.budget = TASK_BUDGETS[self.task]

        if self.warmup_transitions < 0:
            raise ValueError("warmup_transitions cannot be negative")

        if self.train_ratio <= 0:
            raise ValueError("train_ratio must be positive")

        if self.batch_size <= 0 or self.train_length <= 0 or self.burn_in < 0:
            raise ValueError("batch_size and train_length must be positive, burn_in non-negative")

        if self.horizon < 1:
            raise ValueError("horizon must be positive")

    @property
    def domain_task(self) -> tuple[str, str]:
        return TASKS[self.task]

    @property
    def positions_per_update(self) -> int:
        """B*T loss-bearing positions; the burn-in prefix is recomputed, never loss-bearing."""
        return self.batch_size * self.train_length

    @property
    def sequence_length(self) -> int:
        return self.burn_in + self.train_length

    @property
    def eval_seeds(self) -> tuple[int, ...]:
        return tuple(self.eval_seed_base + i for i in range(self.eval_episodes))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "RunConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known

        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")

        return cls(**values)

    @classmethod
    def load(cls, path: Path | str) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
