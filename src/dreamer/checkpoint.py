from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .agent import Agent
from .config import RunConfig

FORMAT_VERSION = 1


def atomic_save(payload: dict, path: Path | str) -> Path:
    """A partial write must never replace a good checkpoint, so rename is the commit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")

    with temporary.open("wb") as handle:
        # protocol 2 (torch's default) latin1-encodes bytes and inflates a uint8 replay by ~1.5x
        torch.save(payload, handle, pickle_protocol=5)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary, path)
    return path


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    task: str
    seed: int
    horizon: int
    created: float

    @classmethod
    def build(cls, config: RunConfig, run_id: str | None = None) -> "RunIdentity":
        created = time.time()
        resolved = run_id or f"{config.task}-h{config.horizon}-s{config.seed}"
        return cls(resolved, config.task, config.seed, config.horizon, created)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "seed": self.seed,
            "horizon": self.horizon,
            "created": self.created,
        }


def model_payload(agent: Agent, identity: RunIdentity, counters) -> dict:
    """Compact: weights and identity only. No replay, no optimizer moments."""
    return {
        "format": FORMAT_VERSION,
        "kind": "model",
        "identity": identity.to_dict(),
        "config": agent.config.to_dict(),
        "model": agent.model_state(),
        "counters": counters.state_dict(),
    }


def resume_payload(trainer, identity: RunIdentity) -> dict:
    """Everything a bit-exact continuation needs, including the 6 GB of replay frames."""
    agent = trainer.agent

    return {
        "format": FORMAT_VERSION,
        "kind": "resume",
        "identity": identity.to_dict(),
        "config": agent.config.to_dict(),
        "model": agent.model_state(),
        "generators": agent.generator_state(),
        "trainer": trainer.state_dict(),
        "replay": trainer.replay.state_dict(),
        "env": trainer.env.state_dict(),
    }


def restore_resume(trainer, payload: dict) -> RunIdentity:
    """Load order matters: the env seed and the scheduler remainder are as load-bearing as weights."""
    if int(payload.get("format", 0)) != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {payload.get('format')}")

    if payload["kind"] != "resume":
        raise ValueError(f"{payload['kind']!r} checkpoint carries no replay and cannot resume")

    saved = RunConfig.from_dict(payload["config"])
    current = trainer.config

    for key in ("task", "batch_size", "train_length", "burn_in", "horizon", "train_ratio"):
        if getattr(saved, key) != getattr(current, key):
            raise ValueError(
                f"configuration drift on {key}: checkpoint {getattr(saved, key)} "
                f"against {getattr(current, key)}"
            )

    if payload["replay"]["current"] is not None:
        raise ValueError(
            "this resume checkpoint was taken mid-episode: replay holds a partial episode whose "
            "successor observation the recreated simulator cannot produce"
        )

    trainer.agent.load_model_state(payload["model"])
    trainer.agent.load_generator_state(payload["generators"])
    trainer.replay.load_state_dict(payload["replay"])
    trainer.env.load_state_dict(payload["env"])
    trainer.load_state_dict(payload["trainer"])

    identity = payload["identity"]
    return RunIdentity(
        run_id=identity["run_id"],
        task=identity["task"],
        seed=int(identity["seed"]),
        horizon=int(identity["horizon"]),
        created=float(identity["created"]),
    )


def load_checkpoint(path: Path | str) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


class Checkpointer:
    """Rotating resume checkpoints beside one compact final model checkpoint."""

    def __init__(
        self,
        out_dir: Path | str,
        identity: RunIdentity,
        keep: int = 2,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.keep = max(1, int(keep))
        self.written: list[Path] = []

    def resume_path(self, env_step: int) -> Path:
        return self.out_dir / f"resume-{env_step:08d}.pt"

    @property
    def model_path(self) -> Path:
        return self.out_dir / "model-final.pt"

    def save(self, trainer, final: bool = False, resumable: bool = True) -> Path | None:
        env_step = trainer.counters.env_step
        path = None

        if resumable:
            path = atomic_save(
                resume_payload(trainer, self.identity), self.resume_path(env_step)
            )

            # a final save landing on the same boundary must not enter the list twice, or
            # rotation would evict a live checkpoint to make room for a duplicate name
            if path in self.written:
                self.written.remove(path)

            self.written.append(path)
            self._rotate()

        if final:
            atomic_save(
                model_payload(trainer.agent, self.identity, trainer.counters), self.model_path
            )

        return path

    def save_model(self, agent: Agent, counters, name: str | None = None) -> Path:
        path = self.out_dir / name if name else self.model_path
        return atomic_save(model_payload(agent, self.identity, counters), path)

    def latest_resume(self) -> Path | None:
        candidates = sorted(self.out_dir.glob("resume-*.pt"))
        return candidates[-1] if candidates else None

    def _rotate(self) -> None:
        while len(self.written) > self.keep:
            stale = self.written.pop(0)
            stale.unlink(missing_ok=True)
