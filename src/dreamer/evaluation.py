from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .agent import Agent
from .config import RunConfig, TASKS
from .env import DMCEnv
from .viz import write_video


@dataclass
class EpisodeResult:
    seed: int
    episode_return: float
    length: int


@dataclass
class EvaluationResult:
    env_step: int
    episodes: list[EpisodeResult]
    seconds: float
    video: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def returns(self) -> list[float]:
        return [e.episode_return for e in self.episodes]

    @property
    def lengths(self) -> list[int]:
        return [e.length for e in self.episodes]

    @property
    def steps(self) -> int:
        return int(sum(self.lengths))

    @property
    def mean(self) -> float:
        return float(np.mean(self.returns)) if self.episodes else float("nan")

    @property
    def stddev(self) -> float:
        # population sd over the evaluation episodes, which are not independent training runs
        return float(np.std(self.returns)) if self.episodes else float("nan")

    def summary(self) -> dict:
        return {
            "env_step": self.env_step,
            "seeds": [e.seed for e in self.episodes],
            "returns": self.returns,
            "lengths": self.lengths,
            "return_mean": self.mean,
            "return_std": self.stddev,
            "eval_seconds": self.seconds,
            "eval_steps": self.steps,
            "video": self.video,
        }


def _agent_fingerprint(agent: Agent) -> dict:
    """Everything an evaluation is forbidden to touch, in a form a test can compare bitwise."""
    return {
        "parameters": [p.detach().clone() for p in agent.parameters()],
        "buffers": [b.detach().clone() for b in agent.buffers()],
        "generators": {
            name: generator.get_state().clone()
            for name, generator in agent.generators.items()
        },
        "provider": agent.provider.state_dict(),
    }


def _fingerprints_match(before: dict, after: dict) -> bool:
    if not all(
        torch.equal(a, b)
        for key in ("parameters", "buffers")
        for a, b in zip(before[key], after[key])
    ):
        return False

    if not all(
        torch.equal(before["generators"][name], after["generators"][name])
        for name in before["generators"]
    ):
        return False

    left, right = before["provider"]["generator"], after["provider"]["generator"]

    if left is None or right is None:
        return left is right

    return bool(torch.equal(left, right))


@torch.no_grad()
def evaluate_episode(
    agent: Agent,
    domain: str,
    task: str,
    seed: int,
    size: tuple[int, int] = (64, 64),
    max_steps: int = 10_000,
    record: bool = False,
) -> tuple[EpisodeResult, np.ndarray | None]:
    """A separate simulator and an isolated generator; the mean is the action (spec 7.8)."""
    env = DMCEnv(domain, task, seed=seed, size=size)
    policy = agent.evaluation_policy(seed)
    frames: list[np.ndarray] = []

    try:
        observation = env.reset()
        total, length = 0.0, 0

        if record:
            frames.append(observation.copy())

        for _ in range(max_steps):
            action = policy(observation)
            step = env.step(action)
            policy.observe_executed(step.action)

            total += float(step.reward)
            length += 1

            if record:
                frames.append(step.next_observation.copy())

            if step.is_last:
                break

            observation = step.next_observation
    finally:
        env.close()

    video = np.stack(frames, axis=0) if record and frames else None
    return EpisodeResult(seed=seed, episode_return=total, length=length), video


def evaluate_agent(
    agent: Agent,
    domain: str,
    task: str,
    seeds,
    size: tuple[int, int] = (64, 64),
    env_step: int = 0,
    video_path: Path | str | None = None,
    verify_isolation: bool = True,
    max_steps: int = 10_000,
) -> EvaluationResult:
    """Diagnostic evaluation. Touches no parameter, no replay, no counter, no training stream."""
    seeds = tuple(int(s) for s in seeds)
    training_mode = agent.training
    agent.eval()

    before = _agent_fingerprint(agent) if verify_isolation else None
    start = time.perf_counter()
    episodes: list[EpisodeResult] = []
    video: str | None = None

    try:
        for position, seed in enumerate(seeds):
            # the video comes from the FIRST evaluation seed, so the same episode is comparable
            record = video_path is not None and position == 0
            result, frames = evaluate_episode(
                agent, domain, task, seed, size=size, max_steps=max_steps, record=record
            )
            episodes.append(result)

            if frames is not None and video_path is not None:
                video = write_video(video_path, frames)
    finally:
        agent.train(training_mode)

    seconds = time.perf_counter() - start

    if before is not None and not _fingerprints_match(before, _agent_fingerprint(agent)):
        raise RuntimeError(
            "evaluation mutated training state: parameters, buffers or a training generator moved"
        )

    return EvaluationResult(env_step=env_step, episodes=episodes, seconds=seconds, video=video)


class PeriodicEvaluator:
    """The callable `OnlineTrainer` invokes every `eval_every` control steps."""

    def __init__(self, config: RunConfig, out_dir: Path | str) -> None:
        self.config = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.domain, self.task = TASKS[config.task]
        self.results: list[EvaluationResult] = []

    def __call__(self, agent: Agent, env_step: int) -> EvaluationResult:
        video_path = (
            self.out_dir / f"eval-{env_step:08d}.mp4" if self.config.eval_video else None
        )
        result = evaluate_agent(
            agent,
            self.domain,
            self.task,
            self.config.eval_seeds,
            size=(self.config.image_size, self.config.image_size),
            env_step=env_step,
            video_path=video_path,
        )
        self.results.append(result)
        self._append(result)
        return result

    def _append(self, result: EvaluationResult) -> None:
        path = self.out_dir / "evaluations.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(result.summary()) + "\n")
