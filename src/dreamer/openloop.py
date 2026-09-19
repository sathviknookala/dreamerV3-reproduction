from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .replay import ReplayBuffer
from .rssm import State, observe_sequence
from .types import Transition


@dataclass(frozen=True, slots=True)
class Episode:
    episode_id: int
    split: str
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    is_last: np.ndarray
    is_terminal: np.ndarray
    discounts: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])

    @property
    def episode_return(self) -> float:
        return float(self.rewards.sum())

    def digest(self) -> str:
        h = hashlib.sha256()
        for array in (self.observations, self.actions, self.rewards, self.is_terminal):
            h.update(np.ascontiguousarray(array).tobytes())
        return h.hexdigest()


def save_episode(path: Path, episode: Episode) -> None:
    np.savez_compressed(
        path,
        episode_id=np.int64(episode.episode_id),
        split=np.array(episode.split),
        observations=episode.observations,
        actions=episode.actions,
        rewards=episode.rewards,
        is_last=episode.is_last,
        is_terminal=episode.is_terminal,
        discounts=episode.discounts,
    )


def load_episode(path: Path) -> Episode:
    with np.load(path, allow_pickle=False) as data:
        return Episode(
            episode_id=int(data["episode_id"]),
            split=str(data["split"]),
            observations=data["observations"],
            actions=data["actions"],
            rewards=data["rewards"],
            is_last=data["is_last"],
            is_terminal=data["is_terminal"],
            discounts=data["discounts"],
        )


def load_split(data_dir: Path | str, split: str) -> list[Episode]:
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())

    if split not in ("train", "holdout"):
        raise ValueError(f"unknown split: {split}")

    episodes = [
        load_episode(data_dir / entry["file"])
        for entry in manifest["episodes"]
        if entry["split"] == split
    ]

    for episode in episodes:
        # the split label travels with the data, so a mislabelled file is loud
        if episode.split != split:
            raise ValueError(
                f"episode {episode.episode_id} is labelled {episode.split!r} "
                f"but the manifest lists it under {split!r}"
            )

    return episodes


def replay_from_episodes(
    episodes: list[Episode],
    seed: int,
    capacity: int = 500_000,
) -> ReplayBuffer:
    replay = ReplayBuffer(capacity=capacity, seed=seed)

    for episode in episodes:
        for j in range(episode.length):
            replay.add(
                Transition(
                    observation=episode.observations[j],
                    action=episode.actions[j],
                    reward=episode.rewards[j],
                    next_observation=episode.observations[j + 1],
                    is_last=bool(episode.is_last[j]),
                    is_terminal=bool(episode.is_terminal[j]),
                    discount=episode.discounts[j],
                )
            )

    return replay


def training_reward_mean(episodes: list[Episode]) -> float:
    if any(episode.split != "train" for episode in episodes):
        raise ValueError("the constant reward baseline may only see training episodes")

    return float(np.concatenate([e.rewards for e in episodes]).mean())


@dataclass(frozen=True, slots=True)
class Context:
    episode_index: int
    start: int

    def cutoff(self, context_length: int) -> int:
        # absolute observation index of the last posterior state
        return self.start + context_length


def make_contexts(
    episodes: list[Episode],
    context_length: int,
    horizon: int,
    per_episode: int,
) -> list[Context]:
    contexts: list[Context] = []

    for index, episode in enumerate(episodes):
        last_start = episode.length - context_length - horizon

        if last_start < 0:
            raise ValueError(
                f"episode {episode.episode_id} is too short for "
                f"context {context_length} + horizon {horizon}"
            )

        starts = np.linspace(0, last_start, per_episode).round().astype(np.int64)
        contexts.extend(Context(index, int(s)) for s in np.unique(starts))

    return contexts


def target_indices(cutoff: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Distance k lands on state index cutoff+k, whose reward target is rewards[cutoff+k-1]."""
    k = np.arange(1, horizon + 1, dtype=np.int64)
    return cutoff + k - 1, cutoff + k


@dataclass(frozen=True, slots=True)
class ContextBatch:
    context_observations: np.ndarray
    context_actions: np.ndarray
    future_actions: np.ndarray
    target_rewards: np.ndarray
    target_images: np.ndarray
    target_terminal: np.ndarray
    last_observed_reward: np.ndarray
    last_observed_image: np.ndarray


def gather_contexts(
    episodes: list[Episode],
    contexts: list[Context],
    context_length: int,
    horizon: int,
) -> ContextBatch:
    fields: dict[str, list[np.ndarray]] = {k: [] for k in ContextBatch.__slots__}

    for context in contexts:
        episode = episodes[context.episode_index]
        s, c = context.start, context.cutoff(context_length)
        reward_index, observation_index = target_indices(c, horizon)

        fields["context_observations"].append(episode.observations[s : c + 1])
        fields["context_actions"].append(episode.actions[s:c])
        fields["future_actions"].append(episode.actions[c : c + horizon])
        fields["target_rewards"].append(episode.rewards[reward_index])
        fields["target_images"].append(episode.observations[observation_index])
        fields["target_terminal"].append(episode.is_terminal[reward_index])
        # persistence uses context data only: the last reward and frame before the cutoff
        fields["last_observed_reward"].append(episode.rewards[c - 1])
        fields["last_observed_image"].append(episode.observations[c])

    return ContextBatch(**{k: np.stack(v, axis=0) for k, v in fields.items()})


class encoder_tripwire:
    """Records every encoder call inside the block, so a leaked future frame is visible."""

    def __init__(self, model) -> None:
        self.model = model
        self.calls: list[tuple[int, ...]] = []

    def __enter__(self) -> "encoder_tripwire":
        self._original = self.model.encoder.forward

        def wrapped(image, *args, **kwargs):
            self.calls.append(tuple(image.shape))
            return self._original(image, *args, **kwargs)

        self.model.encoder.forward = wrapped
        return self

    def __exit__(self, *exc) -> None:
        self.model.encoder.forward = self._original

    @property
    def frames_seen(self) -> int:
        return sum(int(np.prod(shape[:-3])) for shape in self.calls)


@dataclass(frozen=True, slots=True)
class OpenLoopPrediction:
    reward: Tensor
    cont: Tensor
    image: Tensor | None
    states: State


def open_loop_predict(
    model,
    context_observations: Tensor,
    context_actions: Tensor,
    future_actions: Tensor,
    generator: torch.Generator | None = None,
    decode: bool = False,
) -> OpenLoopPrediction:
    """Posterior over the context prefix, then prior only. No future observation is an argument."""
    if context_observations.shape[1] != context_actions.shape[1] + 1:
        raise ValueError(
            f"expected {context_actions.shape[1] + 1} context observations, "
            f"got {context_observations.shape[1]}"
        )

    post, _ = observe_sequence(
        model.encoder,
        model.rssm,
        context_observations,
        context_actions,
        generator=generator,
    )

    states = model.rssm.imagine(post[:, -1], future_actions, generator)
    feat = states.feat

    return OpenLoopPrediction(
        reward=model.reward.predict(feat),
        cont=model.cont.predict(feat),
        image=model.decoder(states.deter, states.stoch) if decode else None,
        states=states,
    )


@dataclass(slots=True)
class OpenLoopMetrics:
    horizon: int
    contexts: int
    samples: int
    reward_mae: np.ndarray
    reward_rmse: np.ndarray
    reward_mae_sample_sd: np.ndarray
    reward_mae_shuffled: np.ndarray
    cont_mae: np.ndarray
    image_mae: np.ndarray
    baseline_reward_mae_mean: np.ndarray
    baseline_reward_mae_persistence: np.ndarray
    baseline_image_mae_persistence: np.ndarray
    target_reward_sd: np.ndarray

    def as_dict(self) -> dict:
        out: dict = {"horizon": self.horizon, "contexts": self.contexts, "samples": self.samples}
        for field in self.__slots__:
            value = getattr(self, field)
            if isinstance(value, np.ndarray):
                out[field] = [float(x) for x in value]
        return out


def _shuffle_time(actions: Tensor, generator: torch.Generator) -> Tensor:
    order = torch.randperm(actions.shape[1], generator=generator, device=actions.device)
    return actions[:, order]


def evaluate_open_loop(
    model,
    episodes: list[Episode],
    contexts: list[Context],
    context_length: int,
    horizon: int,
    constant_reward: float,
    samples: int,
    seed: int,
    device,
    chunk_contexts: int = 8,
) -> OpenLoopMetrics:
    if any(episode.split == "train" for episode in episodes):
        raise ValueError("open-loop evaluation must run on held-out episodes")

    data = gather_contexts(episodes, contexts, context_length, horizon)
    n = len(contexts)

    sums = {
        key: np.zeros(horizon, dtype=np.float64)
        for key in (
            "reward_abs",
            "reward_sq",
            "reward_abs_shuffled",
            "cont_abs",
            "image_abs",
        )
    }
    per_sample_abs = np.zeros((samples, horizon), dtype=np.float64)
    counts = 0

    generator = torch.Generator(device=device)
    shuffle_generator = torch.Generator(device=device)

    model.eval()

    with torch.no_grad():
        for begin in range(0, n, chunk_contexts):
            end = min(begin + chunk_contexts, n)
            width = end - begin

            def to(array, dtype=None):
                tensor = torch.from_numpy(np.ascontiguousarray(array[begin:end])).to(device)
                return tensor if dtype is None else tensor.to(dtype)

            context_observations = to(data.context_observations)
            context_actions = to(data.context_actions, torch.float32)
            future_actions = to(data.future_actions, torch.float32)
            target_rewards = to(data.target_rewards, torch.float32)
            target_images = to(data.target_images).to(torch.float32) / 255.0
            target_cont = (
                (~to(data.target_terminal, torch.bool)).to(torch.float32)
                * (1.0 - 1.0 / model.horizon if model.contdisc else 1.0)
            )

            for sample in range(samples):
                # a fixed stream per (chunk, sample) makes the whole evaluation reproducible
                generator.manual_seed(seed * 1_000_003 + begin * 1_009 + sample)
                shuffle_generator.manual_seed(seed * 7_919 + begin * 131 + sample)

                prediction = open_loop_predict(
                    model,
                    context_observations,
                    context_actions,
                    future_actions,
                    generator,
                    decode=True,
                )

                abs_error = (prediction.reward - target_rewards).abs()
                sums["reward_abs"] += abs_error.sum(0).double().cpu().numpy()
                sums["reward_sq"] += (
                    (prediction.reward - target_rewards).square().sum(0).double().cpu().numpy()
                )
                sums["cont_abs"] += (prediction.cont - target_cont).abs().sum(0).double().cpu().numpy()
                sums["image_abs"] += (
                    (prediction.image - target_images).abs().mean((-1, -2, -3)).sum(0).double().cpu().numpy()
                )
                per_sample_abs[sample] += abs_error.sum(0).double().cpu().numpy()

                generator.manual_seed(seed * 1_000_003 + begin * 1_009 + sample)
                shuffled = open_loop_predict(
                    model,
                    context_observations,
                    context_actions,
                    _shuffle_time(future_actions, shuffle_generator),
                    generator,
                    decode=False,
                )
                sums["reward_abs_shuffled"] += (
                    (shuffled.reward - target_rewards).abs().sum(0).double().cpu().numpy()
                )

            counts += width

    total = counts * samples
    targets = data.target_rewards.astype(np.float64)

    return OpenLoopMetrics(
        horizon=horizon,
        contexts=n,
        samples=samples,
        reward_mae=sums["reward_abs"] / total,
        reward_rmse=np.sqrt(sums["reward_sq"] / total),
        reward_mae_sample_sd=(per_sample_abs / counts).std(axis=0, ddof=0),
        reward_mae_shuffled=sums["reward_abs_shuffled"] / total,
        cont_mae=sums["cont_abs"] / total,
        image_mae=sums["image_abs"] / total,
        baseline_reward_mae_mean=np.abs(targets - constant_reward).mean(axis=0),
        baseline_reward_mae_persistence=np.abs(
            targets - data.last_observed_reward.astype(np.float64)[:, None]
        ).mean(axis=0),
        baseline_image_mae_persistence=np.abs(
            data.target_images.astype(np.float64) / 255.0
            - data.last_observed_image.astype(np.float64)[:, None] / 255.0
        ).mean(axis=(0, 2, 3, 4)),
        target_reward_sd=targets.std(axis=0),
    )
