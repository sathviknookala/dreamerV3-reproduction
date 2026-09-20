"""M9 qualification controls: four policies per task on matched initial conditions.

Evaluation only. No optimizer, no replay, no checkpoint write, no configuration change. The trained
agent's trajectories are kept for the open-loop diagnostic and never enter a training buffer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from dreamer import UniformRandomPolicy
from dreamer.agent import Agent, LatentPolicy
from dreamer.checkpoint import load_checkpoint
from dreamer.config import RunConfig, TASKS
from dreamer.env import DMCEnv
from dreamer.evaluation import _agent_fingerprint, _fingerprints_match
from dreamer.openloop import (
    Episode,
    evaluate_open_loop,
    gather_contexts,
    make_contexts,
    open_loop_predict,
    save_episode,
)
from dreamer.viz import paired_filmstrip, write_png, write_video

CONDITIONS = ("trained", "untrained", "zero", "random")

CONTEXT_LENGTH = 5
OPENLOOP_HORIZON = 30
OPENLOOP_SAMPLES = 8
CONTEXTS_PER_EPISODE = 4
FILMSTRIP_DISTANCES = (1, 5, 10, 15, 20, 25, 30)


def control_policy_seed(condition: str, env_seed: int) -> int:
    """Independent of the simulator seed, so matched initial conditions do not share a stream."""
    digest = hashlib.sha256(f"m9-controls|{condition}|{env_seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_trained_agent(path: Path, device: torch.device) -> tuple[Agent, RunConfig, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; this runner evaluates a trained checkpoint and will not "
            "substitute a freshly initialized model"
        )

    payload = load_checkpoint(path)

    if payload.get("kind") != "model":
        raise ValueError(f"{path} is a {payload.get('kind')!r} checkpoint, not a model checkpoint")

    config = RunConfig.from_dict(payload["config"])
    action_dim = int(payload["model"]["action_dim"])
    agent = Agent(action_dim=action_dim, config=config, device=device)
    agent.load_model_state(payload["model"])
    agent.eval()

    meta = {
        "path": str(path),
        "sha256": file_sha256(path),
        "identity": payload["identity"],
        "counters": payload.get("counters", {}),
        "action_dim": action_dim,
    }
    return agent, config, meta


class ZeroPolicy:
    def __init__(self, action_shape: tuple[int, ...]) -> None:
        self.action_shape = action_shape

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return np.zeros(self.action_shape, dtype=np.float32)


def build_policy(condition: str, agent: Agent, untrained: Agent, env: DMCEnv, seed: int):
    if condition == "trained":
        return agent.evaluation_policy(seed)
    if condition == "untrained":
        return untrained.evaluation_policy(seed)
    if condition == "zero":
        return ZeroPolicy(env.action_shape)
    if condition == "random":
        return UniformRandomPolicy(env.action_low, env.action_high, seed=seed)
    raise ValueError(f"unknown condition {condition!r}")


@torch.no_grad()
def run_episode(
    condition: str,
    agent: Agent,
    untrained: Agent,
    domain: str,
    task: str,
    env_seed: int,
    policy_seed: int,
    size: tuple[int, int],
    max_steps: int,
    keep_trajectory: bool,
    record: bool,
) -> dict:
    env = DMCEnv(domain, task, seed=env_seed, size=size)
    policy = build_policy(condition, agent, untrained, env, policy_seed)
    latent = isinstance(policy, LatentPolicy)

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    is_last: list[bool] = []
    is_terminal: list[bool] = []
    discounts: list[float] = []
    frames: list[np.ndarray] = []

    start = time.perf_counter()

    try:
        observation = env.reset()
        total, length, complete = 0.0, 0, False

        if keep_trajectory:
            observations.append(observation.copy())
        if record:
            frames.append(observation.copy())

        for _ in range(max_steps):
            action = policy(observation)

            if not np.isfinite(action).all():
                raise ValueError(f"{condition} produced a non-finite action at step {length}")

            step = env.step(action)

            if latent:
                policy.observe_executed(step.action)

            total += float(step.reward)
            length += 1

            if keep_trajectory:
                observations.append(step.next_observation.copy())
                actions.append(step.action.copy())
                rewards.append(float(step.reward))
                is_last.append(bool(step.is_last))
                is_terminal.append(bool(step.is_terminal))
                discounts.append(float(step.discount))
            if record:
                frames.append(step.next_observation.copy())

            if step.is_last:
                complete = True
                break

            observation = step.next_observation
    finally:
        env.close()

    if not complete:
        raise RuntimeError(
            f"{condition}/{env_seed} did not reach a terminal step within {max_steps}; a truncated "
            "episode is not a return"
        )

    if not np.isfinite(total):
        raise ValueError(f"{condition}/{env_seed} produced a non-finite return")

    record_out = {
        "condition": condition,
        "env_seed": env_seed,
        "policy_seed": policy_seed,
        "policy_rng_used": condition != "zero",
        "episode_return": total,
        "length": length,
        "elapsed_s": time.perf_counter() - start,
    }

    trajectory = None
    if keep_trajectory:
        trajectory = Episode(
            episode_id=env_seed,
            # never "train": evaluate_open_loop refuses training episodes, and so should this data
            split="holdout",
            observations=np.stack(observations, axis=0),
            actions=np.stack(actions, axis=0).astype(np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            is_last=np.asarray(is_last, dtype=bool),
            is_terminal=np.asarray(is_terminal, dtype=bool),
            discounts=np.asarray(discounts, dtype=np.float32),
        )

    return {"record": record_out, "trajectory": trajectory, "frames": frames}


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def summarize(returns: list[float]) -> dict:
    array = np.asarray(returns, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def run_controls(args, out_dir: Path, agent: Agent, untrained: Agent, config: RunConfig) -> dict:
    domain, task = TASKS[config.task]
    size = (config.image_size, config.image_size)
    seeds = [args.seed_base + i for i in range(args.episodes)]
    episodes_path = out_dir / "control-episodes.jsonl"
    trajectory_dir = out_dir / "trained-trajectories"

    results: dict[str, list[float]] = {}
    trajectories: list[Episode] = []

    for condition in CONDITIONS:
        latent = condition in ("trained", "untrained")
        subject = agent if condition == "trained" else untrained
        before = _agent_fingerprint(subject) if latent else None
        returns: list[float] = []

        for position, env_seed in enumerate(seeds):
            policy_seed = control_policy_seed(condition, env_seed)
            keep = condition == "trained"
            outcome = run_episode(
                condition,
                agent,
                untrained,
                domain,
                task,
                env_seed,
                policy_seed,
                size,
                args.max_steps,
                keep_trajectory=keep,
                record=position == 0,
            )

            row = outcome["record"]
            append_jsonl(episodes_path, row)
            returns.append(row["episode_return"])
            print(
                f"{condition:9s} seed {env_seed} return {row['episode_return']:8.2f} "
                f"length {row['length']} {row['elapsed_s']:.1f}s",
                flush=True,
            )

            if outcome["frames"]:
                write_video(out_dir / f"video-{condition}.mp4", np.stack(outcome["frames"], 0))

            if outcome["trajectory"] is not None:
                trajectory_dir.mkdir(parents=True, exist_ok=True)
                save_episode(trajectory_dir / f"episode-{env_seed}.npz", outcome["trajectory"])
                trajectories.append(outcome["trajectory"])

        if len(returns) != args.episodes:
            raise RuntimeError(f"{condition} produced {len(returns)} episodes, not {args.episodes}")

        if before is not None and not _fingerprints_match(before, _agent_fingerprint(subject)):
            raise RuntimeError(
                f"{condition} mutated agent state: parameters, buffers or a generator moved"
            )

        results[condition] = returns

    summary = {c: summarize(r) for c, r in results.items()}
    trained = np.asarray(results["trained"], dtype=np.float64)

    for control in ("untrained", "zero", "random"):
        paired = trained - np.asarray(results[control], dtype=np.float64)
        summary[f"trained_minus_{control}"] = {
            "mean": float(paired.mean()),
            "std": float(paired.std(ddof=0)),
            "median": float(np.median(paired)),
            "min": float(paired.min()),
            "max": float(paired.max()),
            "wins": int((paired > 0).sum()),
        }

    return {"returns": results, "summary": summary, "trajectories": trajectories}


def run_filmstrips(model, episodes: list[Episode], contexts, out_dir: Path, device, seed: int):
    """Three predetermined contexts, decoded open-loop against the frames actually observed."""
    picks = sorted({0, len(contexts) // 2, len(contexts) - 1})
    written = []

    for rank, index in enumerate(picks):
        context = contexts[index]
        data = gather_contexts(episodes, [context], CONTEXT_LENGTH, OPENLOOP_HORIZON)

        def to(array, dtype=None):
            tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device)
            return tensor if dtype is None else tensor.to(dtype)

        generator = torch.Generator(device=device)
        generator.manual_seed(seed * 65_537 + index)

        prediction = open_loop_predict(
            model,
            to(data.context_observations),
            to(data.context_actions, torch.float32),
            to(data.future_actions, torch.float32),
            generator,
            decode=True,
        )

        if prediction.image is None:
            raise RuntimeError("decode=True returned no image; a filmstrip needs the decoder")

        columns = [d - 1 for d in FILMSTRIP_DISTANCES if d <= OPENLOOP_HORIZON]
        observed = data.target_images[0, columns]
        predicted = (
            prediction.image[0, columns].clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
        )
        strip = paired_filmstrip(observed, predicted)
        path = out_dir / f"filmstrip-{rank}-ep{episodes[context.episode_index].episode_id}-t{context.start}.png"
        write_png(path, strip)
        written.append({"file": path.name, "episode_index": context.episode_index, "start": context.start})

    return written


def run_openloop(agent: Agent, trajectories: list[Episode], out_dir: Path, seed: int) -> dict:
    model = agent.world_model
    device = agent.device
    contexts = make_contexts(
        trajectories, CONTEXT_LENGTH, OPENLOOP_HORIZON, CONTEXTS_PER_EPISODE
    )
    # in-sample: these are the same trajectories being predicted, so it flatters the constant baseline
    constant_reward = float(np.concatenate([e.rewards for e in trajectories]).mean())

    metrics = evaluate_open_loop(
        model,
        trajectories,
        contexts,
        CONTEXT_LENGTH,
        OPENLOOP_HORIZON,
        constant_reward=constant_reward,
        samples=OPENLOOP_SAMPLES,
        seed=seed,
        device=device,
        chunk_contexts=8,
    )

    filmstrips = run_filmstrips(model, trajectories, contexts, out_dir, device, seed)
    result = metrics.as_dict()
    result["constant_reward"] = constant_reward
    result["constant_reward_is_in_sample"] = True
    result["context_length"] = CONTEXT_LENGTH
    result["contexts_per_episode"] = CONTEXTS_PER_EPISODE
    result["filmstrips"] = filmstrips
    result["filmstrip_distances"] = list(FILMSTRIP_DISTANCES)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-openloop", action="store_true")
    args = parser.parse_args()

    if args.seed_base < 3000 or args.seed_base + args.episodes > 3000 + 1000:
        raise ValueError("control seeds live at 3000+; 0-2 and 1000-1019 are reserved for M10")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"device: {device}  torch: {torch.__version__}  cuda: {torch.cuda.is_available()}", flush=True)

    started = time.time()
    status_path = out_dir / "status.json"

    try:
        agent, config, meta = load_trained_agent(Path(args.checkpoint), device)

        if config.task != args.task:
            raise ValueError(f"checkpoint holds task {config.task!r}, not {args.task!r}")

        # same architecture, same init stream: Agent.__init__ seeds from stream_seed(config.seed)
        untrained = Agent(action_dim=meta["action_dim"], config=config, device=device)
        untrained.eval()

        counts = agent.parameter_counts()
        print(
            f"task: {config.task}  seed: {config.seed}  horizon: {config.horizon}  "
            f"action_dim: {meta['action_dim']}",
            flush=True,
        )
        print(
            f"parameters: world model {counts.world_model}  actor {counts.actor}  "
            f"critic {counts.critic}  trainable total {counts.total}",
            flush=True,
        )
        print(f"checkpoint: {meta['path']} sha256 {meta['sha256'][:16]}", flush=True)

        (out_dir / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
        )

        controls = run_controls(args, out_dir, agent, untrained, config)
        control_seconds = time.time() - started

        openloop = None
        if not args.skip_openloop:
            print("open-loop diagnostic on the trained trajectories", flush=True)
            openloop = run_openloop(
                agent, controls["trajectories"], out_dir, seed=args.seed_base
            )

        report = {
            "task": config.task,
            "checkpoint": meta,
            "device": str(device),
            "torch": torch.__version__,
            "episodes_per_condition": args.episodes,
            "env_seeds": [args.seed_base + i for i in range(args.episodes)],
            "conditions": list(CONDITIONS),
            "returns": controls["returns"],
            "summary": controls["summary"],
            "parameters": {
                "world_model": counts.world_model,
                "actor": counts.actor,
                "critic": counts.critic,
                "trainable_total": counts.total,
            },
            "openloop": openloop,
            "control_seconds": control_seconds,
            "total_seconds": time.time() - started,
            "started": started,
            "finished": time.time(),
        }
        (out_dir / "controls-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )

        for condition in CONDITIONS:
            s = controls["summary"][condition]
            print(
                f"{condition:9s} mean {s['mean']:8.2f}  sd {s['std']:7.2f}  "
                f"median {s['median']:8.2f}  min {s['min']:8.2f}  max {s['max']:8.2f}",
                flush=True,
            )

        status_path.write_text(
            json.dumps(
                {"status": "ok", "seconds": report["total_seconds"], "report": "controls-report.json"},
                indent=2,
            )
            + "\n"
        )
        print(f"wrote {out_dir / 'controls-report.json'}", flush=True)
        return 0

    except Exception as error:
        status_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "seconds": time.time() - started,
                },
                indent=2,
            )
            + "\n"
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
