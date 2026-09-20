"""M9 step 5: where a complete update's time goes, at H = 5, 15 and 30.

Every stage is timed with a CUDA synchronize on both sides, warm-up iterations are discarded, and
the decomposition is reported beside a real `training_update` so the unattributed residual is
visible rather than hidden. H=30 peak VRAM is read in a steady-state update, after LaProp has
allocated its moments -- a bare rollout understates it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, "src")

from dreamer import (
    Agent,
    Collector,
    EpisodicEnv,
    LaProp,
    ReplayBuffer,
    RunConfig,
    UniformRandomPolicy,
    apply_update,
    atomic_save,
    behavior_losses,
    compute_losses,
    evaluate_episode,
    imagine_trajectory,
    scatter_imagined_return,
    select_start_states,
    stream_seed,
)
from dreamer.config import TASKS
from dreamer.training import _batch_tensors

HORIZONS = (5, 15, 30)


class Stopwatch:
    def __init__(self, device) -> None:
        self.device = device
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @contextmanager
    def stage(self, name: str):
        self._sync()
        start = time.perf_counter()
        yield
        self._sync()
        elapsed = time.perf_counter() - start
        self.totals[name] = self.totals.get(name, 0.0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + 1

    def mean_ms(self, name: str) -> float:
        if not self.counts.get(name):
            return 0.0

        return 1000.0 * self.totals[name] / self.counts[name]


def decomposed_update(agent, optimizer, batch, horizon, watch) -> None:
    """The same sequence `compute_losses` runs, timed stage by stage."""
    with watch.stage("replay_transfer"):
        tensors = _batch_tensors(batch, agent.device)

    with watch.stage("world_model"):
        world = agent.world_model.loss(
            tensors["observations"],
            tensors["actions"],
            tensors["rewards"],
            tensors["is_terminal"],
            tensors["loss_mask"],
            generator=agent.generators["replay"],
        )

    with watch.stage("imagination_behavior"):
        starts, index = select_start_states(
            world.post, tensors["loss_mask"], tensors["is_terminal"]
        )
        imagination = imagine_trajectory(
            agent.world_model, starts, agent.provider, horizon, agent.generators["imagine"]
        )
        behavior = behavior_losses(
            agent.actor, agent.critic, imagination, agent.normalizer, replay=None
        )
        grid, filled = scatter_imagined_return(
            behavior.imagined.ret, index, batch.batch_size, batch.transition_length
        )
        burn_in = agent.config.burn_in
        replay_critic = agent.critic.replay_loss(
            world.post[:, burn_in + 1 :].feat,
            tensors["rewards"][:, burn_in:],
            tensors["is_last"][:, burn_in:].to(torch.float32),
            tensors["is_terminal"][:, burn_in:].to(torch.float32),
            grid[:, burn_in:],
            filled=filled[:, burn_in:],
        )
        total = (
            world.total + behavior.loss + agent.critic.scales["repval"] * replay_critic.loss
        )

    with watch.stage("backward_optimizer"):
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        agent.critic.update_slow()


def profile_horizon(args, horizon: int, collection_ms: float) -> dict:
    config = RunConfig(
        task=args.task,
        seed=args.seed,
        horizon=horizon,
        batch_size=args.batch_size,
        train_length=args.train_length,
        burn_in=args.burn_in,
        replay_capacity=args.transitions * 2,
        eval_video=False,
    )
    domain, task = config.domain_task
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    env = EpisodicEnv(domain, task, base_seed=config.seed, recreate=False)
    agent = Agent(action_dim=env.action_shape[0], config=config, device=device)
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=stream_seed(config.seed, "replay"))
    Collector(env, replay, UniformRandomPolicy(env.action_low, env.action_high, config.seed)).collect(
        args.transitions
    )

    optimizer = LaProp(
        agent.trainable_parameters(), lr=config.lr, warmup=config.lr_warmup
    )
    watch = Stopwatch(device)

    def sample():
        return replay.sample_sequences(config.batch_size, config.train_length, config.burn_in)

    for _ in range(args.warmup):
        apply_update(agent, optimizer, compute_losses(agent, sample()))

    if device.type == "cuda":
        torch.cuda.synchronize()
        # after warm-up, so LaProp's mu/nu are already allocated and counted
        torch.cuda.reset_peak_memory_stats()

    real_total = 0.0

    for _ in range(args.steps):
        batch = sample()
        watch._sync()
        start = time.perf_counter()
        decomposed_update(agent, optimizer, batch, horizon, watch)
        watch._sync()
        real_total += time.perf_counter() - start

    peak_mib = (
        torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    )
    reserved_mib = (
        torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else 0.0
    )

    with watch.stage("checkpoint"):
        payload = {
            "model": agent.model_state(),
            "generators": agent.generator_state(),
            "replay": replay.state_dict(),
        }
        # atomic_save, not torch.save: the pickle protocol is part of the measured cost
        path = atomic_save(payload, Path(args.out) / f"profile-checkpoint-h{horizon}.pt")

    checkpoint_mib = path.stat().st_size / 2**20
    path.unlink()

    with watch.stage("evaluation"):
        result, _ = evaluate_episode(
            agent, domain, task, seed=2000, max_steps=args.eval_steps
        )

    env.close()

    update_ms = 1000.0 * real_total / args.steps
    stages = ("replay_transfer", "world_model", "imagination_behavior", "backward_optimizer")
    attributed = sum(watch.mean_ms(name) for name in stages)

    row = {
        "horizon": horizon,
        "batch": config.batch_size,
        "train_length": config.train_length,
        "burn_in": config.burn_in,
        "steps": args.steps,
        "update_ms": round(update_ms, 3),
        "collection_render_ms_per_transition": round(collection_ms, 4),
        "replay_transfer_ms": round(watch.mean_ms("replay_transfer"), 3),
        "world_model_ms": round(watch.mean_ms("world_model"), 3),
        "imagination_behavior_ms": round(watch.mean_ms("imagination_behavior"), 3),
        "backward_optimizer_ms": round(watch.mean_ms("backward_optimizer"), 3),
        "unattributed_ms": round(update_ms - attributed, 3),
        "evaluation_ms_per_step": round(watch.mean_ms("evaluation") / max(1, result.length), 4),
        "checkpoint_s": round(watch.totals["checkpoint"], 3),
        "checkpoint_mib": round(checkpoint_mib, 1),
        "peak_vram_mib": round(peak_mib, 1),
        "reserved_vram_mib": round(reserved_mib, 1),
        "imagination_starts": config.positions_per_update,
        "imagined_transitions": config.positions_per_update * horizon,
    }

    del agent, optimizer, replay

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return row


def measure_collection(args) -> float:
    """Simulator step plus 64x64 render, with a learned-policy forward in the loop."""
    config = RunConfig(task=args.task, seed=args.seed, eval_video=False)
    domain, task = config.domain_task
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    env = EpisodicEnv(domain, task, base_seed=config.seed, recreate=False)
    agent = Agent(action_dim=env.action_shape[0], config=config, device=device)
    replay = ReplayBuffer(capacity=args.collection_steps * 2, seed=0)
    collector = Collector(env, replay, agent.collection_policy())

    collector.collect(min(100, args.collection_steps))

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    collector.collect(args.collection_steps)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    env.close()
    return 1000.0 * elapsed / args.collection_steps


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="M9 per-stage profile")
    parser.add_argument("--task", choices=TASKS, default="walker")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20, help="timed steady-state updates")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--transitions", type=int, default=2000)
    parser.add_argument("--collection-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-length", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default="results/m9")
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"profile-{args.task}-{args.tag}.csv"

    if path.exists():
        raise SystemExit(f"{path} exists; a committed measurement is not overwritten")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}  torch: {torch.__version__}")

    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    collection_ms = measure_collection(args)
    print(f"collection + render + policy forward: {collection_ms:.3f} ms/transition\n")

    rows = [profile_horizon(args, horizon, collection_ms) for horizon in HORIZONS]

    header = ["horizon", "update_ms", "world_model_ms", "imagination_behavior_ms",
              "backward_optimizer_ms", "replay_transfer_ms", "unattributed_ms", "peak_vram_mib"]
    print("  " + "".join(f"{c:>26}" for c in header))

    for row in rows:
        print("  " + "".join(f"{row[c]:>26}" for c in header))

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "args": vars(args),
        "collection_render_ms_per_transition": collection_ms,
        "rows": rows,
        "excludes": (
            "the update decomposition excludes collection, evaluation and checkpointing, which are "
            "reported separately; `unattributed_ms` is the residual of the decomposition against a "
            "timed complete update"
        ),
    }
    (out / f"profile-{args.task}-{args.tag}.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
