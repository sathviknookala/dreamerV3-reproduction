"""M5 data: random-policy Walker episodes, split into disjoint train and held-out sets.

Development seed 100 (spec.md 8). The M10 diagnostic corpus (seeds 500-519) stays frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from dreamer import Collector, DMCEnv, ReplayBuffer, UniformRandomPolicy
from dreamer.openloop import Episode, save_episode

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="data/m5-walker-random")
parser.add_argument("--domain", default="walker")
parser.add_argument("--task", default="walk")
parser.add_argument("--train-episodes", type=int, default=120)
parser.add_argument("--holdout-episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=100)
args = parser.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

env = DMCEnv(args.domain, args.task, seed=args.seed)
policy = UniformRandomPolicy(env.action_low, env.action_high, seed=args.seed)
replay = ReplayBuffer(capacity=2_000_000, seed=args.seed)
collector = Collector(env, replay, policy)

total = args.train_episodes + args.holdout_episodes
entries = []
collected = 0

for index in range(total):
    while replay.num_complete_episodes <= index:
        collector.collect_step()

    stored = replay.complete_episodes[index]
    # holdout episodes are the tail block, so a train/holdout swap changes every file name
    split = "train" if index < args.train_episodes else "holdout"

    episode = Episode(
        episode_id=index,
        split=split,
        observations=stored.observations,
        actions=stored.actions,
        rewards=stored.rewards,
        is_last=stored.is_last,
        is_terminal=stored.is_terminal,
        discounts=stored.discounts,
    )

    name = f"ep_{index:04d}.npz"
    save_episode(out / name, episode)
    entries.append(
        {
            "episode_id": index,
            "split": split,
            "file": name,
            "length": episode.length,
            "return": episode.episode_return,
            "sha256": episode.digest(),
        }
    )
    collected += episode.length
    print(f"  {name}  {split:<8} len={episode.length}  return={episode.episode_return:.3f}")

env.close()

returns = {
    split: [e["return"] for e in entries if e["split"] == split]
    for split in ("train", "holdout")
}

manifest = {
    "domain": args.domain,
    "task": args.task,
    "policy": "uniform random",
    "seed": args.seed,
    "action_dim": env.action_shape[0],
    "train_episodes": args.train_episodes,
    "holdout_episodes": args.holdout_episodes,
    "transitions": collected,
    "train_transitions": sum(e["length"] for e in entries if e["split"] == "train"),
    "holdout_transitions": sum(e["length"] for e in entries if e["split"] == "holdout"),
    "return_mean": {k: float(np.mean(v)) for k, v in returns.items()},
    "return_sd": {k: float(np.std(v)) for k, v in returns.items()},
    "episodes": entries,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

digests = [e["sha256"] for e in entries]
assert len(set(digests)) == len(digests), "duplicate episode content"

print(f"\nwrote {total} episodes ({collected} transitions) to {out}")
print(f"return mean  train {manifest['return_mean']['train']:.3f}  "
      f"holdout {manifest['return_mean']['holdout']:.3f}")
