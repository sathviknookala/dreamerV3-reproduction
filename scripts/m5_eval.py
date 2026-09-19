"""M5 open-loop prediction evaluation on held-out episodes.

Posterior over a fixed context prefix, then prior only on recorded actions. No future
observation is an argument to the rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from dreamer import WorldModel
from dreamer.openloop import (
    evaluate_open_loop,
    gather_contexts,
    load_split,
    make_contexts,
    open_loop_predict,
    training_reward_mean,
)
from dreamer.viz import paired_filmstrip, write_png

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/m5-walker-random")
parser.add_argument("--checkpoint", default="runs/m5/world-model.pt")
parser.add_argument("--out", default="results/m5")
parser.add_argument("--context", type=int, default=5, help="context transitions (spec.md 7.5 P)")
parser.add_argument("--horizon", type=int, default=30)
parser.add_argument("--contexts-per-episode", type=int, default=16)
parser.add_argument("--samples", type=int, default=8)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--chunk", type=int, default=32)
parser.add_argument("--filmstrips", type=int, default=3)
parser.add_argument("--tag", default="")
args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
model = WorldModel(action_dim=checkpoint["action_dim"]).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

train_episodes = load_split(args.data, "train")
holdout_episodes = load_split(args.data, "holdout")

train_digests = sorted(e.digest() for e in train_episodes)
holdout_digests = sorted(e.digest() for e in holdout_episodes)

if set(train_digests) & set(holdout_digests):
    raise RuntimeError("train and holdout splits share episode content")

if train_digests != sorted(checkpoint["train_episode_sha256"]):
    raise RuntimeError("checkpoint was not trained on this training split")

constant = training_reward_mean(train_episodes)
contexts = make_contexts(holdout_episodes, args.context, args.horizon, args.contexts_per_episode)

print(f"device {device}  checkpoint {args.checkpoint} ({checkpoint['steps']} steps)")
print(f"holdout episodes {len(holdout_episodes)}  contexts {len(contexts)}  samples {args.samples}")
print(f"training-set mean reward (train split only) {constant:.6f}")

metrics = evaluate_open_loop(
    model,
    holdout_episodes,
    contexts,
    context_length=args.context,
    horizon=args.horizon,
    constant_reward=constant,
    samples=args.samples,
    seed=args.seed,
    device=device,
    chunk_contexts=args.chunk,
)

report = metrics.as_dict()
report.update(
    checkpoint=args.checkpoint,
    checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
    checkpoint_steps=checkpoint["steps"],
    data=args.data,
    context_transitions=args.context,
    contexts_per_episode=args.contexts_per_episode,
    holdout_episodes=len(holdout_episodes),
    seed=args.seed,
    constant_reward_baseline=constant,
    constant_reward_baseline_source="train split only",
    train_episode_sha256=train_digests,
    holdout_episode_sha256=holdout_digests,
)

tag = f"-{args.tag}" if args.tag else ""
(out / f"openloop{tag}.json").write_text(json.dumps(report, indent=2) + "\n")

csv_columns = [
    "distance", "reward_mae", "reward_rmse", "reward_mae_sample_sd",
    "reward_mae_shuffled_actions", "baseline_reward_mae_mean",
    "baseline_reward_mae_persistence", "target_reward_sd",
    "cont_mae", "image_mae", "baseline_image_mae_persistence",
]
lines = [",".join(csv_columns)]
for k in range(args.horizon):
    lines.append(",".join([
        str(k + 1),
        *(f"{v[k]:.6f}" for v in (
            metrics.reward_mae, metrics.reward_rmse, metrics.reward_mae_sample_sd,
            metrics.reward_mae_shuffled, metrics.baseline_reward_mae_mean,
            metrics.baseline_reward_mae_persistence, metrics.target_reward_sd,
            metrics.cont_mae, metrics.image_mae, metrics.baseline_image_mae_persistence,
        )),
    ]))
(out / f"openloop-by-distance{tag}.csv").write_text("\n".join(lines) + "\n")

print(f"\n{'k':>4}{'reward MAE':>14}{'mean base':>13}{'persist':>12}"
      f"{'shuffled':>12}{'ratio':>8}{'img MAE':>10}{'img persist':>13}")
for k in (1, 5, 15, 30):
    if k > args.horizon:
        continue
    i = k - 1
    ratio = metrics.reward_mae[i] / metrics.baseline_reward_mae_mean[i]
    print(f"{k:>4}{metrics.reward_mae[i]:>14.6f}{metrics.baseline_reward_mae_mean[i]:>13.6f}"
          f"{metrics.baseline_reward_mae_persistence[i]:>12.6f}"
          f"{metrics.reward_mae_shuffled[i]:>12.6f}{ratio:>8.3f}"
          f"{metrics.image_mae[i]:>10.5f}{metrics.baseline_image_mae_persistence[i]:>13.5f}")

print("\n## paired open-loop frames")
generator = torch.Generator(device=device)
frames_dir = out / f"frames{tag}"
frames_dir.mkdir(exist_ok=True)
selected = [contexts[i] for i in np.linspace(0, len(contexts) - 1, args.filmstrips).astype(int)]
data = gather_contexts(holdout_episodes, selected, args.context, args.horizon)
saved = []

with torch.no_grad():
    generator.manual_seed(args.seed)
    prediction = open_loop_predict(
        model,
        torch.from_numpy(data.context_observations).to(device),
        torch.from_numpy(data.context_actions).to(device=device, dtype=torch.float32),
        torch.from_numpy(data.future_actions).to(device=device, dtype=torch.float32),
        generator,
        decode=True,
    )
    decoded = (prediction.image.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()

# the selection rule is positional, not by error: evenly spaced over the context list
for row, context in enumerate(selected):
    name = f"holdout-ep{holdout_episodes[context.episode_index].episode_id}-t{context.start}.png"
    write_png(frames_dir / name, paired_filmstrip(data.target_images[row], decoded[row]))
    saved.append(name)
    print(f"  {name}  (top: observed, bottom: open-loop prediction, k=1..{args.horizon})")

np.savez_compressed(
    frames_dir / f"frames{tag}.npz",
    observed=data.target_images,
    predicted=decoded,
    episode_id=np.array([holdout_episodes[c.episode_index].episode_id for c in selected]),
    start=np.array([c.start for c in selected]),
)
print(f"\nwrote {out / f'openloop{tag}.json'} and {out / f'openloop-by-distance{tag}.csv'}")
