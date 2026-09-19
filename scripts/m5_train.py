"""M5 world-model training on the training split only. Held-out episodes are never loaded."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "src")

from dreamer import LaProp, WorldModel
from dreamer.openloop import load_split, replay_from_episodes
from dreamer.world_model import batch_to_tensors

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/m5-walker-random")
parser.add_argument("--out", default="runs/m5")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--burn-in", type=int, default=5)
parser.add_argument("--train-length", type=int, default=64)
parser.add_argument("--train-ratio", type=int, default=64)
parser.add_argument("--steps", type=int, default=0, help="0 = derive from the training ratio")
parser.add_argument("--lr", type=float, default=4e-5)
parser.add_argument("--warmup", type=int, default=1000)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--log-every", type=int, default=25)
args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

data_manifest = json.loads((Path(args.data) / "manifest.json").read_text())
episodes = load_split(args.data, "train")
train_digests = sorted(e.digest() for e in episodes)
transitions = sum(e.length for e in episodes)

positions = args.batch * args.train_length
steps = args.steps or (transitions * args.train_ratio) // positions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device {device}  torch {torch.__version__}")
print(f"train episodes {len(episodes)}  transitions {transitions}")
print(f"train_position/gradient_step {positions}  ratio {args.train_ratio}  -> {steps} gradient steps")

replay = replay_from_episodes(episodes, seed=args.seed)
print(f"replay {replay.stats()}")

torch.manual_seed(args.seed)
model = WorldModel(action_dim=data_manifest["action_dim"]).to(device)
optimizer = LaProp(model.parameters(), lr=args.lr, warmup=args.warmup)
generator = torch.Generator(device=device)

log_path = out / "train-log.csv"
columns = ["step", "elapsed_s", "total", "rec", "rew", "con", "dyn", "rep",
           "dyn_raw", "rep_raw", "post_entropy", "prior_entropy",
           "reward_mae", "cont_mae", "grad_norm"]
log = log_path.open("w")
log.write(",".join(columns) + "\n")

start = time.perf_counter()

for step in range(steps):
    generator.manual_seed(args.seed * 1_000_003 + step)
    batch = replay.sample_sequences(args.batch, args.train_length, args.burn_in)
    output = model.loss(*batch_to_tensors(batch, device), generator=generator)

    optimizer.zero_grad(set_to_none=True)
    output.total.backward()
    # observed only -- spec.md 5.8 specifies no global-norm clip, so nothing is rescaled here
    grad_norm = torch.linalg.vector_norm(
        torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])
    )
    optimizer.step()

    if step % args.log_every == 0 or step == steps - 1:
        metrics = dict(output.metrics, grad_norm=float(grad_norm))
        row = [step, round(time.perf_counter() - start, 3)] + [
            round(float(metrics[c]), 6) for c in columns[2:]
        ]
        log.write(",".join(str(v) for v in row) + "\n")
        log.flush()

        if step % (args.log_every * 20) == 0 or step == steps - 1:
            print(f"  step {step:>6}  total {metrics['total']:>10.3f}  rec {metrics['rec']:>9.2f}  "
                  f"rew {metrics['rew']:>7.4f}  con {metrics['con']:>7.4f}  "
                  f"dyn_raw {metrics['dyn_raw']:>7.4f}  rew_mae {metrics['reward_mae']:.5f}",
                  flush=True)

    if not torch.isfinite(output.total):
        raise RuntimeError(f"non-finite loss at step {step}")

log.close()
elapsed = time.perf_counter() - start

checkpoint = out / "world-model.pt"
torch.save(
    {
        "model": model.state_dict(),
        "action_dim": data_manifest["action_dim"],
        "steps": steps,
        "seed": args.seed,
        "config": vars(args),
        "train_episode_sha256": train_digests,
    },
    checkpoint,
)

manifest = {
    "data": args.data,
    "data_seed": data_manifest["seed"],
    "domain": data_manifest["domain"],
    "task": data_manifest["task"],
    "seed": args.seed,
    "gradient_steps": steps,
    "train_positions": steps * positions,
    "batch": args.batch,
    "burn_in": args.burn_in,
    "train_length": args.train_length,
    "train_ratio": args.train_ratio,
    "train_ratio_convention": "loss-bearing positions per collected transition (spec.md 7.7)",
    "optimizer": {"name": "LaProp", "lr": args.lr, "warmup": args.warmup,
                  "beta1": 0.9, "beta2": 0.999, "eps": 1e-20, "agc": 0.3},
    "train_episodes": len(episodes),
    "train_transitions": transitions,
    "train_episode_sha256": train_digests,
    "elapsed_s": round(elapsed, 1),
    "steps_per_s": round(steps / elapsed, 3),
    "peak_vram_mib": (
        round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None
    ),
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "parameters": sum(p.numel() for p in model.parameters()),
}
(out / "train-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

print(f"\n{steps} gradient steps in {elapsed:.1f}s ({steps / elapsed:.2f} step/s)")
print(f"checkpoint {checkpoint}  sha256 {manifest['checkpoint_sha256'][:16]}")
