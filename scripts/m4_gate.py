"""M4 gate: world-model objective on a real Walker replay batch, plus a fixed-subset overfit."""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "src")

from dreamer import (
    Collector,
    DMCEnv,
    LaProp,
    ReplayBuffer,
    UniformRandomPolicy,
    WorldModel,
)
from dreamer.world_model import batch_to_tensors

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

B, P, T = 16, 5, 64
OVERFIT_STEPS = 300
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}  torch: {torch.__version__}")

env = DMCEnv("walker", "walk", seed=0)
replay = ReplayBuffer(capacity=500_000, seed=0)
collector = Collector(env, replay, UniformRandomPolicy(env.action_low, env.action_high, 0))
collector.collect(1000)
batch = replay.sample_sequences(batch_size=B, train_length=T, burn_in=P)
action_dim = env.action_shape[0]
env.close()

torch.manual_seed(0)
wm = WorldModel(action_dim=action_dim).to(device)
generator = torch.Generator(device=device)
generator.manual_seed(0)

print("\n## parameter counts (measured)")
counts = {
    "enc": sum(p.numel() for p in wm.encoder.parameters()),
    "dyn": sum(p.numel() for p in wm.rssm.parameters()),
    "dec": sum(p.numel() for p in wm.decoder.parameters()),
    "rew": sum(p.numel() for p in wm.reward.parameters()),
    "con": sum(p.numel() for p in wm.cont.parameters()),
}
derived = {"enc": 14_304, "dyn": 376_704, "dec": 80_595, "rew": 57_663, "con": 41_153}
for key in counts:
    print(f"  {key:<5}{counts[key]:>8}  derived {derived[key]:>8}  delta {counts[key]-derived[key]:>4}")
total = sum(counts.values())
print(f"  {'TOTAL':<5}{total:>8}  derived {sum(derived.values()):>8}")

check("per-module counts match the derivation", counts == derived)
check("world-model total == 570408 + heads", total == 570_419, str(total))

print("\n## real replay batch")
out = wm.loss(*batch_to_tensors(batch, device), generator=generator)
for key, value in out.losses.items():
    print(f"  {key:<5}{float(value):>14.4f}")
print(f"  {'total':<5}{float(out.total):>14.4f}")
print(f"  dyn_raw {out.metrics['dyn_raw']:.4f}   rep_raw {out.metrics['rep_raw']:.4f}")
print(f"  post_entropy {out.metrics['post_entropy']:.4f}   prior_entropy {out.metrics['prior_entropy']:.4f}")
print(f"  reward_mae {out.metrics['reward_mae']:.6f}   cont_mae {out.metrics['cont_mae']:.6f}")

check("train_positions == B*T", out.metrics["train_positions"] == float(B * T), str(out.metrics["train_positions"]))
check("reward prediction is exactly 0 at init", out.metrics["reward_mae"] == float(torch.from_numpy(batch.rewards)[:, P:].abs().mean()), "")
check("all losses finite", all(torch.isfinite(v).all() for v in out.losses.values()))

out.total.backward()
grads = [p.grad for p in wm.parameters() if p.grad is not None]
check("gradients finite", all(torch.isfinite(g).all() for g in grads))
check("encoder receives gradient", wm.encoder.convs[0].weight.grad.abs().sum().item() > 0)
check("rssm receives gradient", wm.rssm.dyngru.kernel.grad.abs().sum().item() > 0)
check("decoder receives gradient", wm.decoder.imgout.weight.grad.abs().sum().item() > 0)

if device.type == "cuda":
    print(f"  peak VRAM, one B=16 P=5 T=64 world-model step: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")

print(f"\n## overfit a fixed subset (B=4, {OVERFIT_STEPS} steps, LaProp lr=3e-4)")
small = replay.sample_sequences(batch_size=4, train_length=T, burn_in=P)
tensors = batch_to_tensors(small, device)

torch.manual_seed(0)
wm = WorldModel(action_dim=action_dim).to(device)
opt = LaProp(wm.parameters(), lr=3e-4, warmup=0)

history = []
for step in range(OVERFIT_STEPS):
    generator.manual_seed(step)
    out = wm.loss(*tensors, generator=generator)
    opt.zero_grad(set_to_none=True)
    out.total.backward()
    opt.step()
    if step % 50 == 0 or step == OVERFIT_STEPS - 1:
        history.append((step, out.metrics))

print(f"  {'step':>5}{'rec':>12}{'rew':>10}{'con':>10}{'dyn_raw':>10}{'rep_raw':>10}{'rew_mae':>10}")
for step, m in history:
    print(f"  {step:>5}{m['rec']:>12.2f}{m['rew']:>10.4f}{m['con']:>10.4f}"
          f"{m['dyn_raw']:>10.4f}{m['rep_raw']:>10.4f}{m['reward_mae']:>10.4f}")

first, last = history[0][1], history[-1][1]
for key in ("rec", "rew", "con"):
    ratio = last[key] / first[key]
    check(f"{key} falls materially", ratio < 0.75, f"{first[key]:.4g} -> {last[key]:.4g}  ({ratio:.3f}x)")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
