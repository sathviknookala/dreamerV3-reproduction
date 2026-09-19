"""M2+M3 gate: drive the RSSM from a real M1 replay batch on the GPU."""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "src")

from dreamer import (
    Collector,
    DMCEnv,
    Encoder,
    RSSM,
    ReplayBuffer,
    UniformRandomPolicy,
    observe_replay_batch,
)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

B, P, T = 16, 5, 64
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
env.close()

torch.manual_seed(0)
encoder = Encoder().to(device)
rssm = RSSM(action_dim=env.action_shape[0]).to(device)

generator = torch.Generator(device=device)
generator.manual_seed(0)

post, prior = observe_replay_batch(encoder, rssm, batch, device, generator)

L = P + T + 1
check("posterior deter", tuple(post.deter.shape) == (B, L, 512), str(tuple(post.deter.shape)))
check("posterior stoch", tuple(post.stoch.shape) == (B, L, 32, 4), str(tuple(post.stoch.shape)))
check("posterior feat", tuple(post.feat.shape) == (B, L, 640), str(tuple(post.feat.shape)))
check("prior logit", tuple(prior.logit.shape) == (B, L, 32, 4), str(tuple(prior.logit.shape)))

check("states finite", bool(torch.isfinite(post.feat).all() and torch.isfinite(prior.logit).all()))
check(
    "stoch factors are one-hot",
    bool((post.stoch.sum(-1) == 1).all() and torch.isin(post.stoch, torch.tensor([0.0, 1.0], device=device)).all()),
)
check(
    "posterior probs normalize",
    bool(torch.allclose(rssm.probs(post.logit).sum(-1), torch.ones(B, L, 32, device=device))),
)
# deter at a reset step is exactly zero at init (all biases zero-init), so the
# image must reach t=0 through the posterior head, not through the carry
check(
    "image reaches the posterior at the reset step",
    bool((post.deter[:, 0] == 0).all() and post.logit[:, 0].std(0).max() > 0),
)

loss = post.feat.square().mean() + prior.logit.square().mean()
loss.backward()
grads = [p.grad for p in list(encoder.parameters()) + list(rssm.parameters())]
check("all parameters received gradient", all(g is not None for g in grads))
check("gradients finite", all(torch.isfinite(g).all() for g in grads))
check("gradients nonzero", sum(g.abs().sum().item() for g in grads) > 0.0)

params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in rssm.parameters())
check("parameter count == derived 391008", params == 391_008, str(params))

if device.type == "cuda":
    print(f"peak VRAM for one B=16 P=5 T=64 forward+backward: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
