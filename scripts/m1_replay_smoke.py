"""M1 smoke test: one real Walker episode through the collector, one sampled batch."""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from dreamer import Collector, DMCEnv, ReplayBuffer, UniformRandomPolicy

B, P, T = 16, 5, 64
STEPS = 1000

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


env = DMCEnv("walker", "walk", seed=0)
replay = ReplayBuffer(capacity=500_000, seed=0)
policy = UniformRandomPolicy(env.action_low, env.action_high, seed=0)
collector = Collector(env, replay, policy)

counters = collector.collect(STEPS)
stats = replay.stats()
print(f"collected {counters.env_step} env steps; {stats}")

check("one complete episode stored", stats["complete_episodes"] == 1, str(stats["complete_episodes"]))

batch = replay.sample_sequences(batch_size=B, train_length=T, burn_in=P)

check(
    "observations shape/dtype",
    batch.observations.shape == (B, P + T + 1, 64, 64, 3) and batch.observations.dtype == np.uint8,
    f"{batch.observations.shape} {batch.observations.dtype}",
)
check("actions shape", batch.actions.shape == (B, P + T, 6), str(batch.actions.shape))
for name, arr in (
    ("rewards", batch.rewards),
    ("is_last", batch.is_last),
    ("is_terminal", batch.is_terminal),
    ("discounts", batch.discounts),
):
    check(f"{name} shape", arr.shape == (B, P + T), str(arr.shape))

mask = batch.loss_mask
check(
    "loss mask 5 burn-in / 64 training per sequence",
    mask.shape == (B, P + T)
    and bool((~mask[:, :P]).all())
    and bool(mask[:, P:].all())
    and bool(((~mask).sum(axis=1) == P).all())
    and bool((mask.sum(axis=1) == T).all()),
    f"train_positions={batch.train_positions}",
)
check("train_position == B*T", batch.train_positions == B * T, str(batch.train_positions))

# a boundary anywhere but the final transition would mean the window spans two episodes
check("no sequence crosses an episode boundary", not bool(batch.is_last[:, :-1].any()))

check("rewards finite", bool(np.isfinite(batch.rewards).all()))
check("actions finite and in bounds", bool(np.isfinite(batch.actions).all() and (np.abs(batch.actions) <= 1.0).all()))
check("discounts binary", bool(np.isin(batch.discounts, (0.0, 1.0)).all()))
check("is_terminal == (discount == 0)", bool((batch.is_terminal == (batch.discounts == 0.0)).all()))

env.close()

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
