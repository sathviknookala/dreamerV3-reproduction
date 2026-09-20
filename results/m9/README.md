# M9 — the online loop

**Everything in this directory is implementation validation, not demonstrated learning.** The loop
runs, every component is wired as [spec.md](../../docs/spec.md) §5 and §7 say, resume is exact, and
the per-stage cost is measured. No number here is evidence that the agent learns. **M9 stays open**
until both pixel tasks show sustained improvement over the M1 random floor.

## What is here

| File | What it is |
|---|---|
| [`gate-cartpole-2026-09-19.json`](gate-cartpole-2026-09-19.json) | The integration gate on Cartpole Swingup, 75/75. |
| [`gate-walker-2026-09-19.json`](gate-walker-2026-09-19.json) | The same gate on Walker Walk, 75/75, `action_dim` 6. |
| [`profile-walker-2026-09-19.csv`](profile-walker-2026-09-19.csv) | Per-stage cost of a complete update at H = 5 / 15 / 30, B=16 T=64 P=5. |
| [`profile-walker-2026-09-19.json`](profile-walker-2026-09-19.json) | The same run's device, flags, arguments and exclusions. |
| [`checkpoint-cost-2026-09-19.json`](checkpoint-cost-2026-09-19.json) | Resume-checkpoint write/read at full 500,000-transition occupancy. |

Reproduce with:

```bash
PYTHONPATH=src .venv/bin/python scripts/m9_gate.py --task cartpole --tag <date>
PYTHONPATH=src .venv/bin/python scripts/m9_gate.py --task walker --tag <date>
PYTHONPATH=src .venv/bin/python scripts/m9_profile.py --task walker --tag <date>
```

Both scripts **refuse to overwrite an existing artifact** — pass a new `--tag`.

## The gate

75 checks on the real simulator with real replay data, learned-policy collection, complete training
updates, isolated evaluation and a checkpoint/resume cycle. It covers stream separation, the
optimizer's parameter list, budget and warm-up accounting, the scheduler's realized ratio, rollout
and return layout, the eight pinned loss scales, gradient routing on the combined loss, update
cardinality, evaluation isolation, checkpoint rotation and next-update equivalence after a restore.

The Walker run reproduces the committed parameter counts exactly: world model **570,419**, `pol`
**50,316**, `val` **66,111**, trainable total **686,846**.

**The returns the gate prints are not results.** They come from a policy trained on ~1,200 control
steps at `B=8, T=32`, which is a smoke signal and nothing more. The gate's own JSON says so.

**The gate perturbs `rew` and `val` before any routing or magnitude claim**, because both are
`outscale: 0.0` and the advantage would otherwise be identically zero. Every magnitude the gate
prints after that point — the advantage, the return-normalization scale — is a property of a
perturbed head, not of a trained critic. The gate checks separately that a fresh `ReturnNormalizer`
reads exactly its floor of 1.0.

## The per-stage profile

B=16, T=64, P=5, 1024 imagination starts per update, 20 timed steady-state updates after 5 discarded
warm-up updates, CUDA-synchronized on both sides of every stage, TF32 at PyTorch defaults.
RTX PRO 4000 Blackwell, torch 2.13.0+cu129.

| H | update | world model | imagination + behaviour | backward + optimizer | replay transfer | unattributed | peak VRAM |
|---|---|---|---|---|---|---|---|
| 5 | 206.0 ms | 71.3 ms | 10.5 ms | 121.1 ms | 1.05 ms | 2.0 ms | 1576.3 MiB |
| 15 | 221.2 ms | 75.5 ms | 21.6 ms | 121.6 ms | 0.91 ms | 1.7 ms | 1815.3 MiB |
| 30 | 236.8 ms | 71.2 ms | 39.5 ms | 123.2 ms | 1.03 ms | 1.9 ms | 2373.8 MiB |

Collection, rendering and the policy forward together cost **2.796 ms per control step**.
Evaluation costs **3.29 ms per control step**, measured on a separate simulator with the same agent.

`unattributed_ms` is the residual of the stage decomposition against a separately timed complete
update; at under 1% it says the decomposition accounts for the update.

**H=30 peak VRAM is 2373.8 MiB allocated / 2844.0 MiB reserved in a steady-state update**, read
after LaProp's moments were allocated — a bare rollout understates it. That leaves roughly **22 GB
of headroom** on the 24 GB card, so H=30 is not memory-limited at this batch.

Horizon moves only the imagination + behaviour stage, which is 5–17% of an update. Doubling the
horizon from 15 to 30 costs **7.0%** more wall clock per update, not 2×.

## Checkpoint cost

A resume checkpoint at full occupancy is **5.744 GiB**, written in **10.1 s** and read in **7.4 s**
(583 MiB/s). Two are kept, so peak disk per run is ~11.5 GiB. At `checkpoint_every: 25000` a Walker
run writes 40 of them, ~400 s, **2.3%** of the run. The profile's own `checkpoint_mib` /
`checkpoint_s` columns are the same measurement at the profiler's 2,000-transition occupancy
(26.4 MiB, 0.06 s) and scale linearly to the figure above.

This measures serialization and I/O on synthetic uint8 frames, not collection. `atomic_save` pins
`pickle_protocol=5`; torch's default protocol 2 latin1-encodes bytes and produced **8.613 GiB** and
41.3 s for the same buffer.

## Campaign cost, derived from the numbers above

Per run, from `profile-walker-2026-09-19.csv` and the checkpoint measurement:

| Run | updates | training | collection | evaluation | checkpointing | total |
|---|---|---|---|---|---|---|
| Walker H=5 | 62,187 | 3.56 h | 0.78 h | 0.18 h | 0.11 h | **4.7 h** |
| Walker H=15 | 62,187 | 3.82 h | 0.78 h | 0.18 h | 0.11 h | **4.9 h** |
| Walker H=30 | 62,187 | 4.09 h | 0.78 h | 0.18 h | 0.11 h | **5.2 h** |
| Cartpole H=15 | 30,937 | 1.90 h | 0.39 h | 0.09 h | 0.06 h | **2.5 h** |

**The 12-run final campaign is ≈ 52 GPU-hours** on this card, plus whatever the pilots consume.

This is a derived estimate from measured per-stage costs at a fixed replay occupancy, not a measured
run. It assumes the update cost is independent of replay size beyond the measured 1.03 ms sampling,
and it does not include a failed or restarted run.

## Reading a run's log

`train-log.csv` and `evaluations.jsonl` are **append-only across resumes, and a resume replays**.
A checkpoint is older than wherever the previous segment died, so the rows between the two are
written a second time. Both artifacts therefore carry a `segment` column, incremented on every
restore.

**Dedupe rule for any curve: keep the highest `segment` for each `gradient_step`** (for evaluations,
for each `env_step`). Nothing is deleted, so a crashed segment stays inspectable, and the surviving
rows are the trajectory that actually produced the final weights.

`elapsed_s` is wall clock and **includes** evaluation, which is why it jumps at an evaluation
boundary and why it rewinds at a seam. Training time is `elapsed_s - eval_seconds`; both are
columns, and `evaluations.jsonl` records each evaluation's own seconds and steps.

## What none of this establishes

- **No return.** Nothing here shows the agent learns. The M1 random floor has still not been
  recorded, so there is not yet a floor to clear.
- **The gate's evaluation returns describe a ~1,200-step agent.**
- **The profile is a fixed-occupancy measurement.** Replay was 2,000 transitions of random data; a
  1M-step run samples from up to 500,000 and evicts.
- **Resume is bitwise for the next update, not for a trajectory.** Two restores of the same
  checkpoint compute an identical next update, and then drift at ~1e-6 over hundreds of updates.
  Two restores of the *same* checkpoint drift from *each other* by the same amount, so this is
  nondeterministic CUDA convolution backward, not a checkpoint defect.
- **Replay capacity 500,000 does not hold a 1,000,000-step Walker run.** Eviction begins around the
  half-way point and the second half of the run evicts the first, whole episodes at a time.
