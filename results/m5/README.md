# M5 — open-loop prediction evaluation

**Read before quoting an open-loop number, or before touching the prediction path.** Gate passed
2026-09-18, 26/26: [`gate-2026-09-18.txt`](gate-2026-09-18.txt).

## What was run

| | |
|---|---|
| Data | Walker Walk, 64×64 pixels, uniform-random policy, development seed **100** |
| Episodes | **140 × 1000 transitions** — **120 train / 20 held out**, disjoint by episode and by SHA-256 |
| Training | B=16, P=5, T=64; LaProp lr 4e-5, warmup 1000, β₂=0.999, AGC 0.3 ([spec.md §5.8](../../docs/spec.md)) |
| Gradient steps | **7500**, derived from the specified ratio: 120 000 × 64 ÷ 1024 ([spec.md §7.7](../../docs/spec.md)) |
| Cost | **1624.9 s** (4.62 step/s), peak **1489.5 MiB** VRAM, 570 419 parameters |
| Evaluation | 5 context transitions (6 frames), horizon **30**, 16 contexts × 20 episodes = **320 contexts**, **8 latent samples** each |

Artifacts: [`train-manifest-2026-09-18.json`](train-manifest-2026-09-18.json),
[`train-log-2026-09-18.csv`](train-log-2026-09-18.csv),
[`openloop-2026-09-18.json`](openloop-2026-09-18.json),
[`openloop-by-distance-2026-09-18.csv`](openloop-by-distance-2026-09-18.csv),
[`frames-2026-09-18/`](frames-2026-09-18/).

The checkpoint is **not committed** (`runs/` is ignored); it is identified by
`sha256 1b86998c754d0e52ff7096c4d74ee3ce82590186d20f2578768d844a9843f9cc` and is reproducible from
`scripts/m5_collect.py` → `scripts/m5_train.py` at the recorded seed.

Reproduce:

```bash
.venv/bin/python scripts/m5_collect.py
.venv/bin/python scripts/m5_train.py
.venv/bin/python scripts/m5_eval.py --tag 2026-09-18
.venv/bin/python scripts/m5_gate.py --report results/m5/openloop-2026-09-18.json
```

## Result — open-loop reward MAE by prediction distance

Held-out episodes. Rollouts start from a posterior state and use the **prior only** thereafter;
the recorded actions are the sole input after the cutoff.

| k | model | training-mean baseline | last-reward persistence | shuffled actions | model ÷ mean |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.015872 | 0.021439 | **0.005213** | 0.015935 | 0.740 |
| 5 | **0.015265** | 0.020740 | 0.015510 | 0.016938 | 0.736 |
| 15 | **0.013938** | 0.015810 | 0.022210 | 0.015037 | 0.882 |
| 30 | **0.013482** | 0.015690 | 0.024658 | 0.015425 | 0.859 |

**The model beats the training-set-mean predictor at every distance 1–30**, worst ratio 0.908 at
k=28. Sample-to-sample spread of the MAE is 1.44e-4 to 5.09e-4 across the eight latent samples —
an order of magnitude below the model-versus-baseline gaps, so the ordering is not sampling noise.

**Persistence wins at k=1 and loses from k≈5.** The reward is strongly autocorrelated, so repeating
the last observed reward is a very strong one-step predictor (0.0052 against the model's 0.0159) and
degrades steadily with distance (0.0247 at k=30). The crossover is at **k=5**, where the margin is
narrow: 0.015265 against 0.015510, a 1.6 % improvement. From k=6 onward the margin widens and stays
open. **The honest reading is that this model earns its keep at medium and long distance, not at
k=1.**

### Action sensitivity

Permuting the future actions in time raises the mean MAE from 0.014051 to 0.015437 — **1.099×**.
The predictions are measurably action-conditioned, but **only weakly so at this training budget**.
This is a sensitivity diagnostic, not a causal control-performance result.

### Secondary diagnostics

Decoded-image MAE (per pixel, [0,1]) against last-frame persistence: **0.033 vs 0.012 at k=1**,
0.039 vs 0.033 at k=5, 0.045 vs 0.046 at k=15, 0.049 vs 0.051 at k=30. The same crossover shape as
the reward, later. The paired filmstrips in [`frames-2026-09-18/`](frames-2026-09-18/) show why: the open-loop frames keep
the ground plane and a plausible body pose but are visibly blurred and drift toward an average pose.
**Image quality is not evidence here, and was not used as any.**

Continuation MAE is degenerate on these tasks and is reported only for completeness: Walker Walk ends
by time limit, so `is_terminal` is false for every transition and the target is the constant
0.996997 ([spec.md §7.3](../../docs/spec.md)).

The filmstrip selection rule is **positional** — evenly spaced over the context list — never by error.

## What the gate establishes

26 checks in [`gate-2026-09-18.txt`](gate-2026-09-18.txt), covering the eight requirements:

1. **Observations reach the context only.** The encoder is called exactly once per rollout, on a
   tensor of `C+1 = 6` frames; 240 frames for 40 contexts, and no more.
2. **The rollout structurally cannot consume future frames.** `RSSM.imagine`/`imagine_step` and
   `open_loop_predict` take no image or embedding argument, and **randomizing every source frame
   after the cutoff leaves the predictions bit-identical** while changing the targets.
3. **Actions are aligned with the transitions they cause**, and rolling them moves the rollout.
4. **Distance k pairs state `cutoff+k` with `rewards[cutoff+k-1]`**, verified against the real
   episode arrays for every context.
5. **Reproducible:** the same seed reproduces the metrics exactly; a different seed does not; eight
   samples with nonzero spread.
6. **No split leak:** train and held-out content are disjoint by SHA-256, and the checkpoint records
   exactly the 120 training hashes and none of the 20 held-out ones.
7. **Baselines use training data only:** `training_reward_mean` refuses a held-out episode, and the
   reported constant equals the training-split mean.
8. **A leaking rollout is detected:** a posterior pass over the future frames encodes 1440 frames
   instead of 240 and trips the tripwire.

> On check 7, the numeric guard "the constant is not the held-out mean" is **weak evidence here** —
> both splits are random-policy Walker, so the two means differ by only 3.5e-5 (0.032110 vs
> 0.032075). The provenance guarantee comes from the type-level refusal and the hash disjointness,
> not from that number.

Two complementary leak detectors were confirmed necessary by mutation: an `open_loop_predict` that
takes a `future_observations` argument is caught by the tripwire and the signature checks but
**not** by the corruption check; a `gather_contexts` that slides one future frame into the context
window is caught **only** by the corruption check.

## What this does not establish

- **This is a random-policy world model.** The replay distribution is uniform-random Walker, so these
  numbers say nothing about prediction under a competent policy. M5's own text requires revisiting
  these diagnostics once online learning broadens the distribution.
- **Not the M10 diagnostic corpus.** That corpus uses reserved seeds 500–519 and is frozen after M9
  ([spec.md §8](../../docs/spec.md)); it is untouched. These held-out episodes come from the same
  development seed 100 as the training split.
- **Not a return measurement.** The collected random-policy episode return (train mean 32.110,
  held-out mean 32.075) is a **property of the collected data**, not the M1 random-policy return
  floor, which has its own 20-episode protocol and is still deferred to the M9 measurement pass.
- **One training run, one seed, one task.** No variability estimate, and no comparison to any
  published number — the model is 570 419 parameters against the paper's smallest evaluated row of
  12M.
- **7500 gradient steps is a small budget.** The weak action sensitivity (1.099×) is consistent with
  either a genuinely weak action-conditioned prior or simple undertraining, and this run cannot
  separate the two.
