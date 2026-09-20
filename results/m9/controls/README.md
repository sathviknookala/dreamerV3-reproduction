# M9 qualification controls — four policies on matched initial conditions

**The first controlled comparison of a trained agent in this project.** Each pilot's
`model-final.pt` is evaluated against three reference policies on the *same* 20 initial conditions,
seeds **3000–3019**, 20 complete 1000-step episodes per condition, 160 episodes per task.

Evaluation only: no optimizer, no replay buffer, no checkpoint write, no configuration change. The
trained agent's collected trajectories are saved for the open-loop diagnostic and never enter a
training buffer.

**Both tasks beat every control on every one of the 20 paired episodes.** That is a stronger
statement than clearing the M1 random floor, because the comparison is paired on initial conditions
rather than against a separately-drawn distribution. It does **not** close M9 — see the limits at the
bottom.

## What is here

| File | What it is |
|---|---|
| [`controls-report-walker-2026-09-20.json`](controls-report-walker-2026-09-20.json) | Walker: every return, the summary table, parameter counts, checkpoint sha256, open-loop metrics. |
| [`controls-report-cartpole-2026-09-20.json`](controls-report-cartpole-2026-09-20.json) | The same for Cartpole. |
| [`control-episodes-walker-2026-09-20.jsonl`](control-episodes-walker-2026-09-20.jsonl) | One row per episode as it finished: condition, env seed, policy seed, return, length, elapsed. |
| [`control-episodes-cartpole-2026-09-20.jsonl`](control-episodes-cartpole-2026-09-20.jsonl) | The same for Cartpole. |
| `<task>-seed3000-<condition>.mp4` | Seed 3000 under each of the four conditions, both tasks. |
| `<task>-filmstrip-<n>-ep<seed>-t<start>.png` | Three paired open-loop filmstrips per task: observed frames above their predictions at distances 1, 5, 10, 15, 20, 25, 30. |
| `worker-<task>-2026-09-20.log` | The worker's stdout, including the `device:` line and per-episode returns. |

**The trained trajectories are not committed** — 52 MiB for Walker, 4.2 MiB for Cartpole, in the
ignored `runs/` tree. Regenerate them with the command below.

Reproduce (both workers ran concurrently on the one card):

```bash
setsid nohup env PYTHONPATH=src .venv/bin/python scripts/m9_controls.py --task walker \
  --checkpoint runs/m9-walker-s100/model-final.pt --out runs/m9-controls/<stamp>/walker \
  > runs/m9-controls/<stamp>/walker.log 2>&1 &
```

Checkpoints evaluated, by content hash: Walker `f3e70a158c837614…`, Cartpole `c322cb56487a7d5c…`.

## Control returns

**Walker Walk**, 20 episodes per condition:

| condition | mean | sd | median | min | max |
|---|---|---|---|---|---|
| **trained** | **539.50** | 42.80 | 540.48 | 439.18 | 598.56 |
| untrained | 21.24 | 7.90 | 22.33 | 10.34 | 38.92 |
| zero-action | 20.24 | 7.04 | 22.06 | 10.25 | 39.35 |
| uniform random | 32.50 | 3.64 | 31.64 | 28.04 | 41.40 |

| paired difference | mean | sd | median | min | max | episodes won |
|---|---|---|---|---|---|---|
| trained − untrained | +518.25 | 44.89 | +519.92 | +423.13 | +576.22 | **20/20** |
| trained − zero | +519.26 | 45.98 | +519.97 | +422.83 | +576.21 | **20/20** |
| trained − random | +506.99 | 42.51 | +510.31 | +407.54 | +569.87 | **20/20** |

**Cartpole Swingup**, 20 episodes per condition:

| condition | mean | sd | median | min | max |
|---|---|---|---|---|---|
| **trained** | **153.57** | 29.00 | 145.91 | 86.01 | 230.44 |
| untrained | 0.01 | 0.01 | 0.01 | 0.00 | 0.04 |
| zero-action | 0.01 | 0.01 | 0.00 | 0.00 | 0.05 |
| uniform random | 15.93 | 7.96 | 14.63 | 4.45 | 35.99 |

| paired difference | mean | sd | median | min | max | episodes won |
|---|---|---|---|---|---|---|
| trained − untrained | +153.56 | 29.00 | +145.90 | +85.98 | +230.43 | **20/20** |
| trained − zero | +153.56 | 29.00 | +145.91 | +85.96 | +230.44 | **20/20** |
| trained − random | +137.63 | 29.74 | +129.84 | +81.09 | +216.38 | **20/20** |

**Cartpole's trained policy beats the zero-action policy on all 20 episodes.** That rules out the
reading that it learned nothing and merely idles — it did not consolidate, but it is doing something
the null policies do not.

Runtime: Cartpole **180.0 s**, Walker **183.7 s**, the two concurrent on one RTX PRO 4000.

### Against the 5-episode pilot readings

| | pilot final, 5 ep | controls, 20 ep |
|---|---|---|
| Walker | 517.1 ± 61.3 | **539.50 ± 42.80** |
| Cartpole | 132.6 ± 27.5 | **153.57 ± 29.00** |

Both 5-episode estimates were slightly pessimistic, and Cartpole's true spread is far wider than five
seeds showed — 86.0 to 230.4. Different evaluation seeds (3000–3019 against the pilot's 2000–2004),
so these are different draws from the same policy, not a re-measurement of the same quantity.

### The untrained and zero conditions are one control, not two

The untrained agent is **effectively a zero-action policy**: `pol` initializes at `outscale: 0.01`
with `tanh` on the mean, so its mean action is ≈0 before training. Cartpole shows this starkly
(0.01 against 0.01); Walker's two differ by 1.0 return out of 539. Do not read them as independent
baselines. What the untrained condition *does* establish is that the architecture and the evaluation
path themselves contribute nothing — the return comes from the learned weights.

## Open-loop prediction on the trained trajectories

Context 5 transitions, 30 future steps, 8 latent samples, 4 evenly spaced contexts per episode,
80 contexts per task. Future predictions are conditioned only on the recorded executed actions; no
future observation enters the prediction path.

Reward MAE by prediction distance:

| k | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| **Walker** model | 0.1144 | 0.1112 | 0.1051 | 0.1287 | 0.1847 |
| Walker persistence | 0.0339 | 0.1670 | 0.3395 | 0.4359 | 0.3958 |
| Walker shuffled-action | 0.1154 | 0.1263 | 0.1640 | 0.3288 | 0.4633 |
| Walker target sd | 0.3566 | 0.3463 | 0.3704 | 0.3905 | 0.3738 |
| **Cartpole** model | 0.0526 | 0.0481 | 0.0441 | 0.0545 | 0.0700 |
| Cartpole persistence | 0.0044 | 0.0226 | 0.0470 | 0.0953 | 0.1285 |
| Cartpole shuffled-action | 0.0526 | 0.0481 | 0.0441 | 0.0545 | 0.0700 |
| Cartpole target sd | 0.1122 | 0.1071 | 0.1002 | 0.0885 | 0.0791 |

**Walker's world model is action-conditional.** It beats persistence from k≈5 onward, stays well
under both the constant baseline (~0.34) and the target sd at every distance, and **shuffling the
future actions degrades it 2.5× by k=30** (0.1847 → 0.4633). The predictions depend on which actions
are supplied.

**Cartpole's reward predictions are action-independent on this policy's data.** Shuffled-action error
equals true-action error to four decimals at *every* distance. This is the cleanest mechanical signal
yet for the failure to consolidate — but it is **confounded**: the trained Cartpole policy's actions
are themselves nearly constant, so shuffling them is close to a no-op. This control cannot separate
"the model ignores actions" from "the policy emits no action variation to shuffle." Distinguishing
them needs a diagnostic on data with deliberate action variation, which is not run here.

Persistence beats the model at k=1 on both tasks, and out to k≈10 on Cartpole. Reward is smooth over
one step, so that is expected rather than a defect.

Image MAE: Walker 0.0326 → 0.0608 against persistence 0.0209 → 0.0800, so the model wins from k≈5.
**Cartpole's persistence baseline wins at every distance** (0.0008 → 0.0045 against the model's
0.0072 → 0.0082), because that scene barely moves under a near-constant policy — a stationary
prediction is hard to beat when the truth is nearly stationary.

## What this does not establish

- **Single training seed, and a development one.** Seed 100. Twenty evaluation episodes do not create
  twenty independent training runs; every `sd` above is across initial conditions within one run, not
  across seeds. The core hypothesis is specified against an **across-seed** standard deviation on
  seeds 0/1/2 and remains untested. Seeds 0–2 and 1000–1019 were not touched.
- **M9 does not close on this.** Its clause is improvement *persisting beyond a transient spike*.
  These are end-of-run checkpoint measurements and say nothing about persistence during training;
  [`../pilot/`](../pilot/) holds the curves, and Cartpole's still does not consolidate.
- **The constant-reward baseline is in-sample.** `training_reward_mean` refuses non-train episodes,
  so it is computed from the evaluated trajectories themselves. The report records
  `constant_reward_is_in_sample: true`. It flatters that baseline, and the model beats it anyway.
- **Beating null policies is a low bar.** The project's comparison point is the pinned author
  implementation under matched wrappers and data budget ([experiment.md](../../../docs/experiment.md)),
  which has not been run. A zero-action policy is not a baseline, it is a floor.
- **Nothing here is a horizon result.** Both runs are H=15.
