# M9 pilot runs — the first trained agents

**The first return figures for a trained agent in this project.** Two full-budget pilot runs at the
development seed 100, H=15, 64×64 pixels, no privileged simulator state. Walker Walk reached
1,000,000 control steps; Cartpole Swingup reached 500,000. Both completed their budget and wrote a
final model checkpoint.

**Both tasks clear the M1 random floor, measured 2026-09-20** (Walker 32.21 ± 4.37, Cartpole
24.17 ± 15.92 — [`../../m1/`](../../m1/)). Walker's final checkpoint is **16.1×** its floor;
Cartpole's is **5.5×**. **M9 still does not close on these runs**, because its gate also requires the
improvement to persist beyond a transient spike: Walker's does, **Cartpole's does not consolidate**.

## What is here

| File | What it is |
|---|---|
| [`evaluations-walker-2026-09-20.jsonl`](evaluations-walker-2026-09-20.jsonl) | 40 evaluations, every 25,000 control steps, 5 episodes each. |
| [`evaluations-cartpole-2026-09-20.jsonl`](evaluations-cartpole-2026-09-20.jsonl) | 20 evaluations, same cadence. |
| [`train-log-walker-2026-09-20.csv`](train-log-walker-2026-09-20.csv) | Per-update diagnostics, 36 columns, 62,187 optimizer steps. |
| [`train-log-cartpole-2026-09-20.csv`](train-log-cartpole-2026-09-20.csv) | The same, 30,937 optimizer steps. |
| [`run-manifest-walker-2026-09-20.json`](run-manifest-walker-2026-09-20.json) | Counters, parameter counts, device, wall clock, replay state at exit. |
| [`run-manifest-cartpole-2026-09-20.json`](run-manifest-cartpole-2026-09-20.json) | The same. |
| [`config-walker-2026-09-20.json`](config-walker-2026-09-20.json) | The resolved `RunConfig` the run executed. |
| [`config-cartpole-2026-09-20.json`](config-cartpole-2026-09-20.json) | The same. |

The checkpoints themselves are not committed — `model-final.pt` is 3.0 MiB and the two rotated
resume checkpoints are ~6.2 GiB each, in the ignored `runs/` tree.

Reproduce with:

```bash
setsid nohup env PYTHONPATH=src .venv/bin/python scripts/m9_train.py --task cartpole --seed 100 --out runs/m9-cartpole-s100 > runs/m9-cartpole-s100.log 2>&1 &
setsid nohup env PYTHONPATH=src .venv/bin/python scripts/m9_train.py --task walker   --seed 100 --out runs/m9-walker-s100   > runs/m9-walker-s100.log   2>&1 &
```

## Run summary

| | Walker Walk | Cartpole Swingup |
|---|---|---|
| control steps | 1,000,000 | 500,000 |
| optimizer steps | 62,187 | 30,937 |
| episodes collected | 1,000 | 500 |
| imagined transitions | 955,192,320 | 475,192,320 |
| wall clock | **20,197.7 s = 5.61 h** | **11,069.5 s = 3.07 h** |
| evaluation share | 642.8 s, 3.18% | 333.0 s, 3.01% |
| peak VRAM | 1999.3 MiB | 1997.2 MiB |
| replay at exit | 500,000 / 500 complete episodes | 500,000 / 500 complete episodes |
| trainable parameters | 686,846 | 685,876 |
| **first 5 evals, mean** | **29.6** | **93.9** |
| **last 5 evals, mean** | **542.7** | **159.9** |
| **best eval** | **554.9 ± 24.5 @ 975K** | **217.2 ± 16.7 @ 350K** |
| **final eval** | **517.1 ± 61.3 @ 1M** | **132.6 ± 27.5 @ 500K** |

**Both runs executed concurrently on the one RTX PRO 4000.** Their wall clocks are therefore *not*
comparable to the solo per-run derivations in [`../README.md`](../README.md) (4.9 h Walker,
2.5 h Cartpole). Walker came in 14% over its solo derivation while sharing the card with Cartpole;
that is a single observation of two contending runs, not a measured solo cost.

Cartpole's parameter counts differ from Walker's only in the action dimension (1 vs 6). Walker
reproduces the committed 570,419 / 50,316 / 66,111 / 686,846 exactly.

## Walker — sustained improvement, consolidating late

Evaluation return, 5 episodes at fixed seeds 2000–2004:

| steps | return | steps | return | steps | return | steps | return |
|---|---|---|---|---|---|---|---|
| 25K | 23.8 ± 12.7 | 275K | 124.1 ± 5.0 | 525K | 351.7 ± 75.0 | 775K | 298.6 ± 42.2 |
| 50K | 15.1 ± 7.4 | 300K | 171.5 ± 20.7 | 550K | 355.1 ± 135.5 | 800K | 258.2 ± 103.6 |
| 75K | 25.1 ± 5.6 | 325K | 186.4 ± 26.4 | 575K | 390.8 ± 86.3 | 825K | 235.3 ± 154.0 |
| 100K | 41.0 ± 5.5 | 350K | 262.6 ± 59.7 | 600K | 457.7 ± 20.0 | 850K | 457.6 ± 62.5 |
| 125K | 43.0 ± 11.7 | 375K | 308.3 ± 58.0 | 625K | 329.8 ± 112.1 | 875K | 483.3 ± 89.1 |
| 150K | 94.0 ± 51.1 | 400K | 409.5 ± 107.0 | 650K | 415.4 ± 96.4 | 900K | 548.2 ± 25.5 |
| 175K | 156.6 ± 21.7 | 425K | 393.9 ± 96.7 | 675K | 261.8 ± 94.5 | 925K | 554.6 ± 26.6 |
| 200K | 171.8 ± 33.3 | 450K | 414.5 ± 114.1 | 700K | 322.7 ± 49.6 | 950K | 538.7 ± 38.1 |
| 225K | 182.7 ± 7.1 | 475K | 396.8 ± 102.7 | 725K | 300.4 ± 61.5 | 975K | 554.9 ± 24.5 |
| 250K | 164.2 ± 28.7 | 500K | 402.4 ± 107.2 | 750K | 301.2 ± 27.2 | 1000K | 517.1 ± 61.3 |

Three regimes. It climbs to ~400 by 400K; **oscillates between 235 and 458 from 400K to 825K** with
`return_std` as high as 154.0; then from 850K steps up and holds 457–555 with `return_std` falling to
24–38 across five consecutive evaluations. That std collapse is the strongest single signal in either
run — the behaviour became reliable across all five initial conditions, not merely high on average.

**The final checkpoint is not the best checkpoint.** 1M reads 517.1 ± 61.3 against 975K's
554.9 ± 24.5. This matters because the project's core hypothesis is specified on the *final*
checkpoint.

Two transitions in the curve are bimodal across initial conditions rather than uniform shifts:
at 150K the five returns are 151.7, 156.6 vs 35.4, 47.5, 78.7; at 350K they are 310.3, 311.5, 312.4
vs 187.0, 192.1. The large `return_std` at those points is the signal, not noise.

The 275K dip (124.1, down from 182.7) coincides with `rec` rising 24.1 → 32.0, dynamics KL
8.15 → 9.70 and `reward_mae` 0.021 → 0.033 in `train-log-walker-2026-09-20.csv`. That is consistent
with self-inflicted distribution shift — a newly walking policy visiting states its replay did not
contain — but the alignment is circumstantial, not a controlled test.

Final-row diagnostics: `policy_retnorm_scale` **25.78** (from its floor of 1.0),
`value_value_mean` **169.62**, `policy_entropy` **+2.78**. The actor did **not** collapse to
`minstd`.

## Cartpole — did not consolidate

| steps | return | steps | return |
|---|---|---|---|
| 25K | 72.4 ± 0.9 | 275K | 57.2 ± 5.9 |
| 50K | 76.9 ± 1.1 | 300K | 78.7 ± 1.4 |
| 75K | 175.1 ± 5.1 | 325K | 76.6 ± 1.3 |
| 100K | 71.9 ± 1.2 | 350K | 217.2 ± 16.7 |
| 125K | 73.1 ± 0.5 | 375K | 172.4 ± 23.8 |
| 150K | 72.8 ± 0.9 | 400K | 192.7 ± 30.8 |
| 175K | 73.5 ± 1.4 | 425K | 138.6 ± 5.3 |
| 200K | 73.5 ± 0.6 | 450K | 190.5 ± 28.2 |
| 225K | 79.2 ± 0.7 | 475K | 144.9 ± 14.8 |
| 250K | 129.2 ± 18.3 | 500K | 132.6 ± 27.5 |

Eight of twenty evaluations sit in a 72–79 band with `return_std` under 1.5. Five episodes from five
different initial conditions landing within ~1 point of each other is the signature of a policy whose
output barely depends on the state.

Three excursions — 175.1 at 75K, 129.2 at 250K, 217.2 at 350K — and the first two collapsed back to
the band. After 350K the run never returns to the band, but decays 217.2 → 172.4 → 192.7 → 138.6 →
190.5 → 144.9 → 132.6, with the last five evaluations trending down.

This is **not** a perception failure: final `rec` is **3.38** against Walker's 23.50, and
`reward_mae` is 0.076. `policy_retnorm_scale` reached 15.52 and `value_value_mean` only 46.79,
against Walker's 25.78 and 169.62.

Cartpole is the *validation* task, chosen as the easier of the two, and it is the one that failed to
consolidate — though it clears its floor throughout, so this is an instability, not a failure to
learn. Whether that is a task-specific exploration problem (swingup pays almost nothing until
the pole is up, so the actor has little advantage signal) or a configuration defect shared with
Walker is **not determined by these runs** — Cartpole inherits Walker's H=15, update ratio 64 and
every other setting unchanged.

## Against the measured random floor

Floors from [`../../m1/random-floor-{walker,cartpole}-2026-09-20.json`](../../m1/): Walker
**32.21 ± 4.37** (best single random episode 47.34), Cartpole **24.17 ± 15.92** (best 63.90).

| | Walker | Cartpole |
|---|---|---|
| evaluations above floor mean | 37 / 40 | **20 / 20** |
| above the best random episode | 35 / 40 | **19 / 20** (only 275K's 57.2 is not) |
| first evaluation above the best random episode | **150K** | **25K** |
| final checkpoint vs floor | 517.1 = **16.1×**, 111 floor-sd above | 132.6 = **5.5×**, 6.8 floor-sd above |
| best evaluation vs floor | 554.9 = **17.2×** | 217.2 = **9.0×** |
| last ten evaluations, mean vs floor | 444.7 = **13.8×** | 140.1 = **5.8×** |

**This reframes Cartpole.** It is not failing to learn — it is above its floor at *every* evaluation
from 25K onward, and even the flat 72–79 band is ~3× the floor. What it fails to do is *consolidate*:
it reaches a modest level almost immediately, holds it for 200K steps, spikes three times and decays.
Walker is the opposite shape — it **starts below its floor** (23.8 and 15.1 at 25K and 50K against a
floor of 32.21) and does not clear the best random episode until 150K, but once it climbs it keeps
climbing and then locks in.

Read Cartpole against `return_max` rather than the mean. Its floor has sd 15.92 with one random
episode at 63.90, so its 72–79 band is only just clear of chance; Walker's floor is tight
(sd 4.37, max 47.34) and the same band would be decisively clear.

## What these runs do not establish

- **The floor is a bound, not a baseline.** Clearing it is necessary and not remotely sufficient.
  The project's comparison point is the pinned author implementation under matched wrappers and data
  budget ([experiment.md](../../../docs/experiment.md)), which has not been run.
- **Every return here is 5 episodes, not 20.** The core hypothesis is specified on 20 evaluation
  episodes at the final checkpoint. These are pilot estimates at a fifth of that resolution.
- **`return_std` is not across-seed variability.** It is the population sd over 5 non-independent
  episodes from **one** training run, at fixed evaluation seeds 2000–2004, with the policy acting on
  the distribution **mean** (`src/dreamer/agent.py:81`, [spec.md](../../../docs/spec.md) §7.8). It
  contains no policy sampling noise and says nothing about seed-to-seed spread. The hypothesis' test
  against an across-seed standard deviation needs training seeds 0/1/2 and is M10 work.
- **One training seed, and a development one.** Seed 100. Final seeds 0–2 stay unused until M10.
- **These are pilots, not final runs.** The configuration was not frozen when they launched, and
  they are not part of the M10 run matrix.
- **The wall clocks are contended.** Two runs shared one GPU.

## These two logs predate the segment counter

[`../README.md`](../README.md) states that `train-log.csv` and `evaluations.jsonl` carry a `segment`
column and gives a dedupe rule keyed on it. **That rule does not apply to the two logs here.** Both
processes started 2026-09-19 20:45:01, and commit `a0bc33b`, which added the counter, landed
22:42:15 — nearly two hours later. The running processes held the earlier code.

So in this directory: `evaluations-*.jsonl` has **no `segment` field**, and `train-log-*.csv` has
**neither `segment` nor `eval_seconds`**. Neither run ever resumed, so no row is duplicated and no
dedupe is needed — read every row as-is. `elapsed_s` still includes evaluation, and the per-evaluation
seconds are in `evaluations-*.jsonl` rather than the CSV.

A resumed run started from here would write the newer schema mid-file, leaving `.segment == null` on
the earlier rows. Any plotting code should tolerate that.
