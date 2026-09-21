# M9 confirmation runs — Walker at two more seeds

**Two full-budget Walker Walk runs at development seeds 101 and 102**, H=15, 64×64 pixels, no
privileged simulator state, 1,000,000 control steps each. Both completed their budget, exited 0 and
wrote a final model checkpoint. Together with the seed-100 pilot ([`../pilot/`](../pilot/)) the
project now has **three Walker runs under one configuration**.

Two results, which are separable:

- **The pilot's learning result replicates.** Both new seeds clear the M1 random floor
  (32.21 ± 4.37, [`../../m1/`](../../m1/)) and end far above it — 466.90 and 516.58, i.e. 14.5× and
  16.0×. Final checkpoints across the three seeds are **500.18 with an across-seed sd of 28.82**.
- **These are the project's first measured *solo* full-budget runtimes: 5.036 h and 5.008 h.**
  Every earlier Walker figure was either a derivation (4.9 h) or contended (the pilot's 5.61 h,
  sharing the card with Cartpole). Solo cost is **≈5.02 h, 2.5% above the derivation**, which
  confirms that the pilot's 14% overshoot was contention.

**Consolidation is not uniform across seeds, and that is the finding to carry forward.** s102
consolidates like the pilot did; **s101 does not** — its final checkpoint reads 466.90 ± **222.55**,
five evaluation episodes disagreeing violently, and its last four evaluations span 283 points. So
M9's "improvement persisting beyond a transient spike" clause is satisfied by two of three Walker
seeds, not three.

## What is here

| File | What it is |
|---|---|
| [`evaluations-walker-s101-2026-09-21.jsonl`](evaluations-walker-s101-2026-09-21.jsonl) | 40 evaluations, every 25,000 control steps, 5 episodes each at fixed seeds 2000–2004. |
| [`evaluations-walker-s102-2026-09-21.jsonl`](evaluations-walker-s102-2026-09-21.jsonl) | The same. |
| [`train-log-walker-s101-2026-09-21.csv`](train-log-walker-s101-2026-09-21.csv) | Per-update diagnostics, 38 columns, 1,243 rows, 62,187 optimizer steps. |
| [`train-log-walker-s102-2026-09-21.csv`](train-log-walker-s102-2026-09-21.csv) | The same. |
| [`run-manifest-walker-s101-2026-09-21.json`](run-manifest-walker-s101-2026-09-21.json) | Counters, parameter counts, device, wall clock, replay state at exit. |
| [`run-manifest-walker-s102-2026-09-21.json`](run-manifest-walker-s102-2026-09-21.json) | The same. |
| [`config-walker-s101-2026-09-21.json`](config-walker-s101-2026-09-21.json) | The resolved `RunConfig` the run executed. |
| [`config-walker-s102-2026-09-21.json`](config-walker-s102-2026-09-21.json) | The same. |

Checkpoints and the 40 evaluation videos per run are not committed — they live in the ignored
`runs/walker-confirmation-s{101,102}/` trees.

**Unlike the pilot logs, these carry the `segment` column.** Both runs show `segment` 0 on every row
and neither resumed, so no row is duplicated, but the dedupe rule in [`../README.md`](../README.md)
applies to this schema as written.

## Reproduce

Both runs came from `configs/walker-h15-frozen.json`, committed at `1cb6bea` and byte-identical to
the pilot's resolved config; the only difference from the file is the seed. They ran
**sequentially**, one at a time on the one RTX PRO 4000:

```bash
for seed in 101 102; do
  env PYTHONPATH=src .venv/bin/python -u scripts/m9_train.py --config configs/walker-h15-frozen.json --task walker --seed "$seed" --device cuda --out "runs/walker-confirmation-s${seed}" > "runs/walker-confirmation-s${seed}.log" 2>&1
done
```

`--task walker` is **required** even though the config names the task: `scripts/m9_train.py`
defaults `--task` to `"cartpole"` rather than `None` and `build_config` overrides every non-`None`
argument, so omitting it runs Cartpole at a 500,000 budget instead.

## Run summary

| | s101 | s102 | s100 (pilot) |
|---|---|---|---|
| control steps | 1,000,000 | 1,000,000 | 1,000,000 |
| optimizer steps | 62,187 | 62,187 | 62,187 |
| episodes collected | 1,000 | 1,000 | 1,000 |
| train positions | 63,679,488 | 63,679,488 | 63,679,488 |
| imagined transitions | 955,192,320 | 955,192,320 | 955,192,320 |
| scheduler remainder | 512 | 512 | 512 |
| **wall clock** | **18,128.4 s = 5.036 h** | **18,027.7 s = 5.008 h** | 20,197.7 s = 5.61 h |
| evaluation share | 622.5 s, 3.43% | 623.6 s, 3.46% | 642.8 s, 3.18% |
| peak VRAM | 1999.3 MiB | 1999.3 MiB | 1999.3 MiB |
| replay at exit | 500,000 / 500 episodes | 500,000 / 500 episodes | 500,000 / 500 episodes |
| trainable parameters | 686,846 | 686,846 | 686,846 |
| **first 5 evals, mean** | 35.9 | 40.1 | 29.6 |
| **last 5 evals, mean** | **355.6** | **480.7** | 542.7 |
| **best eval** | 466.9 ± 222.5 @ 1M | **522.6 ± 30.0 @ 975K** | 554.9 ± 24.5 @ 975K |
| **final eval** | **466.9 ± 222.5** | **516.6 ± 38.7** | 517.1 ± 61.3 |

Every counter that should be a deterministic function of the budget is identical across all three
seeds — optimizer steps, train positions, imagined transitions, episodes and the scheduler's
leftover credit. Only the returns and the wall clock differ.

Health, both runs: 1,243 log rows, **one segment** (no crash, no resume), `realized_train_ratio`
**64.0 on every row**, peak VRAM flat at 1999.3 MiB, **zero nan/inf**, `device: cuda`,
`tf32_matmul: False`.

## Evaluation curves

Return, 5 episodes at fixed seeds 2000–2004, policy acting on the distribution mean.

| steps | s101 | s102 | steps | s101 | s102 |
|---|---|---|---|---|---|
| 25K | 20.8 ± 8.5 | 24.9 ± 10.5 | 525K | 294.1 ± 144.9 | 383.0 ± 107.2 |
| 50K | 21.1 ± 8.4 | 25.0 ± 15.7 | 550K | 242.7 ± 138.4 | 321.6 ± 130.0 |
| 75K | 32.1 ± 7.7 | 35.0 ± 7.3 | 575K | 235.1 ± 131.3 | 329.3 ± 86.7 |
| 100K | 53.0 ± 7.1 | 57.0 ± 14.9 | 600K | 364.0 ± 22.3 | 211.5 ± 97.4 |
| 125K | 52.5 ± 7.3 | 58.6 ± 18.3 | 625K | 349.2 ± 146.6 | 283.3 ± 66.7 |
| 150K | 69.3 ± 5.0 | 59.9 ± 12.3 | 650K | 241.2 ± 92.6 | 275.0 ± 125.1 |
| 175K | 86.2 ± 10.6 | 76.7 ± 25.3 | 675K | 235.6 ± 149.8 | 206.7 ± 103.7 |
| 200K | 182.1 ± 82.2 | 129.9 ± 41.2 | 700K | 341.9 ± 23.9 | 439.5 ± 38.9 |
| 225K | 162.9 ± 69.4 | 101.2 ± 16.6 | 725K | 457.5 ± 17.6 | 316.3 ± 137.5 |
| 250K | 107.2 ± 28.8 | 226.0 ± 36.0 | 750K | 359.4 ± 37.6 | 250.0 ± 135.2 |
| 275K | 215.2 ± 110.0 | 179.6 ± 9.2 | 775K | 386.6 ± 9.3 | 306.6 ± 194.5 |
| 300K | 224.1 ± 93.2 | 189.9 ± 9.3 | 800K | 234.9 ± 149.9 | 284.5 ± 188.3 |
| 325K | 231.0 ± 90.9 | 197.9 ± 35.9 | 825K | 346.4 ± 154.0 | 304.2 ± 233.7 |
| 350K | 256.5 ± 92.1 | 220.0 ± 34.2 | 850K | 433.9 ± 64.7 | 199.6 ± 191.6 |
| 375K | 236.6 ± 59.7 | 258.0 ± 57.6 | 875K | 331.6 ± 135.3 | 347.1 ± 154.2 |
| 400K | 241.9 ± 49.5 | 248.1 ± 94.4 | 900K | 309.2 ± 104.4 | 476.5 ± 92.1 |
| 425K | 302.0 ± 30.3 | 273.9 ± 88.1 | 925K | 183.6 ± 79.6 | 443.8 ± 40.8 |
| 450K | 251.9 ± 70.0 | 295.5 ± 78.5 | 950K | 433.4 ± 64.5 | 443.8 ± 74.4 |
| 475K | 260.0 ± 136.3 | 349.3 ± 124.3 | 975K | 385.1 ± 184.6 | **522.6 ± 30.0** |
| 500K | 377.4 ± 129.8 | 380.7 ± 171.7 | 1000K | **466.9 ± 222.5** | **516.6 ± 38.7** |

Both seeds reproduce the pilot's overall shape — a climb through ~400 by 500K, then a long noisy
plateau — but **both clear the best single random episode (47.34) at 100K, where the pilot needed
150K**, and both start *above* the floor rather than below it.

## Consolidation differs by seed

The pilot's clearest signal was `return_std` collapsing to 24–38 across five consecutive late
evaluations: the behaviour became reliable across all five initial conditions, not merely high on
average. Reading the last four evaluations of each run the same way:

| | last four returns | spread | last four `return_std` |
|---|---|---|---|
| s100 (pilot) | 554.6, 538.7, 554.9, 517.1 | **37.9** | 26.6, 38.1, 24.5, 61.3 |
| **s101** | 183.6, 433.4, 385.1, 466.9 | **283.3** | 79.6, 64.5, 184.6, **222.5** |
| **s102** | 443.8, 443.8, 522.6, 516.6 | **78.8** | 40.8, 74.4, 30.0, 38.7 |

s102 consolidates; s101 does not. s101's high final figure is partly a favourable draw — it is the
only one of the three whose peak evaluation *is* its final evaluation, and it carries the largest
episode spread in the whole table. A run whose five episodes range that widely has not locked in a
behaviour, whatever its mean.

This is worth separating from the Cartpole problem in [`../pilot/`](../pilot/). Cartpole never
consolidated at all; s101 reaches a high level and holds it unreliably. Whether they share a cause
is not determined by these runs.

## Final-row diagnostics

| | s101 | s102 | s100 (pilot) |
|---|---|---|---|
| `rec` | 24.99 | 22.72 | 23.50 |
| `reward_mae` | 0.05 | 0.04 | 0.05 |
| dynamics KL | 9.38 | 9.25 | 9.57 |
| `policy_entropy` | **+5.29** | **+5.33** | +2.78 |
| `policy_retnorm_scale` | **49.65** | **40.72** | 25.78 |
| `value_value_mean` | 137.71 | 150.34 | 169.62 |

No actor collapsed to `minstd` — all three entropies are positive, and the two new seeds end with
roughly twice the pilot's entropy. Both also end with a return-normalization scale well above the
pilot's, so `S` is far from its floor of 1 in every run.

## Across the three seeds

| | final checkpoint |
|---|---|
| s100 | 517.06 ± 61.28 |
| s101 | 466.90 ± 222.55 |
| s102 | 516.58 ± 38.69 |
| **mean** | **500.18** |
| **across-seed sd** | **28.82** |

Margin over the floor is 467.97, which is **16.2× that across-seed sd**.

## What these runs do not establish

- **This is not the core hypothesis test, despite having the same shape.** The hypothesis is
  specified on **20 evaluation episodes at the final checkpoint on training seeds 0, 1 and 2**.
  These are **5 episodes on development seeds 100–102**. Seeds 0–2 and evaluation seeds 1000–1019
  remain unused until M10. The 16.2× figure above must not be quoted as the hypothesis being met.
- **`return_std` still is not across-seed variability.** It is the spread over 5 non-independent
  episodes from one run at fixed evaluation seeds, with the policy acting on the distribution mean —
  no policy sampling noise. The 28.82 in the table above *is* an across-seed sd, but over n=3
  development seeds.
- **Three seeds give limited evidence about variability**, and s101 shows the spread is not small.
- **The configuration was not frozen when these launched.** `docs/config.md` is still unfrozen, so
  these are not M10 runs and are not part of its matrix.
- **One task.** Cartpole was not re-run at these seeds, and its consolidation failure
  ([`../pilot/`](../pilot/)) is untouched by anything here.
- **The floor is a bound, not a baseline.** The comparison point remains the pinned author
  implementation under matched wrappers and data budget, which has not been run.
- **The runtimes are solo but single.** One measurement each, on an otherwise idle card; they are
  not a distribution.
