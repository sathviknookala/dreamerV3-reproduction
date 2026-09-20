# Results

**Read before recording a number.** Every quoted number must live in a committed artifact under this
directory, and the quoting text must name the file. A number that exists only in a terminal
scrollback, a chat message, or a notebook that was not committed does not exist.

## Layout

```
results/
  <milestone>/<what>-<date>.{json,csv,md}    diagnostics, profiles, gate evidence
  runs/<task>-h<H>-seed<N>/                  final run: manifest, raw log, checkpoint ref
  figures/                                   regenerated from raw logs, never hand-edited
```

## Rules

- **State what a metric excludes** when it is a floor rather than a measurement.
- **Never pipe a long run through `grep`** — the pipeline reports grep's exit code and can turn a
  crash that lost real rows into an apparent success. Write the full log, then filter it.
- **Launch detached** (`setsid`) any run that must outlive the session.
- **Check the GPU is actually free** before a timed run. Another process's residency shows up as
  inflated timings, not as a clear message.
- **Before re-quoting a committed number, check the tree still reproduces it.**
- Figures regenerate from raw logs. If a figure cannot be regenerated, it is not a result.

Measured so far: environment and compute checks ([`m1/`](m1/)), the RSSM + encoder parameter count
([`m2m3/`](m2m3/)), the world-model parameter count, objective gate and fixed-subset overfit
([`m4/`](m4/)), and the **first real world-model training run** with its open-loop prediction
evaluation on a held-out split ([`m5/`](m5/)) — 7500 gradient steps, 1624.9 s, peak 1489.5 MiB —
the **per-horizon imagination cost** ([`m6/`](m6/)): 1024 rollouts at H = 5 / 15 / 30 cost
3.8 / 10.6 / 20.9 ms and peak 204 / 463 / 858 MiB on an **untrained** model with **random** actions,
which is a feasibility measurement, not the M9 steady-state profile; the **critic's return
arithmetic and parameter count** ([`m7/`](m7/)): `val` measured at 66,111, the three readings of γ
separated at 14.688751 / 14.386389 / 15.000000, and a critic fit from 0.0000 to 0.3907 against a
fixed target mean of 0.4029; and the **actor's arithmetic and parameter count** ([`m8/`](m8/)):
`pol` measured at 50,316, closing the trainable agent total at **686,846 measured** with every
module of [spec.md §4.11](../docs/spec.md) now equal to its derived figure; and the **online loop's
implementation validation and per-stage cost** ([`m9/`](m9/)): 75/75 integration checks on each of
Cartpole and Walker, a complete update costing 206.0 / 221.2 / 236.8 ms at H = 5 / 15 / 30 with
**H=30 peak VRAM 2373.8 MiB** in a steady-state update, collection at 2.796 ms/control step, and a
derived **≈52 GPU-hours** for the 12-run final campaign.

And the **first trained agents** ([`m9/pilot/`](m9/pilot/)): two full-budget runs at development
seed 100, H=15, from 64×64 pixels. Walker Walk over 1,000,000 control steps in 5.61 h goes from a
29.6 opening five-evaluation mean to **542.7 over its last five**, consolidating at 457–555 after
850K with `return_std` down to 24–38; its final checkpoint reads **517.1 ± 61.3** against a
**554.9 ± 24.5** peak at 975K. Cartpole Swingup over 500,000 steps in 3.07 h **does not
consolidate** — it sits in a 72–79 band for eight of twenty evaluations, peaks at **217.2 ± 16.7**
at 350K, and decays to **132.6 ± 27.5** at budget.

**Apart from the parameter count and `m9/pilot/`, every number here was measured on an untrained or
deliberately perturbed model:** `m6/`, `m7/` and `m8/` all run at initialization, `m8/`'s update
diagnostic uses synthetic `U(−1, 1)` returns rather than predictions, `m5/`'s world model saw only
uniform-random data, and `m9/`'s gate returns describe a ~1,200-step agent.

**The random-policy return floor is measured** ([`m1/`](m1/), 2026-09-20): Walker
**32.21 ± 4.37**, Cartpole **24.17 ± 15.92**, 20 complete episodes each. **Both pilots clear it** —
Walker's final checkpoint 16.1× its floor, Cartpole's 5.5×, with all 20 Cartpole evaluations above
the floor mean and 19 above the best single random episode. **M9 still stays open**, because its
clause also requires improvement persisting beyond a transient spike and Cartpole's does not
consolidate. **Environment throughput (§10-6) remains the last M1 deferral outstanding.** The 32.110
mean return recorded in `m5/` is a property of that milestone's collected data, not this floor — but
it is uniform-random Walker at seed 100, and the floor is uniform-random Walker at seed 900, so their
agreement to 0.3% (32.110 vs 32.205) is an independent corroboration that both measure the same
distribution.

And the **qualification controls** ([`m9/controls/`](m9/controls/), 2026-09-20): each pilot's final
checkpoint against an untrained agent, a zero-action policy and a uniform-random policy on the *same*
20 initial conditions (seeds 3000–3019), 160 episodes per task in ~180 s. **Walker 539.50 ± 42.80**
and **Cartpole 153.57 ± 29.00** beat every control on **20 of 20 paired episodes**. The open-loop
diagnostic on those trajectories shows Walker's world model is action-conditional (shuffling future
actions degrades reward MAE 2.5× by k=30) while **Cartpole's reward predictions are
action-independent** — confounded, because that policy's actions barely vary.

**Every pilot return is 5 episodes from one training seed**, at fixed evaluation seeds 2000–2004
with the policy acting on the distribution mean. The core hypothesis is specified on 20 episodes at
the final checkpoint across training seeds 0/1/2; that is M10 work and none of it is done.
