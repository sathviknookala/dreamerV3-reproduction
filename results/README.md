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

**The parameter count is the only figure here that describes the finished agent.** Every other
number was measured on an untrained or deliberately perturbed model: `m6/`, `m7/` and `m8/` all run
at initialization, `m8/`'s update diagnostic uses synthetic `U(−1, 1)` returns rather than
predictions, and `m5/`'s world model saw only uniform-random data.

**No return figure for a trained agent exists.** A learned policy now steps the environment — the
M9 gate collects with one and evaluates with one — but no agent has been trained past a smoke-test
budget, so `m9/`'s evaluation returns describe a ~1,200-step agent and are a smoke signal, not a
result. The random-policy return floor and the environment throughput benchmark are **still
unrecorded**; `scripts/m1_random_floor.py` has been repaired (it was a copy of the throughput script
and never accumulated a reward) and runs in the pilot pass. The 32.1 mean return recorded in `m5/`
is a property of the collected random-policy data, not that floor.
