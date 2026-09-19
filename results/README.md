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
module of [spec.md §4.11](../docs/spec.md) now equal to its derived figure.

**The parameter count is the only figure here that describes the finished agent.** Every other
number was measured on an untrained or deliberately perturbed model: `m6/`, `m7/` and `m8/` all run
at initialization, `m8/`'s update diagnostic uses synthetic `U(−1, 1)` returns rather than
predictions, and `m5/`'s world model saw only uniform-random data.

**No return figure for a trained agent exists**, and nothing has ever stepped the environment with
a learned policy. The random-policy return floor and the environment throughput benchmark are still
deferred to the M9 measurement pass, and the 32.1 mean return recorded in `m5/` is a property of the
collected random-policy data, not that floor.
