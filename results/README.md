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

Measured so far: environment and compute checks ([`m1/`](m1/)), and the RSSM + encoder parameter
count with its gate evidence ([`m2m3/`](m2m3/)). No return, timing or VRAM figure for a *training
run* exists; the random-policy floor and environment throughput are deferred to the M9 measurement
pass.
