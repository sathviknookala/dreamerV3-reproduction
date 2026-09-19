# CLAUDE.md — dreamer

**What this project is:** An independent, reduced-scale PyTorch reimplementation of **DreamerV3** — a
model-based RL agent that learns a recurrent latent world model from pixels, rolls that model forward
*in latent space* to produce imagined trajectories, and trains an actor and critic on those imagined
trajectories rather than on real environment steps. The unit under study is one imagined rollout of
horizon `H` and the actor/critic update it feeds. The artifact is a working online learner plus a
controlled experiment: 12 final runs on one RTX Pro 4000 (24 GB) — Walker Walk from 64×64 pixels at
`H ∈ {5, 15, 30}` and Cartpole Swingup at `H = 15`, three training seeds each. Success is two
separable outcomes — **demonstrated visual-control learning at a measured scale** and **a measured
horizon/return/compute relationship** — not a claim of matching published numbers. The comparison
point is the pinned author implementation run under matched wrappers, observation access, data
budget, and closely matched capacity; the random-policy return is a floor, not a baseline. Priorities
are decided by the milestone dependency chain M0–M10 and its validation gates: a failed gate
redirects work to the diagnosis it exposed, and the final experiment does not start until M9 passes.

## Doc map

This file is the always-loaded hub and stays thin. Detail lives in `docs/`.

| Doc | When to read it |
|---|---|
| [docs/spec.md](docs/spec.md) | **Before writing any model code**, and whenever the paper and the reference implementation appear to disagree. The M0 specification, now complete: pinned sources §1, version manifest §2, paper→code→project mapping §3, exact architecture §4, objectives and gradient routing §5, resolved ambiguities §6, environment and counters §7, seeds §8, deviations §9, deferred measurements §10. |
| [docs/milestones.md](docs/milestones.md) | When starting or closing any milestone M0–M9. Purpose, implementation scope, validation gate, and deliverable for each. |
| [docs/config.md](docs/config.md) | Before changing any hyperparameter or quoting a setting. Initial values with their qualification rules. **Not frozen** until the end of M9. |
| [docs/experiment.md](docs/experiment.md) | Before designing a final run or writing any result claim. The M10 contract: run matrix, controls, metrics, and the limits on what may be concluded. |
| [results/README.md](results/README.md) | Before recording a number. Artifact layout and measurement rules. |
| [results/m0/m0-audit-2026-09-17.md](results/m0/m0-audit-2026-09-17.md) | To check whether an M0 requirement is actually discharged, and what is deferred to where. |
| [results/m5/](results/m5/) | Before quoting an open-loop number, a training-run cost, or touching the prediction path. The M5 gate, the open-loop reward-MAE curve against both baselines, and the paired filmstrips. |
| [results/m4/](results/m4/) | Before quoting a parameter count, a loss value, or touching the world model. The M4 gate, the overfit curve, and the measured world-model `sum(p.numel())`. |
| [results/m2m3/](results/m2m3/) | Before touching the RSSM. The M2+M3 gate output and the measured RSSM + encoder parameter count. |
| [results/m1/](results/m1/) | Environment and compute verification: the two check suites, their outputs, and the captured manifest. Re-run them after any dependency change. |

The original `dreamerv3_implementation_plan.md` was split into the four docs above; it is preserved
unmodified at commit `da9a55a` and no longer exists in the tree, so there is one copy of each claim.

## The System

```
obs_t (64×64×3 uint8), a_{t-1}
  → encoder CNN                           → e_t
  → h_t = f(h_{t-1}, z_{t-1}, a_{t-1})    deterministic, 512
  → z_t ~ q(z_t | h_t, e_t)               posterior — real experience
    z_t ~ p(z_t | h_t)                    prior — imagination, no image input
  → s_t = concat(h_t, z_t)                512 + 32×4
  → heads(s_t)                            → image, reward, continue
  → imagine H prior-only steps from replay s_t
  → λ-returns → critic;  policy gradient → actor
  → a_t ~ π(a_t | s_t)
```

1. **Encoder.** CNN, 64×64×3 → `e_t`. Exact layers, activations, normalization, and **measured**
   parameter count are M0 deliverables.
2. **Recurrence.** `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})`, 512 deterministic features, reset at episode
   boundaries (to literal zeros, not a learned state). **The cell is a block-diagonal GRU with 8
   blocks, not a dense GRU(512)** — the only learned cross-block path is a dense 512→64 bottleneck
   broadcast to every block. Connectivity, gate extraction, and the fan-in trap: [spec.md §4.1](docs/spec.md).
   Single-step and sequence interfaces must agree on identical inputs.
3. **Posterior / prior.** 32 categorical variables × 4 classes, straight-through sampling, uniform
   mixing. The posterior sees `(h_t, e_t)`; **the prior-only API takes no image argument** — this is
   structural, not a calling convention.
4. **Model state.** `s_t = concat(h_t, z_t)`. Every head, the actor, and the critic consume this.
5. **Heads.** Image reconstruction, reward, continuation. Reward and critic use `symexp_twohot`:
   255 exponentially spaced bins over ±4.85e8, **no symlog on the target**, and a **mirror-pair
   readout** whose summation order is part of the contract ([spec.md §5.5](docs/spec.md)). Target
   encoding and scalar decoding are tested independently of each other.
6. **World-model loss.** Reconstruction + reward + continuation + separately weighted dynamics KL
   (1.0) and representation KL (0.1) with free bits. **Reduction order is load-bearing:** sum
   categorical KL over the 32 factors *before* the free-nats threshold, then reduce over valid
   batch/time positions. **Free bits = 1 nat for the whole latent (≈0.031/factor), not 1 per
   factor.** Reconstruction is summed over pixels and meaned over positions, so `rec: 1.0` is ~1e4
   against `rew: 1.0` — deliberate; do not "fix" it to a pixel mean.
7. **Transition convention.**
   `(observation_t, action_t, reward_{t+1}, observation_{t+1}, episode_boundary, environment_discount)`.
   Reward and continuation targets accompanying `observation_{t+1}` are supervised from the resulting
   model state. Reset observations that lack a preceding reward target are masked out.
8. **Terminal ≠ time limit.** DMControl can end an episode with a nonzero discount, so `LAST` alone
   must not imply a zero continuation target. The environment discount carries the distinction.
9. **Imagination.** Start from posterior states drawn from replay, excluding invalid and terminal
   starts; then `H` prior-only steps returning `H` actions/rewards/continuations and `H+1` latent
   states. No decoder in this path, no simulator step, no future observation. Decode only for
   inspection.
10. **Critic.** Distributional, bootstrapped λ-returns (λ=0.95) with the bootstrap at **`v[t+1]`**
    (v2's paper equation prints `v_t`; that is a typo — [spec.md §6.1](docs/spec.md)). The imagined
    path bootstraps the **fast** critic; the slow critic is a regularizer only, never a target. The
    **replay critic is included** at weight 0.3 and bootstraps the *imagined* return.
11. **Actor.** **REINFORCE** with a critic baseline — v1's pathwise rule for continuous actions is a
    *different objective* and must not be substituted. Return normalization (5th/95th percentile,
    clamp at 1), continuation weighting, and entropy **subtracted** in the minimized loss at
    η=3e-4. The distribution is a diagonal Gaussian with `tanh` on the **mean only**; the sample is
    unsquashed, so **there is no density correction to apply** ([spec.md §4.6](docs/spec.md)).
12. **Discount lives in the continuation head.** `contdisc`: the head is trained on the soft label
    `(1−is_terminal)·(1−1/333)` and the return discount is then **1**. Applying γ explicitly *as
    well* double-counts it; using a hard 0/1 label with `disc=1` drops it. Both train silently
    ([spec.md §5.4](docs/spec.md)).
13. **Gradient routing.** Actor gradients through imagined dynamics are **blocked** (which is *why*
    REINFORCE is needed); replay-critic gradients into the encoder and RSSM are **live**. Full table:
    [spec.md §5.9](docs/spec.md).
14. **Online loop.** During real interaction, update the posterior from the current image **before**
    selecting an action. The environment policy uses the learned latent state directly; imagination
    supplies training experience only.

Where a step is conventional but not required: the vector-observation path is a development aid for
isolating control bugs and must stay separate from final pixel results. Image decoding is required
for M4/M5 diagnostics but is deliberately absent from the behavior-training path.

## Why It Is a Target

**TBD — nothing has been profiled, so this project has no cost model and must not state one.**

M9 step 3 produces this section: measured shares for collection, rendering, replay transfer,
world-model updates, and behavior updates, plus representative H=30 peak memory and steady-state
cost estimates including evaluation overhead. Until that run exists and its artifact is committed
under `results/`, every cost statement here would be a guess. Add the measurement date to this
heading when it is filled.

Compute cost is not incidental — it is one of the three response variables in the horizon experiment,
alongside return and prediction error.

## Target Regime / Constraints

The specialization *is* the project. Values below are starting choices with qualification rules in
[docs/config.md](docs/config.md); **none are frozen until the end of M9.**

```
Hardware:        1× RTX Pro 4000, 24 GB VRAM; Ryzen 7950X; 128 GB system RAM
Framework:       PyTorch, eager execution (compile/AMP only after correctness checks)
Observation:     64×64 RGB, identical preprocessing in training and evaluation
Tasks:           DMControl Walker Walk (primary); Cartpole Swingup (validation)
Agents:          one trained separately per task — no shared weights, no transfer
Actions:         bounded continuous, task-specific dimension
Deterministic:   512 features
Stochastic:      32 categorical variables × 4 classes
Sequences:       B=16 × T=64 loss-bearing + P=5 burn-in → 69 transitions / 70 obs per sample
Imagination:     H=15 default; Walker comparison at H ∈ {5, 15, 30}
Discount:        γ=0.997, λ=0.95
Collection:      1 environment, action repeat 1
Replay:          CPU-resident uint8 frames, capacity 500,000 transitions
Update ratio:    64 replay training positions per collected transition
Warm-up:         5,000 random agent transitions
Budget:          Walker 1M control steps/run; Cartpole 500K/run
Final runs:      12 — Walker × {5,15,30} × seeds {0,1,2}; Cartpole H=15 × seeds {0,1,2}
```

A general Dreamer implementation must handle discrete and continuous actions, several observation
modalities, variable image sizes, and one shared configuration that works unmodified across Atari,
DMLab, Minecraft, and proprioceptive control. This one may assume bounded continuous actions, a
single 64×64 RGB modality, two DMControl tasks, and a single GPU. That assumption is what makes a
compact 512 / 32×4 state and an eager-mode implementation small enough to actually *diagnose*
component interactions in — which is the point, since the deliverable is a controlled experiment, not
a fast agent. Explicitly **not** goals: cross-domain generality, matching published absolute returns,
kernel optimization, additional simulators or task suites, and any claim of superiority to model-free
RL. These stay out of scope unless the completed experiment exposes a concrete need for one.

## Core Hypothesis

> **A DreamerV3 implementation at 512 deterministic / 32×4 categorical state, trained on 1M control
> steps of Walker Walk from 64×64 pixels with no privileged simulator state, learns visual control:
> final-checkpoint return over 20 evaluation episodes on each of seeds 0, 1, and 2 exceeds the
> random-policy floor recorded in M1 by a margin larger than the across-seed standard deviation.**

**The horizon study is a question, not a hypothesis.** How does imagination horizon affect
real-environment return, multi-step prediction error, and compute cost in a reduced-capacity world
model? No directional prediction is made. Improvement, plateau, decline, and no clear difference at
n=3 are all reportable outcomes; a null result with clear measurement and stated limitations is a
valid deliverable. Stating a direction now would only create pressure to find it.

**What it cannot claim.** Absolute returns and runtime: **TBD — no run exists.** The parameter count
is **derived** at ~0.69M from the specification ([results/m0/](results/m0/)) but **not measured**;
`sum(p.numel())` over the world model is **measured at 570,419** and matches the derivation per
module ([results/m4/](results/m4/)); with `pol` and `val` it closes at 686,846. Owed at M7–M8.
Structurally, two tasks do not establish the paper's cross-domain result. Three training seeds give
limited evidence about variability, and additional evaluation episodes do not create additional
independent training runs. The horizon comparison holds **real data** fixed, not compute — longer
horizons necessarily consume more computation, so no pure horizon effect at fixed compute can be
isolated. `H=5` is a short-horizon condition, **not** a model-free baseline; a "without imagination"
condition would remove the actor's learning mechanism entirely and is not run.

**The bar is the pinned author implementation, not the random-policy floor.** Matched wrappers,
observation access, data budget, and closely matched capacity, with residual differences recorded.
The random policy is a sanity bound any learning agent must clear. A single reference run is a
diagnostic, not a statistical benchmark. Published results from larger models under different
training settings are context only, never the comparison.

## Workflow Rules

### Plan Before Acting
- Enter plan mode for any non-trivial task (3+ steps or architectural decisions)
- If something goes sideways mid-task, stop and re-plan — don't push through
- Write a spec or checklist upfront to reduce ambiguity; verify with the user before implementing

### Subagent Strategy
- Use subagents to keep the main context window clean
- Offload research, exploration, and parallel analysis to subagents
- One focused task per subagent

### Self-Correction Loop
- After any correction: note the pattern so the same mistake doesn't recur
- Ruthlessly iterate on this until mistake rate drops

### Verification Before Done
- Never consider a task complete without demonstrating it works
- Check logs, run tests, or diff behavior when relevant
- **Always use the repo venv — `.venv/bin/python`, Python 3.12.11.** The system `python3` is 3.13
  and **cannot install this project's dependencies** (see Known Issues). There is no compiled or
  generated step, so no rebuild rule is needed yet.
- Environment and compute verification, copy-pasteable:

  ```bash
  .venv/bin/python results/m1/check_torch_compute.py
  ```

  ```bash
  .venv/bin/python results/m1/check_env_render.py
  ```

  Both exit non-zero on any failed check and print a `PASS`/`FAIL` line per check. Re-run both after
  any dependency change. Recreating the environment from scratch:

  ```bash
  uv venv --python $(command -v python3.12) --seed .venv && .venv/bin/python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129 && .venv/bin/python -m pip install -r requirements.txt
  ```
- Unit tests (58), the M1 smoke test (14 checks), the M2+M3 gate (12), the M4 gate (15) and the
  M5 gate (26), verbatim:

  ```bash
  PYTHONPATH=src:tests .venv/bin/python -m unittest test_env test_replay test_collector test_rssm test_world_model test_openloop test_optim
  ```

  ```bash
  .venv/bin/python scripts/m1_replay_smoke.py
  ```

  ```bash
  .venv/bin/python scripts/m2m3_gate.py
  ```

  ```bash
  .venv/bin/python scripts/m4_gate.py
  ```

  The M5 gate needs the collected split and the trained checkpoint, neither of which is committed
  (`data/` and `runs/` are ignored). Rebuild both, ~30 minutes, then gate:

  ```bash
  .venv/bin/python scripts/m5_collect.py && .venv/bin/python scripts/m5_train.py && .venv/bin/python scripts/m5_eval.py --tag 2026-09-18 && .venv/bin/python scripts/m5_gate.py --report results/m5/openloop-2026-09-18.json
  ```
- A test that passes where the bug cannot occur is not a test — confirm it fails without its fix
- **Domain testing hazards for this project** (each is a validation gate that a naive assertion
  passes while the bug survives):
  - A `LAST → continue=0` assertion passes on Atari-style environments and is **wrong** on DMControl,
    which ends episodes with a nonzero discount. Assert on the environment discount, not the
    boundary flag, and include both a time-limit and a true-terminal example.
  - **Falling reconstruction loss is not evidence of a working world model.** Assert on open-loop
    reward MAE at prediction distance ≥ 5 against a training-set-mean predictor, not on pixel error.
  - Single-step and sequence recurrence paths drift apart silently. Assert they agree on identical
    inputs; a shape-only test will not catch it.
  - Imagination accidentally calling the posterior leaks future observations and looks like excellent
    prediction. Assert **structurally** that no observation tensor can reach the prior path.
  - Latent sampling is stochastic: average metrics over multiple latent samples per context rather
    than selecting attractive rollouts, and fix the seed before comparing conditions.
  - **Block-GRU gate extraction:** the 1536-wide projection must be reshaped to `(B, 8, 192)`
    *before* splitting into three gates. A flat `split(x, 3, -1)` gives a silently wrong, still
    trainable model. Assert the index map, not just the shapes.
  - **`symexp_twohot` readout at init:** with zero-init output weights the softmax is uniform and the
    readout must be **exactly 0**. A naive float32 sum over ±4.85e8 bins returns −1.0 instead. Assert
    the value is 0, not that it is small.
  - **Encoder scales `x/255 − 0.5`; the decoder target is `x/255` with no shift.** The asymmetry is
    real. A test that only checks output range passes either way.
  - **A leaking open-loop rollout needs two different detectors.** An `open_loop_predict` that
    accepts a `future_observations` argument is caught by the encoder tripwire and the signature
    check but **not** by the frame-corruption check; a `gather_contexts` that slides one future frame
    into the context window is caught **only** by the corruption check. Both are in the M5 gate, and
    each was confirmed to pass the other's mutant.
  - **TF32 breaks hand-computed fixtures.** Measured on this GPU: max|GPU−CPU| on a 2048² fp32
    matmul is **6.8e-02 with TF32 on** and **1.3e-04 off** — a 500× difference. Set
    `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False`
    in any test asserting against analytic values (M2+M3, M7). `tests/test_rssm.py` sets both.
- Ask: "Would a senior engineer approve this?"
- Before quoting a committed number, check the tree still reproduces it

### Demand Elegance
- For non-trivial changes: pause and ask "is there a more elegant solution?"
- If a fix feels hacky: "Knowing everything I know now, implement the clean version"
- Skip this for simple, obvious fixes — don't over-engineer

### Autonomous Bug Fixing
- Given a bug report: fix it; don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them

### Measurement Discipline
- Never quote a number that is not in a committed artifact under `results/`, and name the file.
  See [results/README.md](results/README.md) for the layout and the full rules
- State what a metric excludes when it is a floor rather than a measurement
- Don't pipe a long run through `grep` — the pipeline reports grep's exit code and can turn a crash
  that lost real rows into an apparent success
- A run that must outlive the session has to be launched detached (`setsid`), not backgrounded
- Check the shared resource (GPU / port / DB / rate limit) is actually free before a timed run;
  another process's residency shows up as an error or as inflated timings, not as a clear message

## Code Style: Comments

- No paragraph-style or multi-line block comments explaining what code does
- Comments only where intent isn't obvious from the code itself (e.g. non-obvious tradeoffs,
  gotchas, why not what)
- Max 1 line per comment; keep it tight
- No section dividers, no docstrings restating the function signature, no "this function does X"
  fluff
- If the code is self-explanatory, leave it uncommented

## Code Style: Python / PyTorch

No code exists yet, so there are no observed conventions to record — the code will be the style
guide. Two rules are already forced by the spec and should hold from the first file:

- **State the shape contract at every module boundary.** The M2+M3 gate tests shape, layout, and
  reset semantics directly; an implicit `(B, T, ...)` vs `(T, B, ...)` convention is the most likely
  source of a silent temporal misalignment.
- **Every `detach()` / stop-gradient placement gets a one-line comment saying *why*.** The M4, M7,
  and M8 gates all turn on gradient routing — prior vs posterior, critic targets, actor baselines —
  and a missing or extra detach produces a model that trains without erroring.

Add further conventions here only when a session would otherwise get them wrong; duplicating what the
code already says creates drift.

## Current Focus

**M6 — latent imagination engine.** M0 closed 2026-09-17; M1, the merged **M2+M3**, **M4** and **M5**
all closed 2026-09-18. The world model is `implemented` and `validated`, and it is now also
**demonstrated to predict**: open-loop reward MAE on held-out episodes beats the training-set-mean
predictor at every distance 1–30 and overtakes last-reward persistence at k=5
([results/m5/](results/m5/)).

What exists: the full world-model training path, `LaProp` (now gate-tested), the episode-split
dataset, and the prior-only open-loop rollout with its evaluation and baselines. What does not:
**actor, critic, λ-returns, and the online loop.**

Next, in order:

1. M6: begin from posterior states drawn from replay, take an action-provider interface with random
   actions, advance through the prior, and return `H` actions/rewards/continuations and `H+1` latent
   states. `RSSM.imagine_step` takes no embed argument — **keep it that way**; M5's structural
   assertions and `encoder_tripwire` are reusable as-is.
2. Exclude invalid and terminal start states, and implement the continuation weighting.
3. Measure H∈{5,15,30} alignment, finiteness and memory. Image decoding stays **out** of the
   behavior-training path; decode only for inspection.
4. Revisit the M5 diagnostics once online learning broadens the replay distribution — M5's own text
   requires it, and the current numbers describe a uniform-random data distribution only.

**Deferred from M1 by decision, not blocked:** random-policy return floor, throughput benchmark,
random-policy video. Scripts are written; they run in the M9 measurement pass. The 32.1 mean return
in [results/m5/](results/m5/) is a property of the collected data, **not** that floor.

`Why It Is a Target` stays `TBD`: the per-stage profile is M9's, and nothing has been profiled. M5
did record the first real training cost — 7500 gradient steps in 1624.9 s, peak 1489.5 MiB — but a
single world-model-only loop is not a per-stage profile.

## Last Session

**Session 8 — implemented and validated M5, open-loop prediction on held-out episodes.**

- **The world model predicts, it does not only reconstruct.** 120 train / 20 held-out random-policy
  Walker episodes, **7500 gradient steps** derived from the §7.7 ratio, LaProp lr 4e-5, 1624.9 s,
  peak 1489.5 MiB. Open-loop reward MAE over 320 contexts × 8 latent samples beats the
  training-set-mean predictor at **every** distance 1–30 (worst ratio 0.908 at k=28) and overtakes
  last-reward persistence at **k=5**, narrowly (1.6 %). Persistence is far stronger at k=1.
  Action shuffling costs 1.099× — measurably action-conditioned, weakly so.
- **Two leak detectors were both necessary, and mutation proved it.** A `future_observations`
  argument is invisible to the frame-corruption check; a one-frame slide inside `gather_contexts` is
  invisible to the tripwire and the signature check. Seven open-loop mutants and six LaProp mutants
  were each confirmed to fail. One mutation initially "survived" only because the patch text did not
  match — mutation scripts now assert the edit applied.
- **The owed LaProp test is discharged.** `tests/test_optim.py`, 8 hand-computed assertions in
  float64 (float32 rounding at 1e-8 swamped the first attempt), covering the LaProp-vs-Adam ordering,
  `eps` outside the sqrt, whole-tensor AGC, the `pmin` floor, warmup, and bias correction.
- **The sandbox silently strips CUDA from detached processes.** `setsid nohup … &` through the
  sandboxed shell fails `torch.cuda.is_available()` and falls back to **CPU without erroring** — the
  first training launch ran 90 s on CPU before it was caught. Long runs must be launched with the
  sandbox disabled, and the device line in the log must be read, not assumed.
- **No deviation from the specification.** `results/m5/` records what the run cannot claim: nothing
  about prediction under a competent policy, no return measurement, and no variability estimate.

## Known Issues

- **Python 3.13 cannot install this project; use `.venv` (3.12.11) for everything.** `dm-control`
  requires `labmaze`, whose latest release (1.0.6) has wheels only to cp312 and otherwise needs bazel.
  The system `python3` is 3.13.13, so a bare `python3 script.py` will fail on imports. Also: this
  machine's `python3.12` is uv-managed with a broken `ensurepip`, so `python -m venv` cannot create
  the environment — `uv venv --seed` does, and pip does the installs.
- **`pytest` is not installed; the suite is `unittest` and needs `PYTHONPATH=src:tests`.** The package
  is not installed into the venv, and `unittest discover` fails because `tests/` has no `__init__.py`
  — name the five modules explicitly, as in the recorded command.
- **The reward head is invisible to testing at initialization.** `outscale: 0.0` zeroes the output
  kernel, so the two-hot loss is exactly `log(255)` regardless of the target and **no gradient
  reaches the encoder, the RSSM, or even the head's own hidden layer**. Any test of reward alignment
  or reward gradient routing must perturb `reward.mlp.out.weight` first, or it passes vacuously.
- **Stop-gradient directions cannot be asserted on parameters.** Both KL terms reach both heads
  through the recurrence. Assert on the posterior/prior **logit tensors**; §5.9's table describes the
  direct path only, and milestones.md's "accounting for shared recurrent parameters" is the caveat.
- **A `torch.Generator` must be created on the same device as the model.** Passing a CPU generator to
  a CUDA model raises at `torch.multinomial`. §8.1's stream separation must respect this.
- **torch must come from the `cu129` index, not PyPI.** The GPU is Blackwell `sm_120` and the driver
  is 575.64.03; CUDA 13.0 wheels need driver ≥ 580, so the newest PyPI default build is excluded.
- **`size1m` is 570,419 parameters for the world model and 686,846 for the full agent by
  derivation.** Far below the paper's smallest evaluated row (12M), so no result can be compared to a
  published number. `pol` and `val` are still derived, not measured.
- **All four documented transcription traps are now guarded.** Gate split, `BlockLinear` fan-in,
  encoder/decoder scaling asymmetry, and the `sg(…, skip=)` polarity — the last via the KL
  stop-gradient test. Each was mutation-verified to fail without its fix.
- **A detached process launched through the sandboxed shell loses CUDA and falls back to CPU
  silently.** `torch.cuda.is_available()` returns `False` with only a `UserWarning`; the run then
  trains at roughly 1/30 speed and nothing else looks wrong. Launch long runs with the sandbox
  disabled and **read the `device:` line in the log** before walking away.
- **No measurement of a trained *agent* exists — only of a trained world model.** `results/m5/`
  holds open-loop prediction error and the first real training cost. There is still **no return, no
  behaviour, and no per-stage profile**, because the actor, critic and online loop do not exist.
- **Every M5 number describes a uniform-random data distribution.** M5's own text requires the
  diagnostics to be revisited once online learning broadens replay. Do not carry these numbers
  forward as properties of a trained agent's world model.

---

At session end, refresh Current Focus / Last Session / Known Issues here — overwrite in place, 3–5
bullets in Last Session on what was actually done, no appending, no changelogs (git log is for
history). Keep this file thin: when a section grows past what a session needs loaded every time,
move the detail into `docs/` and leave a doc-map line saying **when** to read it.
