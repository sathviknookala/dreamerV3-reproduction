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
Sequences:       batch 16 × length 64 (burn-in prefix counted separately)
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
`sum(p.numel())` over real modules is owed at M3–M4.
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
- No build or test command exists yet. When one does, record it here verbatim and copy-pasteable.
  If the project grows a compiled or generated step, add the rule that a rebuild must precede
  testing — an unrebuilt edit tests the previous binary and passes — and declare inputs as
  dependencies in the build config so an edit triggers a rebuild
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

- **State the shape contract at every module boundary.** The M2 and M3 gates test shape, layout, and
  reset semantics directly; an implicit `(B, T, ...)` vs `(T, B, ...)` convention is the most likely
  source of a silent temporal misalignment.
- **Every `detach()` / stop-gradient placement gets a one-line comment saying *why*.** The M4, M7,
  and M8 gates all turn on gradient routing — prior vs posterior, critic targets, actor baselines —
  and a missing or extra detach produces a model that trains without erroring.

Add further conventions here only when a session would otherwise get them wrong; duplicating what the
code already says creates drift.

## Current Focus

**M1 — environment interface and sequence replay.** M0 closed 2026-09-17; the specification in
[docs/spec.md](docs/spec.md) is implementation-ready. 0 tests, 0 model code, **nothing measured**.

**Two blocking checks come first**, because neither dependency is installed and the GPU is Blackwell
(`sm_120`, compute capability 12.0):

1. A PyTorch build carrying `sm_120` kernels — record `torch.cuda.get_arch_list()` and the exact wheel.
2. `dm_control` + MuJoCo rendering headless under EGL on Python 3.13.

Then M1 proper: pixel rendering, the transition record, sequence replay, and the `is_last` vs
`is_terminal` contract — all **specified** in [spec.md §7](docs/spec.md), so M1 implements and
validates rather than decides. Record the random-policy return floor and environment throughput under
[results/README.md](results/README.md)'s rules.

`Why It Is a Target` in this file stays `TBD` by construction: nothing has been profiled.

## Last Session

**Session 3 — closed M0.** 3 commits.

- **Pinned the reference at `e3f02248693a79dc8b0ebd62c93683888ddaccfe`** after finding the repository
  has two disjoint eras matching the two paper versions: a wholesale rewrite landed two days before
  arXiv v2 was posted, and `size1m` was added later still (2024-12-07). A paper-contemporaneous pin
  would have specified a different model and could not cite the preset this project scales from.
- **Resolved the three ambiguities the PDFs could not settle**, each by code trace with line-anchored
  permalinks: the λ-return bootstraps at `v[t+1]` (v2's `v_t` is a typo); entropy is *subtracted* in
  the minimized actor loss; replay-critic gradients **do** reach the encoder and RSSM, which is why
  v1's "without sharing gradients" sentence was deleted.
- **Wrote the full architecture and objective spec**, including the block-diagonal cell's real
  connectivity, the `contdisc` discount location, the `bounded_normal` policy (no density correction
  needed), LaProp, and a gradient-routing table separating the blocked actor path from the live
  replay-critic path.
- **Two evidence artifacts under `results/m0/`.** The twohot check shows the v1/v2 estimator gap
  (9.05 vs 50 on a case with true mean 50) and that a naive float32 readout returns −1.0 where 0 is
  required; it states explicitly what is *not* established about unbiasedness. The parameter count is
  **derived from the spec** at 686,846 and agreed to the digit with an independent enumeration.
- **Still nothing measured and nothing implemented.** `sm_120` PyTorch, `dm_control` and MuJoCo are
  all verified **absent**; every count, timing and return remains `unmeasured`.

## Known Issues

- **No dependency needed to run anything is installed.** `torch`, `dm_control` and `mujoco` all fail
  to import. The GPU is **Blackwell, compute capability 12.0 (`sm_120`)**, so PyTorch must carry
  `sm_120` kernels (CUDA 12.8+); Python is 3.13.13, which `dm_control` may not support. This pair is
  the largest unvalidated risk in the project and it **blocks M1**.
- **`size1m` is ~0.69M parameters by derivation, not 1M, and the figure is not measured.** It is also
  below the paper's smallest evaluated row (12M), so no result can be compared to a published number.
  The measured count is owed at M3–M4 from `sum(p.numel())`.
- **Specification is not implementation.** [spec.md §11](docs/spec.md) tracks `specified` /
  `implemented` / `validated` separately. Everything is `specified`; nothing is the other two. Do not
  read a completed spec section as working code.
- **Four transcription traps are documented but unguarded** until tests exist: the block-GRU gate
  split, the `BlockLinear` fan-in (2.83× init error if done per-block), the reference's inverted
  `sg(…, skip=)` polarity, and the encoder/decoder image-scaling asymmetry. All are in
  [spec.md](docs/spec.md) and in the hazards list above.
- **No results exist beyond `results/m0/`**, which contains derivations and numerical checks only —
  no measurement of any model. No number may be quoted from anywhere else.

---

At session end, refresh Current Focus / Last Session / Known Issues here — overwrite in place, 3–5
bullets in Last Session on what was actually done, no appending, no changelogs (git log is for
history). Keep this file thin: when a section grows past what a session needs loaded every time,
move the detail into `docs/` and leave a doc-map line saying **when** to read it.
