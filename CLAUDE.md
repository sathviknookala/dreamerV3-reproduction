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
| [docs/spec.md](docs/spec.md) | **Before writing any model code**, and whenever the paper and the reference implementation appear to disagree. The M0 frozen specification: pinned sources, paper-to-code mapping, preserved V3 methods, loss reductions, deviation table. **Currently a skeleton — M0 is not complete.** |
| [docs/milestones.md](docs/milestones.md) | When starting or closing any milestone M0–M9. Purpose, implementation scope, validation gate, and deliverable for each. |
| [docs/config.md](docs/config.md) | Before changing any hyperparameter or quoting a setting. Initial values with their qualification rules. **Not frozen** until the end of M9. |
| [docs/experiment.md](docs/experiment.md) | Before designing a final run or writing any result claim. The M10 contract: run matrix, controls, metrics, and the limits on what may be concluded. |
| [results/README.md](results/README.md) | Before recording a number. Artifact layout and measurement rules. |

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
   boundaries. Single-step and sequence interfaces must agree on identical inputs.
3. **Posterior / prior.** 32 categorical variables × 4 classes, straight-through sampling, uniform
   mixing. The posterior sees `(h_t, e_t)`; **the prior-only API takes no image argument** — this is
   structural, not a calling convention.
4. **Model state.** `s_t = concat(h_t, z_t)`. Every head, the actor, and the critic consume this.
5. **Heads.** Image reconstruction, reward, continuation. Reward uses the transformed distributional
   output chosen in M0; target encoding and scalar decoding are tested independently of each other.
6. **World-model loss.** Reconstruction + reward + continuation + separately weighted dynamics KL and
   representation KL with free bits. **Reduction order is load-bearing:** sum categorical KL over the
   32 factors *before* the free-nats threshold, then reduce over valid batch/time positions.
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
10. **Critic.** Distributional, bootstrapped λ-returns (γ=0.997, λ=0.95), slow-critic regularization,
    v2 replay critic objective preserved — or its omission classified as a deviation in `docs/spec.md`.
11. **Actor.** Bounded continuous policy trained by the **V3** policy-gradient estimator specifically,
    with return normalization, entropy regularization, and continuation weighting. Do not substitute
    an earlier Dreamer variant's gradient rule. Any action transform carries its density correction.
12. **Online loop.** During real interaction, update the posterior from the current image **before**
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

**What it cannot claim.** Absolute returns, parameter counts, and runtime: **TBD — no run exists.**
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

**M0 — freeze the algorithm specification.** 0 tests, 0 artifacts, no code. The project statement,
The System, the constraints, and the Core Hypothesis are now real and sourced from the plan; the cost
model remains `TBD` because nothing has been profiled.

Next: fill [docs/spec.md](docs/spec.md) — pin a `danijar/dreamerv3` commit SHA, record the dependency
and simulator version manifest, write the paper-to-code mapping with an interpretation note on every
row where the paper and the reference code differ, decide the v2 replay critic objective
include-or-deviate question, and label every capacity reduction in the deviation table.

**No model code before `docs/spec.md` is complete.** M1 (environment interface and sequence replay)
is the first milestone that writes implementation, and it depends on M0's transition convention and
step-counter definitions. The layout is the cheapest thing to get right later and the most expensive
thing to have chosen for the wrong problem.

## Last Session

**Session 2 — populate the hub and version the project.** 2 commits.

- **Ingested `dreamerv3_implementation_plan.md` and filled the four banner sections** (statement, The
  System, Target Regime, Core Hypothesis) from it; removed the unpopulated-project banner.
- **Split the plan into `docs/` with no duplicated content** — M0–M9 to `milestones.md`, M10 and the
  final package to `experiment.md`, the scope table to `config.md` — and added the doc map. Wrote
  `docs/spec.md` as an M0 skeleton (all `TBD`) and `results/README.md` for the measurement
  convention. The original file is preserved at `da9a55a` and removed from the tree so each claim
  has exactly one copy.
- **Recorded the domain testing hazards** under Verification Before Done. All five are derived from
  the plan's own validation gates, not invented.
- **Still nothing measured.** Why It Is a Target, the parameter count, and every return remain `TBD`
  by construction, not by omission.

## Known Issues

- **`docs/spec.md` is a skeleton and M0 is not complete.** Every entry is `TBD`. Writing model code
  against it now would resolve open questions by accident — which is exactly the failure M0 exists
  to prevent.
- **Do not call this implementation "1M parameters" from the `size1m` preset name.** The preset is a
  scale reference; its current defaults and architecture are not equivalent to the pinned paper
  version. Count the actual parameters and name the artifact the count came from.
- **Every section marked `TBD` is an admission, not a placeholder to skip past.** A session that
  fills one in from plausible-sounding assumption rather than measurement has made the file worse
  than empty. Fill from artifacts or leave `TBD`.
- **No results exist.** `results/` holds only its README. No number may be quoted from anywhere else.

---

At session end, refresh Current Focus / Last Session / Known Issues here — overwrite in place, 3–5
bullets in Last Session on what was actually done, no appending, no changelogs (git log is for
history). Keep this file thin: when a section grows past what a session needs loaded every time,
move the detail into `docs/` and leave a doc-map line saying **when** to read it.
