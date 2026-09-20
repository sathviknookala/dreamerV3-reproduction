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
| [docs/testing-hazards.md](docs/testing-hazards.md) | **Before writing any test or gate check**, and whenever a new assertion passes on the first try. The 35+ places where the obvious assertion passes while the bug survives, grouped by component. |
| [docs/experiment.md](docs/experiment.md) | Before designing a final run or writing any result claim. The M10 contract: run matrix, controls, metrics, and the limits on what may be concluded. |
| [results/README.md](results/README.md) | Before recording a number. Artifact layout and measurement rules. |
| [results/m0/m0-audit-2026-09-17.md](results/m0/m0-audit-2026-09-17.md) | To check whether an M0 requirement is actually discharged, and what is deferred to where. |
| [results/m9/](results/m9/) | Before quoting a runtime, a per-stage share, a memory figure or a campaign cost, and before touching the online loop, checkpointing or evaluation. The M9 integration gate (75/75 on each task), the per-stage profile at H = 5/15/30, the checkpoint cost, and what none of it establishes. |
| [results/m9/pilot/](results/m9/pilot/) | **Before quoting any return for a trained agent.** The two full-budget pilots at seed 100, H=15 — Walker's learning curve, Cartpole's failure to consolidate, both raw logs, and the limits on reading either. |
| [results/m8/](results/m8/) | Before quoting an entropy, a log-probability, a normalization scale or a parameter count, or touching the policy path. The M8 gate, the `tarval` and `debias` resolutions, and the REINFORCE direction check. |
| [results/m7/](results/m7/) | Before quoting a value, a λ-return or a critic parameter count, or touching the return path. The M7 gate, the three readings of γ, and the value-vs-target fit. |
| [results/m6/](results/m6/) | Before quoting an imagination cost or touching the rollout path. The M6 gate, the per-horizon cost/memory table, and what those numbers exclude. |
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

1. **Encoder.** CNN, 64×64×3 → `e_t`. Layers, activations and normalization are specified in
   [spec.md §4.2](docs/spec.md); the parameter count is **measured** at 14,304 ([results/m2m3/](results/m2m3/)).
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
    *different objective* and must not be substituted. The baseline is the **fast** critic, the same
    estimate the λ-return bootstraps (`slowtar: False`), read from one pre-update critic state.
    Return normalization (5th/95th percentile, `S = max(1, hi−lo)`) is an **uncorrected** EMA at the
    pin, so `S` sits at its floor of 1 early. Continuation weighting, and entropy **subtracted** in
    the minimized loss at η=3e-4. The distribution is a diagonal Gaussian with `tanh` on the **mean
    only**; the sample is unsquashed, so **there is no density correction to apply** — bounding is
    the environment's and the RSSM's job ([spec.md §4.6, §5.6](docs/spec.md)).
12. **Discount lives in the continuation head.** `contdisc`: the head is trained on the soft label
    `(1−is_terminal)·(1−1/333)` and the return discount is then **1**. Applying γ explicitly *as
    well* double-counts it; using a hard 0/1 label with `disc=1` drops it. Both train silently
    ([spec.md §5.4](docs/spec.md)).
13. **Gradient routing.** Actor gradients through imagined dynamics are **blocked** (which is *why*
    REINFORCE is needed); replay-critic gradients into the encoder and RSSM are **live**. Full table:
    [spec.md §5.9](docs/spec.md).
14. **Online loop.** **Implemented and gate-validated; not yet qualified.** During real interaction,
    update the posterior from the current image **before** selecting an action, conditioning on the
    **executed** previous action, not the requested one. The environment policy uses the learned
    latent state directly and acts on the distribution's **mean** at evaluation
    ([spec.md §7.8](docs/spec.md)); imagination supplies training experience only. The simulator is
    rebuilt each episode from `episode_seed(base, index)`, so resume is a pure function of one
    persisted integer — and a **mid-episode resume checkpoint is refused**, because replay's partial
    episode has no successor observation the rebuilt simulator can produce.

Where a step is conventional but not required: the vector-observation path is a development aid for
isolating control bugs and must stay separate from final pixel results. Image decoding is required
for M4/M5 diagnostics but is deliberately absent from the behavior-training path.

## Why It Is a Target — measured 2026-09-19

Per-stage cost of one complete update, B=16 T=64 P=5, 1024 imagination starts, 20 timed
steady-state updates after 5 discarded warm-ups, CUDA-synchronized, on the RTX PRO 4000
([results/m9/profile-walker-2026-09-19.csv](results/m9/profile-walker-2026-09-19.csv)):

| H | update | world model | imagination + behaviour | backward + optimizer | replay transfer | peak VRAM |
|---|---|---|---|---|---|---|
| 5 | 206.0 ms | 71.3 | 10.5 | 121.1 | 1.05 | 1576.3 MiB |
| 15 | 221.2 ms | 75.5 | 21.6 | 121.6 | 0.91 | 1815.3 MiB |
| 30 | 236.8 ms | 71.2 | 39.5 | 123.2 | 1.03 | 2373.8 MiB |

Collection + rendering + the policy forward: **2.796 ms per control step**. Evaluation:
**3.29 ms per control step**. A resume checkpoint at full occupancy: **5.744 GiB, 10.1 s**.

Steady-state shares of a 4.9 h Walker H=15 run: backward + optimizer **42.7%**, world-model
forward/loss **26.5%**, collection + rendering **15.8%**, imagination + behaviour **7.6%**,
evaluation **3.7%**, checkpointing **2.3%**, episode simulator rebuild **0.6%**, replay transfer
**0.3%**.

**The single backward pass over the combined loss is the largest stage**, and it barely moves with
`H`: doubling the horizon from 15 to 30 costs **7.0%** more wall clock per update, not 2×, because
only the imagination + behaviour stage scales. **H=30 is not memory-limited** — 2373.8 MiB allocated
of 23,986 MiB, roughly 22 GB of headroom.

**The 12-run final campaign derives to ≈52 GPU-hours** (Walker 4.7 / 4.9 / 5.2 h per run at
H = 5 / 15 / 30, Cartpole 2.5 h, three seeds each). That is a derivation from measured per-stage
costs at a fixed 2,000-transition replay occupancy, **not a measured run**, and it excludes failed
or restarted runs.

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

**What it cannot claim.** Runtime is now measured and the first returns exist
([results/m9/pilot/](results/m9/pilot/)), but they are **pilot returns at one development seed over
five evaluation episodes**, with **no random floor to compare against** — not a result under this
hypothesis, which is specified on 20 episodes across seeds 0/1/2. The parameter count
is now **measured end to end**: `sum(p.numel())` over the world model is **570,419**
([results/m4/](results/m4/)), `val` **66,111** ([results/m7/](results/m7/)) and `pol` **50,316**
([results/m8/](results/m8/)), each equal to its §4.11 derivation, for a measured trainable total of
**686,846** excluding the untrained `slowval` mirror.
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
- Unit tests (249), the M1 smoke test (14 checks), the M2+M3 gate (12), the M4 gate (15), the
  M5 gate (26), the M6 gate (55), the M7 gate (36), the M8 gate (63) and the M9 gate (75 per task),
  verbatim:

  ```bash
  PYTHONPATH=src:tests .venv/bin/python -m unittest test_env test_replay test_collector test_rssm test_world_model test_openloop test_optim test_imagine test_critic test_actor test_agent test_training test_checkpoint test_evaluation
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

  ```bash
  .venv/bin/python scripts/m6_gate.py
  ```

  ```bash
  .venv/bin/python scripts/m7_gate.py
  ```

  ```bash
  .venv/bin/python scripts/m8_gate.py
  ```

  The M9 gate takes a required `--tag` and **refuses to overwrite an existing artifact**:

  ```bash
  PYTHONPATH=src .venv/bin/python scripts/m9_gate.py --task cartpole --tag $(date +%Y-%m-%d)
  ```

  ```bash
  PYTHONPATH=src .venv/bin/python scripts/m9_gate.py --task walker --tag $(date +%Y-%m-%d)
  ```

  M9's 22 mutants, each confirmed to kill its own test:

  ```bash
  .venv/bin/python scripts/m9_mutants.py
  ```

  The M5 gate needs the collected split and the trained checkpoint, neither of which is committed
  (`data/` and `runs/` are ignored). Rebuild both, ~30 minutes, then gate:

  ```bash
  .venv/bin/python scripts/m5_collect.py && .venv/bin/python scripts/m5_train.py && .venv/bin/python scripts/m5_eval.py --tag 2026-09-18 && .venv/bin/python scripts/m5_gate.py --report results/m5/openloop-2026-09-18.json
  ```
- A test that passes where the bug cannot occur is not a test — confirm it fails without its fix
- **Domain testing hazards live in [docs/testing-hazards.md](docs/testing-hazards.md)** — 35+ traps
  where the obvious assertion passes while the bug survives, grouped by component. **Read it before
  writing a test or gate check**, and whenever a new assertion passes on the first try. The two that
  bite in almost every session:
  - **`rew` and `val` are `outscale: 0.0`, so they are invisible at initialization.** The readout is
    exactly 0 for every input and no gradient leaves the head. Perturb `mlp.out.weight` before
    asserting anything about reward or value alignment, routing, or fast-vs-slow. `pol` at
    `outscale: 0.01` is the exception and needs no perturbation — but a *rollout* still has a
    zero advantage unless the other two are perturbed.
  - **TF32 breaks hand-computed fixtures**: max|GPU−CPU| on a 2048² fp32 matmul is 6.8e-02 with it
    on and 1.3e-04 off. Set `torch.backends.cuda.matmul.allow_tf32 = False` and
    `torch.backends.cudnn.allow_tf32 = False` in any test asserting against analytic values.
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

`src/dreamer/` is the style guide. The conventions the code actually holds to, recorded because a
session that breaks one produces something that still trains:

- **State the shape contract at every module boundary**, and **raise** on a violation rather than
  broadcasting through it. `(B, T, ...)` vs `(T, B, ...)` is the most likely source of a silent
  temporal misalignment, and `lambda_return`, `imagine_trajectory` and `imagined_actor_loss` all
  reject mismatched inputs by hand.
- **Every `detach()` / stop-gradient placement gets a one-line comment saying *why*.** M4, M7 and M8
  all turn on gradient routing — prior vs posterior, critic targets, actor baselines — and a missing
  or extra detach produces a model that trains without erroring.
- **Assert an invariant that already holds by construction; do not re-enforce it.** `imagine_trajectory`
  builds under `no_grad`, so the actor's blocked path needs no new detach — `imagined_actor_loss`
  *raises* if handed features carrying a graph, and the gate asserts the grads are `None`.
- **Losses reduce with a plain position `.mean()`** over the valid grid, never `sum/weight.sum()`.
  Weights multiply, they do not normalize. This is what keeps the loss scale invariant to `H`.
- **Modules take `in_features` / `action_dim` rather than reading a global config**, so a test can
  instantiate a 12-feature toy version of anything. Every gate and unit test relies on this.
- **Stateful non-parameters are `register_buffer`s on an `nn.Module`** (`bins`, the slow-critic
  mirror, the `ReturnNormalizer` EMAs) so they enter `state_dict()` and survive checkpoint/resume,
  which M9 requires.
- **Each milestone owns `scripts/mN_gate.py` and `results/mN/`.** A gate prints one `PASS`/`FAIL` per
  check and exits non-zero on any failure.

Add further conventions here only when a session would otherwise get them wrong; duplicating what the
code already says creates drift.

## Current Focus

**M9 — Walker learns, Cartpole does not. M9 is OPEN.** M0 closed 2026-09-17; M1, the merged
**M2+M3**, **M4**, **M5**, **M6** and **M7** closed 2026-09-18; **M8 closed 2026-09-19**. M9's
infrastructure is gate-validated, and as of **2026-09-20 two full-budget pilots exist**
([results/m9/pilot/](results/m9/pilot/)) at seed 100, H=15, run concurrently on the one card:

- **Walker Walk, 1M steps, 5.61 h.** Opening five-evaluation mean **29.6** → last five **542.7**,
  consolidating at 457–555 after 850K with `return_std` down to 24–38. Final checkpoint
  **517.1 ± 61.3**; peak **554.9 ± 24.5** at 975K — **the final checkpoint is not the best one.**
- **Cartpole Swingup, 500K steps, 3.07 h.** Eight of twenty evaluations sit in a 72–79 band with
  `return_std` under 1.5; three excursions; peak **217.2 ± 16.7** at 350K decaying to
  **132.6 ± 27.5** at budget. **No consolidation.**

This is the first evidence the implementation learns visual control at all, which retires the risk
that some component is silently broken for every task. It does **not** close M9: the gate needs
*both* tasks, and **the M1 random-policy floor is still unrecorded**, so neither curve can yet be
called improvement over a floor.

**M9 closes only when all five hold:** both pixel tasks show sustained improvement over the random
floor (**Walker yes, Cartpole no, floor unmeasured**), resume works (**done**), H=30 has measured
memory headroom (**done**, 2373.8 MiB of 23,986), campaign costs are measured (**done**,
≈52 GPU-hours), and the configuration in [docs/config.md](docs/config.md) is frozen (**not done**).

Next, in order:

1. **Random floors, both tasks.** Nothing can be concluded from either pilot without them. These
   build no torch model, so they are CPU-bound and will not contend for the GPU:
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/m1_random_floor.py --task cartpole --env-seed 901 --policy-seed 901 --episodes 20 --output results/m1/random-floor-cartpole-$(date +%F).json --video results/m1/random-cartpole-$(date +%F).mp4
   PYTHONPATH=src .venv/bin/python scripts/m1_random_floor.py --task walker   --env-seed 900 --policy-seed 900 --episodes 20 --output results/m1/random-floor-walker-$(date +%F).json   --video results/m1/random-walker-$(date +%F).mp4
   ```
2. **Throughput, both tasks** — the last M1 deferral:
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/m1_throughput.py --task walker   --env-seed 902 --policy-seed 902 --steps 10000 --output results/m1/throughput-walker-$(date +%F).json
   PYTHONPATH=src .venv/bin/python scripts/m1_throughput.py --task cartpole --env-seed 903 --policy-seed 903 --steps 10000 --output results/m1/throughput-cartpole-$(date +%F).json
   ```
3. **Re-evaluate both `model-final.pt` at 20 episodes.** Every pilot return is 5 episodes; the core
   hypothesis is specified on 20. **No standalone checkpoint-evaluation entry point exists** —
   `model-final.pt` is loaded only by `m9_train.py` and `m9_gate.py`, neither of which re-evaluates
   a finished run. One needs writing against `src/dreamer/evaluation.py`.
4. **Diagnose Cartpole** — this is milestone step M9-4, and it is now active. Its world model is not
   the suspect: final `rec` **3.38** against Walker's 23.50, `reward_mae` 0.076. Its
   `value_value_mean` reached only **46.79** against Walker's 169.62. It inherits Walker's H=15 and
   update ratio 64 unchanged, so **task-specific exploration and a shared configuration defect are
   not yet separated.** Change one thing at a time and repeat the affected gates.
5. Re-run the M5 open-loop diagnostics against a pilot checkpoint, now that online learning has
   broadened the replay distribution away from uniform-random data.
6. Freeze [docs/config.md](docs/config.md).

Launch any further run detached with the sandbox disabled and **read the `device:` line** before
walking away. Resume from a boundary checkpoint with
`scripts/m9_train.py --task <t> --seed 100 --out <dir> --resume <dir>/resume-<step>.pt`.

Development seeds are **100–102**. **Final training seeds 0–2 and final evaluation seeds 1000–1019
stay unused** until M10.

**Deferred from M1 and still outstanding:** random-policy return floor, throughput benchmark,
random-policy video. **Steps 1–2 discharge all three.**

## Last Session

**Session 14 — ran the first two full-budget pilots to completion. Walker learns visual control;
Cartpole does not consolidate. M9 remains OPEN.**

- **Two full-budget pilots finished cleanly**, concurrently on the one RTX PRO 4000: Walker 1M steps
  / 62,187 optimizer steps / 20,197.7 s, Cartpole 500K / 30,937 / 11,069.5 s. Both hit budget, wrote
  `model-final.pt` and a manifest, and ended with replay at its 500,000 capacity and 500 complete
  episodes — the predicted eviction behaviour. Evaluation was 3.2% and 3.0% of wall clock.
- **Walker is the project's first learning result:** 29.6 → 542.7 comparing first and last five
  evaluations, consolidating at 457–555 after 850K. Its actor did **not** collapse to `minstd`
  (final entropy +2.78) and `policy_retnorm_scale` left its floor of 1.0 for 25.78.
- **Cartpole is the failure to investigate**, and it is the *easier* validation task. Its perception
  is fine; its critic barely grew. Recorded as milestone step M9-4 rather than diagnosed here.
- **The pilot artifacts are committed to [results/m9/pilot/](results/m9/pilot/)** — evaluations,
  per-update logs, manifests and configs — because `runs/` is gitignored and no number may be quoted
  from an uncommitted file.
- **Both pilot logs predate commit `a0bc33b` by ~2 h**, so they carry **no `segment` column** and the
  documented dedupe rule does not apply to them. Neither run resumed, so no row is duplicated. Noted
  in both `results/m9/README.md` and the pilot README so a future plot does not key on a null field.


## Known Issues

- **Python 3.13 cannot install this project; use `.venv` (3.12.11) for everything.** `dm-control`
  requires `labmaze`, whose latest release (1.0.6) has wheels only to cp312 and otherwise needs bazel.
  The system `python3` is 3.13.13, so a bare `python3 script.py` will fail on imports. Also: this
  machine's `python3.12` is uv-managed with a broken `ensurepip`, so `python -m venv` cannot create
  the environment — `uv venv --seed` does, and pip does the installs.
- **`pytest` is not installed; the suite is `unittest` and needs `PYTHONPATH=src:tests`.** The package
  is not installed into the venv, and `unittest discover` fails because `tests/` has no `__init__.py`
  — name the fourteen modules explicitly, as in the recorded command.
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
- **`size1m` is 570,419 parameters for the world model and 686,846 for the full trainable agent,
  all measured.** Far below the paper's smallest evaluated row (12M), so no result can be compared
  to a published number. §10-7 is discharged; nothing in §4.11 is derived-only.
- **All four documented transcription traps are now guarded.** Gate split, `BlockLinear` fan-in,
  encoder/decoder scaling asymmetry, and the `sg(…, skip=)` polarity — the last via the KL
  stop-gradient test. Each was mutation-verified to fail without its fix.
- **A detached process launched through the sandboxed shell loses CUDA and falls back to CPU
  silently.** `torch.cuda.is_available()` returns `False` with only a `UserWarning`; the run then
  trains at roughly 1/30 speed and nothing else looks wrong. Launch long runs with the sandbox
  disabled and **read the `device:` line in the log** before walking away.
- **Only [results/m9/pilot/](results/m9/pilot/) describes a trained agent.** `results/m5/` holds
  open-loop prediction error on uniform-random data; `results/m6/`, `results/m7/` and `results/m8/`
  hold imagination cost, critic arithmetic and policy arithmetic measured on an **untrained** model;
  `results/m9/`'s gate returns describe a ~1,200-step agent and are a smoke signal. Do not quote any
  of those as a property of the trained agent.
- **Cartpole, the easier validation task, is the one that failed to learn.** Walker consolidates at
  457–555; Cartpole peaks at 217.2 and decays to 132.6. Before assuming a Cartpole-specific
  exploration problem, note that it inherits Walker's H=15 and update ratio 64 unchanged — a shared
  configuration defect is not ruled out, and neither explanation is tested.
- **No random floor exists, so no pilot return can be called improvement over one.** Both M9's gate
  and the core hypothesis are stated against that floor. `scripts/m1_random_floor.py` is repaired
  and ready; it has never been run.
- **`scripts/m9_gate.py` is implementation validation, NOT the M9 validation gate.** It proves the
  loop is wired as specified. M9's actual gate is empirical: both pixel tasks must show sustained
  improvement over the random floor. Do not read 75/75 as M9 closing.
- **Resume is bitwise for the next update, not for a trajectory.** Two restores of the same
  checkpoint compute an identical next update; continued for hundreds of updates they drift from
  *each other* at ~1e-6. That is nondeterministic CUDA convolution backward, measured, not a
  checkpoint defect — so a resume test must assert next-update equality, never trajectory equality.
- **The two pilot logs predate the segment counter.** Both processes started 2026-09-19 20:45:01;
  commit `a0bc33b` landed 22:42:15. So `results/m9/pilot/evaluations-*.jsonl` has **no `segment`
  field** and `train-log-*.csv` has **neither `segment` nor `eval_seconds`**. Neither run resumed,
  so nothing is duplicated and the rule below does not apply to them. A resumed run started from
  those checkpoints would write the newer schema mid-file, leaving `.segment == null` on the earlier
  rows — plotting code should tolerate that.
- **A resume replays, so `train-log.csv` and `evaluations.jsonl` contain duplicated rows.** The
  checkpoint is older than wherever the previous segment died, and the rows between the two are
  written again. Both carry a `segment` column incremented on every restore: **plot the highest
  `segment` per `gradient_step`**. Nothing is deleted, so the crashed segment stays inspectable.
  `elapsed_s` is wall clock and includes evaluation; training time is `elapsed_s - eval_seconds`.
- **A resume checkpoint is refused unless replay's current episode is closed.** Mid-episode
  simulator state is not restorable; appending to a partial episode after a rebuild would raise in
  `ReplayBuffer.add`. `OnlineTrainer` therefore only writes a resume checkpoint at a boundary, and
  a final stop mid-episode writes the compact model checkpoint alone.
- **A resume checkpoint is 5.744 GiB and takes 10.1 s to write.** Two are kept, so a run needs
  ~11.5 GiB of disk beside its log. `atomic_save` pins `pickle_protocol=5`; torch's default
  protocol 2 latin1-encodes bytes and made the same buffer 8.613 GiB and 41.3 s.
- **The 500,000-transition buffer does NOT hold a 1M-step Walker run.** Eviction begins near the
  half-way point and the second half evicts the first, whole episodes at a time. `docs/config.md`
  claimed otherwise until 2026-09-19.
- **Every M8 entropy, log-probability and loss is a property of a random actor.** The integration
  section perturbs the reward and value kernels (otherwise the advantage is identically 0 and every
  claim about it is vacuous), and the update diagnostic uses synthetic `U(-1, 1)` returns. The
  100-step trajectory collapses `σ` to `minstd` by design — a fixed advantage with no critic
  learning is not training, and its negative entropy is correct, not a failure.
- **Re-running a gate overwrites its committed artifact.** `m6_gate.py`, `m7_gate.py` and
  `m8_gate.py` write their CSV on every run, so a regression check silently replaces a committed
  measurement with a fresh draw — M6's was overwritten and restored during the M8 session. M7's and
  M8's reproduce bitwise and M6's timings to <1%, but `git checkout` the artifact after a regression
  run unless a re-measurement was intended.
- **Every M6 and M7 reward and value magnitude is meaningless.** Both gates perturb the zero-init
  output kernels so alignment and routing do not pass vacuously, which makes each readout a random
  draw over ±4.85e8 bins. They test that quantities are correctly *placed*, routed and finite, never
  that they are right. A perturbed head is also a pathological starting point for any *fitting*
  diagnostic — fit from a fresh zero-init critic instead.
- **Every M5 number describes a uniform-random data distribution.** M5's own text requires the
  diagnostics to be revisited once online learning broadens replay. Do not carry these numbers
  forward as properties of a trained agent's world model.

---

At session end, refresh Current Focus / Last Session / Known Issues here — overwrite in place, 3–5
bullets in Last Session on what was actually done, no appending, no changelogs (git log is for
history). Keep this file thin: when a section grows past what a session needs loaded every time,
move the detail into `docs/` and leave a doc-map line saying **when** to read it.
