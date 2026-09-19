# Milestones M0–M9

**Read when starting or closing a milestone.** Relocated verbatim from
`dreamerv3_implementation_plan.md` (preserved at commit `c377be1`). M10 lives in
[experiment.md](experiment.md); starting values live in [config.md](config.md).

Keep the dependency order. Complete the implementation and establish learning before launching the
final horizon study. **A milestone is complete when its validation evidence and reproducible
configuration are recorded** — not when the code runs.

## Status

| Milestone | Status |
|---|---|
| **M0** | **CLOSED 2026-09-17.** Gate passed. Evidence: [spec.md](spec.md) and the audit at [`results/m0/m0-audit-2026-09-17.md`](../results/m0/m0-audit-2026-09-17.md). |
| **M1** | **CLOSED 2026-09-18.** Gate passed: 40/40 environment checks, 13/13 unit tests, 14/14 real-data replay smoke. Three empirical measurements deferred by decision — see the M1 status note. |
| **M2+M3** | **CLOSED 2026-09-18.** Merged into one milestone with one gate. 12/12 unit tests, 12/12 real-batch GPU gate, measured parameter count equals the derivation. |
| **M4** | **ACTIVE.** Next milestone. |
| M5–M10 | Not started. |

The environment contract and the RSSM state-transition core are `implemented` and `validated`;
everything downstream of the encoder is still only `specified` — see the status legend in
[spec.md §11](spec.md).

---

## M0 — Freeze the algorithm specification and experiment contract

**Purpose:** Make implementation choices traceable and prevent accidental mixing of Dreamer variants.

**Implement and document:**

- Pin arXiv v2 as the paper specification. Record a concrete commit from the [author-maintained repository](https://github.com/danijar/dreamerv3) before using its implementation as a reference. Record dependencies, hardware, and simulator versions.
- Create a paper-to-code mapping for the recurrent model, probability distributions, losses, actor estimator, critic targets, normalization, and initialization. Where the paper and reference code differ, document the chosen interpretation.
- Specify the exact CNN, recurrent cell, prior/posterior MLPs, reward and continuation heads, actor, and critic. Label each capacity reduction and algorithmic deviation.
- Preserve the relevant V3 methods: categorical straight-through sampling and uniform mixing; separately weighted dynamics and representation KL losses with free bits; transformed reward/value prediction; return normalization; entropy regularization; and slow-critic regularization. Include v2's replay critic objective, or explicitly classify its omission as a deviation. [Algorithm specification](https://arxiv.org/pdf/2301.04104v2)
- Define observations, action bounds, rewards, continuation targets, episode boundaries, time limits, evaluation action selection, and all step counters.
- Specify loss reductions. Sum categorical KL over stochastic factors before applying the free-nats threshold; then reduce over valid batch/time positions. Reconcile reconstruction reductions with loss weights rather than inheriting framework defaults.
- Reserve development seeds separately from final training seeds 0, 1, and 2. Record evaluation seed sets and the final metric definitions.

**Validation gate:** A written specification resolves the major architectural and objective choices. Any remaining resource settings have a named M9 qualification step. No unresolved choice is silently filled in by combining incompatible implementations.

**Deliverable:** Configuration, source/version manifest, and paper-to-code checklist.

> **CLOSED 2026-09-17.** Every bullet above is discharged in [spec.md](spec.md): pinned sources §1,
> version manifest §2, paper-to-code mapping §3, exact architecture §4, preserved V3 methods §5.0,
> loss reductions §5.2–§5.5, environment and counters §7, seeds §8, deviations §9.
> **The replay critic is included, not omitted** (§5.7), so no deviation was needed for it.
> Three paper-level ambiguities that the arXiv PDFs could not settle were resolved by tracing the
> pinned code and are recorded with permalinks in §6. Requirement-by-requirement mapping:
> [`results/m0/m0-audit-2026-09-17.md`](../results/m0/m0-audit-2026-09-17.md).

## M1 — Environment interface and sequence replay

**Purpose:** Establish trustworthy temporal data before training a recurrent model.

**Implement:** Pixel rendering, action conversion, seeded resets, transition collection, sequence replay, and logging. Store transitions with an explicit convention:

`(observation_t, action_t, reward_{t+1}, observation_{t+1}, episode_boundary, environment_discount)`.

Keep the last observation before a reset. Distinguish a true terminal event from a time-limit boundary. DMControl can end an episode with a nonzero discount, so `LAST` alone must not imply a zero continuation target. Reset recurrent state at episode boundaries while retaining the correct bootstrap semantics. [DMControl environment implementation](https://github.com/google-deepmind/dm_control/blob/3e9cd0bf3ec5141f3c6225e77f3ef99b42df2426/dm_control/rl/control.py)

Initially sample contiguous within-episode sequences. Define how a sequence beginning mid-episode obtains recurrent context: use a preceding burn-in prefix with masked losses, or reproduce a documented reference replay-state strategy. Avoid treating an arbitrary mid-episode frame as a real environment reset.

> **Decided at M0.** This choice is no longer open: **recomputed burn-in prefix, `P = 5`,
> loss-masked** — [spec.md §7.5](spec.md), classified as an implementation choice in §9-8 with its
> rationale and its ≈8% compute cost. The reference's alternative (latents persisted in the replay
> buffer and restored via `replay_context: 1`) is documented there and deliberately not adopted.
> The transition record, the `is_last` vs `is_terminal` contract, the action alignment, and all six
> step counters are likewise fixed in [spec.md §7](spec.md); M1 implements and validates them rather
> than deciding them.

**Validation gate:** A deterministic toy trajectory detects one-step action/reward alignment errors. Replay preserves sequence order and boundaries. Time-limit and terminal examples produce the intended targets. Observation shapes, action limits, seed handling, and step counters pass checks. Record a random-policy return floor under the final reward convention.

**Deliverable:** Inspectable replay batches, a random-policy video, and initial environment throughput measurements.

> **Status 2026-09-18: structurally complete — M2 may start.** `src/dreamer/{env,collector,replay,types}.py`
> implement the §7 contracts; 13/13 unit tests cover alignment, boundaries, the time-limit vs
> true-terminal targets, shapes, bounds, seeds, and counters, on top of the 40/40 environment checks
> in [`results/m1/`](../results/m1/). `scripts/m1_replay_smoke.py` runs the whole path on a real
> Walker episode and asserts the §7.5 sequence contract end to end (69 transitions / 70 observations,
> 5 masked + 64 loss-bearing per sequence, 1024 `train_position` per gradient step, no cross-episode
> window): 14/14 pass.
>
> **Deferred by decision, not blocked:** the random-policy return floor, the environment throughput
> benchmark, and the random-policy video are *empirical reporting*, not interface correctness. They
> are measurements of a fixed environment, reproducible at any time from the same code, and nothing
> in M2–M8 consumes them — the floor is only read at M10. Running them now would spend the session's
> remaining budget on numbers that would sit unused; they are scheduled with the M9 measurement pass.
> `scripts/m1_random_floor.py` and `scripts/m1_throughput.py` are written and waiting.

## M2+M3 — RSSM state-transition core

**Merged 2026-09-18.** M2 (deterministic recurrence) and M3 (encoder, posterior, prior) were
specified as separate milestones but share one object: there is no useful recurrence without a
stochastic state to carry, and M2's own text conceded this by allowing "synthetic stochastic states
before the encoder exists". Two gates over one module would have meant testing the recurrence twice —
once against a placeholder and once for real. They are now **one milestone with one validation
suite**; the deliverables of both are preserved.

**Purpose:** Implement the agent's memory, construct latent states from observations, and predict
latent states without observations.

**Implement:** `h_t = f(h_{t-1}, z_{t-1}, a_{t-1})` on the block-diagonal cell of
[spec.md §4.1](spec.md); the CNN encoder of §4.2; the posterior `q(z_t | h_t, e_t)` and prior
`p(z_t | h_t)` of §4.4 with straight-through categorical sampling, uniform mixing and explicit
aggregation over the 32 factors. Expose the complete model state `s_t = [h_t, z_t]`, single-step and
sequence APIs, reset masks, and an observation-conditioned path alongside a prior-only path that
takes no image argument. The decoder, the reward/continuation heads and the objective are **M4**.

**Validation gate — one suite, `tests/test_rssm.py` (12 tests):**

1. single-step recurrence agrees with the sequence API on identical inputs
2. reset masks isolate batch elements
3. changing actions, stochastic states or images changes the appropriate downstream state
4. the posterior depends on the image; the prior-only API has no image argument (asserted
   structurally, not just numerically)
5. categorical probabilities normalize and samples are valid one-hot factors
6. straight-through samples carry finite, nonzero gradients to logits
7. empirical sampling frequencies match a known categorical within tolerance
8. gradients stay finite through a short full RSSM sequence
9. every public API returns the shapes and layouts in [spec.md §4](spec.md)
10. the block-GRU gate **index map** — not just its shapes
11. `BlockLinear` fan-in is the full input width, not the per-block width
12. measured `sum(p.numel())` equals the §4.11 derivation

Gates 10–12 are not in the original M2/M3 text. They were added because mutation testing showed the
first nine pass unchanged against a flat gate split and against a per-block fan-in — the two hazards
[CLAUDE.md](../CLAUDE.md) names as silently wrong but still trainable. Each of 10–12 was confirmed to
fail without its fix.

**Deliverable:** `src/dreamer/{nets,distributions,rssm}.py`; observation and prediction APIs; the
measured parameter count.

> **Status 2026-09-18: PASSED.** 12/12 unit tests, plus `scripts/m2m3_gate.py` driving the RSSM from
> a real Walker replay batch on the GPU — 12/12, artifact
> [`results/m2m3/gate-2026-09-18.txt`](../results/m2m3/gate-2026-09-18.txt). Measured parameter count
> **391,008** (`dyn` 376,704 + `enc` 14,304), equal to the §4.11 derivation to the digit:
> [`results/m2m3/param-count-measured-2026-09-18.txt`](../results/m2m3/param-count-measured-2026-09-18.txt).
> **No deviation from the specification.** M4 may start.

## M4 — World-model heads and training objective

**Purpose:** Learn a latent state that retains useful information and supports prediction.

**Implement:** Image reconstruction, reward prediction, and continuation prediction. Combine their losses with the dynamics and representation KL terms specified in M0. Verify the direction of KL, stop-gradient placement, free-nats clipping, and weighting against the chosen sources. Use the transformed distributional reward output chosen in M0; test target encoding and scalar decoding independently.

For the transition convention in M1, reward and continuation targets accompanying `observation_{t+1}` are supervised from the corresponding resulting model state. Apply an explicit mask to reset observations that lack a preceding reward target.

Start by fitting a small fixed replay subset, then train on a larger training split. Hold out whole episodes for validation. Log separate losses, unclipped KL, posterior/prior entropy, reward error, continuation error, gradient norms, and parameter/activation finiteness.

**Validation gate:** The small subset can be fit; validation predictions improve over simple constant or persistence baselines where appropriate. Isolated loss checks demonstrate the intended prior/posterior gradient routing, accounting for shared recurrent parameters. Good reconstruction alone does not pass the world-model gate.

**Deliverable:** A world-model checkpoint, learning curves, reconstructions, and a record of resolved numerical or temporal bugs.

## M5 — Open-loop prediction evaluation

**Purpose:** Establish that the model predicts action-dependent futures rather than only reconstructing observed frames.

**Implement:** Condition on a fixed image prefix, then predict 30 transitions using only the recorded future actions and the dynamics prior. Future images and rewards are available solely for scoring. Use the same starting contexts for all reported prediction horizons.

Measure reward MAE by prediction distance at 1, 5, 15, and 30 transitions; decoded-image error and continuation quality are secondary diagnostics. Compare reward prediction with a training-set mean predictor and image prediction with last-frame persistence. Include an action-shuffling diagnostic to assess action sensitivity, without treating it as a causal control-performance result.

Use multiple latent samples per context and average prediction metrics rather than selecting attractive rollouts. Preserve paired videos of predicted and observed frames. Keep episode-level validation data out of model training.

**Validation gate:** Automated checks establish that no future observation reaches the prediction path. On sufficiently varied data, action-conditioned predictions add measurable information beyond the simple baselines at short horizons. Inspect error growth at longer horizons without requiring it to be monotonic.

**Deliverable:** Prediction-error curves and paired open-loop videos. Revisit these diagnostics after online learning broadens the replay distribution.

## M6 — Latent imagination engine

**Purpose:** Generate the experience used for behavior learning.

**Implement:** Begin from posterior states obtained from replay. Use an action-provider interface, initially with random actions. At each step, choose an action, advance through the prior, and predict reward and continuation. Return H actions/rewards/continuations and H+1 latent states.

Keep image decoding out of the behavior-training path; decode only selected rollouts for inspection. Separate the posterior context pass from prior-only imagination. Implement discount/continuation weights and exclude invalid or terminal start states.

**Validation gate:** H∈{5,15,30} produces correctly aligned outputs, finite predictions, and measured memory use. Tests detect reward shifts, incorrect terminal weighting, and accidental posterior calls during imagination. Generated trajectories require neither simulator steps nor future images.

**Deliverable:** A reusable imagination function and per-horizon cost measurements.

## M7 — Critic and return targets

**Purpose:** Estimate rewards beyond the finite imagination horizon.

**Implement:** A distributional critic, bootstrapped λ-returns, and slow-critic regularization. Preserve the selected v2 replay critic objective. Calculate scalar values and distributional targets consistently with the chosen reward/value representation; record whether a transformation is applied before or after an expectation.

Use the random action provider from M6 for initial integration. Start with hand-computable synthetic trajectories before evaluating the critic on model-generated targets. Changing the behavior policy later will change its value target.

**Validation gate:** Return calculations match manually derived examples for zero rewards, constant rewards, early termination, and horizon bootstrapping. λ endpoints behave as intended. Critic targets are detached appropriately. Critic fitting reduces error on fixed targets without updating the actor or world model accidentally.

**Deliverable:** Verified return calculations, critic training, and value-versus-target diagnostics.

## M8 — Actor and imagined behavior learning

**Purpose:** Learn actions that improve predicted return.

**Implement:** A bounded continuous-action policy and the pinned V3 policy-gradient estimator, with return normalization, entropy regularization, and continuation weighting. Replace M6's random action provider with the actor. Consult the [paper's actor objective](https://arxiv.org/pdf/2301.04104v2) when fixing the estimator; do not silently substitute an earlier Dreamer variant's gradient rule.

Check action log-probability and entropy calculations against the selected distribution. Define which latent features, sampled actions, returns, and baseline values are detached. If an action transform is used, handle its probability-density correction consistently.

**Validation gate:** On a small analytic action/reward fixture, updates increase the probability of better actions. Actions stay in bounds, entropy remains finite, and an actor update changes only intended parameters. An increase in imagined return is treated as an integration diagnostic until real-environment performance improves.

**Deliverable:** Actor/critic training integrated with imagination, plus gradient-routing and policy diagnostics.

## M9 — Complete online loop and qualify the final configuration

**Purpose:** Demonstrate learning from pixels and determine a feasible experiment budget.

**Implement:** Alternate real collection, replay sampling, world-model updates, imagined behavior updates, checkpointing, and evaluation. During real interaction, update the posterior from the current image before selecting an action. The environment policy uses the learned latent state directly; imagination supplies training experience.

Use Cartpole as the first complete-loop debugging task, followed by Walker. Vector-state runs are optional for isolating control bugs and must remain separate from final pixel results.

Checkpoint model/optimizer states, slow critic, normalization statistics, RNG states, replay state, configuration, and step counters. Resume at an episode boundary unless the simulator state is also restored. Log evaluation time separately from training time, and use evaluation trajectories neither for replay updates nor gradient steps.

**Qualification sequence:**

1. Run the complete learning path under development seeds, with diagnostic evaluation every 25,000 control steps using five evaluation episodes.
2. Verify that learning persists beyond a transient return spike and that each task improves over the random floor.
3. Profile collection, rendering, replay transfer, world-model updates, and behavior updates. Measure representative H=30 peak memory. Estimate final-run cost from steady-state measurements, including evaluation overhead.
4. If learning stalls, inspect replay alignment, reward/value scaling, latent information, prior error, update ratio, and capacity before changing several settings at once. Repeat affected validation gates after fixes.
5. Where feasible, run the pinned author implementation with the same task wrappers and approximate capacity/data budget to qualify the setup. A single reference run is a diagnostic, not a statistical performance benchmark.
6. Finalize architecture, training ratio, replay handling, precision, budgets, evaluation policy, and shared hyperparameters. Create a run manifest before executing final seeds.

**Validation gate:** Both pixel tasks demonstrate real learning in development runs; checkpoint/resume works; H=30 fits with memory headroom; and final runtime estimates are recorded. The final study starts only after this gate. If it fails, the deliverable is an investigated implementation limitation, not a claimed successful reproduction.

**Deliverable:** Frozen configuration, working online learner, resource profile, and final run manifest.
