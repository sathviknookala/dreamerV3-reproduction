# DreamerV3 Reproduction: Updated Implementation Plan

## Project context

Build an independent, reduced-scale PyTorch implementation of DreamerV3 that learns visual control through imagined latent trajectories. The target hardware is one **RTX Pro 4000 with 24 GB VRAM**, supported by the available Ryzen 7950X and 128 GB system RAM.

Dreamer learns a world model from real experience, predicts possible futures in latent space, and uses those predictions to train an actor and critic. DreamerV3 adds stabilization methods intended to make this process work across diverse tasks with a shared configuration. This project will reproduce the central learning mechanism and relevant V3 methods at smaller scale. Its two-task evaluation will support a limited reproduction claim, rather than the paper's full cross-domain result. The algorithm specification is [Mastering Diverse Domains through World Models, arXiv v2](https://arxiv.org/pdf/2301.04104v2).

The project should produce evidence of three capabilities: implementing a research algorithm, diagnosing interactions between its components, and conducting a controlled experiment. It complements existing representation-learning, probabilistic-modeling, CUDA, and serving work by adding action-conditioned dynamics, visual perception, and sequential decision-making.

**Primary task:** DMControl Walker Walk from 64×64 RGB images. **Validation task:** Cartpole Swingup from the same observation format. Train a separate agent for each task. Vector observations are a development aid; final agents receive pixels and their action history, without privileged simulator state.

**Experimental question:** How does imagination horizon affect real-environment return, multi-step prediction error, and compute cost in a reduced-capacity world model?

Keep the dependency order M0–M10. Complete the implementation and establish learning before launching the final horizon study. A milestone is complete when its validation evidence and reproducible configuration are recorded.

## Scope and initial configuration

The values below are starting choices to qualify during development. Hardware fit and learning quality must be measured. Final settings are frozen at the end of M9, before the final experiment seeds are run.

| Item | Initial choice | Qualification rule |
|---|---|---|
| Framework | PyTorch; eager execution first | Add compilation or mixed precision only after correctness checks |
| Observations | 64×64 RGB | Identical preprocessing in training and evaluation |
| Tasks | Walker Walk; Cartpole Swingup | Same shared learning settings; task-specific action dimensions |
| Recurrent state | 512 deterministic features | Increase capacity only if a diagnosed limitation warrants it before final runs |
| Stochastic state | 32 categorical variables, 4 classes each | Record as a deliberate compact configuration |
| Encoder/decoder and MLP widths | Compact networks selected in M0 | Record exact layers, activations, normalization, and measured parameter count |
| Training sequences | Start with batch 16, length 64 | Treat any context/burn-in prefix separately from loss-bearing positions |
| Imagination | H=15 transitions; H+1 latent states | Final Walker comparison uses H∈{5,15,30} |
| Discount / return mixing | γ=0.997; λ=0.95 | Freeze after source reconciliation |
| Collection | One environment first; action repeat 1 | Log native control steps separately from agent decisions and physics substeps |
| Replay | CPU-resident uint8 frames; initial capacity 500,000 transitions | Measure total RAM use and avoid duplicate image storage |
| Update ratio | Start with 64 replay training positions per collected transition | Log the exact definition and realized ratio; qualify learning before freezing |
| Initial random collection | 5,000 agent transitions | Preserve the same warm-up budget in final comparisons |
| Planning budgets | Walker: 1M control steps/run; Cartpole: 500K/run | Qualify cost and learning in M9; all Walker horizon conditions receive the same final budget |

The [author-maintained configuration](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml) supplies a useful compact scale reference through `size1m`, including deterministic size 512 and four classes. Its current defaults and architecture are not automatically equivalent to the pinned paper version. Record the actual parameter count rather than calling this implementation “1M parameters” from the preset name.

The initial update ratio and budgets are project choices, not claims about the paper's settings or guaranteed convergence. Runtime is estimated from measured pilots. If the compact system needs more capacity or updates, resolve that before the final study and apply the resulting configuration consistently.

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

## M1 — Environment interface and sequence replay

**Purpose:** Establish trustworthy temporal data before training a recurrent model.

**Implement:** Pixel rendering, action conversion, seeded resets, transition collection, sequence replay, and logging. Store transitions with an explicit convention:

`(observation_t, action_t, reward_{t+1}, observation_{t+1}, episode_boundary, environment_discount)`.

Keep the last observation before a reset. Distinguish a true terminal event from a time-limit boundary. DMControl can end an episode with a nonzero discount, so `LAST` alone must not imply a zero continuation target. Reset recurrent state at episode boundaries while retaining the correct bootstrap semantics. [DMControl environment implementation](https://github.com/google-deepmind/dm_control/blob/main/dm_control/rl/control.py)

Initially sample contiguous within-episode sequences. Define how a sequence beginning mid-episode obtains recurrent context: use a preceding burn-in prefix with masked losses, or reproduce a documented reference replay-state strategy. Avoid treating an arbitrary mid-episode frame as a real environment reset.

**Validation gate:** A deterministic toy trajectory detects one-step action/reward alignment errors. Replay preserves sequence order and boundaries. Time-limit and terminal examples produce the intended targets. Observation shapes, action limits, seed handling, and step counters pass checks. Record a random-policy return floor under the final reward convention.

**Deliverable:** Inspectable replay batches, a random-policy video, and initial environment throughput measurements.

## M2 — Deterministic recurrent transition

**Purpose:** Implement the agent's memory and action-conditioned state update.

**Implement:** The recurrence `h_{t+1}=f(h_t,z_t,a_t)` using the M0 recurrent architecture. Define separate single-step and sequence interfaces, reset masks, tensor layouts, and initialization. M2 can use synthetic stochastic states before the encoder exists.

**Validation gate:** Single-step iteration agrees with the sequence implementation on identical inputs. Changing prior actions or latent states affects the subsequent hidden state. Resetting one batch element does not alter others. Gradients remain finite through a short recurrent sequence.

**Deliverable:** Tested recurrent transition with explicit shape and state contracts.

## M3 — Visual encoder, posterior, and dynamics prior

**Purpose:** Construct latent states from observations and predict latent states without observations.

**Implement:** A CNN encoder and separate categorical heads. At time t, the posterior receives the current recurrent state and encoded image; the prior receives the current recurrent state. Implement straight-through categorical sampling, uniform mixing, stable log probabilities, and explicit aggregation over factors. Follow the pinned algorithm mapping from M0.

Expose two paths: an observation-conditioned update for real experience, and a prior-only update for prediction. Define the complete model state as the concatenation of deterministic and stochastic features.

**Validation gate:** Probabilities normalize and sampled factors are valid one-hot values. The sampling estimator carries gradients to logits. Altering the image changes the posterior; the prior-only API has no image input. Empirical samples agree with a small known categorical distribution within sampling tolerance.

**Deliverable:** Observation and prediction APIs plus latent entropy diagnostics.

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

## M10 — Reproduction results and imagination-horizon experiment

**Purpose:** Turn a working implementation into quantitative evidence.

### A. Final run matrix

| Task | Imagination horizon | Independent training seeds | Runs |
|---|---:|---|---:|
| Walker Walk pixels | 5 | 0, 1, 2 | 3 |
| Walker Walk pixels | 15 | 0, 1, 2 | 3 |
| Walker Walk pixels | 30 | 0, 1, 2 | 3 |
| Cartpole Swingup pixels | 15 | 0, 1, 2 | 3 |
| **Total main experiment** | | | **12** |

The Walker H=15 runs serve both as reproduction results and as the middle horizon condition. Cartpole demonstrates a second task under the shared learning settings; it is not a transfer experiment. Pilots, random-policy evaluation, and reference-implementation runs are outside this 12-run count.

### B. Controls

- Change only imagination horizon across Walker conditions. Hold model capacity, replay capacity, sequence length, batch size, number of imagination starts, optimizer settings, environment budget, and update schedule fixed.
- Use the same training seed labels across conditions, while recognizing that policies and collected trajectories will diverge. Give components separate RNG streams where practical so extra imagined steps do not directly consume the environment's random stream.
- Keep actor/critic reductions normalized consistently across horizons. Record the effective weighted sample counts so a longer horizon does not inadvertently multiply the loss scale.
- Longer rollouts inherently add imagined transitions and compute. This is an equal-real-data comparison, not an equal-compute comparison. Report cumulative imagined transitions and actual runtime.
- Rotate run order across conditions where practical. Record interruptions, failures, and reruns. If a code fix affects results, rerun every affected condition on the corrected version.
- Freeze a common diagnostic corpus after M9, using separate episodes collected by random and development policies. Exclude it from training. Evaluate every final model on the same contexts and recorded actions.
- Do not run a “without imagination” condition that removes the actor's learning mechanism. H=5 is a short-horizon comparison, not a model-free baseline.

### C. Metrics and evaluation

| Question | Required evidence |
|---|---|
| Does the agent learn? | Return versus training control steps; final-checkpoint return on 20 evaluation episodes per training seed |
| How efficiently does it use real data? | Area under the evaluation learning curve over the fixed training budget; any threshold-based metric defined before final runs |
| What does horizon change? | Per-seed final return and learning-curve area for H=5,15,30 |
| How far does prediction remain useful? | Reward MAE versus prediction distance on the common diagnostic corpus; secondary image/continuation diagnostics |
| What does it cost? | Return versus elapsed training time; total elapsed time including evaluation; peak VRAM; update time; cumulative imagined transitions |
| How stable are results? | All individual training-seed results, mean, standard deviation, and failures |

Use a reserved final evaluation seed set and the fixed final checkpoint. Do not choose the best checkpoint from final evaluation returns. Aggregate episodes within each training seed before aggregating across training seeds. Three training seeds give limited evidence about variability; additional evaluation episodes do not create additional independent training runs.

Refresh on-policy prediction diagnostics separately if useful, but keep them distinct from the common-corpus comparison because the data distributions differ. Associate prediction error with control performance descriptively. A correlation does not establish that prediction error alone caused a return difference.

### D. Reference comparison and interpretation

Use the random policy as a basic floor. Where resources permit, add three seeds of the pinned author implementation with matched wrappers, observation access, data budget, and closely matched capacity. Record residual differences. Published results with larger models or different training settings provide context only.

The main study can establish how the reduced implementation behaves across horizons. It cannot establish superiority to model-free RL without an appropriate baseline, and it cannot isolate a pure horizon effect at fixed compute because longer horizons require more computation.

Possible findings include improvement with longer horizons, a plateau, a decline, or no clear difference at this sample size. Report the observed outcome. If H=30 performs worse, investigate prediction error and optimization behavior before attributing the result to model exploitation. A null result with clear measurement and limitations remains a valid outcome.

**Validation gate:** All planned runs have complete results or documented failure outcomes; figures regenerate from raw logs; comparisons preserve the frozen settings; and each conclusion is limited to the evidence collected.

**Deliverable:** Final experiment report, aggregate tables, per-seed curves, prediction diagnostics, resource measurements, and representative behavior videos.

## Final project package and resume evidence

The completed repository should include:

- Independent implementation with attribution for consulted or reused code.
- Pinned dependencies, configurations, and commands for development checks, training, evaluation, and figure generation.
- A short architecture explanation and a deviation table linking the reduced implementation to the selected sources.
- Raw per-run logs, the experiment manifest, compact result tables, and representative checkpoints.
- Three central figures: learning versus real data, learning versus time, and prediction error versus horizon.
- Videos showing real policy behavior and paired open-loop predictions, with the selection rule stated.
- A concise discussion of one diagnosed implementation failure, its evidence, and its resolution.

The eventual resume entry should draw from two completed outcomes: **demonstrated visual-control learning at a measured scale** and **a measured horizon–performance–compute result**. Record actual task returns, data budgets, parameter counts, runtime, and variability before writing accomplishment claims. Strong implementation and a careful experiment are sufficient; architectural novelty is not a requirement.

The immediate implementation task is **M0**, followed by the environment/replay contract in **M1**. Additional architectures, simulators, task suites, or kernel optimization are outside this plan unless the completed experiment exposes a concrete need.
