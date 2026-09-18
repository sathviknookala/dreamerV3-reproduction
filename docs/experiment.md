# M10 — Reproduction results and imagination-horizon experiment

**Read before designing a final run or writing any result claim.** Relocated verbatim from
`dreamerv3_implementation_plan.md` (preserved at commit `da9a55a`). The M9 gate must pass before
anything here is executed — see [milestones.md](milestones.md).

---

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

The project should produce evidence of three capabilities: **implementing a research algorithm**,
**diagnosing interactions between its components**, and **conducting a controlled experiment**. It
complements existing representation-learning, probabilistic-modeling, CUDA, and serving work by
adding action-conditioned dynamics, visual perception, and sequential decision-making.

The completed repository should include:

- Independent implementation with attribution for consulted or reused code.
- Pinned dependencies, configurations, and commands for development checks, training, evaluation, and figure generation.
- A short architecture explanation and a deviation table linking the reduced implementation to the selected sources.
- Raw per-run logs, the experiment manifest, compact result tables, and representative checkpoints.
- Three central figures: learning versus real data, learning versus time, and prediction error versus horizon.
- Videos showing real policy behavior and paired open-loop predictions, with the selection rule stated.
- A concise discussion of one diagnosed implementation failure, its evidence, and its resolution.

The eventual resume entry should draw from two completed outcomes: **demonstrated visual-control learning at a measured scale** and **a measured horizon–performance–compute result**. Record actual task returns, data budgets, parameter counts, runtime, and variability before writing accomplishment claims. Strong implementation and a careful experiment are sufficient; architectural novelty is not a requirement.
