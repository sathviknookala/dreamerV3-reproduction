# Frozen algorithm specification (M0 deliverable)

**Read before writing any model code, and whenever the paper and the reference implementation appear
to disagree.**

> **STATUS: SKELETON — M0 IS NOT COMPLETE.** Every `TBD` below is an open decision, not a
> formality. The purpose of this document is to make implementation choices traceable and to prevent
> accidental mixing of Dreamer variants; a `TBD` filled in from a plausible-sounding assumption
> defeats that purpose entirely. Fill each entry from the pinned paper or the pinned commit, and
> record which one, or leave it `TBD`. **No model code should be written against this file until the
> M0 validation gate in [milestones.md](milestones.md) passes.**

## 1. Pinned sources

| Source | Pin | Recorded |
|---|---|---|
| Paper | [Mastering Diverse Domains through World Models, arXiv v2](https://arxiv.org/pdf/2301.04104v2) | Pinned |
| Reference implementation | [danijar/dreamerv3](https://github.com/danijar/dreamerv3) @ `TBD` | TBD — record a concrete commit SHA before consulting the code |
| Scale reference only | [`configs.yaml`](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml) `size1m` | Not authoritative — current defaults are not equivalent to the pinned paper version |

The paper is the specification. The repository is a reference consulted under an explicit
interpretation note, never a silent tie-breaker.

## 2. Version manifest

TBD — Python, PyTorch, CUDA, driver, `dm_control`, MuJoCo, and the GPU/CPU/RAM actually used. Record
exact versions; "latest" is not a version.

## 3. Paper-to-code mapping

One row per component. **Where the paper and the reference code differ, record the chosen
interpretation and why** — that column is the whole point of the table.

| Component | Paper (arXiv v2) | Reference code | Chosen | Interpretation note |
|---|---|---|---|---|
| Recurrent cell | TBD | TBD | TBD | TBD |
| Encoder CNN | TBD | TBD | TBD | TBD |
| Decoder | TBD | TBD | TBD | TBD |
| Posterior head | TBD | TBD | TBD | TBD |
| Prior head | TBD | TBD | TBD | TBD |
| Reward head | TBD | TBD | TBD | TBD |
| Continuation head | TBD | TBD | TBD | TBD |
| Actor | TBD | TBD | TBD | TBD |
| Critic | TBD | TBD | TBD | TBD |
| Normalization | TBD | TBD | TBD | TBD |
| Initialization | TBD | TBD | TBD | TBD |
| Optimizer | TBD | TBD | TBD | TBD |

## 4. Preserved V3 methods

Each must be implemented or explicitly classified as a deviation in §6. None may be quietly dropped.

- [ ] Categorical straight-through sampling
- [ ] Uniform mixing of categorical probabilities
- [ ] Separately weighted dynamics KL and representation KL
- [ ] Free bits (free-nats threshold)
- [ ] Transformed reward/value prediction (distributional)
- [ ] Return normalization
- [ ] Entropy regularization
- [ ] Slow-critic regularization
- [ ] v2 replay critic objective — include, **or** classify its omission as a deviation in §6

## 5. Loss reductions

Reduction order is load-bearing and is a known source of silent scale errors.

- **Categorical KL:** sum over the 32 stochastic factors **before** applying the free-nats
  threshold; then reduce over valid batch/time positions. Applying free bits per-factor is a
  different objective.
- **Reconstruction:** TBD — reconcile the reduction with the loss weight explicitly rather than
  inheriting a framework default (`mean` vs `sum` over pixels changes the effective weight by
  orders of magnitude).
- **Masking:** TBD — define which positions are valid. Burn-in prefix positions and reset
  observations without a preceding reward target are excluded from loss-bearing counts.
- **Actor/critic:** TBD — record the effective weighted sample count so a longer imagination horizon
  does not inadvertently multiply the loss scale across the H∈{5,15,30} conditions.

## 6. Deviation table

Every capacity reduction and algorithmic deviation from the pinned paper, labeled deliberately.

| # | Deviation | Paper | This implementation | Reason | Expected effect |
|---|---|---|---|---|---|
| 1 | Scale | TBD | 512 deterministic, 32×4 categorical | Single 24 GB GPU | TBD |
| 2 | Task coverage | Cross-domain, shared config | 2 DMControl tasks | Scope | Supports a limited reproduction claim only |
| — | TBD | | | | |

**Measured parameter count: TBD.** Do not call this implementation "1M parameters" from the `size1m`
preset name — count the actual parameters and record the artifact the count came from.

## 7. Environment and counter definitions

TBD — observations, action bounds, rewards, continuation targets, episode boundaries, time limits,
evaluation action selection, and **every step counter**. Native control steps, agent decisions, and
physics substeps are distinct quantities and must be logged separately; conflating them silently
misstates the data budget.

## 8. Seeds

| Purpose | Seeds | Rule |
|---|---|---|
| Development | TBD | Reserved — must not appear in final results |
| Final training | 0, 1, 2 | Fixed by [experiment.md](experiment.md) |
| Final evaluation | TBD | Reserved set, chosen before final runs |

Give components separate RNG streams where practical, so extra imagined steps at H=30 do not consume
the environment's random stream and confound the horizon comparison.
