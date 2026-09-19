# M6 — latent imagination engine

**Read before quoting an imagination cost, or before touching the rollout path.** Gate passed
2026-09-18, 55/55: [`gate-2026-09-18.txt`](gate-2026-09-18.txt).

## What was run

| | |
|---|---|
| Model | `WorldModel(action_dim=6)` at **initialization** — no training, no checkpoint |
| Data | 1000 uniform-random Walker Walk transitions, one B=16 / P=5 / T=64 replay batch |
| Starts | posterior pass over the batch, then every loss-bearing non-terminal position: **1024 rollouts** |
| Actions | `RandomActionProvider`, uniform on [−1, 1] |
| Horizons | **H ∈ {5, 15, 30}**, 5 timed repeats each |
| Device | RTX Pro 4000, float32, TF32 disabled, eager |

Reproduce (~1 minute, collects its own data):

```bash
.venv/bin/python scripts/m6_gate.py
```

**The reward head is perturbed in the gate.** `outscale: 0.0` reads out exactly 0 for every latent,
so alignment and action-sensitivity checks would pass vacuously against an untouched head. The gate
draws `reward.mlp.out.weight ~ N(0, 0.5)` first. **Every reward magnitude in this milestone is
therefore the output of a random head and means nothing** — only the *differences* it makes
observable are being tested.

## Per-horizon cost — [`horizon-cost-2026-09-18.csv`](horizon-cost-2026-09-18.csv)

1024 rollouts per batch, prior-only, no decoding, no gradient.

| H | states | ms / batch | µs / state | peak MiB |
|---:|---:|---:|---:|---:|
| 5 | 6 144 | 3.8 | 0.613 | 203.7 |
| 15 | 16 384 | 10.6 | 0.647 | 463.2 |
| 30 | 31 744 | 20.9 | 0.658 | 858.3 |

Cost is **linear in H** to within 8% per state; memory likewise. `H=30` at the full 1024-rollout
width costs **858 MiB**, which settles [spec.md §10-8](../../docs/spec.md)'s H=30 memory question for
the imagination tensor itself: it fits with room to spare on 24 GB.

**What these numbers exclude, and why they are not the M9 profile.** They are one imagination call
on an untrained model: no posterior context pass (run once, outside the timed region), no actor, no
critic, no λ-returns, no backward pass, no environment interaction, no replay transfer, and no
decoding. A trained model's rollout costs the same — the cost is structural — but a *training step*
does not. The per-stage steady-state profile is M9's and does not exist yet.

## What the gate establishes

- **Alignment.** `H+1` states, `H` actions, `H` rewards, `H` continuations, `H+1` weights, with
  state 0 equal to the replay start. Reward and continuation at step *i* are read from the state
  that action *i* produced — the same convention as M5's `target_indices`.
- **Prior only.** Zero encoder calls during imagination (M5's `encoder_tripwire`), zero simulator
  calls (a tripwire on `env.step`), and no parameter of the imagination API whose name contains
  `observ`, `image`, `frame`, `embed` or `pixel`.
- **Reuse, not duplication.** A `SequenceActionProvider` fed the same actions and the same seeded
  generator reproduces `RSSM.imagine` **bitwise**, so M6 did not fork M5's transition path.
- **Start exclusion.** Burn-in positions and terminal states are not rollout starts; 1024 of 1104
  candidate positions survive, which is exactly the 64 loss-bearing positions × 16 sequences of
  [spec.md §4.10](../../docs/spec.md). A time-limit boundary (`is_last` with nonzero discount) **is**
  a valid start — the mask is on `is_terminal`, never on `is_last`.
- **Weights.** `weight = cumprod(disc·con)/disc` over the `H+1` positions, so `weight[0] == con[0]`,
  not 1 ([spec.md §5.4](../../docs/spec.md)). A collapsing continuation zeroes every later weight.
- **Gradient routing.** The trajectory is built under `no_grad`: `sg` over the start state and the
  whole imagined path, so the actor gets no pathwise gradient through the dynamics
  ([spec.md §5.9](../../docs/spec.md)). The heads run outside that block, so their own parameters stay
  differentiable.

## Mutation evidence

Ten mutants of `src/dreamer/imagine.py` were each confirmed to fail `tests/test_imagine.py`: reward
read one step early, continuation read one step early, a weight that drops the start continuation,
`disc` hard-coded to 1, terminal starts not excluded, burn-in not excluded, an off-by-one in start
selection, the trajectory left differentiable, the provider polled once and its action reused, and a
`future_observations` argument added to `imagine_trajectory`.

The last one **survived the first version of the suite**: the signature check tested a fixed set of
parameter names, and `future_observations` is not in it. The check now tests **substrings**, which is
the form M5 used and the form that catches it.

## What this milestone cannot claim

Nothing about behaviour, return, or prediction quality. The model is untrained, the actions are
random, and there is no actor, critic or λ-return. The imagined rewards and continuations are the
outputs of an untrained head over an untrained prior — the gate tests that they are *correctly
placed and finite*, not that they are *right*. M5's held-out prediction numbers remain the only
evidence that the prior predicts anything, and they describe a uniform-random data distribution.
