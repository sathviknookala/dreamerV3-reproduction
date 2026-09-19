# M7 — critic and return targets

**Read before quoting a value, a λ-return or a critic parameter count, or before touching the return
path.** Gate passed 2026-09-18, 37/37: [`gate-2026-09-18.txt`](gate-2026-09-18.txt), on top of 39
unit tests in `tests/test_critic.py`.

## What was run

| | |
|---|---|
| Model | `WorldModel(action_dim=6)` + `Critic(in_features=640)` at **initialization** — no training, no checkpoint |
| Data | 1000 uniform-random Walker Walk transitions, one B=16 / P=5 / T=64 replay batch |
| Imagination | 1024 starts, `RandomActionProvider`, **H=15** |
| Imagined critic | weight 1.0, `disc=1`, fast-critic bootstrap, target `sg(λ-return)`, (1024, 15) |
| Replay critic | weight 0.3, `disc = 1 − 1/333`, true flags, bootstrap = the imagined return, (16, 63) |
| Peak VRAM | **1259.5 MiB** — posterior + 1024 rollouts at H=15 + both critic losses |

Reproduce (~1 minute, collects its own data):

```bash
.venv/bin/python scripts/m7_gate.py
.venv/bin/python results/m7/param_count_measure.py
```

## Parameter count — [`param-count-measured-2026-09-18.txt`](param-count-measured-2026-09-18.txt)

`val` **66,111**, equal to the §4.11 derivation. The `slowval` mirror is a second 66,111,
**excluded from the optimizer and untrained** — §4.11 already accounts for it that way. World model
+ `val` = **636,530**; the agent total closes at 686,846 with `pol` 50,316, which is **still derived,
owed at M8**.

## The two silent readings of γ — the reason this milestone is arithmetic-first

`disc=1` on the imagined path with the continuation head carrying the discount ([spec.md §5.4](../../docs/spec.md)).
Constant reward r=1, λ=1, no terminal, H=15, measured in the gate:

| Reading | R₀ | |
|---|---:|---|
| **Correct** — soft label, `disc=1` | **14.688751** | matches `Σ_{j<15} 0.996997ʲ` |
| γ **double-counted** — soft label and `disc=1−1/333` | 14.386389 | matches `Σ_{j<15} 0.996997²ʲ` |
| γ **dropped** — hard 0/1 label, `disc=1` | 15.000000 | the undiscounted sum |

Three separated values from exact arithmetic. All three train without raising; only the fixture
distinguishes them. The same separation is asserted in `tests/test_critic.py`.

## Transform order — recorded per the milestone text

The critic readout is **`Σ pᵢbᵢ` with no outer `symexp`** — expectation of the value-space bins, not
a transform of an expectation ([spec.md §5.5](../../docs/spec.md)). Measured in the gate on the same
bimodal distribution as the M0 artifact (mass 0.5 at 0, 0.5 near 100): this implementation reads
**47.6240**; v1's symlog-uniform bins with `symexp(Σ pᵢbᵢ)` would read ~9.05. These are different
estimators, and the milestone requires the choice to be on the record. **No unbiasedness claim
follows** — §5.5's own caveat applies unchanged to the critic.

## Value versus target — [`value-vs-target-2026-09-18.csv`](value-vs-target-2026-09-18.csv)

300 LaProp steps at lr 1e-3 on **fixed** targets: the discounted sum of **real** Walker rewards over
the 63 supervised replay positions, range [0.011, 3.185], mean 0.4029. No bootstrap, so the target
does not move.

| step | loss | value mean | value MAE | slow gap |
|---:|---:|---:|---:|---:|
| 0 | 11.08 | **0.0000** | 0.4029 | 0.0000 |
| 90 | 2.40 | 0.3440 | 0.1692 | 0.0606 |
| 299 | 1.31 | 0.3907 | **0.0538** | 0.0409 |

The critic starts at **exactly 0** — the `outscale: 0.0` zero-init readout — and closes 87% of the
gap to the target mean. World-model parameters are **bitwise unchanged** across the fit.

> **The fit deliberately uses a fresh, unperturbed critic.** The gate perturbs
> `value.mlp.out.weight` everywhere else so the routing assertions are not vacuous, but a perturbed
> head is a pathological starting point for *fitting*: with a ±4.85e8 two-hot support, any nonzero
> output kernel reads out ~1e6, and the diagnostic would measure recovery from a state training
> never occupies. An earlier version of this gate did exactly that and reported a "passing" MAE of
> 2.7e5 against targets of 0.4.

## What the gate establishes

- **The six §6.1 fixtures**, hand-computed, each failing if the bootstrap moves to `v[t]`: zero
  rewards → 0; constant reward → the closed form; true terminal at `t+1` → `R_t = r_{t+1}` exactly;
  time limit at `t+1` → `R_t = r_{t+1} + disc·V_{t+1}`, trace cut, **bootstrap at full weight**;
  λ=0 → one-step TD; λ=1 → Monte-Carlo to the seed. Output length `L−1`, `R_t` aligned with state `t`.
- **Index 0 of `rew`, `term` and `last` is never read.** Poisoning `rew[:, 0]` with −999, or setting
  `term[:, 0] = last[:, 0] = 1`, leaves the output bitwise identical — which is what makes the
  leading pad on M6's `(N, H)` rewards safe. `cont_start` reaches the **weight** only; `term[:, 1:]`
  is exactly `imagination.cont`.
- **Gradient routing** ([spec.md §5.9, §6.3](../../docs/spec.md)): `value` reaches the critic and
  **not** the encoder, RSSM or reward head; `repval` reaches the encoder **and** the RSSM and the
  critic, and not the decoder; the slow mirror receives no gradient from either.
- **The slow critic is a regularizer, never a bootstrap.** Asserted with fast and slow separated by
  5.4e6 — at initialization both read out exactly 0 and the claim is untestable.
- **The bootstrap scatter addresses the `(B, P+T)` grid.** `select_start_states` builds its mask from
  `loss_mask` of shape (B, P+T), so its index runs to **1103**, not 1023. Burn-in columns are never
  filled, and `check_bootstrap_holes` **raises** on a hole at a non-terminal position.

## The coupling that makes holes safe, which is load-bearing

A zero hole in the replay bootstrap corrupts **every earlier target**, because the λ recursion runs
backwards — measured in `tests/test_critic.py`, a hole at position 3 moves positions 0–2 by >0.1 —
and the position weight `f32(~is_last)[:, :-1]` masks only the hole's own position, never the
corrupted prefix. Holes are harmless here **only because** the drop criterion is `is_terminal`, where
`live[:, t−1] = (1 − term[:, t])·disc` is exactly zero and the same hole provably changes nothing.
`check_bootstrap_holes` enforces that at runtime rather than assuming it. **A future start-exclusion
rule that is not `is_terminal` silently corrupts the targets.**

## Mutation evidence

Fifteen mutants of `src/dreamer/critic.py` were each confirmed to fail the suite: bootstrap at
`v[t]`; λ-return output off by one; `is_last` as the terminal signal; slow critic as the bootstrap;
`slowreg` dropped; EMA rate inverted; EMA update seeing stale parameters; γ on top of `contdisc`;
hard 0/1 continuation with `disc=1`; `repval` at weight 1.0; `repval` features detached; the wrong
imagined weight slice; the hole guard never firing; the scatter marking everything filled; the
imagined target not detached.

**Two survived the first suite** — "slow critic as the bootstrap" and "`slowreg` dropped" — both
because the zero-init readout makes fast and slow identical at 0. Tests that perturb the fast head
to separate them were added until both failed.

## What this milestone cannot claim

No behaviour, no return, no policy. The actions are random, the world model is untrained, and the
actor does not exist, so **every value here is the value of a random policy under an untrained world
model**. The imagined `value_mae` of 2.1e7 in the gate is the output of a deliberately perturbed
head and means nothing. The gate's "without updating the actor" clause is vacuous at M7 — the
enforceable half is the bitwise world-model snapshot, which is asserted.

Return normalization is **not** implemented here: `retnorm` is the actor's advantage normalizer
(§5.6) and `valnorm`/`advnorm` are `impl: none`, so the critic carries no normalization state.
`imagined_loss` **returns** `ret` rather than consuming it, because M8's `retnorm` EMA updates where
`ret` is produced.
