# M8 — actor and imagined behaviour learning

**Read before quoting an entropy, a log-probability, a return-normalization scale or a parameter
count, or before touching the policy path.** Gate passed 2026-09-19, **63/63**:
[`gate-2026-09-19.txt`](gate-2026-09-19.txt), on top of **38** unit tests in `tests/test_actor.py`.

## What was run

| | |
|---|---|
| Model | `WorldModel(action_dim=6)` + `Critic` + `Actor` at **initialization** — no training, no checkpoint |
| Data | 1000 uniform-random Walker Walk transitions, one B=16 / P=5 / T=64 replay batch |
| Imagination | 1024 starts, **`ActorActionProvider`**, H=15 — the actor replaces M6's random provider |
| Actor | 640→64→64→64 trunk, separate `mean`/`stddev` heads at `outscale 0.01`, `actent = 3e-4` |
| Return norm | `perc`, rate 0.01, limit 1.0, percentiles 5/95, **`debias: False`** |
| Seeds | torch 0, env 0, replay 0, imagination 0, provider 0 |
| Peak VRAM | **1928.5 MiB** — posterior + 1024 actor-driven rollouts at H=15 + all three losses |

Reproduce (~1 minute, collects its own data; both outputs reproduce bitwise):

```bash
.venv/bin/python scripts/m8_gate.py
.venv/bin/python results/m8/param_count_measure.py
```

## Parameter count — [`param-count-measured-2026-09-19.txt`](param-count-measured-2026-09-19.txt)

`pol` **50,316**, equal to the §4.11 derivation. With the world model's 570,419 (M4) and `val`'s
66,111 (M7), the trainable agent total is **686,846**, equal to the derived figure and excluding the
untrained `slowval` mirror. **§10-7 is now fully discharged** — every row of the §4.11 table is
measured, none is derived-only. The Cartpole actor is 49,666, exactly `2 × 5 × (64 + 1)` fewer.

> **This remains not evidence of anything about capability.** 686,846 is far below the paper's
> smallest evaluated row of 12M, and the name `size1m` still names no count.

## The two fixtures that decide the signs — [spec.md §6.2](../../docs/spec.md)

| Fixture | Measured |
|---|---|
| η=0, advantages +1 / −1 on one shared state | `logπ(+1)` −1.435319 → **−0.805891**, `logπ(−1)` −1.430359 → **−3.313280** |
| Â=0, η=3e-4 | entropy 1.30550516 → **1.30556691**, and the mean head's gradient is **exactly 0** |

The second is the one that fails if entropy is *added* in the minimized loss. Both are asserted in
`tests/test_actor.py` as well, and both use a zero-init critic so the baseline is exactly 0 and the
advantage is exactly the return.

## `tarval` resolved against the pin

`agent.py#L400` is `tarval = slowval if slowtar else val`, and `configs.yaml#L108` sets
`imag_loss.slowtar: False`. So `tarval` **is the fast critic**, and it is used twice — as the
λ-return bootstrap (`agent.py#L405`) and as the actor's baseline (`agent.py#L408`). The slow critic
enters neither; it is a regularizer only. The gate asserts the baseline equals
`critic.value.predict(feat)[:, :-1]` with fast and slow separated by **8.04e6**, and that the return
reconstructed from that same prediction matches `imagined_loss`'s `ret` bitwise — the returns and the
baseline see one pre-update critic state.

`valnorm` and `advnorm` are `impl: none`, so `adv_normed = adv` and `tar_normed = ret`: there is no
advantage standard-deviation normalization and no percentile offset subtracted from the advantage.
`roffset` exists and is read only by a metric.

## `debias: False` — where the task brief and the pinned source disagree

`embodied/jax/utils.py#L22` defaults `Normalize.debias = True`, but **`configs.yaml#L111` overrides
it to `False` for `retnorm`**, and `retnorm` is the only `Normalize` with `impl != none`. The
uncorrected branch is therefore the live one. Measured in the gate on `linspace(0, 100, 101)`, whose
5th and 95th percentiles are exactly 5.0 and 95.0:

| | after one update | S |
|---|---|---:|
| **Pinned, `debias: False`** | `lo` 0.0500, `hi` 0.9500 | **1.0000** — the `limit` floor, not the spread |
| `debias: True` (not the pin) | `corr` 0.0100, factor 100 | 90.0000 — the true percentile spread at once |

The difference is a startup transient: uncorrected, `S` sits at its floor of 1 and the EMAs climb
geometrically, so early advantages are unnormalized. Both branches are implemented and both are
asserted; the constructor default is the pinned `False`. **The brief's instruction to preserve bias
correction was followed as far as the pin allows — the state and the corrected branch exist and are
tested — but making it the default would deviate from `configs.yaml` and is not done.** The second
uncorrected update is hand-checked at `0.99·0.95 + 0.95 − (0.99·0.05 + 0.05) = 1.791`.

## What the gate establishes

- **`bounded_normal` is a plain diagonal Gaussian.** `σ = 0.9·sigmoid(raw + 2.0) + 0.1` measured at
  **0.892717** for a zero pre-activation; `tanh` on the mean only; `log_prob` of an out-of-range
  action is the plain Gaussian with **no Jacobian term**; entropy is the closed form summed over the
  6 action dims. Raw samples leave `[-1, 1]` — measured max **4.2156**.
- **"Actions stay in bounds" means *executed* actions, not raw samples.** The policy is deliberately
  unsquashed. Two boundaries outside the actor enforce the range and both are asserted: `DMCEnv.step`
  clips to the action spec before the simulator sees it, and the RSSM divides by `max(1, |a|)`, so an
  action of 3.0 drives identical dynamics to 1.0.
- **Alignment.** H actions against H+1 states: the policy runs on `feat[:, :-1]` and `logπ`/entropy
  are `(1024, 15)` with **no second slice**. The reference's `[:, :-1]` exists because it builds H+1
  actions; transcribing it here would drop a position. Perturbing the final state leaves the policy
  loss unchanged; perturbing the first state moves it.
- **Gradient routing** ([spec.md §5.9](../../docs/spec.md)): `policy` reaches the actor trunk and
  both heads and **not** the encoder, RSSM, decoder, reward head, continuation head, fast critic or
  slow critic. `value` does not reach the actor, `repval` does not reach the actor, and `repval`
  still reaches the encoder and RSSM. The blocked path is asserted, not enforced with a new detach —
  `imagine_trajectory` already builds under `no_grad`, and `imagined_actor_loss` **raises** if it is
  handed features that carry a graph.
- **The actor needs no "perturb first" workaround.** `outscale: 0.01` leaves policy gradients nonzero
  at initialization, unlike the reward head and the critic at `outscale: 0.0`. The reward and value
  kernels *are* perturbed for the integration section, because at zero-init the imagined reward, the
  return, the baseline and hence the **advantage** are all exactly 0 and every claim about them would
  hold vacuously.

## REINFORCE at scale — [`actor-update-2026-09-19.csv`](actor-update-2026-09-19.csv)

A **fresh zero-init critic**, so the baseline is exactly 0 and the advantage is exactly `ret / S`,
with returns a fixed seeded draw on `U(−1, 1)`. The perturbed reward head above emits ~1e7 and would
say nothing about the estimator.

One SGD step over the 15,360 real imagined positions moves the **advantage-weighted** log-probability
by **+0.003254** and the policy loss from −0.00921886 to −0.00921908.

> **Why that quantity and not a correlation.** `corr(Δlogπ, Â)` was measured and rejected as an
> acceptance criterion: one 50,316-parameter network cannot raise `logπ` independently at 15,360
> positions, so the correlation saturates around 0.13–0.50 depending only on how long the ascent
> runs. It measures approximator capacity, not whether the estimator is right.
> `Σ w·Â·Δlogπ > 0` is the capacity-free statement of "updates increase the probability of better
> actions" and held at every learning rate tried (1e-4, 1e-3, 1e-2).

The 100-step LaProp trajectory in the CSV is **monitoring only**. With a fixed advantage, no critic
learning and no new data, unconstrained ascent drives `σ` to `minstd` within ~50 steps: entropy falls
7.8326 → −2.9465 and `logp_mean` to −345. **A negative differential entropy is correct** for a
Gaussian with `σ < 1/√(2πe)`, not a failure; the gate asserts entropy stays inside the closed-form
bounds `[−5.3019, 8.5136]` implied by `minstd`/`maxstd`, which is the real detector of a broken
parameterization. The world model and the critic are **bitwise unchanged** across the run, and the
normalizer's `hi` matches `(1 − 0.99¹⁰⁰)·P₉₅` to 1e-4 — exactly one update per step.

## Mutation evidence

Twelve mutants of `src/dreamer/actor.py`, each confirmed to fail `tests/test_actor.py`: entropy added
instead of subtracted; the loss sign flipped; the `+2.0` offset dropped from `σ`; `tanh` moved onto
the sample; `logπ`/entropy meaned over action dims instead of summed; the policy reading
`feat[:, 1:]`; the baseline taken from the slow critic; a weighted mean instead of the position mean;
the `retnorm` floor dropped; the EMA rate inverted; `debias` defaulted to `True`; and the sampled
action left attached.

The last was the interesting one — it **survived** the first suite, because actions produced inside
`imagine_trajectory`'s `no_grad` block already have `requires_grad=False`, which makes the `sg(act)`
in §5.6 a no-op on this path and therefore invisible. A test that hands the loss an imagination whose
actions *do* carry a graph was added until it failed.

## What this milestone cannot claim

**No return, no policy quality, no learning.** The world model is untrained, the critic is untrained
or deliberately perturbed, the returns in the update diagnostic are a synthetic draw, and **no
environment is stepped after the 1000 random warm-up transitions**. Every entropy, log-probability
and loss here is a property of a randomly initialized actor on a randomly initialized world model.
M8's gate text treats an increase in imagined return as an integration diagnostic only; real
environment performance is M9's gate and does not exist yet.

Re-running `scripts/m8_gate.py` **overwrites** `actor-update-2026-09-19.csv` and, if redirected,
`gate-2026-09-19.txt`. Both reproduce bitwise on this machine, but `git checkout` them after a
regression run unless a re-measurement was intended.
