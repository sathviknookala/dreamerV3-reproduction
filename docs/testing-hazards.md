# Domain testing hazards

**Read before writing a test or a gate check for any component listed below, and whenever a new
assertion passes on the first try.** Every entry is a bug this project's design can actually produce
where the obvious assertion **passes while the bug survives**. Most were found by mutation: the
mutant was written, the suite stayed green, and the test was strengthened until it failed.

The general rule lives in [CLAUDE.md](../CLAUDE.md): *a test that passes where the bug cannot occur
is not a test — confirm it fails without its fix.* This file is that rule's accumulated specifics.

---

## Vacuous at initialization

Three of this project's output kernels are zero-initialized, which makes whole classes of assertion
pass for the wrong reason. This is the single most productive source of false confidence here.

| Head | `outscale` | Consequence |
|---|---|---|
| `rew` | **0.0** | The two-hot loss is exactly `log(255)` for any target, and **no gradient reaches the encoder, the RSSM, or even the head's own hidden layer**. Any reward-alignment or reward-routing test passes vacuously. |
| `val` | **0.0** | The readout is exactly 0 for every latent, so `∂L/∂feat` is exactly zero, **every** §6.3 routing assertion is vacuous, and the fast and slow critics are indistinguishable. |
| `pol` (`mean`, `stddev`) | **0.01** | **The exception.** Policy gradients are nonzero at init, so the actor needs no perturbation. |

- **Perturb `reward.mlp.out.weight` / `value.mlp.out.weight` before asserting anything about those
  heads.** M7 had two mutants survive its first suite — "slow critic as the bootstrap" and "`slowreg`
  dropped" — purely because fast and slow both read out 0.
- **The actor's exemption is easy to over-apply.** `pol` needs no perturbation, but an actor test
  running on a *real rollout* still sees an advantage of **exactly 0** unless the reward and value
  heads are perturbed, because the imagined reward, the return and the baseline are all zero. Every
  claim about the advantage, the baseline or the normalizer is then vacuous. The M8 gate perturbs
  both and says so.
- **A perturbed head is a pathological starting point for a *fitting* diagnostic.** With a ±4.85e8
  support, any nonzero output kernel reads out ~1e6. An early M7 gate fit a perturbed critic to its
  own self-bootstrapped targets and "passed" at MAE 2.7e5 against targets of 0.4. Fit from a fresh
  zero-init head against fixed targets instead.

## Environment and transition alignment

- **A `LAST → continue=0` assertion passes on Atari-style environments and is *wrong* on DMControl**,
  which ends episodes with a nonzero discount. Assert on the environment discount, not the boundary
  flag, and include both a time-limit and a true-terminal example.
- **Single-step and sequence recurrence paths drift apart silently.** Assert they agree on identical
  inputs; a shape-only test will not catch it.

## World model, RSSM and heads

- **Block-GRU gate extraction:** the 1536-wide projection must be reshaped to `(B, 8, 192)` *before*
  splitting into three gates. A flat `split(x, 3, -1)` gives a silently wrong, still trainable model.
  Assert the index map, not just the shapes.
- **`symexp_twohot` readout at init:** with zero-init output weights the softmax is uniform and the
  readout must be **exactly 0**. A naive float32 sum over ±4.85e8 bins returns −1.0 instead. Assert
  the value is 0, not that it is small.
- **The bin support is not pinned by a round-trip test.** Encode/decode agrees for any support, so
  narrowing `symexp(±20)` survives unless the endpoints themselves are asserted.
- **Encoder scales `x/255 − 0.5`; the decoder target is `x/255` with no shift.** The asymmetry is
  real. A test that only checks output range passes either way.
- **Falling reconstruction loss is not evidence of a working world model.** Assert on open-loop
  reward MAE at prediction distance ≥ 5 against a training-set-mean predictor, not on pixel error.

## Imagination and open-loop prediction

- **Imagination accidentally calling the posterior leaks future observations** and looks like
  excellent prediction. Assert **structurally** that no observation tensor can reach the prior path.
- **A structural no-observation check must test substrings, not a name set.** A `future_observations`
  argument added to `imagine_trajectory` survived a check comparing parameter names against
  `{"observation", "observations", "image", "embed"}`, because it equals none of them. Assert that no
  parameter name *contains* `observ` / `image` / `frame` / `embed`.
- **A leaking open-loop rollout needs two different detectors.** An `open_loop_predict` that accepts
  a `future_observations` argument is caught by the encoder tripwire and the signature check but
  **not** by the frame-corruption check; a `gather_contexts` that slides one future frame into the
  context window is caught **only** by the corruption check. Both are in the M5 gate, and each was
  confirmed to pass the other's mutant.
- **Latent sampling is stochastic.** Average metrics over multiple latent samples per context rather
  than selecting attractive rollouts, and fix the seed before comparing conditions.

## Critic and returns

- **A zero hole in the replay bootstrap corrupts every *earlier* target** through the backward λ
  recursion, and the position weight masks only the hole's own position. Holes are safe only because
  the drop criterion is `is_terminal`, where `live[:, t−1]` is exactly zero. `check_bootstrap_holes`
  enforces that at runtime; pass it the `filled` mask rather than trusting the caller.
- **An all-zero feature fixture cannot see a feature misalignment.** The replay critic training on
  `feat[:, 1:]` instead of `feat[:, :-1]` survived M7's first suite because its replay fixture used
  zero features and a zero-init head.
- **A single-column return fixture cannot see which column the scatter takes.** `ret[:, -1]` in place
  of `ret[:, 0]` survived until the fixture had more than one imagined step.

## Actor and behaviour

- **`sg(sampled action)` is invisible on the imagination path.** Actions leave
  `imagine_trajectory`'s `no_grad` block already detached, so removing the detach in the actor loss
  changes nothing and passes. Test it by handing the loss an imagination whose actions deliberately
  carry a graph.
- **Differential entropy is negative for `σ < 1/√(2πe)`.** "Entropy remains finite" is not "entropy
  stays positive"; asserting positivity fails on a *correctly* collapsing Gaussian. Assert the
  closed-form `minstd` / `maxstd` bounds instead.
- **A correlation is not an acceptance criterion for a policy-gradient update.** `corr(Δlogπ, Â)`
  over many positions measures how well one network fits them, not whether the estimator is right —
  it saturated at 0.13–0.50 purely as a function of ascent length. The capacity-free statement of
  "updates increase the probability of better actions" is `Σ w·Â·Δlogπ > 0`.
- **"Actions stay in bounds" is about *executed* actions.** The policy is deliberately unsquashed, so
  raw Gaussian samples leave `[-1, 1]` by design. Assert the two boundaries that do enforce it —
  `DMCEnv.step`'s clip and the RSSM's `a / max(1, |a|)` — not the sample.
- **A near-constant policy hides an alignment bug.** At `outscale: 0.01` the policy barely depends on
  its input, so "perturbing state *t* moves the loss" can fail by numerical accident. Scale the
  output kernels up to make the fixture nondegenerate.

## Numerics and reductions

- **TF32 breaks hand-computed fixtures.** Measured on this GPU: max|GPU−CPU| on a 2048² fp32 matmul
  is **6.8e-02 with TF32 on** and **1.3e-04 off** — a 500× difference. Set
  `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False` in
  any test asserting against analytic values. `tests/test_rssm.py`, `test_critic.py` and
  `test_actor.py` all set both.
- **A position mean and a weight-normalized mean agree whenever the weights are all ones.** Every
  reduction test needs a fixture with at least one zero or fractional weight, or the
  `sum / weight.sum()` mutant survives.
- **A metric that reports `weight.numel()` does not pin the reduction.** It is unchanged by the
  mutant it looks like it tests.
