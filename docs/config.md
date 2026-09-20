# Configuration and qualification rules

**Read before changing any hyperparameter or quoting a setting.** Relocated verbatim from
`dreamerv3_implementation_plan.md` (preserved at commit `da9a55a`).

**Freeze status: NOT FROZEN.** Every value below is a starting choice awaiting its qualification
rule. Final settings are frozen at the end of M9, before final seeds run. When that happens, replace
this banner with the freeze date and the run manifest path.

**Architecture is no longer a starting choice.** M0 closed on 2026-09-17: the exact layers,
activations, normalization, initialization and distributions are specified in
[spec.md §4](spec.md), and the objectives and gradient routing in [spec.md §5](spec.md). This file
governs the *empirical* settings below — capacity, ratios, budgets — which M9 still qualifies.
Where the two touch, spec.md is authoritative and this file records the qualification rule.

---

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
| Behaviour objective | REINFORCE, fast-critic baseline, entropy η=3e-4; `retnorm` percentile EMA (5/95, rate 0.01, `S = max(1, hi−lo)`) | Implemented and validated at M8 ([spec.md §5.6](spec.md)). The EMAs are **uncorrected** at the pin, so `S` sits at its floor of 1 for the first few hundred updates and early advantages are effectively unnormalized — **measured 2026-09-20: `S` leaves the floor and rises to 25.78 on Walker and 15.52 on Cartpole** by end of run ([`results/m9/pilot/`](../results/m9/pilot/)), so the floor is an early-training transient, not a stuck state. Qualified; no change needed before freezing |
| Collection | One environment first; action repeat 1 | Log native control steps separately from agent decisions and physics substeps |
| Replay | CPU-resident uint8 frames; initial capacity 500,000 transitions | Measure total RAM use and avoid duplicate image storage |
| Update ratio | Start with 64 replay training positions per collected transition | **Realized ratio measured at exactly 64.0** on every row of both full-budget pilots ([spec.md §10-10](spec.md)); the convention is loss-bearing positions only. The *definition* is discharged. **Learning is not yet qualified** — Walker consolidates, Cartpole does not, so this value is still a candidate cause and cannot be frozen yet |
| Initial random collection | 5,000 agent transitions | Preserve the same warm-up budget in final comparisons |
| Planning budgets | Walker: 1M control steps/run; Cartpole: 500K/run | Qualify cost and learning in M9; all Walker horizon conditions receive the same final budget |
| Diagnostic evaluation | Every 25,000 training control steps; 5 episodes at seeds 2000–2004; mean action; video from the first seed | Isolated simulator and generators; never touches replay, optimizer, normalizer or training counters. Costs 3.29 ms/control step, ~3.7% of a Walker run ([`results/m9/`](../results/m9/)) |
| Resume checkpointing | Every 25,000 control steps, taken at an episode boundary; latest 2 kept plus one compact final model | 5.744 GiB and 10.1 s per write at full occupancy, ~11.5 GiB peak disk per run. Raise the interval if a longer campaign makes 2.3% of wall clock matter ([`checkpoint-cost-2026-09-19.json`](../results/m9/checkpoint-cost-2026-09-19.json)) |
| Development seeds | 100, 101, 102 | Final training seeds 0–2 and final evaluation seeds 1000–1019 stay unused until M10 |
| Per-episode simulator seed | `episode_seed(base, episode_index)`, simulator rebuilt each episode | Makes the next episode a pure function of one persisted integer; costs 0.105 s per episode, ~0.6% of a Walker run |

The [author-maintained configuration](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/configs.yaml) supplies a useful compact scale reference through `size1m`, including deterministic size 512 and four classes. Its current defaults and architecture are not automatically equivalent to the pinned paper version. Record the actual parameter count rather than calling this implementation “1M parameters” from the preset name.

The initial update ratio and budgets are project choices, not claims about the paper's settings or guaranteed convergence. Runtime is estimated from measured pilots. If the compact system needs more capacity or updates, resolve that before the final study and apply the resulting configuration consistently.


---

## Reconciliation against the pinned reference (M0, 2026-09-17)

The link above was a `blob/main/` reference and has been repointed at the pinned commit
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`. `main` moves; the SHA does not. Per
[spec.md §1](spec.md), no `blob/main/...` reference may be added to this repository.

Four corrections to how the table above should be read. None changes a starting value; all change
what the value is being compared against.

### `size1m` alone does not define the run

**`dmc_vision` does not inherit a size preset** at the pinned commit — `dmc_proprio` merges `size1m`,
`dmc_vision` does not. The reference configuration this project scales from is
`--configs dmc_vision size1m`, both presets. Bare `dmc_vision` is the ~200M-equivalent model.
`size1m` itself is three regex patterns whose `.*\.units` clause rewrites **six** keys, not one.
Resolved values: [spec.md §4.0](spec.md).

### The 32 × 4 stochastic state is partly inherited, and the cell is not a plain GRU

`size1m` sets `deter: 512`, `hidden: 64`, `classes: 4`. It does **not** set `stoch`, which is
inherited as `32` — so "32 categorical variables, 4 classes each" is correct, but only one of those
two numbers comes from the preset. It also inherits **`blocks: 8`**: the recurrent cell is a
**block-diagonal GRU**, not a dense GRU(512). The row "Recurrent state — 512 deterministic features"
above is therefore an incomplete description of the architecture; [spec.md §4.1](spec.md) is the
contract.

### Update ratio — units reconciled, and the gap is 4×

The reference defines `train_ratio` as **replayed frames per environment step**
(`embodied/run/train.py#L24-L25`, confirmed by its own tests). The plan's "replay training positions
per collected transition" is the **same unit**, so the comparison is valid:

| | Value | Env steps per gradient step (B=16, T=64) |
|---|---:|---|
| Pinned reference, `dmc_vision` | **256** | 4 |
| This project, initial | **64** | 16 |

The starting ratio is **4× below** the pinned reference for the same task. That is a deliberate
resource restriction ([spec.md §9-9](spec.md)), and it is the most likely single explanation if
learning underperforms at M9. The qualification rule in the table above stands; it now has a
reference value to be qualified against.

**Definitional caveat.** The reference counts *all* replayed frames including its context frame; this
project counts **loss-bearing positions only**, because the burn-in prefix is recomputed rather than
loss-bearing ([spec.md §7.5](spec.md)). State the convention whenever a realized ratio is quoted.

### Replay capacity is reduced, and on Walker it is NOT inert

500,000 against the reference's 5,000,000.

**An earlier version of this section claimed the buffer holds an entire 1e6-step Walker run before
eviction begins. That is arithmetically false** — 500,000 < 1,000,000. Eviction begins around the
half-way point of a Walker run and the second half evicts the first, whole episodes at a time, so
the final replay distribution covers roughly the last 500,000 control steps. Cartpole at a 500,000
budget does fit, and there the reduction genuinely is inert.

This matters for the horizon comparison only insofar as it applies identically to all three Walker
conditions, which hold real data fixed. It matters for any claim about what the world model was
trained on: at the end of a Walker run, the earliest half of the run is gone.

**Measured 2026-09-20, and the arithmetic holds empirically.** Walker collected 1,000 episodes /
1,000,000 transitions and finished holding **500,000 / 500 complete episodes** — exactly half evicted.
Cartpole collected 500 / 500,000 and evicted nothing. Run manifests in
[`results/m9/pilot/`](../results/m9/pilot/); this discharges [spec.md §10-11](spec.md) and refutes
§9-6's inertness expectation for Walker. `OnlineTrainer` logs `replay_occupancy` on every row, and
eviction is episode-safe by construction (`ReplayBuffer._enforce_capacity` drops whole episodes and
raises rather than truncating one).

### Parameter count

**686,846, now MEASURED end to end**, equal to the ~0.69M derived at M0 —
[`results/m0/param-count-derived-2026-09-17.txt`](../results/m0/param-count-derived-2026-09-17.txt).
Below the preset's name and far below the paper's smallest evaluated row of 12M. World model
**570,419**
([`results/m4/param-count-measured-2026-09-18.txt`](../results/m4/param-count-measured-2026-09-18.txt)),
`val` **66,111**
([`results/m7/param-count-measured-2026-09-18.txt`](../results/m7/param-count-measured-2026-09-18.txt)),
`pol` **50,316**
([`results/m8/param-count-measured-2026-09-19.txt`](../results/m8/param-count-measured-2026-09-19.txt)),
every module equal to its derived figure. §10-7 is discharged and the untrained `slowval` mirror is
excluded ([spec.md §10-7](spec.md)).
