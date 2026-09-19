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
| Collection | One environment first; action repeat 1 | Log native control steps separately from agent decisions and physics substeps |
| Replay | CPU-resident uint8 frames; initial capacity 500,000 transitions | Measure total RAM use and avoid duplicate image storage |
| Update ratio | Start with 64 replay training positions per collected transition | Log the exact definition and realized ratio; qualify learning before freezing |
| Initial random collection | 5,000 agent transitions | Preserve the same warm-up budget in final comparisons |
| Planning budgets | Walker: 1M control steps/run; Cartpole: 500K/run | Qualify cost and learning in M9; all Walker horizon conditions receive the same final budget |

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

### Replay capacity is expected to be inert, not merely reduced

500,000 against the reference's 5,000,000. At a 1e6-step Walker budget the buffer holds the entire
run before eviction begins, so the reduction is **hypothesised to have no effect** on these two
tasks. The qualification rule is unchanged — measure occupancy at M9 — but it is now testing a stated
hypothesis rather than an open question.

### Parameter count

**~0.69M derived from the specification, not measured** —
[`results/m0/param-count-derived-2026-09-17.txt`](../results/m0/param-count-derived-2026-09-17.txt).
Below the preset's name and far below the paper's smallest evaluated row of 12M. **The RSSM and
encoder halves are now measured** — 391,008, equal to their derived figures
([`results/m2m3/param-count-measured-2026-09-18.txt`](../results/m2m3/param-count-measured-2026-09-18.txt));
the decoder and heads are still owed at M4 ([spec.md §10-7](spec.md)).
