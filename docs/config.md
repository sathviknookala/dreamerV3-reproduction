# Configuration and qualification rules

**Read before changing any hyperparameter or quoting a setting.** Relocated verbatim from
`dreamerv3_implementation_plan.md` (preserved at commit `c377be1`).

**Freeze status: NOT FROZEN.** Every value below is a starting choice awaiting its qualification
rule. Final settings are frozen at the end of M9, before final seeds run. When that happens, replace
this banner with the freeze date and the run manifest path.

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

The [author-maintained configuration](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml) supplies a useful compact scale reference through `size1m`, including deterministic size 512 and four classes. Its current defaults and architecture are not automatically equivalent to the pinned paper version. Record the actual parameter count rather than calling this implementation “1M parameters” from the preset name.

The initial update ratio and budgets are project choices, not claims about the paper's settings or guaranteed convergence. Runtime is estimated from measured pilots. If the compact system needs more capacity or updates, resolve that before the final study and apply the resulting configuration consistently.
