# Frozen algorithm specification (M0 deliverable)

**Read before writing any model code, and whenever the paper and the reference implementation appear
to disagree.**

> **STATUS: M0 SPECIFIED — see the status legend in §11 and the audit at
> [`results/m0/m0-audit-2026-09-17.md`](../results/m0/m0-audit-2026-09-17.md).**
> Every architectural, objective, gradient-routing and environment-contract decision below is
> resolved against a pinned source. Remaining `unmeasured` entries are empirical quantities that
> [config.md](config.md) and [milestones.md](milestones.md) assign to a later milestone; each one is
> listed in §10 with its milestone and acceptance check. **"Specified" is not "implemented" and not
> "validated"** — §11 tracks those separately, and no mechanism here may be called implemented until
> code exists for it.

## 1. Pinned sources

| Role | Pin | Status |
|---|---|---|
| **Algorithm specification** | arXiv [2301.04104**v2**](https://arxiv.org/abs/2301.04104v2), submitted 2024-04-17 | Pinned. The paper decides. |
| **Reference implementation** | `danijar/dreamerv3` @ [`e3f02248693a79dc8b0ebd62c93683888ddaccfe`](https://github.com/danijar/dreamerv3/tree/e3f02248693a79dc8b0ebd62c93683888ddaccfe) (2026-05-25) | Pinned. Consulted under an interpretation note, never a silent tie-breaker. |
| **Historical reference only** | arXiv [2301.04104**v1**](https://arxiv.org/abs/2301.04104v1) (2023-01-10) and code @ [`8fa35f83eee1ce7e10f3dee0b766587d0a713a60`](https://github.com/danijar/dreamerv3/tree/8fa35f83eee1ce7e10f3dee0b766587d0a713a60) (2023-06-14) | **Do not import v1 mechanisms without an explicit §9 deviation entry.** |

Integrity of the pin, so it stays checkable if URLs change:

```
commit      e3f02248693a79dc8b0ebd62c93683888ddaccfe
tree        a6611dd5cca395eebcd387ebcad2685bb2d9dbdf
configs.yaml sha256
            9dff9c7062e3e33951cb54c6dd4b598aaf7e56e18e2cff39c812eaa797bcfcfc
```

**Why this commit and not a paper-contemporaneous one.** The repository has two disjoint eras, and
they line up with the two paper versions:

| Era | Range | Commits | Corresponds to |
|---|---|---|---|
| v1 | repo creation (2023-01-14) → `8fa35f83` (2023-06-14) | 16 | arXiv v1 |
| **v2** | `2411f7d1` (2024-04-15) → `e3f0224` (2026-05-25) | 15 | **arXiv v2** |

`2411f7d1` "Update agent and infrastructure" replaced the codebase wholesale (177 files,
+10,432/−6,009, the vendored `embodied/` package deleted and rewritten) and landed **two days before
arXiv v2 was posted**. Three consequences:

1. Any 2023 commit specifies a **different model** — see §9 for the list. Pinning one would map this
   spec onto arXiv v1 while claiming v2.
2. `size1m` **does not exist** in `2411f7d1` or its cleanup `1a532a77`. It was added in
   [`f8817c4040ce`](https://github.com/danijar/dreamerv3/commit/f8817c4040ce) (2024-12-07). A
   paper-contemporaneous pin cannot cite the preset this project scales from.
3. `29eb964e2918` (2024-05-16) "Fix previous action in replay context" corrects exactly the
   action-alignment class of bug that M1 is most exposed to. A pin before it would transcribe a
   known-fixed defect.

`e3f0224` is currently also the tip of `main`. That is why every citation below is a **commit
permalink**: `main` will move, the SHA will not. No `blob/main/...` reference may be added to this
repository.

## 2. Version manifest

Two categories, never mixed: what is **installed and verified on this machine today**, and what is a
**proposed requirement** not yet satisfied.

### Installed and verified (2026-09-17)

Captured by [`capture_manifest.py`](../results/m1/capture_manifest.py) →
[`env-manifest-2026-09-17.json`](../results/m1/env-manifest-2026-09-17.json). Exact pins for
reinstallation: [`requirements.txt`](../requirements.txt).

| Item | Value |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-65-generic |
| CPU | AMD Ryzen 9 7950X, 16 cores / 32 threads |
| System RAM | 124 GiB |
| GPU | NVIDIA RTX PRO 4000 Blackwell, 24467 MiB, driver 575.64.03 |
| GPU compute capability | **12.0 (`sm_120`)** |
| CUDA toolkit (system) | 12.9, V12.9.86 |
| **Python** | **3.12.11**, venv at `.venv` |
| **torch** | **2.13.0+cu129**, bundled CUDA 12.9, cuDNN 9.20.0 |
| torch arch list | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` — **`sm_120` present** |
| mujoco | 3.13.0 |
| dm-control | 1.0.46 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| labmaze | 1.0.6 (transitive; unused by the DMControl suite) |
| GL backend | EGL, `libEGL_nvidia.so.0` present, `DISPLAY` unset |

`pip freeze` sha256 `3790431efebf6f6b0452759d9ef675f75b9f8efee400bcdf58ba02c6f788517b`, 52 packages.

### Two environment constraints discovered while building this, both load-bearing

**1. Python 3.13 does not work; 3.12 does.** The risk this section previously flagged was real, but
the mechanism was not the one predicted. `mujoco` publishes cp313 wheels and `dm-control` is pure
Python — both are fine on 3.13. The blocker is the transitive dependency **`labmaze`**, whose latest
release (1.0.6) ships wheels only to **cp312**. On 3.13 pip falls back to the sdist, which builds with
**bazel**, which is not installed:

```
error: command 'bazel' failed: No such file or directory
ERROR: Failed building wheel for labmaze
```

`labmaze` is needed only by `dm_control.locomotion`, which this project never imports — but it is a
hard `install_requires`, so the whole transaction rolls back. **Python 3.12.11 was chosen over a
`--no-deps` workaround** so that `requirements.txt` reinstalls cleanly and `pip check` passes; a
skipped hard dependency would leave the environment permanently inconsistent for anyone else.

**2. torch must come from the `cu129` index, not PyPI.** The GPU is `sm_120` (Blackwell) and the
driver is 575.64.03. CUDA 13.0 wheels require driver ≥ 580, so the newest PyPI default build
(`cu130`) is excluded; `cu129` matches the system toolkit exactly and is driver-compatible.
`sm_120` presence in the arch list is verified, not assumed.

Reinstalling needs one further quirk: this machine's `python3.12` is **uv-managed** and its
`ensurepip --upgrade --default-pip` path fails, so `python -m venv` cannot create the environment.
`uv venv --seed` creates it and `pip` performs every install — no alternative resolver is involved.
The commands are recorded verbatim at the top of [`requirements.txt`](../requirements.txt).

## 3. Paper → pinned code → project decision

Three distinct columns. **Paper** is arXiv v2. **Pinned code** is `e3f0224`, with the fully resolved
`size1m` + `dmc_vision` value where the two differ. **Project** is what this implementation does, and
the rationale column is mandatory whenever Project differs from either source.

| Component | Paper (arXiv v2) | Pinned code (`e3f0224`) | Project decision | Rationale / interpretation |
|---|---|---|---|---|
| Recurrent cell | GRU, block-diagonal recurrent weights, **8 blocks**; inputs are linear embeddings of `z_t`, `a_t`, and the recurrent state "to allow mixing between blocks" | `blocks: 8`, `deter: 512` under `size1m` → 8 × 64 per block; see §4.1 for exact projections | **Adopt the 8-block cell exactly** | The paper names it as a headline v2 change. A plain dense GRU(512) is a *different* model with ~8× the recurrent parameters; substituting one silently would make the deviation table wrong. §4.1 gives the connectivity. |
| Deterministic width | `8d`; 200M row = 8192 | `deter: 512` (`size1m`) | **512** | Below the paper's smallest row. Capacity reduction, §9-1. |
| Stochastic latents | **not stated** — only "fixed across model sizes" | `stoch: 32`, inherited by `size1m` | **32** | Paper-silent, code-speaks. v1 stated 32 explicitly; code agrees. |
| Classes per latent | `d/16`; 200M row = 64 | `classes: 4` (`size1m`) | **4** | Capacity reduction, §9-1. Note the paper makes classes *scale* with size; 4 is far below the 16 floor of the 12M row. |
| Encoder | strided convs to 6×6 or 4×4; base channels `d/16`; kernel/padding **not stated** | `SimpleEncoder`, `depth: 4`, `mults: [2,3,4,4]`, `kernel: 5`, `layers: 3`, `units: 64`, `symlog: True` | See §4.2 | Kernel size is paper-silent; take 5 from code. |
| Decoder | transposed strided convs, **sigmoid output** | `SimpleDecoder`, `bspace: 8`, mirrored mults, `kernel: 5` | See §4.3 | Sigmoid output confirmed in both sources. |
| Posterior | `q(z_t \| h_t, x_t)`, categorical | `obslayers: 1` | See §4.4 | |
| Prior | `p(z_t \| h_t)`, categorical | `imglayers: 2`, `dynlayers: 1` | See §4.4 | Prior net is **2 layers**, deeper than the posterior's 1. Easy to get wrong by symmetry assumption. |
| Reward head | twohot over exponentially spaced bins, bin count **not stated** | `rewhead: layers: 1, units: 64, output: symexp_twohot, bins: 255, outscale: 0.0` | **1 layer, 255 bins, zero-init output** | Bin count is paper-silent; 255 from code and from v1's explicit `K = 255`. |
| Continuation head | logistic regression on continuation | `conhead: layers: 1, output: binary, outscale: 1.0`; **`contdisc: True`** | **Adopt `contdisc`** — see §5.4 | The discount folded into the head is **not in the paper at all**. Getting this wrong double-counts or drops γ. §9-4. |
| Actor | Reinforce for **both** discrete and continuous; baseline `v_ψ(s_t)` | `policy: layers: 3, units: 64, outscale: 0.01`; `policy_dist_cont: bounded_normal`, `minstd: 0.1`, `maxstd: 1.0`, `unimix: 0.01` | **Reinforce, continuous, with critic baseline** | v1 used pathwise backprop for continuous actions. Walker and Cartpole are continuous, so this is the single highest-risk row: v1's rule is a *different objective*. §9-3. |
| Critic | distribution over returns, max-likelihood on twohot targets | `value: layers: 3, units: 64, output: symexp_twohot, bins: 255, outscale: 0.0` | **3 layers, 255 bins, zero-init output** | |
| Normalization | **RMSNorm** + SiLU | `norm: rms` everywhere | **RMSNorm, scale only, no mean subtraction** | v1 was LayerNorm. §4.7. |
| Initialization | only the zero-init of twohot output weights is specified | `winit: trunc_normal_in` | See §4.8 | Paper is silent beyond zero-init; code decides. |
| Optimizer | LaProp, ε=1e-20, β₁=0.9, **β₂=0.99**, AGC(0.3), one LR 4e-5 | `opt: {lr: 4e-5, agc: 0.3, eps: 1e-20, beta1: 0.9, beta2: 0.999, momentum: True, wd: 0.0, warmup: 1000}` | See §5.8 | **Paper and code disagree on β₂ (0.99 vs 0.999).** Recorded, not averaged; §5.8 states which is taken and why. |
| Batch / length | B=16, T=64 | `batch_size: 16, batch_length: 64, replay_context: 1` | **B=16, T=64** | Matches both. |
| Replay capacity | 5e6 | `replay.size: 5e6` | **5e5** | Resource restriction, §9-6. |
| Discount | γ=0.997 via `horizon: 333` | `horizon: 333` | **333 → γ=1−1/333** | Identical in v1, v2 and code. |
| λ | 0.95 | `imag_loss.lam: 0.95`, `repl_loss.lam: 0.95` | **0.95** | |
| Imagination horizon | H=15 | `imag_length: 15` | **15** default; {5,15,30} in the study | |
| Entropy coefficient | η=3e-4 | `imag_loss.actent: 3e-4` | **3e-4** | Identical in v1, v2 and code. |
| Return normalization | `S = EMA(Per(R,95) − Per(R,5), 0.99)`, clamp `max(1,S)` | `retnorm: {impl: perc, rate: 0.01, limit: 1.0, perclo: 5.0, perchi: 95.0}` | **As specified** | `rate: 0.01` ≡ `decay: 0.99`. `valnorm` and `advnorm` are `none` in the pinned config and are **not** implemented. |
| Slow critic | EMA regularizer weight 1, decay 0.98 | `slowvalue: {rate: 0.02, every: 1}`, `slowreg: 1.0` | **rate 0.02 every step, reg weight 1** | `rate 0.02` ≡ `decay 0.98`. Identical across v1/v2. |
| Replay critic | β_val=1, **β_repval=0.3** | `loss_scales.repval: 0.3`, `repval_loss: True`, `repval_grad: True` | **Included at 0.3** | v2-only. Included per the M0 requirement; no reason to deviate was found. Gradient consequence in §5.6/§6. |

Rows marked "see §4/§5" are specified in full there rather than compressed into a table cell.

## 4. Architecture specification

Every value here is the **fully resolved** effective value for this project's configuration, not a
preset override. Permalink base:
`https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/`

### 4.0 How the configuration resolves, and one trap

**`dmc_vision` does NOT inherit a size preset** (`configs.yaml#L178-L187`). `dmc_proprio` merges
`*size1m`; `dmc_vision` does not, so `--configs dmc_vision` alone runs at base defaults
(deter 8192, classes 64, depth 64, units 1024) — the ~200M-equivalent model. The reference
invocation this project scales from is therefore **`--configs dmc_vision size1m`**, both presets,
and they write disjoint keys so order is irrelevant.

`size1m` is three regex patterns matched against **flattened dotted keys**, prefix-matched. `.*\.units`
therefore hits **six** keys (`enc.simple.units`, `dec.simple.units`, `rewhead.units`, `conhead.units`,
`policy.units`, `value.units`) and `.*\.depth` hits two. A reader expecting it to affect only the
RSSM would under-shrink the model by a large factor.

**Resolved configuration.** `base` = repository default, `size1m` / `dmc_vision` = preset override,
`class` = a Python class default absent from `configs.yaml`.

```
dyn.rssm.deter        512    size1m      dyn.rssm.stoch       32     base
dyn.rssm.hidden        64    size1m      dyn.rssm.classes      4     size1m
dyn.rssm.blocks         8    base        dyn.rssm.free_nats   1.0    base
dyn.rssm.act         silu    base        dyn.rssm.norm       rms     base
dyn.rssm.unimix      0.01    base        dyn.rssm.outscale    1.0    base
dyn.rssm.winit   trunc_normal_in base    dyn.rssm.absolute  False    base
dyn.rssm.imglayers      2    base        dyn.rssm.obslayers    1     base
dyn.rssm.dynlayers      1    base        dyn.rssm.unroll    False    class

enc.simple.depth        4    size1m   -> depths (8,12,16,16)
enc.simple.mults  (2,3,4,4)  base        enc.simple.kernel     5     base
enc.simple.outer    False    base        enc.simple.strided  False   base
enc.simple.units       64    size1m      (unused: no vector observations)
enc.simple.layers       3    base        (unused: no vector observations)
enc.simple.symlog    True    base        (unused: no vector observations)

dec.simple.depth        4    size1m   -> depths (8,12,16,16)
dec.simple.bspace       8    base        dec.simple.units     64     size1m (USED)
dec.simple.outscale   1.0    base        dec.simple.kernel     5     base

rewhead  {layers:1, units:64, output:symexp_twohot, bins:255, outscale:0.0}
conhead  {layers:1, units:64, output:binary,        outscale:1.0}
policy   {layers:3, units:64, minstd:0.1, maxstd:1.0, outscale:0.01}
value    {layers:3, units:64, output:symexp_twohot, bins:255, outscale:0.0}
policy_dist_cont  bounded_normal        imag_last  0  -> K = T
contdisc  True                          imag_length 15
```

**Observation and action spaces.** With `proprio: False`, `obs_space` is
`{is_first, is_last, is_terminal, reward, image: uint8(64,64,3)}` (`dmc.py#L50-L57`), and
`main.py#L131-L132` **strips `reset` from the action space**:

```python
act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
```

So `act_space = {'action': Space(float32, (A,), -1, 1)}` — **one key, no `reset` head, no `reset`
one-hot in the action embedding**. Walker Walk: `A = 6`. Encoder and decoder see `{'image'}` only,
so **`veckeys` is empty** and the encoder's MLP stack and the decoder's vector head **do not exist**
in this configuration. Do not build them.

**Starting architecture vs final configuration.** Everything in §4 is the **selected starting
architecture**, chosen to match the pinned reference at `size1m`. [config.md](config.md) governs what
may still change and §10 lists what must be measured; the final configuration is frozen at the end of
M9. Nothing here is frozen yet.

### 4.1 Block-diagonal recurrent cell

This is the component most likely to be silently wrong. "8 × 64" is not an implementation.

#### `BlockLinear` (`embodied/jax/nets.py#L254-L281`)

```python
insize = x.shape[-1]
shape = (self.blocks, insize // self.blocks, self.units // self.blocks)
kernel = self.value('kernel', self._scaled_winit, shape)
x = x.reshape((*x.shape[:-1], self.blocks, insize // self.blocks))
x = jnp.einsum('...ki,kio->...ko', x, kernel)
x = x.reshape((*x.shape[:-2], self.units))
if self.bias:
    x += self.value('bias', init(self.binit), self.units)
```

| Property | Value |
|---|---|
| Kernel rank | **3**: `(blocks, in/blocks, out/blocks)` |
| Blocking | **contiguous** slices — block `b` reads input channels `[b·in/g, (b+1)·in/g)` |
| Mixing | `einsum '...ki,kio->...ko'` has **no `k→k'` path**; exactly equivalent to a dense matmul with a block-diagonal `(in, out)` matrix |
| Bias | full `(out,)` vector, **not** per-block, zero-init, added after reshape-back |
| Internal norm | **none** |
| `outscale` | multiplies the **kernel init only**, never the bias |

`block_fans` and `block_norm` **do not exist at HEAD** — they were 2024-rewrite options. Do not
implement them.

> **Fan-in trap.** `Initializer.compute_fans` (`nets.py#L187-L197`) takes the rank-≥3 branch for a
> BlockLinear kernel: `space = prod(shape[:-2]) = blocks`, so
> `fanin = shape[-2] · space = (in/g) · g = in` — **the full input width, not the per-block width**.
> `dynhid0`'s kernel `(8, 256, 64)` therefore gets `fanin = 2048`, std
> `1.1368·sqrt(1/2048) = 0.02512`. Implementing this with per-block fan-in (256) gives
> `0.05024` — **2.83× too large**. This is what the code does; no claim is made about intent.

#### The recurrence (`dreamerv3/rssm.py#L135-L159`)

```python
stoch = stoch.reshape((stoch.shape[0], -1))
action /= sg(jnp.maximum(1, jnp.abs(action)))
x0 = silu(rmsnorm(dynin0(deter)))     # Linear 512 -> 64,  DENSE
x1 = silu(rmsnorm(dynin1(stoch)))     # Linear 128 -> 64,  DENSE
x2 = silu(rmsnorm(dynin2(action)))    # Linear   6 -> 64,  DENSE
x = concat([x0, x1, x2], -1)[..., None, :].repeat(8, -2)      # (B,192) -> (B,8,192)
x = group2flat(concat([flat2group(deter), x], -1))            # (B,8,256) -> (B,2048)
x = silu(rmsnorm(dynhid0(x)))         # BlockLinear 2048 -> 512, g=8
x = dyngru(x)                         # BlockLinear  512 -> 1536, g=8, NO norm
reset, cand, update = [group2flat(v) for v in jnp.split(flat2group(x), 3, -1)]
reset  = sigmoid(reset)
cand   = tanh(reset * cand)
update = sigmoid(update - 1)
deter  = update * cand + (1 - update) * deter
```

**Projection inventory** (`deter`=512, `hidden`=64, `g`=8, per-block width **64**, `A`=6):

| Name | Type | Input | in→out | Kernel | Bias | Then |
|---|---|---|---|---|---|---|
| `dynin0` | **dense** Linear | `deter` | 512→64 | `(512, 64)` | `(64,)` | RMSNorm → silu |
| `dynin1` | **dense** Linear | `stoch` flat | 128→64 | `(128, 64)` | `(64,)` | RMSNorm → silu |
| `dynin2` | **dense** Linear | `action` | 6→64 | `(6, 64)` | `(64,)` | RMSNorm → silu |
| `dynhid0` | **BlockLinear** g=8 | `[deter_block ‖ bcast(x0‖x1‖x2)]` | 2048→512 | `(8, 256, 64)` | `(512,)` | RMSNorm over **all 512** → silu |
| `dyngru` | **BlockLinear** g=8 | `dynhid0` out | 512→1536 | `(8, 64, 192)` | `(1536,)` | **nothing — gates are raw** |

**The broadcast.** `concat([x0,x1,x2])` is `(B, 192)`; `[..., None, :].repeat(8, -2)` makes
`(B, 8, 192)` — **the identical 192-d vector goes to every block**. Concatenating
`flat2group(deter)` gives `(B, 8, 256)` = `[own 64 deter channels ‖ shared 192]`, and `group2flat`
lays that out block-major, which is exactly the blocking `BlockLinear` then re-derives.

**Where cross-block mixing happens — and where it does not.** Exactly two places:

1. **`dynin0`**, a dense `Linear(512 → 64)` over the whole `deter`, compressed to a **64-d
   bottleneck** and broadcast identically to all blocks. **This is the only *learned* cross-block
   path in the recurrence.**
2. **`dynhid0norm`**, an RMSNorm over the **full 512** vector, i.e. a global mean-square statistic.
   A scalar, non-learned coupling. It is **not** per-block normalization.

Not mixed: `dynhid0`'s matmul (block `b`: its 256 → its 64), `dyngru`'s matmul (block `b`: its 64 →
its 192), the gate arithmetic (elementwise), and the residual `(1−update)·deter` (elementwise,
block-aligned). So `deter[b·64+u]` at `t+1` depends on `deter[b·64 : b·64+64]` at `t` directly, on
all 512 channels through the 64-d bottleneck, and on the global RMS scalar.

> **Gate-extraction alignment — the single easiest thing to get wrong.** The split is
> `flat2group` **first** — reshape `(B, 1536) → (B, 8, 192)` — then split the 192 into 3×64, then
> re-flatten each to `(B, 512)`. The index map is
> `dyngru output index b·192 + gate·64 + u  →  gate vector index b·64 + u`.
> A naive `split(x, 3, -1)` on the **flat** 1536 vector would assign `reset` all of blocks 0–2 and
> part of block 3 — silently wrong, no exception, trains anyway.
> PyTorch: view `(B, 8, 3, 64)` and index dim 2, or `chunk(3, dim=-1)` on a `(B, 8, 192)` view.

**Gate arithmetic, exactly:**

- `reset = sigmoid(reset)` and it multiplies the **candidate pre-activation**, not `h_{t-1}`. There
  is no `r ⊙ h_{t-1}` term — `h_{t-1}` already entered through `dynhid0`. This is **not** a standard
  GRU.
- Candidate activation is **`tanh`**, not `silu`. `silu` is the activation of the `dynin*` / `dynhid*`
  layers only.
- `update = sigmoid(update - 1)` — a **hard-coded `-1` offset inside the sigmoid**, biasing toward
  retention (`sigmoid(-1) ≈ 0.269` at zero pre-activation). Not a config field.
- **`deter` is not normalized before the gates.** It enters `dynhid0`'s concat raw; `dynhid0`'s
  *output* is normed and `dyngru`'s output is **not normed at all**.

`nets.GRU` (`nets.py#L634-L669`) is **dead code** — unreferenced, and it normalizes
`concat([carry, inp])` *before* the gate projection, which `_core` does not. Do not reimplement it.

**Action preprocessing.** `action /= sg(max(1, |action|))` is an elementwise soft clip applied to the
concatenated action embedding. The divisor is stop-gradient, so for `|a| > 1` the gradient is
**scaled by `1/|a|`, not zeroed** — not the same as `clamp(-1, 1)`.

**Layer-count fields**, all at width `hidden = 64`:

| Field | Value | Controls |
|---|---|---|
| `dynlayers` | **1** | `dynhid{i}` BlockLinear layers inside `_core` |
| `obslayers` | **1** | `obs{i}` dense layers in the **posterior** head |
| `imglayers` | **2** | `prior{i}` dense layers in the **prior** head |

### 4.2 Encoder (`rssm.py#L179-L250`)

`depths = depth × mults = 4 × (2,3,4,4) = (8, 12, 16, 16)`.

With `outer: False` and `strided: False`, every stage is a **stride-1 `same` conv followed by 2×2
max-pooling**, implemented as `x.reshape(B, H//2, 2, W//2, 2, C).max((2, 4))`. There is **no** strided
convolution anywhere in the encoder.

```
input            (N, 64, 64, 3)  uint8
                 cast to bfloat16, then x/255 - 0.5   ->  [-0.5, +0.5]
cnn0  Conv2D(8,  k5, s1, same) + bias -> maxpool2 -> (N,32,32, 8) -> RMSNorm -> silu
cnn1  Conv2D(12, k5, s1, same) + bias -> maxpool2 -> (N,16,16,12) -> RMSNorm -> silu
cnn2  Conv2D(16, k5, s1, same) + bias -> maxpool2 -> (N, 8, 8,16) -> RMSNorm -> silu
cnn3  Conv2D(16, k5, s1, same) + bias -> maxpool2 -> (N, 4, 4,16) -> RMSNorm -> silu
flatten                                             -> (N, 256)      tokens
```

- Order is **conv → bias → pool → norm → activation**. Normalization is **after** pooling.
- Every conv carries a bias even though a norm follows. See §4.9.
- `minres = 4`, token width **256**.
- Layout is **NHWC** (channels-last); kernel layout `HWIO`, shape `(5, 5, Cin, Cout)`. PyTorch:
  `Conv2d(Cin, Cout, 5, stride=1, padding=2)` (kernel 5 is odd so `same` is symmetric padding 2),
  with the weight permuted `HWIO → OIHW`.
- **The `/255 - 0.5` is computed in bfloat16**, after the cast, not in float32.

### 4.3 Decoder (`rssm.py#L253-L359`)

`factor = 2^4 = 16`, `minres = 4`, spatial seed shape `(4, 4, 16)`, `u = 256`, `g = bspace = 8`.

**The `deter` and `stoch` halves take separate paths and are SUMMED, not concatenated:**

| Stage | Type | Detail |
|---|---|---|
| `sp0` | **BlockLinear** `(8, 64, 32)` on `deter` (512) | → 256, then `rearrange('... (g h w c) -> ... h w (g c)', h=4, w=4, g=8)` so `c = 2`: **deter block `b` owns output channels `[2b, 2b+1]` at every spatial location** |
| `sp1` | dense `Linear(128 → 128)` on `stoch` flat | width is `2 × dec.units` — the only use of `dec.units` here — then RMSNorm → silu |
| `sp2` | dense `Linear(128 → 256)` | reshaped to `(N, 4, 4, 16)` |
| `spnorm` | RMSNorm on `x0 + x1`, scale `(16,)` | then silu |

So the stoch path is fully mixing and the deter path is block-diagonal; they merge **additively**.

**Upsampling stack.** With `strided: False` and `outer: False` this is **nearest-neighbour upsample
then stride-1 conv** — `x.repeat(2, -2).repeat(2, -3)` — and **never** a transposed convolution.
`depths[:-1] = (8, 12, 16)` enumerated then reversed gives `[(2,16), (1,12), (0,8)]`:

```
(N, 4, 4,16)  -> up2 -> conv2 Conv2D(16, k5, s1) -> RMSNorm -> silu -> (N, 8, 8,16)
              -> up2 -> conv1 Conv2D(12, k5, s1) -> RMSNorm -> silu -> (N,16,16,12)
              -> up2 -> conv0 Conv2D( 8, k5, s1) -> RMSNorm -> silu -> (N,32,32, 8)
              -> up2 -> imgout Conv2D(3, k5, s1, outscale 1.0)      -> (N,64,64, 3)
              -> sigmoid                                            -> [0, 1]
```

Note the **channel-count asymmetry** against the encoder: `conv{i}` outputs `depths[i]`, so the
decoder runs 16→16→12→8→3 rather than mirroring the encoder exactly.

**Image likelihood.** `MSE(out)` then `Agg(out, 3, jnp.sum)`:

- `loss = square(mean − sg(target))` — **plain squared error, no `0.5` factor, no log-normalizer**,
  computed in float32.
- Summed over `(C, W, H)` = **12288 terms per `(B, T)` position**, then mean over positions. See
  §5.3 for why this must not be changed to a pixel mean.
- **Target is `f32(obs)/255`, in `[0, 1]`** — matching the sigmoid output. **The encoder uses
  `/255 − 0.5` but the decoder target uses `/255` with no shift.** This asymmetry is real; copying
  one into the other is a silent bug.

### 4.4 Posterior and prior heads

**Posterior** (`rssm.py#L75-L92`, `obslayers = 1`, `absolute = False`):

| Step | Detail |
|---|---|
| Input | `concat([deter (512), tokens (256)])` = **768**, **`deter` first** |
| Timing | consumes the **post-GRU `deter` of the same timestep** |
| `obs0` | `Linear(768 → 64)` + RMSNorm + silu |
| `obslogit` | `Linear(64 → 128)`, `outscale = 1.0`, reshaped to **`(B, 32, 4)`** |

**Prior** (`rssm.py#L161-L176`, `imglayers = 2`), input is **`deter` alone** — not `stoch`, not
`action`:

| Step | Detail |
|---|---|
| `prior0` | `Linear(512 → 64)` + RMSNorm + silu |
| `prior1` | `Linear(64 → 64)` + RMSNorm + silu |
| `priorlogit` | `Linear(64 → 128)`, `outscale = 1.0`, reshaped to **`(B, 32, 4)`** |

The prior is **deeper than the posterior** (2 hidden layers vs 1). Assuming symmetry is a common
error.

> **Unimix is applied inside `_dist`, on every call.** `feat['logit']` — the tensor used as `post` in
> the KL — is **raw, unmixed**. Unimix must be re-applied at **every** sampling, KL and entropy site,
> not once at the head. Applying it at the head instead would double-mix at some sites and
> single-mix at others.

Straight-through sampling: `value = sg(onehot(argmax_sample)) + (probs − sg(probs))`, where `probs`
is the **post-unimix** distribution.

**Reset masking.** `nn.mask((deter, stoch, action), ~reset)` zeroes all three on `is_first`, and the
**concatenated action embedding is zeroed again** after `DictConcat` (which would otherwise emit
one-hots for a zeroed input). The initial carry is **literal zeros**, not a learned embedding — the
v1 `initial: learned` option does not exist at HEAD.

### 4.5 Reward and continuation heads

Input to both is `feat2tensor = concat([deter (512), stoch.flatten() (128)])` = **640**.
MLP block order is **`Linear → Norm → activation`**.

| Head | Stack | Output |
|---|---|---|
| `rew` | `linear0 (640→64)` + RMSNorm + silu | `logits (64→255)`, `outscale = 0.0` ⇒ **kernel exactly zero at init** |
| `con` | `linear0 (640→64)` + RMSNorm + silu | `logit (64→1)`, `outscale = 1.0` |

Because the continuation space has `shape == ()`, the kernel is `(64, 1)` and the final reshape
**drops the trailing axis**, giving output shape `(B, T)`. Neither head is wrapped in `Agg`, so
neither performs a reduction.

**Bin construction** (`heads.py#L132-L144`). `bins = 255` is odd, so:

```python
half = symexp(linspace(-20, 0, 128))
bins = concat([half, -half[:-1][::-1]])        # 255 values, symmetric
```

| Property | Value |
|---|---|
| `bins[0]` | `-485165194.4097903` = `-(e^20 − 1)` |
| `bins[126]`, `bins[127]`, `bins[128]` | `-0.17055772`, **`0.0` exactly**, `+0.17055772` |
| `bins[254]` | `+485165194.4097903` |
| Parity effect | changes the **construction**, not the output width — odd guarantees an exact zero bin |
| Squash | **none.** No `squash`/`unsquash` is passed, so targets are compared in **raw reward space**. This is *not* symlog-then-twohot; the nonlinearity lives entirely in the bin spacing. |

**Scalar readout** (`outs.py#L285-L309`) — and the reference carries a comment explaining exactly why:

```python
m = (n - 1) // 2
wavg = (p[m:m+1] * b[m:m+1]).sum(-1) + ((p[:m] * b[:m])[..., ::-1] + (p[m+1:] * b[m+1:])).sum(-1)
```

The negative-half products are **reversed and added elementwise** to the positive-half products
**before** summing, plus the centre term. Each mirror pair therefore cancels in one operation, so
`pred() == 0.0` exactly at initialization where `outscale = 0.0` makes the logits uniform. Verified
against alternatives in [`twohot-readout-2026-09-17.txt`](../results/m0/twohot-readout-2026-09-17.txt):
on a uniform `p` whose exact readout is `0`, sequential float32 gives `+0.25`, `np.sum` float32 gives
**`−1.0`**, and the mirror algorithm gives `0`. **Implement the mirror form**; it is exact by
construction rather than by luck of ordering. §5.5.

### 4.6 Actor

Stack: `linear0 (640→64)`, `linear1 (64→64)`, `linear2 (64→64)`, each + RMSNorm + silu. Then two
**separate** output heads, `mean (64→6)` and `stddev (64→6)`, both `outscale = 0.01`.

`bounded_normal` (`heads.py#L146-L155`):

```python
mean = Linear(space.shape)(x)
stddev = Linear(space.shape)(x)
stddev = (maxstd - minstd) * sigmoid(stddev + 2.0) + minstd
output = outs.Normal(jnp.tanh(mean), stddev)
```

| Property | Value |
|---|---|
| Family | **plain diagonal Gaussian** |
| `stddev` | `0.9 · sigmoid(raw + 2.0) + 0.1` — the **`+2.0` offset is hard-coded**, giving `≈0.893` at zero pre-activation |
| `tanh` | applied to the **mean only** |
| Sample | `ε · stddev + mean` — **unsquashed and unbounded** |
| `logp` | plain `Normal.logpdf` |
| **Density correction** | **none, and none is needed** |
| Reduction | `space.shape == (6,)` is truthy ⇒ wrapped in `Agg(·, 1, sum)`, so `logp` and `entropy` are **summed over the 6 action dimensions** |
| `unimix` | present in the config but **dead** — read only by the discrete path, and `reset` was stripped so there is no discrete head |

> **This settles the plan's M8 question about action transforms.** There is **no squashing bijector
> on the sample**, so there is **no change-of-variables / log-det Jacobian correction anywhere**.
> `tanh` sits inside the *parameterization* of the mean, not on the sampled action. Bounding is
> enforced outside the agent by the `ClipAction` wrapper (`wrappers.py#L76-L86`) and inside the RSSM
> by `action /= max(1, |action|)`. Implementing a tanh-Normal with a density correction would be a
> **different policy class** — and a plausible-looking mistake, since tanh-squashed Gaussians are the
> convention elsewhere in continuous control.

### 4.7 Critic and slow critic

`value` is structurally the reward head with `layers: 3`: `linear0 (640→64)`, `linear1 (64→64)`,
`linear2 (64→64)`, each + RMSNorm + silu, then `logits (64→255)` with `outscale = 0.0` (**kernel
exactly zero at init**). Identical bin construction and mirror readout as §4.5.

`slowval` is a **byte-identical second module**, excluded from the optimizer's module list so it
receives no gradient. `SlowModel` (`embodied/jax/utils.py#L94-L127`):

```python
mix = jnp.where(count % every == 0, rate, 0)
slow <- mix * fast + (1 - mix) * slow
```

With `rate = 0.02`, `every = 1`: **`slow ← 0.02·fast + 0.98·slow` every step**, called **after** the
optimizer step. Parameters are hard-copied from `value` on first use, so the two start identical.

### 4.8 Normalization — exact RMSNorm

`nets.py#L361-L409`:

```python
mean2 = jnp.square(x).mean(axis, keepdims=True)
x = x * (jax.lax.rsqrt(mean2 + self.eps) * scale)
```

| Property | Value |
|---|---|
| Axis | `(-1,)` — last axis only |
| `eps` | **`1e-4`** |
| Placement of `eps` | added to the **mean-square, inside the `rsqrt`** — **not** `x / (rms + eps)` |
| Mean subtraction | **none** |
| Learned params | **scale only**, no shift. The `shift` field exists but the `rms` branch never creates it. |
| Scale init | **ones**, stored in float32 |
| Precision | **computed in float32 regardless of input dtype**, result cast back |

PyTorch equivalent, including the upcast, which matters under bf16:

```python
w = x.float()
y = w * torch.rsqrt(w.pow(2).mean(-1, keepdim=True) + 1e-4) * self.scale
return y.to(x.dtype)
```

### 4.9 Initialization

`winit: trunc_normal_in` on every module (`nets.py#L144-L197`):

```python
x = jax.random.truncated_normal(seed, -2, 2, shape)
x *= 1.1368 * sqrt(1 / fanin)
x *= outscale            # separate per-layer multiplier, kernel only
```

| Property | Value |
|---|---|
| Distribution | standard normal truncated to **`[-2, +2]`**, bounds applied **before** scaling |
| Correction factor | `1.1368` — this is `1 / std(TruncNormal(-2,2)) = 1 / 0.8796257`, so the **resulting std is exactly `sqrt(1/fanin)`** |
| Fan mode | **`in`** |
| `outscale` | a **separate** multiplier on the kernel only; never on the bias |
| Bias init | **zeros**, everywhere |
| `compute_fans` | rank 1 → `(1, n)`; rank 2 → `shape`; rank ≥ 3 → `space = prod(shape[:-2])`, `(shape[-2]·space, shape[-1]·space)`. Conv2D `(5,5,Cin,Cout)` ⇒ `fanin = 25·Cin`. BlockLinear ⇒ `fanin = in` (§4.1). |

> **Bias is NOT suppressed under normalization.** Every `Linear`, `BlockLinear` and `Conv2D` has
> `bias = True`, and nothing in the config ever disables it — **including layers immediately followed
> by an RMSNorm**, where it is mathematically near-redundant. v1 disabled bias under norm
> (`self._bias = bias and norm == 'none'`). Reproduce the HEAD behaviour or both the parameter count
> and the dynamics will differ.

Resolved init std per kernel (`std = 1.1368·sqrt(1/fanin)·outscale`):

```
dyn/dynin0      (512,64)      fanin  512   0.050240
dyn/dynin1      (128,64)      fanin  128   0.100480
dyn/dynin2      (6,64)        fanin    6   0.464100
dyn/dynhid0     (8,256,64)    fanin 2048   0.025120   <- full width, see 4.1
dyn/dyngru      (8,64,192)    fanin  512   0.050240
dyn/obs0        (768,64)      fanin  768   0.041021
dyn/obslogit    (64,128)      fanin   64   0.142100
dyn/prior0      (512,64)      fanin  512   0.050240
dyn/prior1      (64,64)       fanin   64   0.142100
dyn/priorlogit  (64,128)      fanin   64   0.142100
enc/cnn0        (5,5,3,8)     fanin   75   0.131266
enc/cnn1        (5,5,8,12)    fanin  200   0.080384
enc/cnn2        (5,5,12,16)   fanin  300   0.065633
enc/cnn3        (5,5,16,16)   fanin  400   0.056840
dec/sp0         (8,64,32)     fanin  512   0.050240
dec/sp1         (128,128)     fanin  128   0.100480
dec/sp2         (128,256)     fanin  128   0.100480
dec/conv2       (5,5,16,16)   fanin  400   0.056840
dec/conv1       (5,5,16,12)   fanin  400   0.056840
dec/conv0       (5,5,12,8)    fanin  300   0.065633
dec/imgout      (5,5,8,3)     fanin  200   0.080384
{rew,con,pol,val}/mlp/linear0 (640,64) fanin 640  0.044936
{pol,val}/mlp/linear{1,2}     (64,64)  fanin  64  0.142100
rew/head/logits (64,255)      outscale 0.0   EXACTLY ZERO
val/head/logits (64,255)      outscale 0.0   EXACTLY ZERO
con/head/logit  (64,1)        outscale 1.0   0.142100
pol/head/{mean,stddev} (64,6) outscale 0.01  0.0014210
all norm scales: ones (float32)        all biases: zeros
```

### 4.10 Tensor layouts and imagination shapes

- **Batch-first `(B, T, ...)` everywhere.** `nj.scan(..., axis=1)` transposes to `(T, B, ...)` for
  the scan and back afterwards, so **inside the step function every tensor is `(B, ...)` with no time
  axis**. `_core`, `_observe` and `_prior` are single-timestep functions.
- `_core` hard-codes `stoch.reshape((stoch.shape[0], -1))`, assuming exactly **one** leading batch
  dimension.
- Images are **NHWC / channels-last**.
- Encoder and decoder collapse `(B, T) → N = B·T` before the convolutions and restore afterwards.
- **Imagination starts.** `K = min(imag_last or T, T)` with `imag_last = 0` resolves to **`K = T`**,
  not `K = 0` — the `or` makes `0` fall through. So **every** replay timestep is a rollout start:
  `B·K = 16 · 64 = 1024` rollouts of `H + 1 = 16` states, shaped `(1024, 16, ...)`.
  **This drives the H=30 memory question in §10-8**: at H=30 the imagined tensor is `(1024, 31, ...)`.
- Compute dtype in the reference is bfloat16 with float32 parameters and float32 losses. This project
  starts in **float32 throughout** per [config.md](config.md), so bf16-specific effects are out of
  scope until precision is revisited after correctness.

### 4.11 Parameter count — derived, not measured

Summing the shapes above gives **686,846 trainable parameters** for Walker Walk (`A = 6`), excluding
the `slowval` mirror (66,111 more, untrained), optimizer state, and return-norm scalars.

| Module | Params |
|---|---:|
| `dyn` (RSSM incl. posterior + prior) | 376,704 |
| `enc` | 14,304 |
| `dec` | 80,595 |
| `rew` | 57,663 |
| `con` | 41,153 |
| `pol` | 50,316 |
| `val` | 66,111 |
| **Total** | **686,846** |

> **This is arithmetic over the specification, not a measurement**, and it is labelled that way in
> [`param-count-derived-2026-09-17.txt`](../results/m0/param-count-derived-2026-09-17.txt), which
> regenerates it from [`param_count_derive.py`](../results/m0/param_count_derive.py). It was produced
> independently of the source-tracing pass and agreed with it to the digit, which is evidence the
> shape table is **complete and self-consistent** — not evidence that any code is correct.
> §10-7 requires `sum(p.numel())` over instantiated modules. **Discharged for `dyn` and `enc` at
> M2+M3, for `dec`, `rew` and `con` at M4, and for `val` at M7** — every figure in this table
> reproduced exactly from real modules, which is evidence that the §4.1–§4.5, §4.7 and §4.9 shape,
> bias and `outscale` rules were transcribed correctly. Only `pol` remains, at M8.
>
> **The name `size1m` remains not evidence of any count.** The derived figure is ~0.69M, below the
> preset's name and far below the paper's smallest evaluated row of 12M.


## 5. Objectives and gradient boundaries

Permalink base for this section:
`https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/`

### 5.0 Preserved V3 methods

Every item is **specified**; none is implemented. Each links to the subsection that fixes it.

| Method | Status | Where specified |
|---|---|---|
| Categorical straight-through sampling | specified | §5.1 |
| Uniform mixing (`unimix = 0.01`) | specified | §5.1 |
| Separately weighted dynamics / representation KL | specified | §5.2 |
| Free bits (1 nat, post-summation) | specified | §5.2 |
| `symexp_twohot` reward and value prediction | specified | §5.5 |
| Return normalization | specified | §5.6 |
| Entropy regularization | specified | §5.6 |
| Slow-critic regularization | specified | §5.7 |
| **v2 replay critic objective** | specified — **included**, not deviated | §5.7 |

### 5.1 Latent distribution

`rssm.py#L173-L176` builds the latent distribution as `OneHot(logits, unimix)` wrapped in
`Agg(·, 1, sum)`. Logits have shape `(B, T, stoch, classes)`; for this project `(B, T, 32, 4)`.

- **Unimix:** `probs = (1 - u) * softmax(logits) + u / classes` with `u = 0.01`, then
  `logits = log(probs)`. Applied to the **posterior and the prior both**, before any KL.
- **Straight-through sample:** `sample = sg(onehot_sample) + (probs - sg(probs))`
  (`outs.py#L269`). The forward value is a valid one-hot; the gradient flows to `probs`.
- Aggregation over the `stoch` axis is a **sum** of per-factor quantities, never a mean.

### 5.2 World-model KL — direction, stop-gradients, free bits

Verbatim, `rssm.py#L120-L133`:

```python
prior = self._prior(feat['deter'])
post = feat['logit']
dyn = self._dist(sg(post)).kl(self._dist(prior))
rep = self._dist(post).kl(self._dist(sg(prior)))
if self.free_nats:
    dyn = jnp.maximum(dyn, self.free_nats)
    rep = jnp.maximum(rep, self.free_nats)
```

| Term | Expression | Trains | Weight |
|---|---|---|---|
| `dyn` | `KL( sg(q) ‖ p )` | the **prior** toward the posterior | `1.0` |
| `rep` | `KL( q ‖ sg(p) )` | the **posterior** toward the prior | `0.1` |

**Reduction order, which is load-bearing.** `Categorical.kl` reduces the `classes` axis
(`outs.py#L236-L240`) giving `(B, T, 32)`; `Agg.kl` then **sums the 32 factors**
(`outs.py#L73-L76`) giving `(B, T)`. Only then is `maximum(·, 1.0)` applied. Therefore:

> **Free bits = 1.0 nat for the whole latent vector, ≈0.031 nats per factor.**
> Applying 1.0 nat per factor would be a 32× stronger floor and a different objective.

Below the threshold the term is a constant, so its gradient is **exactly zero** — not merely small.
`free_nats` lives at `configs.yaml#L91` inside `agent.dyn.rssm`, and is an RSSM constructor field at
HEAD (it was top-level `rssm_loss: {free: 1.0}` in the 2024 rewrite — do not look for that key).

### 5.3 Reconstruction, reward, continuation

| Loss | Target | Reduction | Weight | Gradient into enc + RSSM |
|---|---|---|---|---|
| `rec` | raw observation, `sg(target)` (`agent.py#L182`) | MSE summed over the 3 image axes, then mean over valid `(B,T)` | `1.0` | yes |
| `rew` | `reward_{t+1}` via `symexp_twohot` | mean over valid `(B,T)` | `1.0` | **yes** — `reward_grad: True`, so no `sg` at `agent.py#L172` |
| `con` | see §5.4 | mean over valid `(B,T)` | `1.0` | **yes, unconditionally** — `agent.py#L177` has no flag and no `sg` |

**Reconstruction reduction is sum-over-pixels, mean-over-batch/time.** For 64×64×3 that is 12288
squared terms summed per position, so the effective weight of `rec: 1.0` against `rew: 1.0` is ~1e4.
This is deliberate in the reference and must **not** be "fixed" by switching to a pixel mean — doing
so would silently reweight the world model by four orders of magnitude.

**Masking.** Loss-bearing positions exclude (a) the `P = 5` burn-in prefix (§7.5) and (b) any reset
observation with no preceding reward target. Denominators count only loss-bearing positions; a
burn-in frame must not appear in any mean.

### 5.4 Continuation target and the location of γ

`agent.py#L174-L177`:

```python
con = f32(~obs['is_terminal'])
if self.config.contdisc:
    con *= 1 - 1 / self.config.horizon
losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
```

**The discount is folded into the continuation target.** With `contdisc: True` and `horizon: 333`:

- soft label = `0.996997` on a non-terminal step, `0.0` on a terminal one;
- the head is a Bernoulli whose `logp` accepts the **fractional** label as
  `y·log σ(x) + (1−y)·log σ(−x)` (`outs.py#L197-L201`);
- in the imagined return `disc = 1` (`agent.py#L401`), so the per-step discount **is** the
  continuation head's output, ≈`0.997 · P(not terminal)`, with **no separate γ factor**;
- the trajectory weight is `cumprod(disc·con, 1)/disc = cumprod(con, 1)`, hence `weight[0] = con[0]`.

**`contdisc` does not apply to the replay path**, which hard-codes `disc = 1 − 1/333`
(`agent.py#L464`) and uses the true `is_terminal` / `is_last` flags.

**Readout is the MEAN, not the mode.** `agent.py#L206` uses `con(...).prob(1) = σ(logit)`.
`Binary.pred()` is the mode (`outs.py#L194-L195`) and is used in **no** loss.

> **Failure mode this creates.** Applying γ explicitly *and* training on the soft label double-counts
> the discount; training on a hard 0/1 label while setting `disc = 1` drops it entirely. Neither
> raises an exception and both train. The M7 gate — constant rewards, early termination, horizon
> bootstrapping against hand-computed values — is the detector.

### 5.5 `symexp_twohot`

Support and bins, matching the reference and reproduced in
[`results/m0/twohot_readout_check.py`](../results/m0/twohot_readout_check.py):

```python
half = symexp(linspace(-20, 0, (bins - 1) // 2 + 1))   # 128 points
bins = concat([half, -half[:-1][::-1]])                # 255, symmetric
```

| Item | Value |
|---|---|
| Bin count | **255** (paper-silent; from code and from v1's explicit `K = 255`) |
| Support | `±symexp(20) = ±4.851652e8` |
| Spacing | **exponential in value space** — *not* uniform in symlog space |
| Target encoding | two-hot on the **untransformed** value; weights are distances in value space |
| Scalar readout | `E[b] = Σ p_i b_i` — **no outer `symexp`** |
| Output init | `outscale: 0.0` — zero-init so predictions start at exactly 0 |

**Transform-of-expectation vs expectation-of-transform.** v1 used symlog-uniform bins with readout
`symexp(Σ p_i b_i)`; v2 uses value-space bins with readout `Σ p_i b_i`. These are different
estimators. From the committed artifact
[`twohot-readout-2026-09-17.txt`](../results/m0/twohot-readout-2026-09-17.txt), on a bimodal
prediction with mass 0.5 on 0 and 0.5 on 100 (true mean 50): v1 reads **9.05**, v2 reads **50**.
Both are exact for their own two-hot target; they diverge on the spread distributions a trained
critic actually emits, because `symexp` is convex on the positive half.

> **What this does not establish.** v2 removes the *transform-order* bias in reading the mean of the
> **predicted** distribution. It does **not** make the value an unbiased estimator of true
> environment return: the predicted distribution is a learned approximation, λ-returns bootstrap off
> the critic, the two-hot target is a projection onto a finite support that saturates beyond
> `±symexp(20)`, and the slow-critic regularizer biases toward an older parameter set by
> construction. No unbiasedness claim is made anywhere in this repository.

**Numerical accumulation is part of the contract, not an optimization.** Sum the negative and the
positive bins **separately, each from small `|b|` to large**, then add the two partial sums. The
extreme bins are `±4.85e8`; in float32 a partial sum of that magnitude has an ulp near 32, so an
`O(1)` term added between the two tail terms is absorbed and the cancellation leaves a residue.
Measured in the artifact on a uniform `p` whose exact readout is `0`: sequential float32 gives
`+0.25`, `np.sum` float32 gives **`-1.0`**, split-and-sorted gives `0`. Sequential and pairwise
reductions fail on **different** inputs, so a framework's default reduction is not a safeguard. This
directly protects the zero-init guarantee above.

### 5.6 Actor

`agent.py#L411-L415`:

```python
logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
policy_loss = sg(weight[:, :-1]) * -(
    logpi * sg(adv_normed) + actent * sum(ents.values()))
```

Minimized loss, per position, with the unary minus distributed:

```
L_policy = - w_t · logπ(a_t) · Â_t   -   w_t · η · H[π(·|s_t)]
```

| Item | Value / rule | Source |
|---|---|---|
| Estimator | **REINFORCE for continuous actions.** The sampled action is `sg`'d, so there is no pathwise term. | `agent.py#L411` |
| Baseline | `Â_t = (R_t^λ − v(s_t)) / S`, with `tarval[:, :-1]` as the baseline | `agent.py#L408` |
| **Entropy sign** | **SUBTRACTED in the minimized loss.** `η = +3e-4`. Minimizing maximizes entropy. | `agent.py#L413-L414`, `configs.yaml#L108` |
| Return norm `S` | `S = max(1, EMA(Per(R,95) − Per(R,5), rate 0.01))` | `retnorm: {impl: perc, rate: 0.01, limit: 1.0, perclo: 5, perchi: 95}` |
| Continuation weighting | `w_t = cumprod(con, 1)`, `sg`'d | `agent.py#L402`, `#L413` |
| Distribution | `bounded_normal`, `minstd: 0.1`, `maxstd: 1.0`, `outscale: 0.01` — see §4.6 | `configs.yaml` |
| `valnorm`, `advnorm` | `impl: none` — **not implemented** | `configs.yaml#L111-L113` |

**No `scale_by_actent` exists at HEAD** — the inverted-parameterization branch present in the 2024
rewrite was deleted. Do not implement it.

**Sign audit through to the update.** `losses['policy']` is summed with `loss_scales.policy = 1.0`
(`agent.py#L240`) and never negated; the only sign flip in the whole path is inside
`optax.scale_by_learning_rate` (`flip_sign=True`), giving `θ ← θ − lr·μ̂`. So the pipeline performs
plain gradient **descent on the loss exactly as written above**. A PyTorch port must therefore
`loss.backward()` on that same expression with no additional negation.

### 5.7 Critic — imagined loss, replay loss, slow critic

**One shared λ-return kernel**, `agent.py#L482-L490`, used by both paths — see §6.1 for the full
recursion and index proof.

| | Imagined critic (`value`) | Replay critic (`repval`) |
|---|---|---|
| Weight | `1.0` | **`0.3`** |
| Sequence | H+1 = 16 imagined states from a replay posterior start | the real replay sequence |
| Rewards | predicted by the reward head | **real** replay rewards |
| Bootstrap | the **fast** critic (`slowtar: False`) | the **imagined return** `ret[:, 0]` reshaped to `(B, K)` (`agent.py#L222`) |
| Discount | `disc = 1`; per-step discount is `con` (§5.4) | hard-coded `1 − 1/333` |
| Boundary flags | `last = 0` everywhere; `term = 1 − con` (predicted) | **true** `is_last` / `is_terminal` |
| Position weight | `sg(weight[:, :-1])` | `f32(~is_last)[:, :-1]` |
| Target | `sg(λ-return)` | `sg(λ-return)` |
| Gradient into enc + RSSM | **no** | **yes** — see §5.9 |

**Slow critic.** An EMA copy updated every step at rate `0.02` (`slowvalue: {rate: 0.02, every: 1}`,
equivalently decay 0.98). It is **not** used as a bootstrap target; it enters **only** as a
regularizer added to both critic losses:

```
+ slowreg · value.loss( sg(slowvalue.pred()) ),   slowreg = 1.0
```

The slow copy is excluded from the optimizer's module list and is advanced by
`self.slowval.update()` after the gradient step (`agent.py#L142`).

### 5.8 Optimizer — LaProp

`agent.py#L342-L379` plus the three hand-rolled transforms at `embodied/jax/opt.py#L109-L164`. The
chain is `clip_by_agc → scale_by_rms → scale_by_momentum → scale_by_learning_rate`.

Per parameter tensor `θ`, as implemented:

```
g  ←  g · 1 / max(1,  ‖g‖₂ / (agc · max(pmin, ‖θ‖₂)) )      # AGC, whole-tensor norm
ν  ←  β₂·ν + (1−β₂)·g²          ;   ν̂ = ν / (1 − β₂ᵗ)
p  ←  g / (√ν̂ + ε)                                          # ε OUTSIDE the sqrt
μ  ←  β₁·μ + (1−β₁)·p           ;   μ̂ = μ / (1 − β₁ᵗ)
θ  ←  θ − lr(t) · μ̂
```

**This is LaProp, not Adam.** The second moment normalizes the **raw gradient first**, and momentum
accumulates on the *already-normalized* update. Adam accumulates momentum on the raw gradient and
divides at the end. The two are not interchangeable, and `torch.optim.Adam` is **not** a valid
substitute.

| Setting | Value | Note |
|---|---|---|
| `lr` | `4e-5` | single LR for all modules; per-module LRs do not exist at HEAD |
| `eps` | `1e-20` | added **outside** the square root |
| `β₁` | `0.9` | |
| `β₂` | **`0.999`** (code) vs **`0.99`** (paper Table 4) | **Take the code value, 0.999.** The paper is a rounded statement of an implementation whose config is explicit; §9-13. |
| AGC | `0.3`, `pmin = 1e-3` | **whole-tensor** L2 norm, not the per-row variant of the AGC paper |
| Global-norm clip | **none** | absent at HEAD; the `globclip` knob was removed |
| Warmup | linear `0 → lr` over **1000** steps, then constant | |
| Weight decay | `0.0` | the `if wd:` branch never executes; pattern would be `/kernel$` |
| Nesterov | `False` | |

**Two dead switches in the reference — do not port them as features.** `_make_opt`'s `momentum: bool`
is never read (momentum is unconditional), and `lambda_return`'s `val` argument is never read, which
makes `repl_loss`'s `slowtar` a no-op. A reader of `configs.yaml` would reasonably assume both are
live. §6.1 and §9-14.

### 5.9 Gradient-routing table

Single optimizer over `[dyn, enc, dec, rew, con, pol, val]` (`agent.py#L74-L78`); `slowval` is
excluded and EMA-updated.

| Loss | Weight | Updates | Detached inputs / targets |
|---|---:|---|---|
| `rec` | 1.0 | enc, RSSM, dec | target observation |
| `rew` | 1.0 | enc, RSSM, rewhead | twohot target |
| `con` | 1.0 | enc, RSSM, conhead | soft label |
| `dyn` | 1.0 | **prior only** | posterior logits |
| `rep` | 0.1 | enc, RSSM posterior | prior logits |
| `policy` | 1.0 | **actor only** | imagined features, sampled actions, advantage, weight |
| `value` | 1.0 | **critic only** | imagined features, λ-return, slow-critic target |
| `repval` | **0.3** | **enc, RSSM, critic** | λ-return target, slow-critic target — **features NOT detached** |

**The two gradient paths that must not be confused:**

| | Actor gradients through imagined dynamics | Replay-critic gradients into representations |
|---|---|---|
| Status at HEAD | **BLOCKED** | **LIVE** |
| Mechanism | `imgfeat = concat([sg(first, skip=ac_grads), sg(imgfeat)], 1)` at `agent.py#L196` with `ac_grads: False`, so `sg` applies to the start state **and** unconditionally to the whole imagined trajectory | `feat = sg(repfeat, skip=repval_grad)` at `agent.py#L220` with `repval_grad: True`, so `sg` is **not** applied |
| Consequence | The actor receives **no** pathwise gradient through the world model. This is why v2 needs REINFORCE — there is no differentiable path to exploit. | Value error shapes the encoder and RSSM. This path **does not exist in v1**, where the critic optimizer was scoped to the critic MLP alone. |

**Transcription hazard.** The reference helper has an **inverted** sense:
`sg = lambda xs, skip=False: xs if skip else stop_gradient(xs)` (`agent.py#L17`). `skip=True` means
**no** stop-gradient. Reading `sg(repfeat, skip=True)` as "stop gradient" inverts three of the rows
above. A PyTorch port must use explicit `.detach()` at the points named in this table and must not
reproduce the helper's polarity.


## 6. Resolved ambiguities

Three items that the paper alone cannot decide. Each is settled by tracing the pinned code, with the
chosen equation, the reasoning, and the module-level test that will verify the decision later. The
tests are **specified, not implemented** (§11).

### 6.1 λ-return indexing and the bootstrap state

**Ambiguity.** arXiv v2 Eq (5) prints the bootstrap as `v_t`; arXiv v1 Eq (7) prints
`v_ψ(s_{t+1})`. The PDFs cannot resolve which is intended.

**Resolution: the bootstrap is `v[t+1]`. v2's `v_t` is a typographical regression; implement v1's
reading.** `agent.py#L482-L490`:

```python
def lambda_return(last, term, rew, val, boot, disc, lam):
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)
```

The array is `boot[:, 1:]` and the seed is `boot[:, -1]`. **`boot[:, :-1]` never appears.** In
conventional notation, with `γ_{t+1} = (1 − term_{t+1})·disc` and `λ_{t+1} = (1 − last_{t+1})·λ`:

```
R_t = r_{t+1} + γ_{t+1} · [ (1 − λ_{t+1}) · V_{t+1} + λ_{t+1} · R_{t+1} ]
R_{L-1} := V_{L-1}                          (scan seed)
output length L-1;  R_t aligns with state t
```

**Boundary behaviour under our §7.3 convention** — this is why the two flags must stay separate:

| Case | Effect | Resulting target |
|---|---|---|
| `is_last[t+1] = 1`, `is_terminal[t+1] = 0` (**time limit**) | `λ_{t+1} = 0`, `γ_{t+1} = disc` | `R_t = r_{t+1} + disc · V_{t+1}` — trace cut, **bootstrap at full weight** |
| `is_terminal[t+1] = 1` (**true terminal**) | `γ_{t+1} = 0` | `R_t = r_{t+1}` exactly — **no bootstrap** |

Both DMControl tasks end by time limit, so the first row is the one that fires every episode. Using
`is_last` as the terminal signal would take the second row instead and truncate every return.

**Which value estimate.** Imagined path: the **fast** critic (`slowtar: False`, `configs.yaml#L108`);
the slow critic enters only through `slowreg`. Replay path: the bootstrap is the **imagined return**
`imgloss_out['ret'][:, 0]` reshaped to `(B, K)` (`agent.py#L222`) — so the replay critic's target at
`t` is real rewards from `t+1` onward, bootstrapped by the *imagined* return at `t+1`.

**Dead argument, recorded so it is not mistaken for a feature.** `lambda_return`'s `val` parameter is
never read — it appears only in a shape assertion. In `imag_loss` the same tensor is passed as both
`val` and `boot`, so nothing is lost; in `repl_loss` it means `slowtar`, `val`, `slowval` and
`tarval` (`agent.py#L461-L463`) are **all dead**. Our implementation takes a single `boot` argument
and no `val`, removing the trap.

**Planned test (M7).** On hand-computed fixtures with `disc` and `λ` fixed: (a) all rewards zero →
`R_t = 0` for all `t`; (b) constant reward `r`, no boundary → geometric sum matching the closed form;
(c) `is_terminal` at `t+1` → `R_t = r_{t+1}` exactly; (d) `is_last` without `is_terminal` at `t+1` →
`R_t = r_{t+1} + disc·V_{t+1}`; (e) `λ = 0` → one-step TD; (f) `λ = 1` → Monte-Carlo sum to the seed.
Each asserted to exact arithmetic, and each must fail if the bootstrap index is changed to `t`.

### 6.2 Entropy sign in the minimized actor loss

**Ambiguity.** In arXiv v2 Eq (6) the scope of `+ ηH` relative to the leading `−Σ` is not bracketed,
so the printed sign of the entropy term is indeterminate.

**Resolution: in the quantity that is minimized, entropy is SUBTRACTED, with `η = +3e-4`.**
`agent.py#L413-L414` distributes the unary minus over both terms:

```python
policy_loss = sg(weight[:, :-1]) * -(
    logpi * sg(adv_normed) + actent * sum(ents.values()))
```

so

```
L_policy = − w_t · logπ(a_t) · Â_t  −  w_t · η · H[π(·|s_t)]
```

Minimizing `L_policy` therefore **maximizes** entropy — an entropy bonus, as intended. `w_t ≥ 0`
(a cumulative product of probabilities), so the prefactor introduces no sign flip.

**Verified through to the parameter update**, because a downstream negation would invert the
conclusion: `losses['policy']` is summed with scale `1.0` (`agent.py#L240`) and is never negated; the
single sign flip in the path is inside `optax.scale_by_learning_rate` (`flip_sign=True`), producing
`θ ← θ − lr·μ̂`. The pipeline is plain descent on the loss as written.

**Planned test (M8).** A two-action analytic fixture with fixed advantages: (a) with `η = 0`, one
update must increase `logπ` of the higher-advantage action and decrease the other; (b) with `Â = 0`
and `η > 0`, one update must **increase** the policy's entropy — this fails if the sign is flipped;
(c) entropy of a `bounded_normal` must match a closed-form value for known parameters.

### 6.3 Where replay-critic gradients go

**Ambiguity.** arXiv v1 states the components train *"without sharing gradients"*; arXiv v2 **deletes
that sentence** and its ablation figure is phrased as if value gradients shape the representation.
Neither version gives a stop-gradient equation at the component boundary.

**Resolution: at HEAD the replay-critic loss backpropagates into both the encoder and the RSSM. v1's
"without sharing gradients" was deleted because it stopped being true.**

The mechanism, in order:

1. `agent.py#L17` — `sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)`.
   **`skip=True` means no stop-gradient.**
2. `agent.py#L164-L167` — `repfeat` is produced by `self.enc(...)` then `self.dyn.loss(...)`, so it
   is a function of encoder **and** RSSM parameters.
3. `agent.py#L220` — `feat = sg(repfeat, skip=self.config.repval_grad)` with `repval_grad: True`
   (`configs.yaml#L116`) ⇒ **`sg` is not applied**; the gradient is live.
4. `agent.py#L471-L473` — both target terms are `sg`'d (`sg(ret_padded)` and
   `sg(slowvalue.pred())`), but `value.loss` differentiates w.r.t. its **input**, so
   `∂L/∂repfeat ≠ 0`.
5. `agent.py#L74-L78` — one optimizer covers `[dyn, enc, dec, rew, con, pol, val]`.
6. `loss_scales.repval = 0.3` scales the contribution.

**Two adjacent paths share the mechanism:** the reward head also backprops into the world model
(`reward_grad: True`, no `sg` at `agent.py#L172`), and the continuation head does so
**unconditionally** — `agent.py#L177` has no flag at all.

**And the path that is blocked**, which is the one usually assumed to be open:
`imgfeat = concat([sg(first, skip=ac_grads), sg(imgfeat)], 1)` at `agent.py#L196` with
`ac_grads: False` applies `sg` to the imagination start state **and** unconditionally to the whole
imagined trajectory. **The actor gets no pathwise gradient through the world model** — which is
exactly why v2's actor must be REINFORCE. See the two-column table in §5.9.

**Planned tests (M4, M8).** (a) Backward from `repval` alone; assert encoder and RSSM parameters have
non-`None`, non-zero grads and that actor parameters have none. (b) Backward from `policy` alone;
assert **zero** grad on encoder, RSSM, decoder, reward and continuation parameters, and non-zero on
the actor. (c) Backward from `dyn` alone; assert zero grad on the posterior-only projections and
non-zero on the prior net, accounting for parameters the two genuinely share. (d) Backward from
`value` alone; assert zero grad on encoder and RSSM. Each assertion must fail if the corresponding
`.detach()` is removed or added.


## 7. Environment and counter definitions

Source for every reference claim below:
[`embodied/envs/dmc.py`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/envs/dmc.py),
[`embodied/envs/from_dm.py`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/envs/from_dm.py),
[`embodied/core/wrappers.py`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/core/wrappers.py).

### 7.1 Observations and preprocessing

| Item | Decision | Source |
|---|---|---|
| Image | 64×64×3, **uint8**, HWC | `dmc.py#L56` declares `Space(np.uint8, size + (3,))` |
| Render | `physics.render(64, 64, camera_id=camera)` — rendered **directly at 64×64**, not downsampled from larger | `dmc.py#L72` |
| Camera | **0** for both walker and cartpole | `dmc.py#L27-L28`: `camera == -1` → `DEFAULT_CAMERAS.get(domain, 0)`; neither domain is in that dict |
| GL backend | `MUJOCO_GL=egl` | `dmc.py#L23-L24` |
| Privileged state | **excluded.** `proprio: False` drops every non-basic observation, leaving only `is_first`/`is_last`/`is_terminal`/`reward` plus `image` | `dmc.py#L53-L54`, and `dmc_vision` sets `env.dmc.proprio: False` |
| Model input scaling | uint8 → float32, `x/255 − 0.5`, on GPU at the model boundary | Project choice. Replay stores uint8 only; see §9-7. |
| Vector symlog | not applicable — no vector observations reach the final agent | `enc.simple.symlog: True` governs vector inputs only |

Preprocessing is a single function used by **both** collection and evaluation. No evaluation-only
resize, crop, or normalization is permitted; M1's gate checks this by identity.

### 7.2 Actions

| Item | Decision |
|---|---|
| Space | bounded continuous, `[low, high]` taken from `dm_control`'s `action_spec()`; asserted finite |
| Dimensions | Walker Walk **6**, Cartpole Swingup **1** — **verified** against `action_spec()` on the installed suite ([`env-render-2026-09-17.txt`](../results/m1/env-render-2026-09-17.txt)) |
| Bounds | **`[-1, 1]`** for both, `float64` — **verified**; asserted finite |
| Clipping | actions clipped to the spec range before `env.step` |
| Repeat | **1.** No aggregation occurs at R=1. |

**Reward aggregation under repeat.** The reference **sums** reward over the repeat and breaks early on
boundary (`wrappers.py#L57-L73`):

```python
reward = 0.0
for _ in range(self._repeat):
    obs = self.env.step(action)
    reward += obs['reward']
    if obs['is_last'] or obs['is_terminal']:
        break
obs['reward'] = np.float32(reward)
```

At R=1 this is a pass-through, which is why R=1 is chosen: it removes reward-scale coupling to the
repeat. **If R is ever changed, the reward scale changes with it** and return numbers stop being
comparable across conditions. R is therefore frozen at 1 for all 12 final runs.

### 7.3 Continuation, terminal, and time limit

This is the contract the M1 gate exists to protect. The reference derives the two flags separately
(`from_dm.py#L74-L76`):

```python
is_last=time_step.last(),
is_terminal=False if time_step.first() else time_step.discount == 0,
```

| Flag | Meaning | DMControl suite behaviour |
|---|---|---|
| `is_first` | reset observation | reward forced to 0 (`from_dm.py#L73`) |
| `is_last` | **episode boundary of any kind** | true at the time limit |
| `is_terminal` | **true termination**, i.e. `discount == 0` | **false** at a time limit |

`from_dm.py#L64` asserts `discount in (0, 1)` — the discount is binary in `dm_control`, so no
fractional-discount case arises.

**Therefore:** continuation target `c_t = 1 − is_terminal`, **never** `1 − is_last`. Walker Walk and
Cartpole Swingup end by time limit, so at the boundary `is_last=True, is_terminal=False` and the
continuation target is **1** — the value must be bootstrapped, not zeroed. Using `is_last` would
zero the bootstrap on every episode of both tasks and silently truncate every return.

`TimeLimit` (`wrappers.py#L28-L55`) sets only `is_last`, never `is_terminal`, consistent with the
above. For DMControl the limit comes from `dm_control` itself via `time_step.last()`, so no extra
wrapper is used.

> **Confirmed empirically, not just read from source.** A full random-policy episode was run on each
> task ([`env-render-2026-09-17.txt`](../results/m1/env-render-2026-09-17.txt)). Both end at step
> 1000 with `last() == True` and **`discount == 1.0`**, and **no step in either episode had
> `discount == 0`**. So `is_terminal` is `False` for the entire episode including its final step, and
> a `1 − is_last` continuation target would zero the bootstrap on every episode of both tasks.

### 7.4 Transition record and alignment

Stored per transition, matching [milestones.md](milestones.md) M1:

```
(observation_t, action_t, reward_{t+1}, observation_{t+1}, is_last, is_terminal, discount)
```

The reward delivered with an observation is the reward received **on arriving at** it, and is forced
to 0 on a reset observation (`from_dm.py#L73`). Previous-action alignment follows the reference: at
sequence position `t` the model consumes action `t−1`, with position 0 taking the action carried over
from the preceding chunk (`agent.py#L317-L318`). The last observation before a reset is kept.

### 7.5 Sequence initialization and burn-in

**The reference does not recompute a burn-in.** It persists latent states in the replay buffer and
restores the carry from them (`agent.py#L312-L340`): sampled length is
`consec * batch_length + replay_context` = 65 at `replay_context: 1`, the first `K=1` frame restores
the carry via `dyn.truncate(...)`, and updated latents are written back after each training step.
The restored carry is used only where `consec == 0`.

**Project decision: recomputed burn-in prefix with masked losses. Classified as an implementation
choice (§9-8).** Prefix length `P = 5`, loss-masked. Rationale:

1. Stored latents go stale as the model trains. The reference tolerates this; for a project whose M1
   gate is specifically about *trustworthy temporal data*, a context that is always model-current is
   easier to reason about when a bug is suspected.
2. M9 must checkpoint and resume model, optimizer, and replay state consistently. A recomputed prefix
   is reproducible from the stored data alone, so resume correctness does not depend on latent
   versioning.
3. It is testable in isolation: the M2 gate compares single-step against sequence recurrence on
   identical inputs, which requires the context to be a pure function of the data.
4. Cost: `P × B = 80` extra RSSM steps per gradient step against `B × T = 1024` loss-bearing
   positions, ≈8% of the recurrent forward pass. Accepted.

Burn-in positions carry **no** loss of any kind and are excluded from every reduction denominator.

**Sequence contract.** `P` and `T` are counted separately and never overlap:

| Quantity | Value at `P = 5`, `T = 64` | Meaning |
|---|---:|---|
| `P` | 5 | burn-in transitions, extra context, loss-masked |
| `T` | 64 | loss-bearing training transitions |
| transitions per sample | `P + T` = **69** | actions, rewards, `is_last`, `is_terminal`, discounts |
| observations per sample | `P + T + 1` = **70** | one leading observation plus one per transition |
| `train_position` per sample | `T` = 64 | positions entering a loss denominator |
| `train_position` per gradient step | `B × T` = **1024** at `B = 16` | loss-reduction denominator |

`T` is **not** reduced by `P`: a batch is 69 transitions long so that 64 of them bear loss. Any
expression of the form `B × (T − P)` is wrong and is not used anywhere in this project.

### 7.6 Step counters

Five distinct counters. Conflating any two misstates the data budget.

| Counter | Definition | Role |
|---|---|---|
| `env_step` | one `env.step()` on the underlying `dm_control` environment = one control timestep | **The budget unit.** Walker 1e6, Cartpole 5e5 per run. |
| `agent_step` | one action selected by the policy; `1 agent_step = R env_steps` | Equals `env_step` at R=1, but logged separately so a future R change cannot silently rescale the budget. |
| `physics_substep` | MuJoCo integrator steps inside one control timestep | Never a budget unit. Recorded once at M1 for the record. |
| `replay_transition` | one stored transition record (§7.4); one per `agent_step` | Replay occupancy. |
| `gradient_step` | one optimizer update | Compute accounting. |
| `train_position` | one `(batch, time)` position that carries loss | Loss-reduction denominator. `B × T` per gradient step = 1024 at `B=16, T=64`; the `P=5` burn-in prefix is additional and excluded (§7.5). |

**Verified on the installed simulator** ([`env-render-2026-09-17.txt`](../results/m1/env-render-2026-09-17.txt)):

| | Walker Walk | Cartpole Swingup |
|---|---|---|
| Episode length | **1000** control steps | **1000** control steps |
| `control_timestep` | 0.025 s | 0.01 s |
| Physics timestep | 0.0025 s | 0.01 s |
| **Physics substeps per control step** | **10** | **1** |
| Per-step reward range observed | [0.0050, 0.1537] | [8.06e-09, 0.0359] |
| Throughput incl. 64×64 render | 923 steps/s | 996 steps/s |

The substep counts **differ between the two tasks** — Cartpole's physics timestep equals its control
timestep. This is exactly why `physics_substep` is never a budget unit.

### 7.7 Training ratio — units reconciled

The reference defines it in [`embodied/run/train.py#L24-L25`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/run/train.py#L24-L25):

```python
batch_steps = args.batch_size * args.batch_length
should_train = elements.when.Ratio(args.train_ratio / batch_steps)
```

and its tests assert `replay_steps = env_steps * train_ratio`
([`test_train.py#L20`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/tests/test_train.py#L20),
[`test_parallel.py#L39-L40`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/embodied/tests/test_parallel.py#L39-L40)).

**So `train_ratio` = replayed frames per environment step.** The plan's "replay training positions per
collected transition" is **the same unit**. The comparison is therefore valid:

| | Value | Gradient steps per `env_step` | `env_steps` per gradient step |
|---|---:|---|---|
| Pinned reference `dmc_vision` | **256** | 256/1024 = 0.25 | 4 |
| This project, initial | **64** | 64/1024 = 0.0625 | 16 |

The project's starting ratio is **4× lower** than the pinned reference for the same task. This is a
resource restriction with a learning-rate-of-progress consequence, not a neutral choice; §9-9 and its
M9 qualification.

**Definitional caution:** the reference counts *all* replayed frames including the context frame
(`batch_size * batch_length`). This project's `train_ratio` counts **loss-bearing positions only**,
i.e. `B × T` = 1024, excluding the `P = 5` burn-in prefix that each sample additionally carries
(§7.5), because that prefix is recomputed rather than loss-bearing. When quoting a
realized ratio, state which convention is used; §10 requires the realized ratio to be logged, not
assumed.

### 7.8 Evaluation behaviour

| Item | Decision |
|---|---|
| Action selection | **mean of the actor distribution**, no sampling, no entropy injection |
| Episodes | 5 per diagnostic evaluation (M9); **20** at the final checkpoint per training seed (M10) |
| Checkpoint | the **fixed final** checkpoint. Best-checkpoint selection on evaluation return is forbidden. |
| Data use | evaluation trajectories enter **neither** replay nor any gradient step |
| Timing | evaluation wall-clock logged separately from training wall-clock |

## 8. Seeds

Disjoint by construction. No seed appears in two rows.

| Purpose | Seeds | Rule |
|---|---|---|
| Development training | **100, 101, 102** (reserve 103–119) | Debugging and pilots. Must not appear in any final result. |
| Development evaluation (M9, 5 episodes) | **2000–2004** | Periodic diagnostic evaluation during development runs. |
| **Final training** | **0, 1, 2** | Fixed by [experiment.md](experiment.md). Same labels across all four conditions. |
| **Final evaluation** (20 episodes/seed) | **1000–1019** | Reserved. Identical set for every condition and every training seed. |
| Diagnostic corpus — random policy | **500–509** | Frozen after M9; excluded from training. |
| Diagnostic corpus — development policy | **510–519** | Frozen after M9; excluded from training. |

### 8.1 RNG stream separation

Five independent generators, each seeded by a fixed derivation from the run seed. Rationale: at
H=30 the actor and prior sample far more often per `env_step` than at H=5, so a shared stream would
make the environment's trajectory a function of the imagination horizon and confound the study.

| Stream | Consumes |
|---|---|
| `env` | simulator reset and any environment stochasticity |
| `collect` | action sampling during real interaction |
| `replay` | sequence sampling from the buffer |
| `imag` | latent sampling and actor sampling inside imagination |
| `init` | parameter initialization |

Streams are derived as `seed_derive(run_seed, stream_name)`, not by incrementing a single counter, so
adding a stream later cannot shift existing ones.

## 9. Deviation register

Each entry is classified: **CR** capacity reduction · **AC** algorithmic change · **IC**
implementation choice · **RR** evaluation/resource restriction. "Expected consequence" is a
**hypothesis stated before measurement**, never an observed result.

| # | Class | Deviation | Pinned source | This project | Rationale | Expected consequence (hypothesis) |
|---|---|---|---|---|---|---|
| 1 | **CR** | Model capacity | Paper Table 3 smallest row is **12M** (`d`=256, deter 1024, classes 16); code `size1m` is deter 512 / classes 4 | deter 512, stoch 32, classes 4, CNN depth 4, MLP units 64 | Single 24 GB GPU and a 1e6-step budget | Lower asymptotic return than any published row; the horizon effect may be *larger* than at full scale because a weaker model's prior degrades faster with rollout length. Return level `unmeasured`. |
| 2 | **CR** | `size1m` has no paper counterpart | Paper's evaluated sizes are 12M–400M | `size1m`, a preset added to the repo **after** arXiv v2, in `f8817c4040ce` (2024-12-07) | It is the only compact preset the author published | Results cannot be compared to any paper row. The reproduction claim is about *mechanism*, not about matching a published number. |
| 3 | **AC** | Actor estimator | v2: Reinforce for both action types, baseline `v_ψ(s_t)` | Same as v2 | v1 used pathwise backprop for continuous actions — a different objective for these tasks | None relative to v2. Recorded because silently using v1's rule is the most likely single error, and it would not raise an exception. |
| 4 | **AC** | Discount location (`contdisc`) | Code folds γ into the continuation head's soft target and sets the return discount to 1; **the paper does not mention this** | Adopt the code behaviour — §5.4 | Paper-silent; the code is the only specification | Double-counting or dropping γ if mis-implemented. Neither failure raises an exception; the M7 gate is the detector. |
| 5 | **AC** | Replay critic included | v2 `β_repval = 0.3`, `repval_grad: True` | Included at 0.3, gradients into representations enabled | Required by M0 unless a concrete reason to deviate is found; none was | Value gradients shape the representation, unlike v1. Improves value fitting where reward is hard to predict; couples critic error into the encoder. |
| 6 | **RR** | Replay capacity | 5e6 | **5e5** | 124 GiB RAM; 5e6 × 64×64×3 uint8 ≈ 61 GB for images alone before any other field | At a 1e6-step budget the buffer holds the **whole run**, so FIFO eviction never triggers and the reduction is expected to be inert for Walker and Cartpole. Verify occupancy at M9. |
| 7 | **IC** | Replay dtype | latents and frames persisted | uint8 frames only; no stored latents; no duplicate image storage | RAM, and §7.5 | None on the objective. Removes a staleness coupling. |
| 8 | **IC** | Sequence context | stored-latent carry restore, `replay_context: 1` | recomputed burn-in prefix `P=5`, loss-masked | §7.5 | ≈8% more recurrent compute per gradient step; context always model-current. |
| 9 | **RR** | Training ratio | `dmc_vision` uses **256** | **64** initially, units reconciled in §7.7 | Project starting choice; compute budget | 4× fewer updates per env step ⇒ slower progress per env step and plausibly a lower final return at a fixed 1e6 budget. This is the most likely single cause if learning underperforms. M9 must qualify it. |
| 10 | **RR** | Task coverage | cross-domain, one shared config | 2 DMControl tasks | Scope | Supports a limited reproduction claim only; no cross-domain generality claim. |
| 11 | **IC** | Framework | JAX, bfloat16 compute, one fused optimizer over all modules | PyTorch, eager, fp32 first | [config.md](config.md) requires correctness before precision or compilation | Slower per step than the reference. Numerical differences from JAX are expected and are **not** by themselves evidence of a bug. |
| 12 | **AC** | Exploration machinery | `behaviors.py` / `expl.py` deleted at HEAD; `policy()` accepts `mode` and ignores it | none; warm-up is 5000 uniform-random transitions | Matches the pinned code, which has no exploration bonus | None relative to the pinned reference. |
| 13 | **IC** | Optimizer β₂ | **Paper Table 4 says `0.99`; the pinned code says `0.999`** (`configs.yaml#L87`) | **`0.999`**, the code value | The paper states a rounded value for an implementation whose config is explicit and machine-read. Recorded rather than averaged or silently picked. | Slower second-moment adaptation than `0.99`. If training is unstable in a way that implicates the optimizer, this is the first knob to test, and the test is a deliberate experiment, not a bug fix. |
| 14 | **IC** | Dead switches not ported | `_make_opt`'s `momentum` flag is never read (momentum is unconditional); `lambda_return`'s `val` argument is never read, making `repl_loss`'s `slowtar` a no-op | Implement the **live** behaviour only: momentum always on; a single `boot` argument and no `val`; no `slowtar` on the replay path | A reader of `configs.yaml` would reasonably take both for working switches | None on the objective. Removes two traps where toggling a config field would appear to do something and would not. |

## 10. Deferred measurements and acceptance checks

Every quantity here is **`unmeasured`** and must stay that way in all documents until a committed
artifact under `results/` supports it. Each has a milestone and a check that decides pass/fail.

| # | Quantity | Milestone | Acceptance check |
|---|---|---|---|
| 1 | PyTorch `sm_120` support | M1 | **VALIDATED 2026-09-17.** `sm_120` present in torch 2.13.0+cu129's arch list and matches the device; matmul, the encoder conv, the block-diagonal einsum and float32 RMSNorm all agree GPU-vs-CPU with TF32 off. 14/14 in [`torch-compute-2026-09-17.txt`](../results/m1/torch-compute-2026-09-17.txt) |
| 2 | `dm_control` + MuJoCo, EGL offscreen render | M1 | **VALIDATED 2026-09-17 on Python 3.12.11** (3.13 is not viable — §2). Both tasks render 64×64×3 uint8 non-constant frames headless under `MUJOCO_GL=egl`. 26/26 in [`env-render-2026-09-17.txt`](../results/m1/env-render-2026-09-17.txt) |
| 3 | Action dimensions and bounds | M1 | **VALIDATED 2026-09-17.** Walker 6-dim, Cartpole 1-dim, both `[-1, 1]` float64 — observed, matching §7.2's prediction |
| 4 | Episode length, physics substeps per control step | M1 | **VALIDATED 2026-09-17.** 1000 control steps for both; substeps **10** (Walker) vs **1** (Cartpole). Terminal-vs-time-limit semantics also confirmed empirically — §7.3 |
| 5 | Random-policy return floor | M1 | 20 episodes per task under the final reward convention; mean and sd recorded. |
| 6 | Environment throughput (`env_step`/s incl. render) | M1 | Steady-state rate over ≥10k steps, GPU confirmed free first. |
| 7 | **Instantiated parameter count** | **M2+M3, M4, M7 (done); M8** | **World model MEASURED 2026-09-18: 570,419** — `dyn` 376,704, `enc` 14,304, `dec` 80,595, `rew` 57,663, `con` 41,153, each equal to the §4.11 derivation — [`results/m4/param-count-measured-2026-09-18.txt`](../results/m4/param-count-measured-2026-09-18.txt). **`val` MEASURED 2026-09-18: 66,111**, equal to the derivation, with its untrained `slowval` mirror a second 66,111 excluded from the optimizer — [`results/m7/param-count-measured-2026-09-18.txt`](../results/m7/param-count-measured-2026-09-18.txt). World model + `val` = 636,530. With `pol` 50,316, **still derived and owed at M8**, the total closes at **686,846**. **The name `size1m` is not evidence of any count.** |
| 8 | Peak VRAM at H=30 | M9 | `torch.cuda.max_memory_allocated()` during a steady-state update; must leave headroom below 24467 MiB. **Partially discharged at M6: the imagination tensor alone peaks at 858.3 MiB for 1024 rollouts × 31 states** ([`results/m6/horizon-cost-2026-09-18.csv`](../results/m6/horizon-cost-2026-09-18.csv)). That is one forward rollout on an untrained model — no posterior pass, no backward, no actor or critic — so the steady-state figure is still owed. |
| 9 | Per-stage time share | M9 | Collection, render, replay transfer, world-model update, behaviour update — shares summing to wall-clock. |
| 10 | Realized training ratio | M9 | Logged, with the §7.7 convention named; compared to the requested 64. |
| 11 | Replay RAM occupancy | M9 | Measured at steady state; confirms or refutes §9-6's inertness hypothesis. |
| 12 | Learning above the random floor | M9 gate | Both tasks, development seeds, improvement persisting beyond a transient spike. |
| 13 | Final returns, curve areas, prediction error, cost | M10 | Per [experiment.md](experiment.md). |

## 11. Status legend

Three independent axes. A row is only promoted by evidence, and **specification never implies
implementation**.

| Status | Meaning |
|---|---|
| **specified** | Decided against a pinned source and written down here with its rationale. |
| **implemented** | Code exists that realizes the specification. |
| **validated** | The relevant milestone gate passed, with a committed artifact under `results/`. |

Current state, as of 2026-09-18:

| Scope | Status |
|---|---|
| Objectives, gradient routing, actor (§5–§6) | **`specified`** |
| Environment and transition contract (§7) | **`validated`** — 40/40 checks + M1 gate, [`results/m1/`](../results/m1/) |
| Recurrence, encoder, posterior, prior, norm, init (§4.1, §4.2, §4.4, §4.8, §4.9) | **`validated`** — M2+M3 gate, [`results/m2m3/`](../results/m2m3/) |
| Decoder, reward/continuation heads, `symexp_twohot`, world-model objective (§4.3, §4.5, §5.1–§5.5) | **`validated`** — M4 gate, [`results/m4/`](../results/m4/) |
| Latent imagination and the continuation weight (§4.10 starts, §5.4 weight) | **`validated`** — M6 gate 55/55, [`results/m6/`](../results/m6/) |
| Critic, λ-returns, slow critic, both critic losses (§4.7, §5.7, §6.1, §6.3) | **`validated`** — M7 gate 37/37, [`results/m7/`](../results/m7/) |
| Actor (§4.6, §5.6, §6.2) | **`specified`** — not `implemented` |
| Open-loop prediction from the prior (§4.4 prior path, §7.4 alignment) | **`validated`** — M5 gate 26/26, [`results/m5/`](../results/m5/) |
| LaProp optimizer (§5.8) | **`validated`** — 8/8 hand-computed update tests, `tests/test_optim.py`; six mutants confirmed to fail |
| §10-7 for `dyn`, `enc`, `dec`, `rew`, `con` | **measured** — 570,419 |
| Everything else in §10 | **`unmeasured`** |

The M0 per-requirement matrix is
[`results/m0/m0-audit-2026-09-17.md`](../results/m0/m0-audit-2026-09-17.md). Note that a `validated`
environment says the toolchain runs correctly; it says nothing about whether this project's
architecture is implemented against it.
