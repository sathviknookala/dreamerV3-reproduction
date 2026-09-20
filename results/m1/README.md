# M1 — environment and compute verification

Discharges deferred items **1–5** from [spec.md §10](../../docs/spec.md). **40/40 checks pass**, plus the random-policy return floor measured 2026-09-20. **§10-6, throughput, is still outstanding.**

| Suite | Covers | Result |
|---|---|---|
| [`check_torch_compute.py`](check_torch_compute.py) → [`torch-compute-2026-09-17.txt`](torch-compute-2026-09-17.txt) | §10-1 — PyTorch on Blackwell | **14/14** |
| [`check_env_render.py`](check_env_render.py) → [`env-render-2026-09-17.txt`](env-render-2026-09-17.txt) | §10-2/3/4 — simulator, EGL render, env contract | **26/26** |
| [`capture_manifest.py`](capture_manifest.py) → [`env-manifest-2026-09-17.json`](env-manifest-2026-09-17.json) | exact versions and hardware | — |
| [`m1_random_floor.py`](../../scripts/m1_random_floor.py) → [`random-floor-walker-2026-09-20.json`](random-floor-walker-2026-09-20.json), [`random-floor-cartpole-2026-09-20.json`](random-floor-cartpole-2026-09-20.json) | §10-5 — random-policy return floor, 20 complete episodes per task | — |

Re-run both after any dependency change:

```bash
.venv/bin/python results/m1/check_torch_compute.py && .venv/bin/python results/m1/check_env_render.py
```

Each exits non-zero if any check fails.

## The random-policy return floor — measured 2026-09-20

Uniform random over the action bounds, action repeat 1, 20 complete 1000-step episodes per task,
64×64 RGB rendered every control step. Discharges the floor deferred from M1 by decision.

| Task | mean | sd | min | max | seeds |
|---|---|---|---|---|---|
| Walker Walk | **32.21** | 4.37 | 27.44 | 47.34 | env 900, policy 900 |
| Cartpole Swingup | **24.17** | 15.92 | 4.95 | 63.90 | env 901, policy 901 |

```bash
PYTHONPATH=src .venv/bin/python scripts/m1_random_floor.py --task walker --env-seed 900 --policy-seed 900 \
  --episodes 20 --output results/m1/random-floor-walker-<date>.json --video results/m1/random-walker-<date>.mp4
```

The script **raises** rather than reporting a truncation if an episode does not end within
`--max-steps`; all 40 episodes ran the full 1000 steps. `--video` records episode 0 only, which
discharges the random-policy video also deferred from M1.

**Cartpole's floor is far noisier than Walker's** — sd 15.92 on a mean of 24.17, with a single
episode reaching 63.90. A Cartpole agent scoring in the 60s is therefore not clearly above chance;
a Walker agent scoring in the 60s is, since the best random Walker episode is 47.34. Compare agent
returns against `return_max`, not only against the mean.

This is the floor, not a baseline. **The comparison point for the project is the pinned author
implementation** under matched wrappers, observation access and data budget
([experiment.md](../../docs/experiment.md)); a random policy is only the bound any learning agent
must clear.

## What these establish

- `sm_120` is in torch 2.13.0+cu129's arch list and matches the device's compute 12.0.
- The ops this project actually needs — the encoder conv, the block-diagonal einsum, float32
  RMSNorm — agree GPU-vs-CPU with TF32 off.
- Both tasks render 64×64×3 uint8 non-constant frames headless under EGL.
- Both tasks end at step 1000 with `discount == 1.0`, and **no step in either episode has
  `discount == 0`** — the time-limit-not-terminal contract in [spec.md §7.3](../../docs/spec.md),
  confirmed against the simulator rather than read from source.

## What they do NOT establish

- **No model of this project is implemented or measured.** These suites exercise the toolchain.
- The memory figures in the torch suite are **floors**, not measurements: they exclude the rest of
  the model, optimizer state, and activations they do not themselves create. Real peak VRAM at H=30
  is deferred to M9 (§10-8).
- Throughput (~920–1000 control steps/s with render) is a single-process figure from a 200-step
  probe, not the steady-state collection rate under training load (§10-6). **`m1_throughput.py` has
  still not been run**; it is the last M1 deferral outstanding.
- **The floor is a floor.** Clearing it is necessary, not sufficient, and says nothing about whether
  a policy is good — only that it is not acting at chance.
