# M1 — environment and compute verification

Discharges deferred items **1–4** from [spec.md §10](../../docs/spec.md). **40/40 checks pass.**

| Suite | Covers | Result |
|---|---|---|
| [`check_torch_compute.py`](check_torch_compute.py) → [`torch-compute-2026-09-17.txt`](torch-compute-2026-09-17.txt) | §10-1 — PyTorch on Blackwell | **14/14** |
| [`check_env_render.py`](check_env_render.py) → [`env-render-2026-09-17.txt`](env-render-2026-09-17.txt) | §10-2/3/4 — simulator, EGL render, env contract | **26/26** |
| [`capture_manifest.py`](capture_manifest.py) → [`env-manifest-2026-09-17.json`](env-manifest-2026-09-17.json) | exact versions and hardware | — |

Re-run both after any dependency change:

```bash
.venv/bin/python results/m1/check_torch_compute.py && .venv/bin/python results/m1/check_env_render.py
```

Each exits non-zero if any check fails.

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
  probe, not the steady-state collection rate under training load (§10-6).
