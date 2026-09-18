import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")  # must precede the mujoco import

import json
import platform

import numpy as np

TASKS = [("walker", "walk"), ("cartpole", "swingup")]
SIZE = (64, 64)
CAMERA = 0
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


print("# M0 deferred items 2, 3, 4 — simulator, headless render, env contract")
print(f"# python {platform.python_version()}  MUJOCO_GL={os.environ['MUJOCO_GL']}")
print()

try:
    import mujoco
    from dm_control import suite
except Exception as e:
    print(f"  [FAIL] import — {type(e).__name__}: {e}")
    sys.exit(1)

print("## Versions")
from importlib import metadata

print(f"  mujoco       {mujoco.__version__}")
for pkg in ("dm-control", "numpy", "scipy", "labmaze"):
    try:
        print(f"  {pkg:<12} {metadata.version(pkg)}")
    except metadata.PackageNotFoundError:
        print(f"  {pkg:<12} not installed")
print()

summary = {}
for domain, task in TASKS:
    label = f"{domain}_{task}"
    print(f"## {label}")
    try:
        env = suite.load(domain, task)
    except Exception as e:
        check(f"{label}: suite.load", False, f"{type(e).__name__}: {e}")
        continue
    check(f"{label}: suite.load", True)

    spec = env.action_spec()
    lo, hi = np.asarray(spec.minimum), np.asarray(spec.maximum)
    dim = int(np.prod(spec.shape))
    print(f"  action_spec  shape={spec.shape} dim={dim} dtype={spec.dtype}")
    print(f"               min={np.unique(lo)} max={np.unique(hi)}")
    check(f"{label}: action bounds finite", bool(np.isfinite(lo).all() and np.isfinite(hi).all()))
    check(f"{label}: action bounds are [-1, 1]",
          bool(np.allclose(lo, -1) and np.allclose(hi, 1)))

    ctrl_dt = env.control_timestep()
    phys_dt = env.physics.model.opt.timestep
    substeps = ctrl_dt / phys_dt
    print(f"  control_timestep {ctrl_dt:.6g} s   physics timestep {phys_dt:.6g} s")
    check(f"{label}: physics substeps per control step is an integer ({substeps:.6g})",
          abs(substeps - round(substeps)) < 1e-9, f"{round(substeps)} substeps")

    ts = env.reset()
    check(f"{label}: reset gives first()==True and reward None",
          ts.first() and ts.reward is None)

    frame = env.physics.render(*SIZE, camera_id=CAMERA)
    print(f"  render       shape={frame.shape} dtype={frame.dtype} "
          f"min={frame.min()} max={frame.max()} unique={len(np.unique(frame))}")
    check(f"{label}: EGL offscreen render succeeds at {SIZE[0]}x{SIZE[1]}",
          frame.shape == (SIZE[0], SIZE[1], 3))
    check(f"{label}: frame dtype is uint8", frame.dtype == np.uint8)
    check(f"{label}: frame is non-constant", len(np.unique(frame)) > 1)

    rng = np.random.default_rng(0)
    n, rews, discounts = 0, [], []
    last_ts = None
    while True:
        a = rng.uniform(lo, hi, size=spec.shape).astype(spec.dtype)
        ts = env.step(a)
        n += 1
        rews.append(float(ts.reward))
        discounts.append(None if ts.discount is None else float(ts.discount))
        if ts.last():
            last_ts = ts
            break
        if n > 5000:
            break

    print(f"  episode      {n} control steps, reward range "
          f"[{min(rews):.4g}, {max(rews):.4g}], sum {sum(rews):.4g}")
    check(f"{label}: episode length is 1000 control steps", n == 1000, f"observed {n}")
    check(f"{label}: per-step reward within [0, 1]", min(rews) >= 0.0 and max(rews) <= 1.0)

    # the central M1 hazard: does the episode end by TIME LIMIT, not termination?
    d = last_ts.discount
    print(f"  final step   last()={last_ts.last()}  discount={d}")
    check(f"{label}: ends with last()==True", bool(last_ts.last()))
    check(f"{label}: final discount is 1.0, NOT 0 — a TIME LIMIT, not a terminal",
          d is not None and float(d) == 1.0, f"discount={d}")
    zeros = [i for i, x in enumerate(discounts) if x == 0.0]
    check(f"{label}: no step in the episode had discount 0", not zeros,
          f"{len(zeros)} steps with discount 0" if zeros else "")

    env.reset()
    t0 = time.perf_counter()
    for _ in range(200):
        env.step(rng.uniform(lo, hi, size=spec.shape).astype(spec.dtype))
        env.physics.render(*SIZE, camera_id=CAMERA)
    fps = 200 / (time.perf_counter() - t0)
    print(f"  throughput   {fps:.1f} control steps/s including a 64x64 render each step")

    summary[label] = {
        "action_dim": dim, "bounds": [float(lo.min()), float(hi.max())],
        "control_timestep": ctrl_dt, "physics_timestep": float(phys_dt),
        "substeps": round(substeps), "episode_steps": n,
        "final_discount": None if d is None else float(d),
        "reward_min": min(rews), "reward_max": max(rews),
        "render_shape": list(frame.shape), "render_dtype": str(frame.dtype),
        "steps_per_s_with_render": round(fps, 1),
    }
    print()

npass = sum(1 for _, ok, _ in results if ok)
print(f"## Verdict: {npass}/{len(results)} checks passed")
failed = [n for n, ok, _ in results if not ok]
if failed:
    print("   FAILED: " + "; ".join(failed))
print()
print(json.dumps({
    "python": platform.python_version(), "mujoco": mujoco.__version__,
    "mujoco_gl": os.environ["MUJOCO_GL"], "camera_id": CAMERA,
    "tasks": summary, "checks_passed": npass, "checks_total": len(results),
    "failed": failed,
}, indent=2))
sys.exit(0 if not failed else 1)
