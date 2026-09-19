"""M6 gate: latent imagination from a real Walker replay batch, and per-horizon cost."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "src")

from dreamer import (
    Collector,
    DMCEnv,
    RandomActionProvider,
    ReplayBuffer,
    SequenceActionProvider,
    UniformRandomPolicy,
    WorldModel,
    imagine_trajectory,
    select_start_states,
    trajectory_weight,
)
from dreamer.openloop import encoder_tripwire
from dreamer.rssm import observe_sequence
from dreamer.world_model import batch_to_tensors

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

B, P, T = 16, 5, 64
HORIZONS = (5, 15, 30)
REPEATS = 5
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}  torch: {torch.__version__}")

env = DMCEnv("walker", "walk", seed=0)
replay = ReplayBuffer(capacity=500_000, seed=0)
collector = Collector(env, replay, UniformRandomPolicy(env.action_low, env.action_high, 0))
collector.collect(1000)
batch = replay.sample_sequences(batch_size=B, train_length=T, burn_in=P)
action_dim = env.action_shape[0]

torch.manual_seed(0)
model = WorldModel(action_dim=action_dim).to(device)
model.eval()
# outscale 0.0 makes the reward readout exactly 0 for every latent, hiding all alignment
with torch.no_grad():
    model.reward.mlp.out.weight.normal_(0.0, 0.5)

print("reward head perturbed: outscale 0.0 reads out exactly 0 for every latent, so\n      alignment and action sensitivity would pass vacuously. Reward magnitudes below\n      are those of a random head and mean nothing.")

generator = torch.Generator(device=device)
generator.manual_seed(0)

observations, actions, rewards, is_terminal, loss_mask = batch_to_tensors(batch, device)

simulator_steps: list = []
_env_step = env.step
# a tripwire on the simulator itself: imagination must never reach it
env.step = lambda action: (simulator_steps.append(action), _env_step(action))[1]

print("\n## posterior context pass")
with torch.no_grad():
    post, _ = observe_sequence(model.encoder, model.rssm, observations, actions, generator=generator)
check("posterior covers P+T+1 positions", tuple(post.deter.shape) == (B, P + T + 1, 512), str(tuple(post.deter.shape)))

starts, index = select_start_states(post, loss_mask.to(torch.bool), is_terminal)
print(f"  start states: {starts.deter.shape[0]} of {B * (P + T)} candidate positions")
check("every loss-bearing non-terminal position is a start", starts.deter.shape[0] == B * T, str(starts.deter.shape[0]))
check("starts are a flat batch", starts.deter.dim() == 2 and starts.stoch.dim() == 3)
check("burn-in positions are excluded", bool((index % (P + T) >= P).all()))
check("terminal positions are excluded", bool(~is_terminal.reshape(-1)[index].any()))
check("start states are finite", bool(torch.isfinite(starts.feat).all()))

print("\n## per-horizon rollouts")
rows = []
for horizon in HORIZONS:
    provider = RandomActionProvider(action_dim, seed=0)
    generator.manual_seed(0)

    simulator_steps.clear()
    with encoder_tripwire(model) as tripwire:
        out = imagine_trajectory(model, starts, provider, horizon, generator)

    check(f"H={horizon}: no encoder call during imagination", tripwire.calls == [], str(tripwire.calls))
    check(f"H={horizon}: no simulator step during imagination", simulator_steps == [])
    check(f"H={horizon}: H+1 states", tuple(out.states.deter.shape) == (B * T, horizon + 1, 512), str(tuple(out.states.deter.shape)))
    check(f"H={horizon}: H actions", tuple(out.actions.shape) == (B * T, horizon, action_dim), str(tuple(out.actions.shape)))
    check(f"H={horizon}: H rewards and H continuations", tuple(out.reward.shape) == (B * T, horizon) and tuple(out.cont.shape) == (B * T, horizon))
    check(f"H={horizon}: H+1 weights", tuple(out.weight.shape) == (B * T, horizon + 1), str(tuple(out.weight.shape)))
    check(f"H={horizon}: state 0 is the replay start", bool((out.states.deter[:, 0] == starts.deter).all()))
    check(f"H={horizon}: outputs finite", all(bool(torch.isfinite(t).all()) for t in (out.feat, out.reward, out.cont, out.weight, out.actions)))
    check(f"H={horizon}: weight[0] is the start continuation", bool((out.weight[:, :1] == out.cont_start).all()))
    check(
        f"H={horizon}: weight is the cumulative product of the continuations",
        bool(torch.allclose(out.weight, trajectory_weight(torch.cat([out.cont_start, out.cont], 1), 1.0), rtol=0, atol=1e-6)),
    )
    check(f"H={horizon}: continuations are probabilities", bool(((out.cont >= 0) & (out.cont <= 1)).all()))
    check(f"H={horizon}: imagined trajectory carries no world-model gradient", not out.states.deter.requires_grad)
    check(f"H={horizon}: no image is decoded", out.image is None)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    for repeat in range(REPEATS):
        generator.manual_seed(repeat)
        imagine_trajectory(model, starts, RandomActionProvider(action_dim, seed=repeat), horizon, generator)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / REPEATS
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")

    rows.append(
        {
            "horizon": horizon,
            "rollouts": B * T,
            "states": B * T * (horizon + 1),
            "seconds_per_batch": round(elapsed, 4),
            "us_per_state": round(elapsed * 1e6 / (B * T * (horizon + 1)), 4),
            "peak_mib": round(peak, 1),
            "outputs_finite": bool(torch.isfinite(out.feat).all() and torch.isfinite(out.reward).all()),
            "cont_in_unit_interval": bool(((out.cont >= 0) & (out.cont <= 1)).all()),
        }
    )
    print(
        f"  H={horizon:>2}  {B * T} rollouts  {B * T * (horizon + 1):>6} states"
        f"  {elapsed * 1e3:8.1f} ms/batch  {rows[-1]['us_per_state']:6.3f} us/state"
        f"  peak {peak:7.1f} MiB"
    )

print("\n## reuse of the M5 prior-only path")
scripted = torch.rand(B * T, 15, action_dim, device=device) * 2 - 1
generator.manual_seed(11)
viaprovider = imagine_trajectory(model, starts, SequenceActionProvider(scripted), 15, generator)
generator.manual_seed(11)
with torch.no_grad():
    reference = model.rssm.imagine(starts.detach(), scripted, generator)
check("a scripted provider reproduces RSSM.imagine exactly", bool((viaprovider.states.deter[:, 1:] == reference.deter).all() and (viaprovider.states.stoch[:, 1:] == reference.stoch).all()))
check("the scripted actions are returned verbatim", bool((viaprovider.actions == scripted).all()))

print("\n## action sensitivity")
generator.manual_seed(0)
low = imagine_trajectory(model, starts, SequenceActionProvider(torch.full((B * T, 5, action_dim), -1.0, device=device)), 5, generator)
generator.manual_seed(0)
high = imagine_trajectory(model, starts, SequenceActionProvider(torch.full((B * T, 5, action_dim), 1.0, device=device)), 5, generator)
state_gap = float((low.states.deter - high.states.deter).abs().max())
reward_gap = float((low.reward - high.reward).abs().max().detach())
print(f"  max|deter| gap {state_gap:.4f}   max|reward| gap {reward_gap:.4f}")
check("opposite actions change the imagined states", state_gap > 1e-4, f"{state_gap:.4g}")
check("opposite actions change the predicted rewards", reward_gap > 1e-6, f"{reward_gap:.4g}")

print("\n## decoding is available for inspection only")
generator.manual_seed(0)
inspected = imagine_trajectory(model, starts[:4], RandomActionProvider(action_dim, seed=0), 5, generator, decode=True)
check("decode=True returns H+1 frames", tuple(inspected.image.shape) == (4, 6, 64, 64, 3), str(tuple(inspected.image.shape)))
check("decoded frames are finite and in [0, 1]", bool(torch.isfinite(inspected.image).all() and (inspected.image >= 0).all() and (inspected.image <= 1).all()))

print("\n## gradient routing")
generator.manual_seed(0)
out = imagine_trajectory(model, starts[:64], RandomActionProvider(action_dim, seed=0), 5, generator)
model.zero_grad(set_to_none=True)
out.reward.sum().backward()
check("the reward head receives gradient", model.reward.mlp.out.weight.grad is not None and float(model.reward.mlp.out.weight.grad.abs().sum()) > 0)
check("the RSSM receives none through imagination", model.rssm.dyngru.kernel.grad is None)
check("the encoder receives none through imagination", model.encoder.convs[0].weight.grad is None)

env.close()

if rows:
    out_dir = Path("results/m6")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"horizon-cost-{sys.argv[1] if len(sys.argv) > 1 else '2026-09-18'}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
