"""M7 gate: critic, lambda-returns and the two critic losses on a real Walker replay batch."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")

from dreamer import (
    Collector,
    Critic,
    DMCEnv,
    LaProp,
    RandomActionProvider,
    ReplayBuffer,
    UniformRandomPolicy,
    ValueHead,
    WorldModel,
    check_bootstrap_holes,
    imagine_trajectory,
    lambda_return,
    scatter_imagined_return,
    select_start_states,
)
from dreamer.rssm import observe_sequence
from dreamer.twohot import twohot_readout
from dreamer.world_model import batch_to_tensors

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

B, P, T, H = 16, 5, 64, 15
FIT_STEPS = 300
CONTDISC = 1.0 - 1.0 / 333
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
env.close()

torch.manual_seed(0)
model = WorldModel(action_dim=action_dim).to(device)
critic = Critic(in_features=model.rssm.feat_size).to(device)

print("""
heads perturbed: outscale 0.0 reads out exactly 0 for every latent, so imagined rewards
      would be identically 0 and every gradient-routing claim below would hold vacuously
      (dL/dfeat = dL/dlogits @ 0 = 0). Reward and value magnitudes here are those of
      random heads and mean nothing.""")
with torch.no_grad():
    model.reward.mlp.out.weight.normal_(0.0, 0.5)
    critic.value.mlp.out.weight.normal_(0.0, 0.1)

generator = torch.Generator(device=device)
generator.manual_seed(0)

print("\n## parameter count (measured)")
val = sum(p.numel() for p in critic.trainable_parameters())
mirror = sum(p.numel() for p in critic.parameters()) - val
print(f"  val    {val:>8}  derived {66_111:>8}  delta {val - 66_111:>4}")
print(f"  slowval{mirror:>8}  (excluded from the optimizer, untrained)")
check("val closes the 4.11 derivation", val == 66_111, str(val))
check("the slow mirror is excluded from trainable_parameters", mirror == 66_111, str(mirror))
check("world model + val = 636,530", sum(p.numel() for p in model.parameters()) + val == 636_530)

print("\n## posterior context pass and imagination starts")
observations, actions, rewards, is_terminal, loss_mask = batch_to_tensors(batch, device)
is_last = torch.from_numpy(batch.is_last).to(device)

post, _ = observe_sequence(model.encoder, model.rssm, observations, actions, generator=generator)
starts, index = select_start_states(post, loss_mask.to(torch.bool), is_terminal)
check("1024 starts from the 1104 candidate positions", starts.deter.shape[0] == B * T, str(starts.deter.shape[0]))
check("the index addresses the (B, P+T) grid, not (B, T)", bool((index % (P + T) >= P).all()))
check("the largest index needs the full grid", int(index.max()) >= B * T, str(int(index.max())))

print("\n## imagined critic — weight 1.0")
imagination = imagine_trajectory(model, starts, RandomActionProvider(action_dim, seed=0), H, generator)
imagined = critic.imagined_loss(imagination)

check("ret is (N, H)", tuple(imagined.ret.shape) == (B * T, H), str(tuple(imagined.ret.shape)))
check("the weight slice is weight[:, :-1]", bool((imagined.weight == imagination.weight[:, :-1]).all()))
check("the target is detached", not imagined.ret.requires_grad)
check("ret and loss are finite", bool(torch.isfinite(imagined.ret).all() and torch.isfinite(imagined.loss)))

fast = critic.value.predict(imagination.feat).detach()
slow = critic.slow.predict(imagination.feat).detach()
con = torch.cat([imagination.cont_start, imagination.cont], dim=1)
rew = torch.cat([torch.zeros_like(imagination.reward[:, :1]), imagination.reward], dim=1)
from_fast = lambda_return(torch.zeros_like(con), 1.0 - con, rew, fast, 1.0, critic.lam)
from_slow = lambda_return(torch.zeros_like(con), 1.0 - con, rew, slow, 1.0, critic.lam)
check("the imagined bootstrap is the FAST critic", bool((imagined.ret == from_fast).all()))
check("fast and slow are genuinely separated here", float((from_fast - from_slow).abs().max().detach()) > 1.0,
      f"{float((from_fast - from_slow).abs().max().detach()):.4g}")
print(f"  value_mae {imagined.metrics['value_value_mae']:.4g}   weight_mean {imagined.metrics['value_weight_mean']:.4f}")

print("\n## replay critic — weight 0.3")
grid, filled = scatter_imagined_return(imagined.ret, index, B, P + T)
check("burn-in columns are never filled", not bool(filled[:, :P].any()))
check_bootstrap_holes(filled[:, P:], is_terminal[:, P:])
check("no bootstrap hole at a non-terminal position", True)

boot = grid[:, P:]
feat_t = post[:, P + 1 :].feat
replay_out = critic.replay_loss(
    feat_t, rewards[:, P:], is_last[:, P:].to(torch.float32), is_terminal[:, P:].to(torch.float32), boot
)

check("replay ret is (B, T-1)", tuple(replay_out.ret.shape) == (B, T - 1), str(tuple(replay_out.ret.shape)))
check("the replay denominator is B*(T-1) = 1008", replay_out.metrics["repval_positions"] == float(B * (T - 1)),
      str(replay_out.metrics["repval_positions"]))
check("replay ret and loss are finite", bool(torch.isfinite(replay_out.ret).all() and torch.isfinite(replay_out.loss)))
check("the replay features are NOT detached", feat_t.requires_grad)
print(f"  value_mae {replay_out.metrics['repval_value_mae']:.4g}   target_mean {replay_out.metrics['repval_target_mean']:.4g}")

total = critic.total(imagined, replay_out)
check("total = 1.0*value + 0.3*repval", bool(torch.isclose(total, imagined.loss + 0.3 * replay_out.loss, rtol=0, atol=0)))

print("\n## gradient routing (spec 5.9, 6.3)")
model.zero_grad(set_to_none=True)
critic.zero_grad(set_to_none=True)
imagined.loss.backward(retain_graph=True)
check("value reaches the critic", float(critic.value.mlp.out.weight.grad.abs().sum()) > 0)
check("value does not reach the encoder", model.encoder.convs[0].weight.grad is None)
check("value does not reach the RSSM", model.rssm.dyngru.kernel.grad is None)
check("value does not reach the slow mirror", all(p.grad is None for p in critic.slow.parameters()))

model.zero_grad(set_to_none=True)
critic.zero_grad(set_to_none=True)
replay_out.loss.backward()
check("repval reaches the encoder", float(model.encoder.convs[0].weight.grad.abs().sum()) > 0)
check("repval reaches the RSSM", float(model.rssm.dyngru.kernel.grad.abs().sum()) > 0)
check("repval reaches the critic", float(critic.value.mlp.out.weight.grad.abs().sum()) > 0)
check("repval does not reach the decoder", model.decoder.imgout.weight.grad is None)
check("repval does not reach the slow mirror", all(p.grad is None for p in critic.slow.parameters()))

print("\n## the two silent readings of gamma (spec 5.4)")
L = 16
ones, zeros_ = torch.ones(1, L, device=device), torch.zeros(1, L, device=device)
def geometric(x):
    return sum(x**j for j in range(L - 1))
correct = float(lambda_return(zeros_, 1.0 - CONTDISC * ones, ones, zeros_, 1.0, 1.0)[0, 0])
doubled = float(lambda_return(zeros_, 1.0 - CONTDISC * ones, ones, zeros_, CONTDISC, 1.0)[0, 0])
dropped = float(lambda_return(zeros_, zeros_, ones, zeros_, 1.0, 1.0)[0, 0])
print(f"  correct {correct:.6f}   double-counted {doubled:.6f}   dropped {dropped:.6f}")
check("correct matches the closed form", abs(correct - geometric(CONTDISC)) < 1e-3, f"{geometric(CONTDISC):.6f}")
check("double-counting matches its own closed form", abs(doubled - geometric(CONTDISC**2)) < 1e-3)
check("the three readings are separated", correct - doubled > 0.2 and dropped - correct > 0.2)

print("\n## transform order (spec 5.5) — recorded, not asserted about return")
bins = critic.value.bins
probs = torch.zeros(1, 255, device=device)
near = int((bins - 100.0).abs().argmin())
probs[0, 127] = 0.5
probs[0, near] = 0.5
v2 = float(twohot_readout(probs, bins))
print(f"  v2 readout  sum(p_i b_i)          = {v2:.4f}   (this implementation)")
print(f"  v1 readout  symexp(sum p_i symlog_bins) would read ~9.05 on the same distribution")
check("the readout is the expectation, no outer symexp", v2 > 40.0, f"{v2:.4f}")
check("the zero-init readout is exactly 0", float(ValueHead().to(device).predict(torch.randn(8, 640, device=device)).abs().max().detach()) == 0.0)

print(f"\n## critic fitting on fixed targets ({FIT_STEPS} LaProp steps)")
# a FRESH critic: outscale 0.0 reads out exactly 0, which is how training actually starts.
# The perturbed head above is pathological for fitting -- with a +-4.85e8 support, any nonzero
# output kernel reads out ~1e6, so it would measure recovery from an unreachable state.
torch.manual_seed(1)
critic = Critic(in_features=model.rssm.feat_size).to(device)
snapshot = [p.detach().clone() for p in model.parameters()]
# fixed targets on the REAL reward scale: the discounted sum of actual Walker rewards with no
# critic bootstrap. The imagined return here is the self-bootstrapped output of a random head
# (~1e7) and fitting it would measure nothing about whether the critic can fit fixed targets.
fit_feat = feat_t[:, :-1].detach()
fit_target = lambda_return(
    is_last[:, P:].to(torch.float32),
    is_terminal[:, P:],
    rewards[:, P:],
    torch.zeros_like(rewards[:, P:]),
    CONTDISC,
    critic.lam,
).detach()
fit_weight = (~is_last[:, P:].bool()).to(torch.float32)[:, :-1]
print(f"  targets: discounted real-reward sums, range "
      f"[{float(fit_target.min()):.3f}, {float(fit_target.max()):.3f}]")

opt = LaProp(critic.trainable_parameters(), lr=1e-3, warmup=0)
rows = []
for step in range(FIT_STEPS):
    out = critic._fit(fit_feat, fit_target, fit_weight, "value")
    opt.zero_grad(set_to_none=True)
    out.loss.backward()
    opt.step()
    critic.update_slow()
    if step % 30 == 0 or step == FIT_STEPS - 1:
        rows.append({
            "step": step,
            "loss": round(out.metrics["value"], 6),
            "value_mean": round(out.metrics["value_value_mean"], 4),
            "target_mean": round(out.metrics["value_target_mean"], 4),
            "value_mae": round(out.metrics["value_value_mae"], 4),
            "slow_gap": round(out.metrics["value_slow_gap"], 4),
        })

print(f"  {'step':>5}{'loss':>12}{'value_mean':>14}{'target_mean':>14}{'value_mae':>14}{'slow_gap':>14}")
for r in rows:
    print(f"  {r['step']:>5}{r['loss']:>12.4f}{r['value_mean']:>14.4g}{r['target_mean']:>14.4g}"
          f"{r['value_mae']:>14.4g}{r['slow_gap']:>14.4g}")

first, last = rows[0], rows[-1]
check("value error falls materially", last["value_mae"] < 0.5 * first["value_mae"],
      f"{first['value_mae']:.4g} -> {last['value_mae']:.4g}")
check("the world model is bitwise unchanged", all(torch.equal(a, b) for a, b in zip(snapshot, model.parameters())))
check("the slow critic trails rather than tracks", last["slow_gap"] > 0.0)

if device.type == "cuda":
    print(f"\n  peak VRAM, posterior + 1024 imagined rollouts at H={H} + both critic losses: "
          f"{torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")

out_dir = Path("results/m7")
out_dir.mkdir(parents=True, exist_ok=True)
tag = sys.argv[1] if len(sys.argv) > 1 else "2026-09-18"
path = out_dir / f"value-vs-target-{tag}.csv"
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
