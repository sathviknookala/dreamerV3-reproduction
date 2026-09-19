"""M8 gate: the continuous actor, REINFORCE, return normalization, and behaviour integration."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from dreamer import (
    ACTENT,
    Actor,
    ActorActionProvider,
    BoundedNormal,
    Collector,
    Critic,
    DMCEnv,
    LaProp,
    ReplayBuffer,
    ReturnNormalizer,
    UniformRandomPolicy,
    WorldModel,
    behavior_losses,
    imagine_trajectory,
    imagined_actor_loss,
    scatter_imagined_return,
    select_start_states,
)
from dreamer.imagine import Imagination
from dreamer.rssm import State, observe_sequence
from dreamer.world_model import batch_to_tensors

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

B, P, T, H = 16, 5, 64, 15
FIT_STEPS = 100
SEED = 0
DERIVED_POL = 50_316
AGENT_TOTAL = 686_846
LOG2PI = math.log(2.0 * math.pi)
failures: list[str] = []
checks = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}  torch: {torch.__version__}")
print(f"seeds: torch {SEED}, env {SEED}, replay {SEED}, imagination {SEED}, provider {SEED}")
print(f"batch: B={B} burn-in P={P} train T={T}   horizon H={H}   actent={ACTENT}")

env = DMCEnv("walker", "walk", seed=SEED)
replay = ReplayBuffer(capacity=500_000, seed=SEED)
collector = Collector(env, replay, UniformRandomPolicy(env.action_low, env.action_high, SEED))
collector.collect(1000)
batch = replay.sample_sequences(batch_size=B, train_length=T, burn_in=P)
action_dim = env.action_shape[0]

print("\n## executed actions are bounded outside the policy (spec 4.6)")
env.reset()
raw = np.full(env.action_shape, 5.0, dtype=np.float32)
step = env.step(raw)
check("the ClipAction boundary clips an out-of-range sample", bool((step.action <= 1.0).all()))
check("the raw sample itself was out of range", float(raw.max()) > 1.0, f"{float(raw.max())}")
env.close()

torch.manual_seed(SEED)
model = WorldModel(action_dim=action_dim).to(device)
critic = Critic(in_features=model.rssm.feat_size).to(device)
actor = Actor(in_features=model.rssm.feat_size, action_dim=action_dim).to(device)
normalizer = ReturnNormalizer().to(device)

print("""
The ACTOR needs no perturbation: outscale 0.01 leaves its gradients nonzero at init, so the
      reward/critic 'perturb first' workaround does not apply to any policy claim below.""")

print("\n## parameter count (measured)")
pol = sum(p.numel() for p in actor.parameters())
val = sum(p.numel() for p in critic.trainable_parameters())
world = sum(p.numel() for p in model.parameters())
print(f"  pol    {pol:>8}  derived {DERIVED_POL:>8}  delta {pol - DERIVED_POL:>4}")
print(f"  total  {world + val + pol:>8}  derived {AGENT_TOTAL:>8}   (slowval mirror excluded)")
check("pol closes the 4.11 derivation", pol == DERIVED_POL, str(pol))
check("the trainable agent total closes at 686,846", world + val + pol == AGENT_TOTAL)
check("cartpole's actor differs only in the two heads",
      pol - sum(p.numel() for p in Actor(action_dim=1).parameters()) == 2 * 5 * 65)

print("\n## bounded_normal closed forms (spec 4.6)")
probe = Actor(in_features=model.rssm.feat_size, action_dim=action_dim).to(device)
with torch.no_grad():
    probe.stddev.weight.zero_()
    probe.stddev.bias.zero_()
flat = torch.randn(32, model.rssm.feat_size, device=device)
with torch.no_grad():
    zero_std = float(probe(flat).stddev.mean())
    policy = actor(flat)
check("stddev at zero pre-activation is 0.9*sigmoid(2)+0.1", abs(zero_std - 0.8926) < 1e-3, f"{zero_std:.6f}")
check("the mean is tanh-bounded", float(policy.mean.abs().max()) <= 1.0)
check("stddev stays inside [minstd, maxstd]",
      actor.minstd <= float(policy.stddev.min()) and float(policy.stddev.max()) <= actor.maxstd)

hand_std = torch.tensor([[0.5, 0.9, 0.1, 1.0, 0.3, 0.7]], device=device)
hand = BoundedNormal(torch.zeros_like(hand_std), hand_std)
closed = float((0.5 * torch.log(2 * math.pi * hand_std.pow(2)) + 0.5).sum(-1))
check("entropy is the closed form summed over the 6 action dims",
      abs(float(hand.entropy()) - closed) < 1e-5, f"{float(hand.entropy()):.6f}")
outside = torch.full((1, action_dim), 3.0, device=device)
z = (outside - hand.mean) / hand_std
plain = float((-0.5 * z.pow(2) - hand_std.log() - 0.5 * LOG2PI).sum(-1))
check("log_prob of an out-of-range action is the plain Gaussian, no Jacobian term",
      abs(float(hand.log_prob(outside)) - plain) < 1e-4, f"{float(hand.log_prob(outside)):.4f}")

big = BoundedNormal(torch.full((4096, 1), 0.9, device=device), torch.full((4096, 1), 1.0, device=device))
sample = big.sample(torch.Generator(device=device).manual_seed(SEED))
check("raw samples are unsquashed and leave [-1, 1]", float(sample.max()) > 1.0, f"{float(sample.max()):.4f}")

start = model.rssm.initial(2, device=device)
bounded = model.rssm.imagine_step(start, torch.full((2, action_dim), 3.0, device=device), None,
                                  torch.Generator(device=device).manual_seed(SEED))
unit = model.rssm.imagine_step(start, torch.full((2, action_dim), 1.0, device=device), None,
                               torch.Generator(device=device).manual_seed(SEED))
check("the RSSM bounds the action it consumes", bool((bounded.deter == unit.deter).all()))

print("\n## return normalization (spec 5.6 retnorm)")
ramp = torch.linspace(0.0, 100.0, 101, device=device).reshape(1, -1)
probe_norm = ReturnNormalizer().to(device)
offset, scale = probe_norm(ramp)
print(f"  pinned (debias False): lo {float(probe_norm.lo):.4f}  hi {float(probe_norm.hi):.4f}  S {float(scale):.4f}")
check("the first update is rate * the percentiles", abs(float(probe_norm.lo) - 0.05) < 1e-5
      and abs(float(probe_norm.hi) - 0.95) < 1e-5)
check("S sits at the floor of 1.0 on the first batch", abs(float(scale) - 1.0) < 1e-6, f"{float(scale):.6f}")
_, second = probe_norm(ramp)
check("the second update is the hand-computed EMA", abs(float(second) - (1.8905 - 0.0995)) < 1e-3,
      f"{float(second):.6f}")

debiased = ReturnNormalizer(debias=True).to(device)
_, dscale = debiased(ramp)
print(f"  debias True (NOT the pin): corr {float(debiased.corr):.4f}  S {float(dscale):.4f}")
check("bias correction would read the true percentiles at once", abs(float(dscale) - 90.0) < 1e-2)
check("configs.yaml#L111 pins debias False, so the uncorrected branch is the live one",
      ReturnNormalizer().debias is False)
frozen = ReturnNormalizer().to(device)
frozen(ramp)
before = (float(frozen.lo), float(frozen.hi))
frozen(ramp * 100.0, update=False)
check("a read-only call leaves the state alone", (float(frozen.lo), float(frozen.hi)) == before)
restored = ReturnNormalizer().to(device)
restored.load_state_dict(frozen.state_dict())
check("the normalizer state round-trips", float(restored.hi) == float(frozen.hi))
check("the normalizer holds no trainable parameters", list(frozen.parameters()) == [])

print("\n## analytic update fixtures (spec 6.2)")
torch.manual_seed(1)
fix_actor = Actor(in_features=8 + 4, action_dim=1)
fix_critic = Critic(in_features=8 + 4)
fix_feat = torch.randn(1, 1, 12, generator=torch.Generator().manual_seed(3)).expand(1, 3, 12).contiguous()
fix_actions = torch.tensor([[[1.0], [-1.0]]])


def fixture(advantage):
    con = torch.ones(1, 3)
    return Imagination(
        states=State(fix_feat[..., :8], fix_feat[..., 8:].reshape(1, 3, 2, 2),
                     fix_feat[..., 8:].reshape(1, 3, 2, 2)),
        actions=fix_actions,
        reward=torch.zeros(1, 2),
        cont=con[:, 1:],
        cont_start=con[:, :1],
        weight=torch.cumprod(con, dim=1),
    ), torch.tensor([advantage])


imag, ret = fixture([1.0, -1.0])
with torch.no_grad():
    before_logp = fix_actor(imag.feat[:, :-1]).log_prob(imag.actions).clone()
out = imagined_actor_loss(fix_actor, fix_critic, imag, ret, ReturnNormalizer(), actent=0.0, update=False)
check("the zero-init critic makes the advantage exactly the return",
      bool(torch.allclose(out.advantage, ret, atol=1e-6)))
opt = torch.optim.SGD(fix_actor.parameters(), lr=0.5)
opt.zero_grad(set_to_none=True)
out.loss.backward()
opt.step()
with torch.no_grad():
    after_logp = fix_actor(imag.feat[:, :-1]).log_prob(imag.actions)
print(f"  logpi  a=+1: {float(before_logp[0, 0]):+.6f} -> {float(after_logp[0, 0]):+.6f}"
      f"   a=-1: {float(before_logp[0, 1]):+.6f} -> {float(after_logp[0, 1]):+.6f}")
check("eta=0: the higher-advantage action gains log-probability",
      float(after_logp[0, 0]) > float(before_logp[0, 0]))
check("eta=0: the lower-advantage action loses log-probability",
      float(after_logp[0, 1]) < float(before_logp[0, 1]))

torch.manual_seed(1)
fix_actor = Actor(in_features=12, action_dim=1)
imag, ret = fixture([0.0, 0.0])
with torch.no_grad():
    before_ent = float(fix_actor(imag.feat[:, :-1]).entropy().mean())
out = imagined_actor_loss(fix_actor, fix_critic, imag, ret, ReturnNormalizer(), actent=ACTENT, update=False)
opt = torch.optim.SGD(fix_actor.parameters(), lr=1.0)
opt.zero_grad(set_to_none=True)
out.loss.backward()
check("A=0: the score term contributes no gradient to the mean head",
      float(fix_actor.mean.weight.grad.abs().sum()) == 0.0)
check("A=0: the entropy term does reach the stddev head",
      float(fix_actor.stddev.weight.grad.abs().sum()) > 0.0)
opt.step()
with torch.no_grad():
    after_ent = float(fix_actor(imag.feat[:, :-1]).entropy().mean())
print(f"  entropy {before_ent:.8f} -> {after_ent:.8f}")
check("A=0, eta>0: one update INCREASES entropy (fails if the sign is flipped)", after_ent > before_ent)

print("""
## integration on one real replay batch
      The reward and value output kernels ARE perturbed here. At outscale 0.0 the imagined
      reward is identically 0, so the return, the baseline and therefore the ADVANTAGE are all
      exactly 0 and every claim about them holds vacuously. Each readout is then a random draw
      over +-4.85e8 bins: the placement and routing are tested, the magnitudes mean nothing.""")
with torch.no_grad():
    model.reward.mlp.out.weight.normal_(0.0, 0.5)
    critic.value.mlp.out.weight.normal_(0.0, 0.1)

generator = torch.Generator(device=device)
generator.manual_seed(SEED)
observations, actions, rewards, is_terminal, loss_mask = batch_to_tensors(batch, device)
is_last = torch.from_numpy(batch.is_last).to(device)

post, _ = observe_sequence(model.encoder, model.rssm, observations, actions, generator=generator)
starts, index = select_start_states(post, loss_mask.to(torch.bool), is_terminal)
provider = ActorActionProvider(actor, seed=SEED)
imagination = imagine_trajectory(model, starts, provider, H, generator)

check("the rollout is actor-driven and (N, H, A)",
      tuple(imagination.actions.shape) == (B * T, H, action_dim), str(tuple(imagination.actions.shape)))
check("imagined features carry no world-model graph", not imagination.feat.requires_grad)
check("the sampled actions are finite", bool(torch.isfinite(imagination.actions).all()))

behavior = behavior_losses(actor, critic, imagination, normalizer)
pol_out = behavior.actor
print(f"  policy {float(pol_out.loss.detach()):+.6f}   value {float(behavior.imagined.loss.detach()):.4f}"
      f"   entropy {pol_out.metrics['policy_entropy']:.4f}"
      f"   S {pol_out.metrics['policy_retnorm_scale']:.4f}"
      f"   |A| {pol_out.metrics['policy_adv_mag']:.4g}")
check("logpi and entropy are (N, H) with no second slice",
      tuple(pol_out.log_prob.shape) == (B * T, H) and tuple(pol_out.entropy.shape) == (B * T, H),
      str(tuple(pol_out.log_prob.shape)))
check("the weight slice is weight[:, :-1]",
      bool((pol_out.weight == imagination.weight[:, :-1]).all()))
check("the advantage and the weight are detached",
      not pol_out.advantage.requires_grad and not pol_out.weight.requires_grad)
check("the policy loss and its entropy are finite",
      bool(torch.isfinite(pol_out.loss) and torch.isfinite(pol_out.entropy).all()))
check("the normalizer consumed its post-update scale",
      float(pol_out.scale) == float(normalizer.stats()[1]))

fast = critic.value.predict(imagination.feat).detach()
check("the advantage is not identically zero, so the claims below are not vacuous",
      float(pol_out.advantage.abs().max()) > 0.0, f"{pol_out.metrics['policy_adv_mag']:.4g}")
slow_base = critic.slow.predict(imagination.feat).detach()
check("the fast and slow critics are genuinely separated here",
      float((fast - slow_base).abs().max()) > 1.0, f"{float((fast - slow_base).abs().max()):.4g}")
check("the baseline is the FAST critic at feat[:, :-1]",
      bool(torch.allclose(pol_out.advantage, (behavior.imagined.ret - fast[:, :-1]) / pol_out.scale, atol=1e-5)))

grid, filled = scatter_imagined_return(behavior.imagined.ret, index, B, P + T)
replay_out = critic.replay_loss(
    post[:, P + 1 :].feat, rewards[:, P:], is_last[:, P:].to(torch.float32),
    is_terminal[:, P:].to(torch.float32), grid[:, P:], filled[:, P:],
)
combined = behavior_losses(actor, critic, imagination, normalizer, replay=replay_out, update=False)
check("total = 1.0*policy + 1.0*value + 0.3*repval",
      bool(torch.allclose(combined.loss,
                          combined.actor.loss + combined.imagined.loss + 0.3 * replay_out.loss,
                          rtol=1e-6, atol=1e-7)))

with torch.no_grad():
    first = ActorActionProvider(actor, seed=7)(starts[:8])
    second = ActorActionProvider(actor, seed=7)(starts[:8])
    other = ActorActionProvider(actor, seed=8)(starts[:8])
check("seeded actor sampling reproduces", bool((first == second).all()))
check("a different seed draws differently", float((first - other).abs().max()) > 0.0)

print("\n## gradient routing (spec 5.9)")
model.zero_grad(set_to_none=True)
critic.zero_grad(set_to_none=True)
actor.zero_grad(set_to_none=True)
routed = behavior_losses(actor, critic, imagination, normalizer, update=False)
routed.actor.loss.backward(retain_graph=True)
check("policy reaches the actor trunk", float(actor.linears[0].weight.grad.abs().sum()) > 0)
check("policy reaches the mean head", float(actor.mean.weight.grad.abs().sum()) > 0)
check("policy reaches the stddev head", float(actor.stddev.weight.grad.abs().sum()) > 0)
check("policy does not reach the encoder", model.encoder.convs[0].weight.grad is None)
check("policy does not reach the RSSM", model.rssm.dyngru.kernel.grad is None)
check("policy does not reach the decoder", model.decoder.imgout.weight.grad is None)
check("policy does not reach the reward head", model.reward.mlp.out.weight.grad is None)
check("policy does not reach the continuation head", model.cont.mlp.out.weight.grad is None)
check("policy does not reach the fast critic", critic.value.mlp.out.weight.grad is None)
check("policy does not reach the slow critic", all(p.grad is None for p in critic.slow.parameters()))

actor.zero_grad(set_to_none=True)
critic.zero_grad(set_to_none=True)
routed.imagined.loss.backward()
check("value reaches the critic", float(critic.value.mlp.out.weight.grad.abs().sum()) > 0)
check("value does not reach the actor", all(p.grad is None for p in actor.parameters()))

model.zero_grad(set_to_none=True)
critic.zero_grad(set_to_none=True)
actor.zero_grad(set_to_none=True)
critic.replay_loss(
    post[:, P + 1 :].feat, rewards[:, P:], is_last[:, P:].to(torch.float32),
    is_terminal[:, P:].to(torch.float32), grid[:, P:], filled[:, P:],
).loss.backward()
check("repval still reaches the encoder", float(model.encoder.convs[0].weight.grad.abs().sum()) > 0)
check("repval still reaches the RSSM", float(model.rssm.dyngru.kernel.grad.abs().sum()) > 0)
check("repval does not reach the actor", all(p.grad is None for p in actor.parameters()))

print(f"\n## REINFORCE on the real rollout with CONTROLLED returns")
print("  A FRESH zero-init critic, so the baseline is exactly 0 and the advantage is exactly")
print("  ret / S. The returns are a fixed seeded draw on U(-1, 1), not predictions: the perturbed")
print("  reward head above emits ~1e7 and would say nothing about the estimator.")
torch.manual_seed(2)
actor = Actor(in_features=model.rssm.feat_size, action_dim=action_dim).to(device)
fit_critic = Critic(in_features=model.rssm.feat_size).to(device)
model_snapshot = [p.detach().clone() for p in model.parameters()]
critic_snapshot = [p.detach().clone() for p in fit_critic.parameters()]

fit_ret = (
    2.0 * torch.rand((B * T, H), generator=torch.Generator(device=device).manual_seed(4), device=device) - 1.0
)
with torch.no_grad():
    fresh_baseline = float(fit_critic.value.predict(imagination.feat).abs().max())
check("the fresh critic's baseline is exactly 0", fresh_baseline == 0.0)

with torch.no_grad():
    before_logp = actor(imagination.feat[:, :-1]).log_prob(imagination.actions).clone()

step_out = imagined_actor_loss(actor, fit_critic, imagination, fit_ret, ReturnNormalizer().to(device), actent=0.0)
before_loss = float(step_out.loss.detach())
opt = torch.optim.SGD(actor.parameters(), lr=1e-3)
opt.zero_grad(set_to_none=True)
step_out.loss.backward()
opt.step()

with torch.no_grad():
    delta = actor(imagination.feat[:, :-1]).log_prob(imagination.actions) - before_logp
    ascent = float((step_out.weight * step_out.advantage * delta).sum())
    after_loss = float(
        imagined_actor_loss(actor, fit_critic, imagination, fit_ret,
                            ReturnNormalizer().to(device), actent=0.0, update=False).loss
    )

print(f"  one SGD step over {B * T * H} real imagined positions:")
print(f"    sum(w * A * delta logpi) = {ascent:+.6f}    policy loss "
      f"{before_loss:+.8f} -> {after_loss:+.8f}")
# capacity-free: one shared network cannot raise every position independently, but the
# advantage-weighted total must rise for any ascent direction. A correlation threshold would
# measure how well 50k parameters fit 15360 positions, not whether the estimator is right.
check("one update raises the ADVANTAGE-WEIGHTED log-probability at scale", ascent > 0.0, f"{ascent:+.6f}")
check("the policy loss falls under that same step", after_loss < before_loss)

print(f"\n## a {FIT_STEPS}-step LaProp trajectory on the same fixed batch")
print("  Monitoring only. With a FIXED advantage, no critic learning and no new data, unconstrained")
print("  ascent drives stddev to minstd within ~50 steps and the differential entropy goes")
print("  NEGATIVE, which is correct for a Gaussian with sigma < 1/sqrt(2*pi*e), not a failure.")
torch.manual_seed(2)
actor = Actor(in_features=model.rssm.feat_size, action_dim=action_dim).to(device)
normalizer = ReturnNormalizer().to(device)
minent = action_dim * (0.5 * math.log(2 * math.pi * actor.minstd ** 2) + 0.5)
maxent = action_dim * (0.5 * math.log(2 * math.pi * actor.maxstd ** 2) + 0.5)

opt = LaProp(actor.parameters(), lr=1e-3, warmup=0)
rows = []
for step_i in range(FIT_STEPS):
    out = imagined_actor_loss(actor, fit_critic, imagination, fit_ret, normalizer)
    opt.zero_grad(set_to_none=True)
    out.loss.backward()
    opt.step()
    if step_i % 10 == 0 or step_i == FIT_STEPS - 1:
        rows.append({
            "step": step_i,
            "policy_loss": round(out.metrics["policy"], 6),
            "entropy": round(out.metrics["policy_entropy"], 6),
            "logp_mean": round(out.metrics["policy_logp_mean"], 4),
            "adv_mag": round(out.metrics["policy_adv_mag"], 6),
            "retnorm_scale": round(out.metrics["policy_retnorm_scale"], 6),
        })

print(f"  entropy bounds from minstd/maxstd: [{minent:.4f}, {maxent:.4f}]")
print(f"  {'step':>5}{'policy_loss':>14}{'entropy':>12}{'logp_mean':>12}{'adv_mag':>12}{'S':>12}")
for r in rows:
    print(f"  {r['step']:>5}{r['policy_loss']:>14.6f}{r['entropy']:>12.4f}{r['logp_mean']:>12.4f}"
          f"{r['adv_mag']:>12.4g}{r['retnorm_scale']:>12.4f}")

check("every logged row is finite", all(math.isfinite(v) for r in rows for v in r.values()))
check("entropy never leaves the closed-form bounds of the stddev parameterization",
      all(minent - 1e-4 <= r["entropy"] <= maxent + 1e-4 for r in rows),
      f"min {min(r['entropy'] for r in rows):.4f}")
check("the world model is bitwise unchanged",
      all(torch.equal(a, b) for a, b in zip(model_snapshot, model.parameters())))
check("the critic is bitwise unchanged",
      all(torch.equal(a, b) for a, b in zip(critic_snapshot, fit_critic.parameters())))

expected_hi = (1.0 - 0.99 ** FIT_STEPS) * float(torch.quantile(fit_ret.reshape(-1), 0.95))
check("the normalizer advanced exactly once per step",
      abs(float(normalizer.hi) - expected_hi) < 1e-4, f"{float(normalizer.hi):.6f} vs {expected_hi:.6f}")

if device.type == "cuda":
    print(f"\n  peak VRAM, posterior + {B * T} actor-driven rollouts at H={H} + all three losses: "
          f"{torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")

out_dir = Path("results/m8")
out_dir.mkdir(parents=True, exist_ok=True)
tag = sys.argv[1] if len(sys.argv) > 1 else "2026-09-19"
path = out_dir / f"actor-update-{tag}.csv"
with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nwrote {path}")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print(f"ALL {checks} CHECKS PASSED")
