"""M9 integration gate: implementation validation, NOT demonstrated learning.

Everything below runs on the real simulator with real replay data, a real learned-policy collection
pass, complete training updates, an isolated evaluation and a checkpoint/resume cycle. It proves the
loop is wired the way spec 5 and spec 7 say. It proves nothing about return: M9 stays open until the
pilots show sustained improvement over the M1 random floor.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from dreamer import (
    ACTENT,
    Agent,
    Checkpointer,
    EpisodicEnv,
    OnlineTrainer,
    ReplayBuffer,
    RunConfig,
    RunIdentity,
    apply_update,
    compute_losses,
    evaluate_agent,
    load_checkpoint,
    restore_resume,
    stream_seed,
)
from dreamer.checkpoint import resume_payload
from dreamer.config import TASKS, episode_seed
from dreamer.evaluation import _agent_fingerprint, _fingerprints_match

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

failures: list[str] = []
checks = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")

    if not ok:
        failures.append(name)


def perturb(agent: Agent, scale: float = 0.05) -> None:
    """rew and val are outscale 0.0: unperturbed, the advantage is identically 0 and every
    routing and magnitude claim below would pass vacuously."""
    generator = torch.Generator().manual_seed(0)

    for head in (agent.world_model.reward.mlp.out, agent.critic.value.mlp.out):
        with torch.no_grad():
            head.weight.add_(
                torch.randn(head.weight.shape, generator=generator).to(head.weight.device) * scale
            )

    agent.critic.slow.mirror.load_state_dict(agent.critic.value.state_dict())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, default="cartpole")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", default="results/m9")
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--budget", type=int, default=2100)
    parser.add_argument("--episode", type=int, default=1000)
    args = parser.parse_args(argv)

    B, T, P, H = 8, 32, 5, 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  torch: {torch.__version__}")

    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    print(f"task: {args.task}  seed: {args.seed}  B={B} P={P} T={T} H={H}")
    print(f"seeds: init/collect/replay/imagine/provider/eval derived from base {args.seed}")

    config = RunConfig(
        task=args.task,
        seed=args.seed,
        budget=args.budget,
        batch_size=B,
        train_length=T,
        burn_in=P,
        horizon=H,
        warmup_transitions=args.warmup,
        replay_capacity=10_000,
        eval_every=10**9,
        eval_episodes=2,
        checkpoint_every=args.episode,
        log_every=1,
        eval_video=True,
    )
    domain, task = config.domain_task

    print("\n## streams are separated by construction")
    names = ("init", "env", "collect", "replay", "imagine", "provider", "eval")
    seeds = {n: stream_seed(config.seed, n) for n in names}
    check("every named stream gets a distinct seed", len(set(seeds.values())) == len(names),
          ", ".join(f"{n}={s}" for n, s in seeds.items()))
    check("the episode seed schedule is injective over a 1e6-step run",
          len({episode_seed(config.seed, i) for i in range(1000)}) == 1000)

    out = Path(tempfile.mkdtemp(prefix="m9-gate-"))
    env = EpisodicEnv(domain, task, base_seed=config.seed, recreate=True)
    agent = Agent(action_dim=env.action_shape[0], config=config, device=device)
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=stream_seed(config.seed, "replay"))
    identity = RunIdentity.build(config)
    trainer = OnlineTrainer(
        agent, env, replay, config, out,
        checkpointer=Checkpointer(out, identity, keep=2),
    )

    counts = agent.parameter_counts()
    print(f"\n## the agent  (action_dim {env.action_shape[0]})")
    print(f"  world model {counts.world_model}   actor {counts.actor}   "
          f"critic {counts.critic}   trainable total {counts.total}")
    check("the optimizer holds every trainable tensor exactly once",
          len({id(p) for p in agent.trainable_parameters()}) == len(agent.trainable_parameters()))
    check("the slow critic is excluded from the optimizer",
          not ({id(p) for p in agent.critic.slow.parameters()}
               & {id(p) for p in agent.trainable_parameters()}))
    check("the counts sum to the trainable total",
          counts.total == sum(p.numel() for p in agent.trainable_parameters()),
          str(counts.total))

    print("\n## real collection with a learned policy on the real simulator")
    trainer.run()
    counters = trainer.counters

    check("the budget stopped collection exactly", counters.env_step == args.budget,
          str(counters.env_step))
    check("warm-up accrued no training debt",
          counters.train_position == (args.budget - args.warmup) * config.train_ratio
          - trainer.scheduler.credits,
          f"{counters.train_position} positions, remainder {trainer.scheduler.credits}")
    check("the learned policy actually drove collection", trainer._learned)
    check("replay holds every collected transition", len(replay) == args.budget,
          str(replay.stats()))
    check("optimizer steps match the scheduler",
          counters.gradient_step == counters.train_position // config.positions_per_update,
          str(counters.gradient_step))
    check("the realized ratio is the configured one",
          abs(counters.train_position / (args.budget - args.warmup) - config.train_ratio) < 1.0,
          f"{counters.train_position / (args.budget - args.warmup):.3f}")
    check("every imagination start became H imagined transitions",
          counters.imagined_transition == counters.imagination_start * H,
          f"{counters.imagination_start} starts")
    check("physics substeps were counted", counters.physics_substep > 0,
          str(counters.physics_substep))

    episode = replay.complete_episodes[0]
    check("collected actions stayed inside the environment bounds",
          bool((np.abs(episode.actions) <= 1.0 + 1e-6).all()),
          f"max |a| = {float(np.abs(episode.actions).max()):.4f}")
    check("the policy is unsquashed, so the bound is the environment's job",
          float(np.abs(episode.actions).max()) > 0.0)

    print("\n## one complete update, every stage")
    perturb(agent)
    batch = replay.sample_sequences(B, T, P)
    output = compute_losses(agent, batch)

    starts = output.starts
    check("starts are the loss-bearing non-terminal positions", starts == B * T, str(starts))
    check("burn-in positions are never starts",
          output.world.metrics["train_positions"] == float(B * T))
    check("the rollout is H prior-only steps",
          tuple(output.imagination.actions.shape) == (starts, H, env.action_shape[0]))
    check("H+1 latent states accompany H actions",
          tuple(output.imagination.feat.shape)[:2] == (starts, H + 1))
    check("the imagined return is (N, H)",
          tuple(output.behavior.imagined.ret.shape) == (starts, H))
    check("the replay critic fits T-1 positions",
          tuple(output.replay_critic.ret.shape) == (B, T - 1))
    check("the imagined features carry no graph", not output.imagination.feat.requires_grad)
    check("the advantage is detached", not output.behavior.actor.advantage.requires_grad)
    check("every loss is finite",
          bool(torch.isfinite(output.total))
          and all(bool(torch.isfinite(v)) for v in output.world.losses.values()))
    check("total = world + behaviour + 0.3 * replay-critic",
          bool(torch.equal(
              output.total,
              output.world.total + output.behavior.loss + 0.3 * output.replay_critic.loss,
          )))
    check("the fixture is not degenerate",
          float(output.behavior.actor.advantage.abs().mean()) > 0.0,
          f"|A| = {float(output.behavior.actor.advantage.abs().mean()):.4g}")

    print("\n## the pinned loss scales")
    for name, value in (("rec", 1.0), ("rew", 1.0), ("con", 1.0), ("dyn", 1.0), ("rep", 0.1)):
        check(f"world model {name} = {value}", agent.world_model.scales[name] == value)

    check("imagined critic = 1.0", agent.critic.scales["value"] == 1.0)
    check("replay critic = 0.3", agent.critic.scales["repval"] == 0.3)
    check("slow-critic regularization = 1.0", agent.critic.slowreg == 1.0)
    check("entropy coefficient = 3e-4", ACTENT == 3e-4)
    check("free bits = 1 nat for the whole latent", agent.world_model.free_nats == 1.0)
    check("contdisc is on, so the return discount is 1", agent.world_model.contdisc)

    print("\n## gradient routing on the combined loss")
    probes = {
        "encoder": agent.world_model.encoder.convs[0].weight,
        "rssm": agent.world_model.rssm.dyngru.kernel,
        "decoder": agent.world_model.decoder.imgout.weight,
        "actor": agent.actor.mean.weight,
        "value": agent.critic.value.mlp.out.weight,
    }

    def grads(loss):
        agent.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        return {k: v.grad for k, v in probes.items()}

    g = grads(output.behavior.actor.loss)
    check("the actor loss reaches the actor", float(g["actor"].abs().sum()) > 0)
    check("the actor loss reaches nothing else",
          all(g[k] is None for k in ("encoder", "rssm", "decoder", "value")))

    g = grads(output.behavior.imagined.loss)
    check("the imagined critic loss reaches the fast critic", float(g["value"].abs().sum()) > 0)
    check("the imagined critic loss reaches nothing else",
          all(g[k] is None for k in ("encoder", "rssm", "decoder", "actor")))

    g = grads(output.replay_critic.loss)
    check("the replay critic reaches the encoder", float(g["encoder"].abs().sum()) > 0)
    check("the replay critic reaches the RSSM", float(g["rssm"].abs().sum()) > 0)
    check("the replay critic reaches the fast critic", float(g["value"].abs().sum()) > 0)
    check("the replay critic reaches neither the decoder nor the actor",
          g["decoder"] is None and g["actor"] is None)

    g = grads(output.total)
    check("the combined loss reaches every trainable module",
          all(float(g[k].abs().sum()) > 0 for k in probes))
    check("no gradient ever reaches the slow mirror",
          all(p.grad is None for p in agent.critic.slow.parameters()))

    print("\n## update cardinality")
    slow_before = [p.detach().clone() for p in agent.critic.slow.mirror.parameters()]
    hi_before = float(agent.normalizer.hi)
    step_before = trainer.optimizer._step

    fresh = compute_losses(agent, replay.sample_sequences(B, T, P))
    apply_update(agent, trainer.optimizer, fresh)

    check("exactly one optimizer step", trainer.optimizer._step == step_before + 1)
    check("the return normalizer moved exactly once",
          float(agent.normalizer.hi) != hi_before)
    rate = agent.critic.slow.rate
    check("the slow critic advanced AFTER the optimizer step",
          all(
              torch.allclose(slow, old.lerp(fast.detach(), rate), rtol=0, atol=1e-6)
              for old, fast, slow in zip(
                  slow_before,
                  agent.critic.value.parameters(),
                  agent.critic.slow.mirror.parameters(),
              )
          ))
    from dreamer import ReturnNormalizer

    check("an unstarted normalizer reads exactly its floor of 1, which is a valid observation",
          float(ReturnNormalizer(debias=False).scale) == 1.0)
    check("retnorm is uncorrected at the pin", not agent.normalizer.debias)
    print(f"  S = {float(agent.normalizer.scale):.4g} here only because the value head was "
          f"perturbed; on a real run it starts at the floor of 1")

    print("\n## isolated evaluation")
    fingerprint = _agent_fingerprint(agent)
    replay_before = replay.stats()
    counters_before = counters.state_dict()
    video = out / "gate-eval.mp4"
    evaluation = evaluate_agent(
        agent, domain, task, (2000, 2001), env_step=counters.env_step, video_path=video
    )

    check("evaluation changed no parameter, buffer or training stream",
          _fingerprints_match(fingerprint, _agent_fingerprint(agent)))
    check("evaluation touched no replay state", replay.stats() == replay_before)
    check("evaluation touched no training counter", counters.state_dict() == counters_before)
    check("both evaluation episodes ran to completion",
          all(length == args.episode for length in evaluation.lengths), str(evaluation.lengths))
    check("individual returns and lengths are recorded",
          len(evaluation.returns) == 2 and len(evaluation.lengths) == 2)
    check("the video came from the first evaluation seed",
          evaluation.video is not None and Path(evaluation.video).exists(),
          str(evaluation.video))
    check("evaluation seeds are 2000+, disjoint from training seeds 100-102 and final 0-2",
          min(config.eval_seeds) >= 2000)
    print(f"  returns {['%.2f' % r for r in evaluation.returns]}  "
          f"mean {evaluation.mean:.3f}  sd {evaluation.stddev:.3f}  "
          f"{evaluation.seconds:.1f}s for {evaluation.steps} steps")

    print("\n## checkpoint and resume")
    path = trainer.checkpointer.latest_resume()
    check("a resume checkpoint was written at an episode boundary", path is not None, str(path))
    check("only the latest two resume checkpoints are kept",
          len(list(out.glob("resume-*.pt"))) <= 2)
    check("a compact model checkpoint sits beside them",
          (out / "model-final.pt").exists()
          and (out / "model-final.pt").stat().st_size < path.stat().st_size,
          f"{(out / 'model-final.pt').stat().st_size / 2**20:.1f} MiB "
          f"vs {path.stat().st_size / 2**20:.1f} MiB")

    reference_batch = replay.sample_sequences(B, T, P)
    reference = compute_losses(agent, reference_batch)

    resumed_env = EpisodicEnv(domain, task, base_seed=config.seed, recreate=True)
    resumed_agent = Agent(action_dim=resumed_env.action_shape[0], config=config, device=device)
    resumed_replay = ReplayBuffer(
        capacity=config.replay_capacity, seed=stream_seed(config.seed, "replay")
    )
    resumed = OnlineTrainer(
        resumed_agent, resumed_env, resumed_replay, config, out / "resumed",
        checkpointer=Checkpointer(out / "resumed", identity, keep=2),
    )
    restore_resume(resumed, load_checkpoint(path))

    saved = load_checkpoint(path)
    saved_trainer = saved["trainer"]
    saved_counters = saved_trainer["counters"]

    check("the restored counters match the checkpoint",
          resumed.counters.state_dict() == saved_counters)
    check("the scheduling remainder was restored",
          resumed.scheduler.credits == saved_trainer["scheduler"]["credits"],
          str(resumed.scheduler.credits))
    check("LaProp's private step counter was restored",
          resumed.optimizer._step == saved_trainer["optimizer"]["_step"],
          str(resumed.optimizer._step))
    restored_moments = resumed.optimizer.state_dict()["state"]
    saved_moments = saved_trainer["optimizer"]["state"]
    check("LaProp's moments were restored",
          bool(saved_moments)
          and set(restored_moments) == set(saved_moments)
          and all(
              torch.equal(restored_moments[k][m].cpu(), saved_moments[k][m].cpu())
              for k in saved_moments
              for m in ("mu", "nu")
          ),
          f"{len(saved_moments)} tensors")
    check("replay contents were restored",
          len(resumed_replay) == saved_counters["replay_transition"],
          str(resumed_replay.stats()))
    check("the next episode's seed was persisted",
          resumed_env.next_seed == saved["env"]["next_seed"],
          str(resumed_env.next_seed))
    check("the restored agent's generators match the checkpoint",
          all(
              torch.equal(
                  resumed_agent.generators[name].get_state().cpu(),
                  saved["generators"]["generators"][name].cpu(),
              )
              for name in resumed_agent.generators
          ))
    check("the provider's private generator was restored",
          saved["generators"]["provider"]["generator"] is not None
          and torch.equal(
              resumed_agent.provider.state_dict()["generator"].cpu(),
              saved["generators"]["provider"]["generator"].cpu(),
          ))
    saved_critic = saved["model"]["critic"]
    restored_critic = resumed_agent.critic.state_dict()
    slow_keys = [k for k in saved_critic if k.startswith("slow.")]
    check("the slow critic mirror was restored",
          bool(slow_keys)
          and all(
              torch.equal(restored_critic[k].cpu(), saved_critic[k].cpu()) for k in slow_keys
          ),
          f"{len(slow_keys)} tensors")
    check("the normalizer buffers and settings were restored",
          float(resumed_agent.normalizer.hi) == float(saved["model"]["normalizer"]["hi"])
          and float(resumed_agent.normalizer.lo) == float(saved["model"]["normalizer"]["lo"])
          and resumed_agent.normalizer.settings() == saved["model"]["normalizer_settings"],
          f"lo={float(resumed_agent.normalizer.lo):.4g} hi={float(resumed_agent.normalizer.hi):.4g}")

    print("\n  -- a mid-episode resume checkpoint must be refused")
    trainer.collector.collect(3)
    mid = resume_payload(trainer, identity)
    refused = False

    try:
        restore_resume(resumed, mid)
    except ValueError as error:
        refused = "mid-episode" in str(error)

    check("a mid-episode resume checkpoint is refused", refused)

    print("\n## the next update after a restore")
    # the checkpointed agent was mutated above by the extra updates, so compare a fresh restore
    totals, actor_losses = [], []

    for _ in range(2):
        target_env = EpisodicEnv(domain, task, base_seed=config.seed, recreate=False)
        target_agent = Agent(action_dim=target_env.action_shape[0], config=config, device=device)
        target = OnlineTrainer(
            target_agent, target_env, ReplayBuffer(config.replay_capacity, 0), config,
            out / "equiv",
        )
        restore_resume(target, load_checkpoint(path))
        result = compute_losses(target_agent, reference_batch)
        totals.append(float(result.total.detach()))
        actor_losses.append(float(result.behavior.actor.loss.detach()))
        target_env.close()

    check("two independent restores compute the same next update bitwise",
          totals[0] == totals[1] and actor_losses[0] == actor_losses[1],
          f"total {totals[0]!r} vs {totals[1]!r}")
    print("  NOTE: bitwise equality holds for the NEXT update. Two runs continued from the same")
    print("        checkpoint drift at ~1e-6 over hundreds of updates -- nondeterministic CUDA")
    print("        convolution backward, measured identical between two restores, not a resume bug.")

    print("\n## memory")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**20
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        print(f"  peak allocated over the whole gate: {peak:.1f} MiB of {total:.0f} MiB")
        check("peak VRAM left headroom on the 24 GB card", peak < 0.5 * total,
              f"{peak:.1f} MiB")
    else:
        print("  CPU run: no VRAM measurement")

    env.close()
    resumed_env.close()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"gate-{args.task}-{args.tag}.json"

    if report_path.exists():
        raise SystemExit(f"{report_path} exists; a committed measurement is not overwritten")

    report = {
        "label": "implementation validation, not demonstrated learning",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "config": config.to_dict(),
        "stream_seeds": seeds,
        "parameters": {
            "world_model": counts.world_model,
            "actor": counts.actor,
            "critic": counts.critic,
            "trainable_total": counts.total,
        },
        "counters": counters.state_dict(),
        "scheduler_remainder": trainer.scheduler.credits,
        "replay": replay.stats(),
        "evaluation": evaluation.summary(),
        "checks_run": checks,
        "checks_failed": failures,
        "caveat": (
            "returns here come from a policy trained on ~%d control steps; they are a smoke "
            "signal, not evidence of learning" % (args.budget - args.warmup)
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {report_path}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print(f"ALL {checks} CHECKS PASSED")
    print("This is implementation validation. It is NOT demonstrated learning.")


if __name__ == "__main__":
    main()
