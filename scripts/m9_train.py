"""M9 online training. Importing this module starts nothing; `main()` does."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "src")

from dreamer import (
    Agent,
    Checkpointer,
    EpisodicEnv,
    OnlineTrainer,
    PeriodicEvaluator,
    ReplayBuffer,
    RunConfig,
    RunIdentity,
    load_checkpoint,
    restore_resume,
    stream_seed,
)
from dreamer.config import TASKS


def build_config(args) -> RunConfig:
    config = RunConfig.load(args.config) if args.config else RunConfig(task=args.task)
    original_task = config.task

    for key in (
        "task", "seed", "budget", "horizon", "warmup_transitions", "train_ratio",
        "eval_every", "eval_episodes", "checkpoint_every", "log_every", "replay_capacity",
    ):
        value = getattr(args, key, None)

        if value is not None:
            setattr(config, key, value)

    # an unstated budget follows the task, so overriding the task must re-resolve it
    if args.budget is None and config.task != original_task:
        config.budget = 0

    return RunConfig.from_dict(config.to_dict())


def parse(argv=None):
    parser = argparse.ArgumentParser(description="M9 online DreamerV3 training")
    parser.add_argument("--task", choices=TASKS, default="cartpole")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--budget", type=int, default=None, help="training control steps")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="a resume-*.pt checkpoint")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--warmup-transitions", dest="warmup_transitions", type=int, default=None)
    parser.add_argument("--train-ratio", dest="train_ratio", type=int, default=None)
    parser.add_argument("--eval-every", dest="eval_every", type=int, default=None)
    parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=None)
    parser.add_argument(
        "--checkpoint-every", dest="checkpoint_every", type=int, default=None
    )
    parser.add_argument("--log-every", dest="log_every", type=int, default=None)
    parser.add_argument(
        "--replay-capacity", dest="replay_capacity", type=int, default=None
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse(argv)
    config = build_config(args)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # a detached launch through a sandboxed shell silently loses CUDA -- read this line
    print(f"device: {device}  torch: {torch.__version__}  cuda: {torch.cuda.is_available()}")
    print(f"task: {config.task}  seed: {config.seed}  horizon: {config.horizon}")
    print(f"budget: {config.budget}  warm-up: {config.warmup_transitions}  ratio: {config.train_ratio}")

    domain, task = config.domain_task
    env = EpisodicEnv(
        domain,
        task,
        base_seed=config.seed,
        size=(config.image_size, config.image_size),
        recreate=config.recreate_env_per_episode,
    )
    agent = Agent(action_dim=env.action_shape[0], config=config, device=device)
    replay = ReplayBuffer(
        capacity=config.replay_capacity, seed=stream_seed(config.seed, "replay")
    )
    identity = RunIdentity.build(config)
    trainer = OnlineTrainer(
        agent,
        env,
        replay,
        config,
        out,
        evaluator=PeriodicEvaluator(config, out),
        checkpointer=Checkpointer(out, identity, keep=config.keep_resume_checkpoints),
    )

    if args.resume:
        identity = restore_resume(trainer, load_checkpoint(args.resume))
        trainer.checkpointer.identity = identity
        print(
            f"resumed {identity.run_id} at env_step {trainer.counters.env_step}, "
            f"gradient_step {trainer.counters.gradient_step}, "
            f"scheduler remainder {trainer.scheduler.credits}"
        )

    counts = agent.parameter_counts()
    print(
        f"parameters: world model {counts.world_model}  actor {counts.actor}  "
        f"critic {counts.critic}  trainable total {counts.total}"
    )

    config.save(out / "config.json")
    started = time.time()

    try:
        counters = trainer.run()
    finally:
        env.close()

    manifest = {
        "identity": identity.to_dict(),
        "config": config.to_dict(),
        "counters": counters.state_dict(),
        "parameters": {
            "world_model": counts.world_model,
            "actor": counts.actor,
            "critic": counts.critic,
            "trainable_total": counts.total,
        },
        "device": str(device),
        "torch": torch.__version__,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "started": started,
        "finished": time.time(),
        "wall_seconds": time.time() - started,
        "replay": replay.stats(),
        "scheduler_remainder": trainer.scheduler.credits,
        "evaluations": trainer.evaluations,
        "peak_vram_mib": (
            round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if device.type == "cuda"
            else None
        ),
        "train_ratio_convention": "loss-bearing positions per collected transition (spec 7.7)",
    }
    (out / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n{counters.env_step} control steps, {counters.gradient_step} optimizer steps")
    print(f"replay {replay.stats()}")
    print(f"wrote {out / 'run-manifest.json'}")


if __name__ == "__main__":
    main()
