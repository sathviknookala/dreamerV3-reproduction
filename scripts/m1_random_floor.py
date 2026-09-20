"""M1 random-policy return floor: 20 COMPLETE episodes per task.

The previous version of this file was a copy of `m1_throughput.py` and never accumulated a reward,
so it reported a step rate under the name of a return floor. Throughput stays in that script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from dreamer import DMCEnv, UniformRandomPolicy
from dreamer.config import TASKS
from dreamer.viz import write_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--env-seed", type=int, required=True)
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--video", type=str, default=None, help="record the first episode")
    args = parser.parse_args()

    domain, task = TASKS[args.task]
    env = DMCEnv(domain, task, seed=args.env_seed)
    policy = UniformRandomPolicy(env.action_low, env.action_high, seed=args.policy_seed)

    returns: list[float] = []
    lengths: list[int] = []
    frames: list[np.ndarray] = []
    start = time.perf_counter()

    try:
        for episode in range(args.episodes):
            observation = env.reset()
            episode_return, length = 0.0, 0
            record = args.video is not None and episode == 0

            if record:
                frames.append(observation.copy())

            for _ in range(args.max_steps):
                step = env.step(policy(observation))
                episode_return += float(step.reward)
                length += 1

                if record:
                    frames.append(step.next_observation.copy())

                if step.is_last:
                    break

                observation = step.next_observation
            else:
                raise RuntimeError(
                    f"episode did not end within {args.max_steps} steps; the floor would be "
                    "a truncation, not an episode return"
                )

            returns.append(episode_return)
            lengths.append(length)
    finally:
        env.close()

    elapsed = time.perf_counter() - start
    video = write_video(args.video, np.stack(frames)) if frames else None

    result = {
        "task": args.task,
        "domain": domain,
        "dmc_task": task,
        "env_seed": args.env_seed,
        "policy_seed": args.policy_seed,
        "episodes": args.episodes,
        "returns": returns,
        "lengths": lengths,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "total_control_steps": int(sum(lengths)),
        "elapsed_seconds": elapsed,
        "policy": "uniform random over the action bounds, action repeat 1",
        "render": "64x64 RGB every control step",
        "video": video,
        "measures": "return floor only; throughput is measured by scripts/m1_throughput.py",
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
