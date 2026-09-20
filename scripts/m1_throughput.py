from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, "src")

from dreamer import DMCEnv, UniformRandomPolicy


from dreamer.config import TASKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=TASKS,
        required=True,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--env-seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--policy-seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    if args.steps < 10_000:
        raise SystemExit("throughput must be measured over at least 10,000 control steps")

    domain, task = TASKS[args.task]

    env = DMCEnv(
        domain,
        task,
        seed=args.env_seed,
    )

    policy = UniformRandomPolicy(
        env.action_low,
        env.action_high,
        seed=args.policy_seed,
    )

    observation = env.reset()

    try:
        for _ in range(args.warmup):
            action = policy(observation)
            step = env.step(action)

            if step.is_last:
                observation = env.reset()
            else:
                observation = step.next_observation

        start = time.perf_counter()

        episodes = 0

        for _ in range(args.steps):
            action = policy(observation)
            step = env.step(action)

            if step.is_last:
                episodes += 1
                observation = env.reset()
            else:
                observation = step.next_observation

        elapsed = time.perf_counter() - start

    finally:
        env.close()

    result = {
        "task": args.task,
        "env_seed": args.env_seed,
        "policy_seed": args.policy_seed,
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "episodes_completed": episodes,
        "elapsed_seconds": elapsed,
        "control_steps_per_second": args.steps / elapsed,
        "render": "64x64 RGB every control step",
        "measures": "control steps per second only; the return floor is scripts/m1_random_floor.py",
    }

    with open(args.output, "w") as file:
        json.dump(
            result,
            file,
            indent=2,
            sort_keys=True,
        )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
