import unittest

import numpy as np

from dreamer import Collector, EnvStep, ReplayBuffer


def image(value: int) -> np.ndarray:
    return np.full(
        (2, 2, 1),
        value,
        dtype=np.uint8,
    )


class ToyEnv:
    physics_substeps = 3

    def __init__(self, episode_length: int = 4) -> None:
        self.episode_length = episode_length
        self.t = 0
        self.needs_reset = True

    def reset(self) -> np.ndarray:
        self.t = 0
        self.needs_reset = False
        return image(0)

    def step(self, action: np.ndarray) -> EnvStep:
        if self.needs_reset:
            raise RuntimeError("reset required")

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        self.t += 1
        is_last = self.t == self.episode_length

        if is_last:
            self.needs_reset = True

        return EnvStep(
            next_observation=image(self.t),
            action=action.copy(),
            reward=np.float32(1000 + self.t),
            is_last=is_last,
            is_terminal=False,
            discount=np.float32(1.0),
        )


class ToyPolicy:
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        t = float(observation[0, 0, 0])

        return np.array(
            [100.0 + t],
            dtype=np.float32,
        )


class TestCollector(unittest.TestCase):
    def test_temporal_alignment(self) -> None:
        replay = ReplayBuffer(
            capacity=100,
            seed=0,
        )

        collector = Collector(
            env=ToyEnv(episode_length=4),
            replay=replay,
            policy=ToyPolicy(),
        )

        transitions = [
            collector.collect_step()
            for _ in range(4)
        ]

        for t, transition in enumerate(transitions):
            self.assertEqual(
                int(transition.observation[0, 0, 0]),
                t,
            )
            self.assertEqual(
                float(transition.action[0]),
                100.0 + t,
            )
            self.assertEqual(
                float(transition.reward),
                1001.0 + t,
            )
            self.assertEqual(
                int(
                    transition.next_observation[
                        0, 0, 0
                    ]
                ),
                t + 1,
            )

        self.assertFalse(
            transitions[0].is_last
        )
        self.assertTrue(
            transitions[-1].is_last
        )

    def test_counters(self) -> None:
        replay = ReplayBuffer(
            capacity=100,
            seed=0,
        )

        collector = Collector(
            env=ToyEnv(episode_length=4),
            replay=replay,
            policy=ToyPolicy(),
        )

        collector.collect(4)

        counters = collector.counters

        self.assertEqual(
            counters.env_step,
            4,
        )
        self.assertEqual(
            counters.agent_step,
            4,
        )
        self.assertEqual(
            counters.replay_transition,
            4,
        )
        self.assertEqual(
            counters.physics_substep,
            12,
        )
        self.assertEqual(
            counters.gradient_step,
            0,
        )
        self.assertEqual(
            counters.train_position,
            0,
        )

    def test_collector_resets_after_boundary(self) -> None:
        replay = ReplayBuffer(
            capacity=100,
            seed=0,
        )

        collector = Collector(
            env=ToyEnv(episode_length=2),
            replay=replay,
            policy=ToyPolicy(),
        )

        collector.collect_step()
        collector.collect_step()

        transition = collector.collect_step()

        self.assertEqual(
            int(transition.observation[0, 0, 0]),
            0,
        )
        self.assertEqual(
            float(transition.action[0]),
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
