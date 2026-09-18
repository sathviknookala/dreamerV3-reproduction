import unittest

import numpy as np

from dreamer import DMCEnv


class TestDMCEnv(unittest.TestCase):
    def test_walker_contract(self) -> None:
        env = DMCEnv(
            "walker",
            "walk",
            seed=100,
        )

        try:
            observation = env.reset()

            self.assertEqual(
                observation.shape,
                (64, 64, 3),
            )
            self.assertEqual(
                observation.dtype,
                np.uint8,
            )
            self.assertEqual(
                env.action_shape,
                (6,),
            )
            self.assertEqual(
                env.physics_substeps,
                10,
            )

            action = np.full(
                env.action_shape,
                2.0,
                dtype=np.float32,
            )

            step = env.step(action)

            np.testing.assert_array_equal(
                step.action,
                np.ones(6, dtype=np.float32),
            )

        finally:
            env.close()

    def test_cartpole_contract(self) -> None:
        env = DMCEnv(
            "cartpole",
            "swingup",
            seed=100,
        )

        try:
            observation = env.reset()

            self.assertEqual(
                observation.shape,
                (64, 64, 3),
            )
            self.assertEqual(
                env.action_shape,
                (1,),
            )
            self.assertEqual(
                env.physics_substeps,
                1,
            )

        finally:
            env.close()

    def test_seed_reproducibility(self) -> None:
        first = DMCEnv(
            "walker",
            "walk",
            seed=101,
        )
        second = DMCEnv(
            "walker",
            "walk",
            seed=101,
        )

        try:
            first_observation = first.reset()
            second_observation = second.reset()

            np.testing.assert_array_equal(
                first_observation,
                second_observation,
            )

            action = np.zeros(
                first.action_shape,
                dtype=np.float32,
            )

            for _ in range(10):
                first_step = first.step(action)
                second_step = second.step(action)

                np.testing.assert_array_equal(
                    first_step.next_observation,
                    second_step.next_observation,
                )

                self.assertEqual(
                    float(first_step.reward),
                    float(second_step.reward),
                )

        finally:
            first.close()
            second.close()


if __name__ == "__main__":
    unittest.main()
