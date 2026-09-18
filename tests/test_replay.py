import unittest

import numpy as np

from dreamer import ReplayBuffer, Transition


def image(value: int) -> np.ndarray:
    return np.full(
        (2, 2, 1),
        value,
        dtype=np.uint8,
    )


def add_episode(
    replay: ReplayBuffer,
    length: int,
    base: int = 0,
    terminal: bool = False,
) -> None:
    for t in range(length):
        is_last = t == length - 1
        is_terminal = bool(is_last and terminal)

        replay.add(
            Transition(
                observation=image(base + t),
                action=np.array(
                    [float(base + t)],
                    dtype=np.float32,
                ),
                reward=np.float32(t + 0.5),
                next_observation=image(base + t + 1),
                is_last=is_last,
                is_terminal=is_terminal,
                discount=np.float32(
                    0.0 if is_terminal else 1.0
                ),
            )
        )


class TestReplayBuffer(unittest.TestCase):
    def test_sequence_chronology(self) -> None:
        replay = ReplayBuffer(
            capacity=1000,
            seed=0,
        )

        add_episode(
            replay,
            length=30,
        )

        batch = replay.sample_sequences(
            batch_size=32,
            train_length=6,
            burn_in=2,
        )

        values = batch.observations[:, :, 0, 0, 0]
        differences = np.diff(
            values.astype(np.int32),
            axis=1,
        )

        np.testing.assert_array_equal(
            differences,
            np.ones_like(differences),
        )

    def test_sequence_shapes_and_mask(self) -> None:
        replay = ReplayBuffer(
            capacity=1000,
            seed=0,
        )

        add_episode(
            replay,
            length=20,
        )

        batch = replay.sample_sequences(
            batch_size=4,
            train_length=6,
            burn_in=2,
        )

        self.assertEqual(
            batch.observations.shape,
            (4, 9, 2, 2, 1),
        )
        self.assertEqual(
            batch.actions.shape,
            (4, 8, 1),
        )
        self.assertEqual(
            batch.loss_mask.shape,
            (4, 8),
        )

        self.assertFalse(
            batch.loss_mask[:, :2].any()
        )
        self.assertTrue(
            batch.loss_mask[:, 2:].all()
        )
        self.assertEqual(
            batch.train_positions,
            4 * 6,
        )

    def test_samples_do_not_cross_boundaries(self) -> None:
        replay = ReplayBuffer(
            capacity=1000,
            seed=4,
        )

        add_episode(
            replay,
            length=20,
            base=0,
        )
        add_episode(
            replay,
            length=20,
            base=100,
        )

        batch = replay.sample_sequences(
            batch_size=100,
            train_length=5,
            burn_in=2,
        )

        values = batch.observations[:, :, 0, 0, 0]
        differences = np.diff(
            values.astype(np.int32),
            axis=1,
        )

        np.testing.assert_array_equal(
            differences,
            np.ones_like(differences),
        )

        self.assertFalse(
            batch.is_last[:, :-1].any()
        )

    def test_rejects_shifted_transition(self) -> None:
        replay = ReplayBuffer(
            capacity=100,
            seed=0,
        )

        replay.add(
            Transition(
                observation=image(0),
                action=np.array([0.0], dtype=np.float32),
                reward=np.float32(1.0),
                next_observation=image(1),
                is_last=False,
                is_terminal=False,
                discount=np.float32(1.0),
            )
        )

        with self.assertRaises(ValueError):
            replay.add(
                Transition(
                    observation=image(9),
                    action=np.array([1.0], dtype=np.float32),
                    reward=np.float32(2.0),
                    next_observation=image(10),
                    is_last=True,
                    is_terminal=False,
                    discount=np.float32(1.0),
                )
            )

    def test_terminal_requires_zero_discount(self) -> None:
        replay = ReplayBuffer(
            capacity=100,
            seed=0,
        )

        with self.assertRaises(ValueError):
            replay.add(
                Transition(
                    observation=image(0),
                    action=np.array([0.0], dtype=np.float32),
                    reward=np.float32(0.0),
                    next_observation=image(1),
                    is_last=True,
                    is_terminal=True,
                    discount=np.float32(1.0),
                )
            )

    def test_time_limit_is_not_terminal(self) -> None:
        replay = ReplayBuffer(
            capacity=100,
            seed=0,
        )

        replay.add(
            Transition(
                observation=image(0),
                action=np.array([0.0], dtype=np.float32),
                reward=np.float32(0.0),
                next_observation=image(1),
                is_last=True,
                is_terminal=False,
                discount=np.float32(1.0),
            )
        )

        self.assertEqual(
            replay.num_complete_episodes,
            1,
        )

    def test_capacity_evicts_old_episodes(self) -> None:
        replay = ReplayBuffer(
            capacity=10,
            seed=0,
        )

        add_episode(
            replay,
            length=6,
            base=0,
        )
        add_episode(
            replay,
            length=6,
            base=100,
        )

        self.assertEqual(
            replay.num_complete_episodes,
            1,
        )
        self.assertEqual(
            len(replay),
            6,
        )


if __name__ == "__main__":
    unittest.main()
