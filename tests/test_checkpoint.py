import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dreamer import (
    Agent,
    Checkpointer,
    LaProp,
    OnlineTrainer,
    ReplayBuffer,
    RunConfig,
    RunIdentity,
    Transition,
    atomic_save,
    compute_losses,
    load_checkpoint,
    restore_resume,
    training_update,
)
from dreamer.checkpoint import resume_payload

from test_agent import ACTION_DIM, FakeEnv, frame
from test_training import make_batch, perturb

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

A, B, P, T = ACTION_DIM, 2, 2, 6
EPISODE = 12


class StubEpisodicEnv(FakeEnv):
    """A FakeEnv with EpisodicEnv's seed bookkeeping, so no simulator is built in a unit test."""

    def __init__(self, base_seed: int = 100, episode_length: int = EPISODE) -> None:
        super().__init__(episode_length=episode_length)
        self.domain, self.task = "cartpole", "swingup"
        self.base_seed = base_seed
        self.episode_index = 0

    @property
    def next_seed(self) -> int:
        return self.base_seed + 10_007 * self.episode_index

    def reset(self) -> np.ndarray:
        self.episode_index += 1
        return super().reset()

    def state_dict(self) -> dict:
        return {
            "domain": self.domain,
            "task": self.task,
            "base_seed": self.base_seed,
            "episode_index": self.episode_index,
            "next_seed": self.next_seed,
        }

    def load_state_dict(self, state: dict) -> None:
        self.base_seed = int(state["base_seed"])
        self.episode_index = int(state["episode_index"])

    @property
    def action_low(self) -> np.ndarray:
        return np.full(A, -1.0, dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.full(A, 1.0, dtype=np.float32)


def make_config(**overrides) -> RunConfig:
    defaults = dict(
        task="cartpole",
        seed=100,
        budget=60,
        batch_size=B,
        train_length=T,
        burn_in=P,
        horizon=5,
        replay_capacity=500,
        warmup_transitions=24,
        train_ratio=64,
        eval_every=10**9,
        checkpoint_every=EPISODE * 3,
        log_every=10**9,
        eval_video=False,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def build_trainer(out: Path, **overrides):
    config = make_config(**overrides)
    env = StubEpisodicEnv()
    agent = Agent(action_dim=A, config=config, device="cpu")
    perturb(agent)
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=7)
    identity = RunIdentity.build(config)
    trainer = OnlineTrainer(
        agent, env, replay, config, out, checkpointer=Checkpointer(out, identity, keep=2)
    )
    return trainer, agent, identity


class TestAtomicSave(unittest.TestCase):
    def test_a_partial_file_is_never_left_behind(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.pt"
            atomic_save({"a": 1}, path)
            self.assertTrue(path.exists())
            self.assertEqual(list(Path(directory).glob("*.partial")), [])
            self.assertEqual(load_checkpoint(path)["a"], 1)

    def test_a_rewrite_replaces_the_file_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.pt"
            atomic_save({"a": 1}, path)
            atomic_save({"a": 2}, path)
            self.assertEqual(load_checkpoint(path)["a"], 2)


class TestOptimizerRoundTrip(unittest.TestCase):
    def test_the_private_step_counter_survives(self):
        parameter = torch.nn.Parameter(torch.ones(3))
        optimizer = LaProp([parameter], lr=1e-3, warmup=10)

        for _ in range(4):
            parameter.grad = torch.ones(3)
            optimizer.step()

        state = optimizer.state_dict()
        self.assertEqual(state["_step"], 4)

        restored = LaProp([torch.nn.Parameter(torch.ones(3))], lr=1e-3, warmup=10)
        restored.load_state_dict(state)
        self.assertEqual(restored._step, 4)

    def test_the_moments_survive(self):
        parameter = torch.nn.Parameter(torch.ones(3))
        optimizer = LaProp([parameter], lr=1e-3, warmup=1)
        parameter.grad = torch.tensor([1.0, 2.0, 3.0])
        optimizer.step()

        clone = torch.nn.Parameter(torch.ones(3))
        restored = LaProp([clone], lr=1e-3, warmup=1)
        restored.load_state_dict(optimizer.state_dict())

        self.assertTrue(
            torch.equal(restored.state[clone]["mu"], optimizer.state[parameter]["mu"])
        )
        self.assertTrue(
            torch.equal(restored.state[clone]["nu"], optimizer.state[parameter]["nu"])
        )


class TestReplayRoundTrip(unittest.TestCase):
    def _fill(self, replay: ReplayBuffer, episodes: int, length: int) -> None:
        for episode in range(episodes):
            for j in range(length):
                replay.add(
                    Transition(
                        observation=np.full((4, 4, 3), (episode * 50 + j) % 256, dtype=np.uint8),
                        action=np.zeros(A, dtype=np.float32),
                        reward=np.float32(j),
                        next_observation=np.full(
                            (4, 4, 3), (episode * 50 + j + 1) % 256, dtype=np.uint8
                        ),
                        is_last=j == length - 1,
                        is_terminal=False,
                        discount=np.float32(1.0),
                    )
                )

    def test_contents_episode_metadata_and_the_sampler_stream_all_survive(self):
        replay = ReplayBuffer(capacity=500, seed=7)
        self._fill(replay, episodes=3, length=20)
        replay.sample_sequences(2, 8, 2)

        restored = ReplayBuffer(capacity=500, seed=999)
        restored.load_state_dict(replay.state_dict())

        self.assertEqual(restored.stats(), replay.stats())

        left = restored.sample_sequences(2, 8, 2)
        right = replay.sample_sequences(2, 8, 2)
        np.testing.assert_array_equal(left.observations, right.observations)
        np.testing.assert_array_equal(left.rewards, right.rewards)

    def test_a_partial_episode_round_trips(self):
        replay = ReplayBuffer(capacity=500, seed=7)
        self._fill(replay, episodes=1, length=20)
        self._fill(replay, episodes=1, length=5)  # never closed: is_last is False throughout

        restored = ReplayBuffer(capacity=500, seed=7)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(restored.stats(), replay.stats())

    def test_a_capacity_mismatch_raises(self):
        replay = ReplayBuffer(capacity=500, seed=7)

        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=400, seed=7).load_state_dict(replay.state_dict())


class TestResumeRoundTrip(unittest.TestCase):
    def test_the_full_state_survives_and_the_next_update_matches_bitwise(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            trainer, agent, _ = build_trainer(out)
            trainer.run()

            path = trainer.checkpointer.latest_resume()
            self.assertIsNotNone(path)

            reference = compute_losses(agent, make_batch(batch=B, burn_in=P, train_length=T))

            restored, restored_agent, _ = build_trainer(out / "second")
            restored_identity = restore_resume(restored, load_checkpoint(path))

            self.assertEqual(restored.counters.state_dict(), trainer.counters.state_dict())
            self.assertEqual(restored.scheduler.credits, trainer.scheduler.credits)
            self.assertEqual(restored.optimizer._step, trainer.optimizer._step)
            self.assertEqual(restored.env.episode_index, trainer.env.episode_index)
            self.assertEqual(restored_identity.seed, 100)
            self.assertEqual(len(restored.replay), len(trainer.replay))

            candidate = compute_losses(
                restored_agent, make_batch(batch=B, burn_in=P, train_length=T)
            )
            self.assertEqual(float(candidate.total.detach()), float(reference.total.detach()))
            self.assertEqual(
                float(candidate.behavior.actor.loss.detach()),
                float(reference.behavior.actor.loss.detach()),
            )

    def test_the_restored_optimizer_takes_the_same_next_step(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            trainer, agent, _ = build_trainer(out)
            trainer.run()
            path = trainer.checkpointer.latest_resume()

            batch = make_batch(batch=B, burn_in=P, train_length=T, seed=3)
            training_update(agent, trainer.optimizer, batch)
            expected = [p.detach().clone() for p in agent.trainable_parameters()]

            restored, restored_agent, _ = build_trainer(out / "second")
            restore_resume(restored, load_checkpoint(path))
            training_update(restored_agent, restored.optimizer, batch)

            for left, right in zip(expected, restored_agent.trainable_parameters()):
                self.assertTrue(torch.equal(left, right.detach()))

    def test_the_slow_critic_and_the_normalizer_buffers_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            trainer, agent, _ = build_trainer(out)
            trainer.run()
            path = trainer.checkpointer.latest_resume()

            self.assertNotEqual(float(agent.normalizer.hi), 0.0)

            restored, restored_agent, _ = build_trainer(out / "second")
            restore_resume(restored, load_checkpoint(path))

            self.assertEqual(float(restored_agent.normalizer.hi), float(agent.normalizer.hi))
            self.assertEqual(float(restored_agent.normalizer.lo), float(agent.normalizer.lo))
            self.assertEqual(restored_agent.normalizer.settings(), agent.normalizer.settings())

            for left, right in zip(
                agent.critic.slow.mirror.parameters(),
                restored_agent.critic.slow.mirror.parameters(),
            ):
                self.assertTrue(torch.equal(left, right))

    def test_every_generator_including_the_providers_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            trainer, agent, _ = build_trainer(out)
            trainer.run()
            path = trainer.checkpointer.latest_resume()

            restored, restored_agent, _ = build_trainer(out / "second")
            restore_resume(restored, load_checkpoint(path))

            for name, generator in agent.generators.items():
                self.assertTrue(
                    torch.equal(generator.get_state(), restored_agent.generators[name].get_state()),
                    name,
                )

            state = agent.provider.state_dict()
            self.assertIsNotNone(state["generator"])
            self.assertTrue(
                torch.equal(state["generator"], restored_agent.provider.state_dict()["generator"])
            )

            left = agent.world_model.rssm.initial(3, torch.device("cpu"))
            self.assertTrue(torch.equal(agent.provider(left), restored_agent.provider(left)))

    def test_a_model_checkpoint_cannot_be_resumed_from(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            trainer, agent, identity = build_trainer(out)
            trainer.run()

            model = trainer.checkpointer.model_path
            self.assertTrue(model.exists())

            with self.assertRaises(ValueError):
                restore_resume(trainer, load_checkpoint(model))

    def test_a_model_checkpoint_is_far_smaller_than_a_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            trainer, agent, _ = build_trainer(out)
            trainer.run()

            model = trainer.checkpointer.model_path.stat().st_size
            resume = trainer.checkpointer.latest_resume().stat().st_size
            self.assertLess(model, resume)

    def test_configuration_drift_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            trainer, _, _ = build_trainer(out)
            trainer.run()
            payload = load_checkpoint(trainer.checkpointer.latest_resume())

            other, _, _ = build_trainer(out / "second", horizon=30)

            with self.assertRaises(ValueError) as raised:
                restore_resume(other, payload)

            self.assertIn("horizon", str(raised.exception))

    def test_a_mid_episode_resume_checkpoint_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            trainer, _, identity = build_trainer(out)
            trainer.collector.collect(5)  # stops inside an episode

            payload = resume_payload(trainer, identity)
            self.assertIsNotNone(payload["replay"]["current"])

            other, _, _ = build_trainer(out / "second")

            with self.assertRaises(ValueError) as raised:
                restore_resume(other, payload)

            self.assertIn("mid-episode", str(raised.exception))


class TestRotation(unittest.TestCase):
    def test_only_the_latest_two_resume_checkpoints_are_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            trainer, _, _ = build_trainer(out, budget=EPISODE * 6, checkpoint_every=EPISODE)
            trainer.run()

            resumes = sorted(out.glob("resume-*.pt"))
            self.assertEqual(len(resumes), 2)
            self.assertTrue((out / "model-final.pt").exists())

    def test_the_final_model_checkpoint_is_always_written(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            trainer, _, _ = build_trainer(out, budget=EPISODE * 2 + 3)
            trainer.run()
            self.assertTrue((out / "model-final.pt").exists())


if __name__ == "__main__":
    unittest.main()
