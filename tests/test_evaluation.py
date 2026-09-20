import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dreamer import (
    Agent,
    EvaluationResult,
    PeriodicEvaluator,
    ReplayBuffer,
    RunConfig,
    evaluate_agent,
    evaluate_episode,
)
from dreamer.evaluation import EpisodeResult, _agent_fingerprint, _fingerprints_match

from test_training import perturb

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# cartpole swingup has a one-dimensional action; the agent must be built to match
ACTION_DIM = 1
MAX_STEPS = 12


def make_agent(**overrides) -> Agent:
    config = RunConfig(
        task="cartpole",
        seed=100,
        batch_size=2,
        train_length=6,
        burn_in=2,
        horizon=5,
        eval_episodes=2,
        eval_seed_base=2000,
        warmup_transitions=0,
        **overrides,
    )
    return Agent(action_dim=ACTION_DIM, config=config, device="cpu")


class TestEvaluationResult(unittest.TestCase):
    def setUp(self):
        self.result = EvaluationResult(
            env_step=25_000,
            episodes=[
                EpisodeResult(2000, 10.0, 1000),
                EpisodeResult(2001, 20.0, 1000),
                EpisodeResult(2002, 30.0, 1000),
            ],
            seconds=4.0,
        )

    def test_individual_returns_and_lengths_are_kept_not_only_the_mean(self):
        self.assertEqual(self.result.returns, [10.0, 20.0, 30.0])
        self.assertEqual(self.result.lengths, [1000, 1000, 1000])
        self.assertEqual(self.result.steps, 3000)

    def test_the_summary_carries_every_recorded_field(self):
        summary = self.result.summary()
        self.assertEqual(summary["return_mean"], 20.0)
        self.assertAlmostEqual(summary["return_std"], float(np.std([10.0, 20.0, 30.0])))
        self.assertEqual(summary["seeds"], [2000, 2001, 2002])
        self.assertEqual(summary["eval_steps"], 3000)
        json.dumps(summary)


class TestFingerprint(unittest.TestCase):
    def test_a_moved_generator_is_detected(self):
        agent = make_agent()
        before = _agent_fingerprint(agent)
        self.assertTrue(_fingerprints_match(before, _agent_fingerprint(agent)))

        torch.rand(4, generator=agent.generators["collect"])
        self.assertFalse(_fingerprints_match(before, _agent_fingerprint(agent)))

    def test_a_moved_parameter_is_detected(self):
        agent = make_agent()
        before = _agent_fingerprint(agent)

        with torch.no_grad():
            agent.actor.mean.weight.add_(1.0)

        self.assertFalse(_fingerprints_match(before, _agent_fingerprint(agent)))

    def test_a_moved_normalizer_buffer_is_detected(self):
        agent = make_agent()
        before = _agent_fingerprint(agent)
        agent.normalizer.update(torch.arange(10.0))
        self.assertFalse(_fingerprints_match(before, _agent_fingerprint(agent)))


class TestEvaluationIsolation(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        perturb(self.agent)
        self.config = self.agent.config

    def test_evaluation_changes_no_parameter_buffer_or_training_stream(self):
        before = _agent_fingerprint(self.agent)
        result = evaluate_agent(
            self.agent,
            "cartpole",
            "swingup",
            self.config.eval_seeds,
            max_steps=MAX_STEPS,
        )
        self.assertTrue(_fingerprints_match(before, _agent_fingerprint(self.agent)))
        self.assertEqual(len(result.episodes), 2)
        self.assertEqual(result.lengths, [MAX_STEPS, MAX_STEPS])

    def test_evaluation_touches_neither_replay_nor_the_step_counters(self):
        replay = ReplayBuffer(capacity=100, seed=0)
        before = replay.stats()
        evaluate_agent(
            self.agent, "cartpole", "swingup", (2000,), max_steps=MAX_STEPS
        )
        self.assertEqual(replay.stats(), before)

    def test_the_module_is_returned_to_its_previous_train_flag(self):
        self.agent.train()
        evaluate_agent(self.agent, "cartpole", "swingup", (2000,), max_steps=MAX_STEPS)
        self.assertTrue(self.agent.training)

        self.agent.eval()
        evaluate_agent(self.agent, "cartpole", "swingup", (2000,), max_steps=MAX_STEPS)
        self.assertFalse(self.agent.training)

    def test_the_same_seed_gives_the_same_episode_twice(self):
        first, _ = evaluate_episode(
            self.agent, "cartpole", "swingup", 2000, max_steps=MAX_STEPS
        )
        second, _ = evaluate_episode(
            self.agent, "cartpole", "swingup", 2000, max_steps=MAX_STEPS
        )
        self.assertEqual(first.episode_return, second.episode_return)

    def test_different_seeds_give_different_episodes(self):
        first, _ = evaluate_episode(
            self.agent, "cartpole", "swingup", 2000, max_steps=MAX_STEPS
        )
        second, _ = evaluate_episode(
            self.agent, "cartpole", "swingup", 2001, max_steps=MAX_STEPS
        )
        self.assertNotEqual(first.episode_return, second.episode_return)

    def test_the_isolation_guard_is_armed(self):
        # the guard must be able to fire, or "evaluation is isolated" is an untested claim
        original = self.agent.evaluation_policy

        def leaking(seed):
            policy = original(seed)
            policy.generator = self.agent.generators["collect"]
            return policy

        self.agent.evaluation_policy = leaking

        with self.assertRaises(RuntimeError):
            evaluate_agent(
                self.agent, "cartpole", "swingup", (2000,), max_steps=MAX_STEPS
            )


class TestVideo(unittest.TestCase):
    def test_only_the_first_seed_is_recorded(self):
        agent = make_agent()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.mp4"
            result = evaluate_agent(
                agent,
                "cartpole",
                "swingup",
                (2000, 2001),
                video_path=path,
                max_steps=MAX_STEPS,
            )
            self.assertIsNotNone(result.video)
            self.assertTrue(Path(result.video).exists())
            self.assertEqual(len(list(Path(directory).iterdir())), 1)


class TestPeriodicEvaluator(unittest.TestCase):
    def test_every_evaluation_is_appended_to_one_jsonl(self):
        agent = make_agent()

        with tempfile.TemporaryDirectory() as directory:
            evaluator = PeriodicEvaluator(agent.config, directory)
            evaluator.config.eval_video = False

            for env_step in (25_000, 50_000):
                result = evaluator(agent, env_step)
                self.assertEqual(result.env_step, env_step)

            lines = (Path(directory) / "evaluations.jsonl").read_text().strip().split("\n")
            self.assertEqual(len(lines), 2)
            self.assertEqual(
                [json.loads(line)["env_step"] for line in lines], [25_000, 50_000]
            )


if __name__ == "__main__":
    unittest.main()
