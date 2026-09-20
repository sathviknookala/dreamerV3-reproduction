import unittest

import numpy as np
import torch

from dreamer import (
    Agent,
    Collector,
    EnvStep,
    EpisodicEnv,
    ReplayBuffer,
    RunConfig,
    UniformRandomPolicy,
    episode_seed,
    stream_seed,
)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

ACTION_DIM = 2
CLIP = 0.1


def frame(value: int) -> np.ndarray:
    return np.full((64, 64, 3), value % 256, dtype=np.uint8)


class FakeEnv:
    """Clips hard at ±0.1 so a requested action and the executed one are never the same tensor."""

    physics_substeps = 2

    def __init__(self, episode_length: int = 4, terminal: bool = False) -> None:
        self.episode_length = episode_length
        self.terminal = terminal
        self.t = 0
        self.resets = 0
        self.requested: list[np.ndarray] = []

    def reset(self) -> np.ndarray:
        self.t = 0
        self.resets += 1
        return frame(0)

    def step(self, action: np.ndarray) -> EnvStep:
        action = np.asarray(action, dtype=np.float32)
        self.requested.append(action.copy())
        executed = np.clip(action, -CLIP, CLIP).astype(np.float32)

        self.t += 1
        is_last = self.t == self.episode_length
        is_terminal = bool(is_last and self.terminal)

        return EnvStep(
            next_observation=frame(self.t),
            action=executed,
            reward=np.float32(0.5),
            is_last=is_last,
            is_terminal=is_terminal,
            discount=np.float32(0.0 if is_terminal else 1.0),
        )


def make_agent(**overrides) -> Agent:
    config = RunConfig(
        task="cartpole",
        seed=100,
        batch_size=2,
        train_length=8,
        burn_in=2,
        horizon=5,
        replay_capacity=2000,
        warmup_transitions=0,
        **overrides,
    )
    return Agent(action_dim=ACTION_DIM, config=config, device="cpu")


class RecordingRSSM:
    """Wraps observe_step so the (previous action, current embedding) pair is inspectable."""

    def __init__(self, rssm) -> None:
        self.rssm = rssm
        self.calls: list[dict] = []
        self._original = rssm.observe_step
        rssm.observe_step = self

    def __call__(self, state, action, embed, is_first=None, generator=None):
        self.calls.append(
            {
                "action": action.detach().clone(),
                "embed": embed.detach().clone(),
                "is_first": None if is_first is None else bool(is_first.item()),
            }
        )
        return self._original(state, action, embed, is_first, generator)

    def restore(self) -> None:
        self.rssm.observe_step = self._original


class TestLatentPolicyAlignment(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.env = FakeEnv(episode_length=4)
        self.replay = ReplayBuffer(capacity=100, seed=0)
        self.policy = self.agent.collection_policy()
        self.collector = Collector(self.env, self.replay, self.policy)
        self.spy = RecordingRSSM(self.agent.world_model.rssm)

    def tearDown(self):
        self.spy.restore()

    def test_posterior_sees_the_current_image_before_the_action_is_chosen(self):
        for _ in range(3):
            self.collector.collect_step()

        self.assertEqual(len(self.spy.calls), 3)

        for step, call in enumerate(self.spy.calls):
            expected = self.agent.world_model.encoder(
                torch.from_numpy(frame(step)).unsqueeze(0)
            )
            self.assertTrue(torch.allclose(call["embed"], expected))

    def test_the_actor_reads_the_state_that_absorbed_the_current_image(self):
        captured = []
        original = self.agent.actor.forward

        def spy(feat):
            captured.append(feat.detach().clone())
            return original(feat)

        self.agent.actor.forward = spy

        try:
            for _ in range(3):
                self.collector.collect_step()
                # policy.state is the posterior the image just produced; reading the carry
                # instead would choose an action one step stale
                self.assertTrue(torch.equal(captured[-1], self.policy.state.feat))
        finally:
            self.agent.actor.forward = original

        self.assertEqual(len(captured), 3)

    def test_the_conditioning_action_is_the_executed_one_not_the_requested_one(self):
        for _ in range(3):
            self.collector.collect_step()

        requested = self.env.requested
        self.assertTrue(
            any(abs(float(a).__abs__()) > CLIP for step in requested for a in step),
            "the fixture never exercised the clip, so this assertion would be vacuous",
        )

        for step in range(1, 3):
            executed = np.clip(requested[step - 1], -CLIP, CLIP)
            recorded = self.spy.calls[step]["action"].squeeze(0).numpy()
            np.testing.assert_allclose(recorded, executed, rtol=0, atol=0)

    def test_the_first_call_of_an_episode_is_flagged_and_carries_a_zero_action(self):
        for _ in range(6):
            self.collector.collect_step()

        flags = [call["is_first"] for call in self.spy.calls]
        # episode_length 4, so calls 0 and 4 open an episode
        self.assertEqual(flags, [True, False, False, False, True, False])
        self.assertTrue(bool((self.spy.calls[4]["action"] == 0).all()))

    def test_a_time_limit_resets_the_recurrence_exactly_as_a_termination_does(self):
        for terminal in (False, True):
            env = FakeEnv(episode_length=3, terminal=terminal)
            agent = make_agent()
            policy = agent.collection_policy()
            collector = Collector(env, ReplayBuffer(capacity=100, seed=0), policy)
            spy = RecordingRSSM(agent.world_model.rssm)

            try:
                for _ in range(4):
                    collector.collect_step()
            finally:
                spy.restore()

            self.assertEqual(
                [c["is_first"] for c in spy.calls],
                [True, False, False, True],
                f"terminal={terminal}",
            )

    def test_a_policy_call_without_executed_feedback_raises(self):
        observation = frame(0)
        self.policy(observation)

        with self.assertRaises(RuntimeError):
            self.policy(observation)

    def test_reset_clears_state_and_the_pending_feedback_flag(self):
        self.policy(frame(0))
        self.policy.reset()
        self.assertIsNone(self.policy.state)
        self.policy(frame(1))


class TestPolicyModes(unittest.TestCase):
    def test_evaluation_acts_on_the_mean_and_training_samples(self):
        agent = make_agent()
        observation = frame(7)
        captured = []
        original = agent.actor.forward

        def spy(feat):
            distribution = original(feat)
            captured.append(distribution)
            return distribution

        agent.actor.forward = spy

        try:
            chosen = agent.evaluation_policy(2000)(observation)
            mean = captured[-1].mean.squeeze(0).detach().numpy()
            np.testing.assert_allclose(chosen, mean, rtol=0, atol=0)

            sampled = agent.collection_policy()(observation)
            drawn_mean = captured[-1].mean.squeeze(0).detach().numpy()
            self.assertFalse(np.allclose(sampled, drawn_mean))
            # the spread must be real or "sampled != mean" would be luck, not a property
            self.assertGreater(float(captured[-1].stddev.min()), 0.0)
        finally:
            agent.actor.forward = original

    def test_two_evaluation_policies_with_the_same_seed_agree(self):
        agent = make_agent()
        runs = []

        for _ in range(2):
            policy = agent.evaluation_policy(2000)
            actions = []

            for value in range(4):
                actions.append(policy(frame(value)))
                policy.observe_executed(actions[-1])

            runs.append(np.stack(actions))

        np.testing.assert_allclose(runs[0], runs[1], rtol=0, atol=0)

    def test_two_evaluation_policies_do_not_share_a_stream_with_collection(self):
        agent = make_agent()
        before = agent.generators["collect"].get_state().clone()
        policy = agent.evaluation_policy(2000)

        for value in range(3):
            policy(frame(value))
            policy.observe_executed(np.zeros(ACTION_DIM, dtype=np.float32))

        self.assertTrue(torch.equal(before, agent.generators["collect"].get_state()))


class TestAgentParameters(unittest.TestCase):
    def test_the_optimizer_list_holds_every_trainable_tensor_exactly_once(self):
        agent = make_agent()
        params = agent.trainable_parameters()
        self.assertEqual(len({id(p) for p in params}), len(params))

        expected = (
            set(map(id, agent.world_model.parameters()))
            | set(map(id, agent.actor.parameters()))
            | set(map(id, agent.critic.value.parameters()))
        )
        self.assertEqual({id(p) for p in params}, expected)

    def test_the_slow_critic_is_excluded(self):
        agent = make_agent()
        slow = {id(p) for p in agent.critic.slow.parameters()}
        self.assertTrue(slow)
        self.assertFalse(slow & {id(p) for p in agent.trainable_parameters()})

    def test_the_counts_sum_to_the_trainable_total(self):
        agent = make_agent()
        counts = agent.parameter_counts()
        self.assertEqual(
            counts.total, sum(p.numel() for p in agent.trainable_parameters())
        )


class TestStreams(unittest.TestCase):
    def test_every_named_stream_gets_a_distinct_seed(self):
        seeds = {
            name: stream_seed(100, name)
            for name in ("init", "env", "collect", "replay", "imagine", "provider", "eval")
        }
        self.assertEqual(len(set(seeds.values())), len(seeds))

    def test_an_unknown_stream_name_raises_instead_of_silently_colliding(self):
        with self.assertRaises(KeyError):
            stream_seed(100, "not-a-stream")

    def test_episode_seeds_advance_with_the_episode_index(self):
        seeds = [episode_seed(100, i) for i in range(50)]
        self.assertEqual(len(set(seeds)), 50)
        self.assertEqual(episode_seed(100, 7), episode_seed(100, 7))

    def test_the_provider_generator_is_not_recreated_between_calls(self):
        agent = make_agent()
        state = agent.world_model.rssm.initial(3, torch.device("cpu"))
        first = agent.provider(state)
        second = agent.provider(state)
        self.assertFalse(torch.allclose(first, second))


class TestEpisodicEnv(unittest.TestCase):
    def test_the_seed_schedule_is_a_pure_function_of_the_persisted_index(self):
        env = EpisodicEnv.__new__(EpisodicEnv)
        env.domain, env.task = "cartpole", "swingup"
        env.base_seed, env.episode_index = 100, 3
        self.assertEqual(env.next_seed, episode_seed(100, 3))

        env.load_state_dict(
            {"domain": "cartpole", "task": "swingup", "base_seed": 100, "episode_index": 11}
        )
        self.assertEqual(env.next_seed, episode_seed(100, 11))

    def test_restoring_a_different_task_raises(self):
        env = EpisodicEnv.__new__(EpisodicEnv)
        env.domain, env.task = "cartpole", "swingup"
        env.base_seed, env.episode_index = 100, 0

        with self.assertRaises(ValueError):
            env.load_state_dict(
                {"domain": "walker", "task": "walk", "base_seed": 100, "episode_index": 0}
            )


class TestCollectorLifecycle(unittest.TestCase):
    def test_a_stateless_policy_still_collects(self):
        env = FakeEnv(episode_length=3)
        replay = ReplayBuffer(capacity=100, seed=0)
        policy = UniformRandomPolicy(
            np.full(ACTION_DIM, -1.0, dtype=np.float32),
            np.full(ACTION_DIM, 1.0, dtype=np.float32),
            seed=0,
        )
        Collector(env, replay, policy).collect(5)
        self.assertEqual(len(replay), 5)

    def test_replay_stores_the_executed_action(self):
        env = FakeEnv(episode_length=8)
        replay = ReplayBuffer(capacity=100, seed=0)
        agent = make_agent()
        collector = Collector(env, replay, agent.collection_policy())

        for _ in range(4):
            transition = collector.collect_step()
            self.assertLessEqual(float(np.abs(transition.action).max()), CLIP + 1e-6)

    def test_switching_the_policy_resets_the_new_one(self):
        env = FakeEnv(episode_length=8)
        replay = ReplayBuffer(capacity=100, seed=0)
        agent = make_agent()
        collector = Collector(
            env,
            replay,
            UniformRandomPolicy(
                np.full(ACTION_DIM, -1.0, dtype=np.float32),
                np.full(ACTION_DIM, 1.0, dtype=np.float32),
                seed=0,
            ),
        )
        collector.collect(2)

        policy = agent.collection_policy()
        spy = RecordingRSSM(agent.world_model.rssm)

        try:
            collector.set_policy(policy)
            collector.collect_step()
        finally:
            spy.restore()

        self.assertTrue(spy.calls[0]["is_first"])
        self.assertEqual(len(replay), 3)


if __name__ == "__main__":
    unittest.main()
