import unittest

import numpy as np
import torch

from dreamer import (
    Agent,
    LaProp,
    ReplayBuffer,
    RunConfig,
    SequenceBatch,
    Transition,
    TrainingScheduler,
    apply_update,
    compute_losses,
    training_update,
)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

A, B, P, T = 2, 2, 2, 6


def make_agent(horizon: int = 5, **overrides) -> Agent:
    config = RunConfig(
        task="cartpole",
        seed=100,
        batch_size=B,
        train_length=T,
        burn_in=P,
        horizon=horizon,
        replay_capacity=2000,
        warmup_transitions=0,
        **overrides,
    )
    return Agent(action_dim=A, config=config, device="cpu")


def make_batch(
    batch: int = B,
    burn_in: int = P,
    train_length: int = T,
    seed: int = 0,
    terminal_at: int | None = None,
    mask_out: int | None = None,
) -> SequenceBatch:
    rng = np.random.default_rng(seed)
    length = burn_in + train_length

    is_last = np.zeros((batch, length), dtype=np.bool_)
    is_terminal = np.zeros((batch, length), dtype=np.bool_)
    discounts = np.ones((batch, length), dtype=np.float32)

    if terminal_at is not None:
        is_last[:, terminal_at] = True
        is_terminal[:, terminal_at] = True
        discounts[:, terminal_at] = 0.0

    loss_mask = np.ones((batch, length), dtype=np.bool_)
    loss_mask[:, :burn_in] = False

    if mask_out is not None:
        loss_mask[:, mask_out] = False

    return SequenceBatch(
        observations=rng.integers(0, 256, (batch, length + 1, 64, 64, 3), dtype=np.uint8),
        actions=rng.uniform(-1, 1, (batch, length, A)).astype(np.float32),
        rewards=rng.uniform(0, 1, (batch, length)).astype(np.float32),
        is_last=is_last,
        is_terminal=is_terminal,
        discounts=discounts,
        loss_mask=loss_mask,
    )


def perturb(agent: Agent, scale: float = 0.05) -> None:
    """rew and val are outscale 0.0: without this the advantage is identically 0 and every
    routing claim below passes vacuously."""
    generator = torch.Generator().manual_seed(0)

    for head in (agent.world_model.reward.mlp.out, agent.critic.value.mlp.out):
        with torch.no_grad():
            head.weight.add_(torch.randn(head.weight.shape, generator=generator) * scale)

    agent.critic.slow.mirror.load_state_dict(agent.critic.value.state_dict())


class TestTrainingScheduler(unittest.TestCase):
    def test_sixteen_transitions_buy_exactly_one_update_at_ratio_sixty_four(self):
        scheduler = TrainingScheduler(train_ratio=64, positions_per_update=1024)

        for _ in range(15):
            scheduler.credit()
            self.assertEqual(scheduler.take(), 0)

        scheduler.credit()
        self.assertEqual(scheduler.take(), 1)
        self.assertEqual(scheduler.credits, 0)

    def test_the_remainder_is_carried_not_discarded(self):
        scheduler = TrainingScheduler(train_ratio=100, positions_per_update=1024)
        scheduler.credit(11)
        self.assertEqual(scheduler.take(), 1)
        self.assertEqual(scheduler.credits, 1100 - 1024)

        scheduler.credit(10)
        self.assertEqual(scheduler.take(), 1)
        self.assertEqual(scheduler.credits, 76 + 1000 - 1024)

    def test_a_burst_of_credit_releases_every_update_it_bought(self):
        scheduler = TrainingScheduler(train_ratio=64, positions_per_update=1024)
        scheduler.credit(160)
        self.assertEqual(scheduler.take(), 10)

    def test_the_remainder_survives_a_round_trip(self):
        scheduler = TrainingScheduler(train_ratio=64, positions_per_update=1024)
        scheduler.credit(10)
        state = scheduler.state_dict()

        restored = TrainingScheduler(train_ratio=64, positions_per_update=1024)
        restored.load_state_dict(state)
        self.assertEqual(restored.credits, 640)

        restored.credit(6)
        self.assertEqual(restored.take(), 1)

    def test_a_restored_ratio_that_disagrees_with_the_configuration_raises(self):
        scheduler = TrainingScheduler(train_ratio=64, positions_per_update=1024)

        with self.assertRaises(ValueError):
            scheduler.load_state_dict(
                {"train_ratio": 32, "positions_per_update": 1024, "credits": 0}
            )

    def test_a_non_positive_ratio_raises(self):
        with self.assertRaises(ValueError):
            TrainingScheduler(train_ratio=0)


class TestUpdateLayout(unittest.TestCase):
    def test_the_update_is_finite_and_correctly_shaped_at_every_horizon(self):
        for horizon in (5, 15, 30):
            agent = make_agent(horizon=horizon)
            perturb(agent)
            output = compute_losses(agent, make_batch())

            starts = B * T
            self.assertEqual(output.starts, starts)
            self.assertEqual(tuple(output.imagination.actions.shape), (starts, horizon, A))
            self.assertEqual(tuple(output.imagination.feat.shape)[:2], (starts, horizon + 1))
            self.assertEqual(tuple(output.behavior.imagined.ret.shape), (starts, horizon))
            self.assertEqual(tuple(output.replay_critic.ret.shape), (B, T - 1))
            self.assertTrue(torch.isfinite(output.total))
            self.assertEqual(output.metrics["imagined_transitions"], starts * horizon)

    def test_burn_in_positions_are_never_rollout_starts_or_critic_positions(self):
        agent = make_agent()
        perturb(agent)
        output = compute_losses(agent, make_batch())

        self.assertEqual(output.starts, B * T)
        self.assertEqual(output.replay_critic.metrics["repval_positions"], float(B * (T - 1)))
        self.assertEqual(output.world.metrics["train_positions"], float(B * T))

    def test_a_terminal_position_is_excluded_from_the_starts(self):
        agent = make_agent()
        perturb(agent)
        output = compute_losses(agent, make_batch(terminal_at=P + T - 1))
        self.assertEqual(output.starts, B * T - B)

    def test_a_non_terminal_bootstrap_hole_is_rejected(self):
        agent = make_agent()
        perturb(agent)

        with self.assertRaises(ValueError) as raised:
            compute_losses(agent, make_batch(mask_out=P + 2))

        self.assertIn("bootstrap hole", str(raised.exception))

    def test_a_batch_of_the_wrong_length_raises_instead_of_misaligning(self):
        agent = make_agent()

        with self.assertRaises(ValueError):
            compute_losses(agent, make_batch(train_length=T + 1))

    def test_the_loss_scales_are_the_pinned_ones(self):
        agent = make_agent()
        self.assertEqual(agent.world_model.scales["rec"], 1.0)
        self.assertEqual(agent.world_model.scales["rew"], 1.0)
        self.assertEqual(agent.world_model.scales["con"], 1.0)
        self.assertEqual(agent.world_model.scales["dyn"], 1.0)
        self.assertEqual(agent.world_model.scales["rep"], 0.1)
        self.assertEqual(agent.critic.scales["value"], 1.0)
        self.assertEqual(agent.critic.scales["repval"], 0.3)
        self.assertEqual(agent.critic.slowreg, 1.0)

    def test_total_is_exactly_the_weighted_sum_of_the_three_parts(self):
        agent = make_agent()
        perturb(agent)
        output = compute_losses(agent, make_batch())
        expected = (
            output.world.total
            + output.behavior.loss
            + 0.3 * output.replay_critic.loss
        )
        self.assertTrue(torch.equal(output.total, expected))


class TestGradientRouting(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        perturb(self.agent)
        self.batch = make_batch()

    def _grads(self, loss):
        self.agent.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        return {
            "encoder": self.agent.world_model.encoder.convs[0].weight.grad,
            "rssm": self.agent.world_model.rssm.dyngru.kernel.grad,
            "decoder": self.agent.world_model.decoder.imgout.weight.grad,
            "actor": self.agent.actor.mean.weight.grad,
            "value": self.agent.critic.value.mlp.out.weight.grad,
            "slow": self.agent.critic.slow.mirror.mlp.out.weight.grad,
        }

    def test_the_fixture_is_not_degenerate(self):
        output = compute_losses(self.agent, self.batch)
        self.assertGreater(float(output.behavior.actor.advantage.abs().mean()), 0.0)
        self.assertNotEqual(float(output.world.metrics["rew"]), 0.0)

    def test_the_actor_loss_updates_only_the_actor(self):
        output = compute_losses(self.agent, self.batch)
        grads = self._grads(output.behavior.actor.loss)
        self.assertGreater(float(grads["actor"].abs().sum()), 0.0)

        for name in ("encoder", "rssm", "decoder", "value", "slow"):
            self.assertIsNone(grads[name], name)

    def test_the_imagined_critic_loss_updates_only_the_fast_critic(self):
        output = compute_losses(self.agent, self.batch)
        grads = self._grads(output.behavior.imagined.loss)
        self.assertGreater(float(grads["value"].abs().sum()), 0.0)

        for name in ("encoder", "rssm", "decoder", "actor", "slow"):
            self.assertIsNone(grads[name], name)

    def test_the_replay_critic_loss_reaches_the_encoder_and_the_rssm(self):
        output = compute_losses(self.agent, self.batch)
        grads = self._grads(output.replay_critic.loss)

        for name in ("encoder", "rssm", "value"):
            self.assertGreater(float(grads[name].abs().sum()), 0.0, name)

        for name in ("decoder", "actor", "slow"):
            self.assertIsNone(grads[name], name)

    def test_the_combined_loss_reaches_every_trainable_tensor_and_no_other(self):
        output = compute_losses(self.agent, self.batch)
        grads = self._grads(output.total)

        for name in ("encoder", "rssm", "decoder", "actor", "value"):
            self.assertGreater(float(grads[name].abs().sum()), 0.0, name)

        self.assertIsNone(grads["slow"])
        self.assertTrue(
            all(p.grad is None for p in self.agent.critic.slow.parameters())
        )

    def test_the_imagined_features_carry_no_graph(self):
        output = compute_losses(self.agent, self.batch)
        self.assertFalse(output.imagination.feat.requires_grad)
        self.assertFalse(output.behavior.actor.advantage.requires_grad)
        self.assertFalse(output.behavior.imagined.ret.requires_grad)
        self.assertFalse(output.replay_critic.ret.requires_grad)


class TestUpdateCardinality(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        perturb(self.agent)
        self.optimizer = LaProp(self.agent.trainable_parameters(), lr=1e-3, warmup=1)

    def test_one_optimizer_step_and_one_normalizer_update_per_batch(self):
        steps, updates = [], []
        original_step = self.optimizer.step
        original_update = self.agent.normalizer.update

        def counting_step(*args, **kwargs):
            steps.append(1)
            return original_step(*args, **kwargs)

        def counting_update(ret):
            updates.append(1)
            return original_update(ret)

        self.optimizer.step = counting_step
        self.agent.normalizer.update = counting_update

        training_update(self.agent, self.optimizer, make_batch())

        self.assertEqual(sum(steps), 1)
        self.assertEqual(sum(updates), 1)
        self.assertEqual(self.optimizer._step, 1)

    def test_the_slow_critic_advances_after_the_optimizer_step(self):
        before = [p.detach().clone() for p in self.agent.critic.slow.mirror.parameters()]
        output = compute_losses(self.agent, make_batch())
        apply_update(self.agent, self.optimizer, output)

        rate = self.agent.critic.slow.rate
        after_fast = list(self.agent.critic.value.parameters())
        after_slow = list(self.agent.critic.slow.mirror.parameters())

        moved = False

        for old, fast, slow in zip(before, after_fast, after_slow):
            expected = old.lerp(fast.detach(), rate)
            self.assertTrue(torch.allclose(slow, expected, rtol=0, atol=1e-7))
            moved = moved or not torch.equal(old, fast.detach())

        # the fast critic must actually have moved, or "post-step" and "pre-step" agree
        self.assertTrue(moved)

    def test_the_normalizer_advances_by_exactly_one_ema_step(self):
        agent = self.agent
        self.assertEqual(float(agent.normalizer.hi), 0.0)
        output = compute_losses(agent, make_batch())
        expected = agent.normalizer.rate * float(
            torch.quantile(output.behavior.imagined.ret.reshape(-1), 0.95)
        )
        self.assertAlmostEqual(
            float(agent.normalizer.hi), expected, delta=abs(expected) * 1e-5 + 1e-6
        )

    def test_the_scale_floor_of_one_is_a_valid_observation(self):
        self.assertEqual(float(self.agent.normalizer.scale), 1.0)
        self.assertFalse(self.agent.normalizer.debias)


class TestReplayBehaviour(unittest.TestCase):
    def _episode(self, replay: ReplayBuffer, length: int, start: int) -> None:
        for j in range(length):
            replay.add(
                Transition(
                    observation=np.full((4, 4, 3), (start + j) % 256, dtype=np.uint8),
                    action=np.zeros(A, dtype=np.float32),
                    reward=np.float32(1.0),
                    next_observation=np.full((4, 4, 3), (start + j + 1) % 256, dtype=np.uint8),
                    is_last=j == length - 1,
                    is_terminal=False,
                    discount=np.float32(1.0),
                )
            )

    def test_eviction_drops_whole_episodes(self):
        replay = ReplayBuffer(capacity=25, seed=0)

        for index in range(4):
            self._episode(replay, 10, start=index * 20)

        self.assertLessEqual(len(replay), 25)
        self.assertEqual(len(replay) % 10, 0)
        self.assertEqual(replay.num_complete_episodes, 2)

    def test_can_sample_is_false_until_a_long_enough_episode_closes(self):
        replay = ReplayBuffer(capacity=100, seed=0)
        self.assertFalse(replay.can_sample(8))
        self._episode(replay, 5, start=0)
        self.assertFalse(replay.can_sample(8))
        self._episode(replay, 12, start=50)
        self.assertTrue(replay.can_sample(8))


if __name__ == "__main__":
    unittest.main()
