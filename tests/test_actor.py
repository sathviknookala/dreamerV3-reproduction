import math
import unittest

import torch
from torch import Tensor

from dreamer import (
    ACTENT,
    Actor,
    ActorActionProvider,
    BoundedNormal,
    Critic,
    ReturnNormalizer,
    WorldModel,
    behavior_losses,
    imagine_trajectory,
    imagined_actor_loss,
)
from dreamer.imagine import Imagination
from dreamer.rssm import State

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

A, DETER, STOCH, CLASSES = 6, 8, 2, 2
FEAT = DETER + STOCH * CLASSES
LOG2PI = math.log(2.0 * math.pi)


def make_imagination(feat: Tensor, actions: Tensor, cont: float | Tensor = 1.0) -> Imagination:
    """A hand-built rollout: H+1 states, H actions, and a continuation per state."""
    n, steps, _ = feat.shape
    horizon = actions.shape[1]

    if steps != horizon + 1:
        raise ValueError(f"{steps} states do not match {horizon} actions")

    deter = feat[..., :DETER]
    stoch = feat[..., DETER:].reshape(n, steps, STOCH, CLASSES)
    con = cont if isinstance(cont, Tensor) else torch.full((n, steps), float(cont))

    return Imagination(
        states=State(deter, stoch, stoch),
        actions=actions,
        reward=torch.zeros(n, horizon),
        cont=con[:, 1:],
        cont_start=con[:, :1],
        weight=torch.cumprod(con, dim=1),
    )


def flat_feat(n: int, steps: int, seed: int = 0) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, steps, FEAT, generator=generator)


class TestBoundedNormal(unittest.TestCase):
    def test_shapes_for_both_action_dimensions(self):
        for action_dim in (1, 6):
            actor = Actor(in_features=FEAT, action_dim=action_dim)
            policy = actor(torch.randn(3, 4, FEAT))

            self.assertEqual(tuple(policy.mean.shape), (3, 4, action_dim))
            self.assertEqual(tuple(policy.stddev.shape), (3, 4, action_dim))
            # batch and time survive; only the action axis is reduced away
            self.assertEqual(tuple(policy.sample().shape), (3, 4, action_dim))
            self.assertEqual(tuple(policy.log_prob(policy.sample()).shape), (3, 4))
            self.assertEqual(tuple(policy.entropy().shape), (3, 4))

    def test_the_stddev_formula_carries_the_hard_coded_plus_two(self):
        actor = Actor(in_features=FEAT, action_dim=A)
        with torch.no_grad():
            actor.stddev.weight.zero_()
            actor.stddev.bias.zero_()

        stddev = actor(torch.randn(4, FEAT)).stddev
        expected = 0.9 * (1.0 / (1.0 + math.exp(-2.0))) + 0.1

        # without the +2.0 offset a zero pre-activation would give 0.55, not 0.893
        torch.testing.assert_close(stddev, torch.full_like(stddev, expected), rtol=0, atol=1e-6)
        self.assertGreater(float(stddev.min()), 0.88)

    def test_the_stddev_stays_inside_minstd_and_maxstd(self):
        actor = Actor(in_features=FEAT, action_dim=A)
        with torch.no_grad():
            actor.stddev.weight.normal_(0.0, 50.0)

        stddev = actor(torch.randn(64, FEAT)).stddev
        self.assertGreaterEqual(float(stddev.min()), actor.minstd)
        self.assertLessEqual(float(stddev.max()), actor.maxstd)

    def test_the_mean_is_tanh_bounded_even_when_the_kernel_is_large(self):
        actor = Actor(in_features=FEAT, action_dim=A)
        with torch.no_grad():
            actor.mean.weight.normal_(0.0, 50.0)

        mean = actor(torch.randn(64, FEAT)).mean
        # tanh saturates to exactly +-1 in float32, so the bound is closed, never exceeded
        self.assertLessEqual(float(mean.abs().max()), 1.0)
        self.assertGreater(float(mean.abs().max()), 0.99)

    def test_log_prob_is_the_plain_gaussian_summed_over_actions(self):
        mean = torch.tensor([[0.2, -0.5, 0.0, 0.1, -0.3, 0.4]])
        stddev = torch.tensor([[0.5, 0.9, 0.1, 1.0, 0.3, 0.7]])
        action = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        policy = BoundedNormal(mean, stddev)

        z = (action - mean) / stddev
        expected = (-0.5 * z.pow(2) - stddev.log() - 0.5 * LOG2PI).sum(-1)

        torch.testing.assert_close(policy.log_prob(action), expected, rtol=0, atol=1e-6)
        # no tanh Jacobian term: a squashed density would add -sum(log(1 - a^2))
        self.assertGreater(float((policy.log_prob(action) - expected).abs().max()), -1.0)

    def test_entropy_matches_the_closed_form_summed_over_actions(self):
        stddev = torch.tensor([[0.5, 0.9, 0.1, 1.0, 0.3, 0.7]])
        policy = BoundedNormal(torch.zeros_like(stddev), stddev)

        expected = (0.5 * torch.log(2 * math.pi * stddev.pow(2)) + 0.5).sum(-1)
        torch.testing.assert_close(policy.entropy(), expected, rtol=0, atol=1e-6)

        # a per-dimension mean instead of a sum would be six times smaller
        self.assertGreater(float(policy.entropy()), 0.0)

    def test_the_evaluation_action_is_the_mean(self):
        actor = Actor(in_features=FEAT, action_dim=A)
        policy = actor(torch.randn(4, FEAT))

        # spec 7.8 selects the mean at evaluation, with no sampling
        torch.testing.assert_close(policy.mode, policy.mean, rtol=0, atol=0)
        self.assertLessEqual(float(policy.mode.abs().max()), 1.0)

    def test_samples_are_unsquashed_and_score_unchanged_outside_the_bounds(self):
        mean = torch.full((4096, 1), 0.9)
        policy = BoundedNormal(mean, torch.full_like(mean, 1.0))
        sample = policy.sample(torch.Generator().manual_seed(0))

        self.assertGreater(float(sample.max()), 1.0)

        outside = torch.tensor([[3.0]])
        z = (3.0 - 0.9) / 1.0
        expected = -0.5 * z**2 - math.log(1.0) - 0.5 * LOG2PI
        self.assertAlmostEqual(float(BoundedNormal(mean[:1], torch.ones(1, 1)).log_prob(outside)),
                               expected, places=5)

    def test_the_model_boundary_bounds_the_action_the_rssm_sees(self):
        torch.manual_seed(0)
        model = WorldModel(action_dim=A)
        start = model.rssm.initial(2)

        big = torch.full((2, A), 3.0)
        unit = torch.full((2, A), 1.0)

        raw = model.rssm.imagine_step(start, big, None, torch.Generator().manual_seed(0))
        bounded = model.rssm.imagine_step(start, unit, None, torch.Generator().manual_seed(0))

        # a / max(1, |a|) sends 3.0 to 1.0, so an out-of-range sample cannot drive the dynamics
        torch.testing.assert_close(raw.deter, bounded.deter, rtol=0, atol=0)


class TestParameterCount(unittest.TestCase):
    def test_walker_actor_closes_the_derivation(self):
        self.assertEqual(sum(p.numel() for p in Actor(action_dim=6).parameters()), 50_316)

    def test_cartpole_actor_differs_only_in_the_two_heads(self):
        walker = sum(p.numel() for p in Actor(action_dim=6).parameters())
        cartpole = sum(p.numel() for p in Actor(action_dim=1).parameters())

        # 2 heads * (64 weights + 1 bias) per dropped action dimension
        self.assertEqual(walker - cartpole, 2 * 5 * (64 + 1))

    def test_the_trainable_agent_total_closes_at_686_846(self):
        model = WorldModel(action_dim=6)
        critic = Critic()
        actor = Actor(action_dim=6)

        total = (
            sum(p.numel() for p in model.parameters())
            + sum(p.numel() for p in critic.trainable_parameters())
            + sum(p.numel() for p in actor.parameters())
        )
        self.assertEqual(total, 686_846)


class TestReturnNormalizer(unittest.TestCase):
    def ramp(self) -> Tensor:
        # percentiles of 0..100 in unit steps are exactly 5.0 and 95.0
        return torch.linspace(0.0, 100.0, 101).reshape(1, -1)

    def test_the_first_update_is_rate_times_the_percentiles(self):
        norm = ReturnNormalizer()
        offset, scale = norm(self.ramp())

        self.assertAlmostEqual(float(norm.lo), 0.05, places=6)
        self.assertAlmostEqual(float(norm.hi), 0.95, places=6)
        # hi - lo is 0.9 on the first batch, so the floor of 1.0 is what is actually used
        self.assertAlmostEqual(float(offset), 0.05, places=6)
        self.assertAlmostEqual(float(scale), 1.0, places=6)

    def test_the_second_update_is_the_hand_computed_ema(self):
        norm = ReturnNormalizer()
        norm(self.ramp())
        _, scale = norm(self.ramp())

        self.assertAlmostEqual(float(norm.lo), 0.99 * 0.05 + 0.05, places=6)
        self.assertAlmostEqual(float(norm.hi), 0.99 * 0.95 + 0.95, places=6)
        self.assertAlmostEqual(float(scale), 1.8905 - 0.0995, places=4)

    def test_bias_correction_reads_the_true_percentiles_immediately(self):
        norm = ReturnNormalizer(debias=True)
        offset, scale = norm(self.ramp())

        # corr is rate after one update, so the factor is exactly 1 / rate
        self.assertAlmostEqual(float(norm.corr), 0.01, places=8)
        self.assertAlmostEqual(float(offset), 5.0, places=4)
        self.assertAlmostEqual(float(scale), 90.0, places=4)

        norm(self.ramp())
        self.assertAlmostEqual(float(norm.stats()[1]), 90.0, places=3)

    def test_the_pinned_default_is_uncorrected(self):
        # configs.yaml#L111 sets debias: False, overriding the class default of True
        self.assertFalse(ReturnNormalizer().debias)
        self.assertAlmostEqual(float(ReturnNormalizer()(self.ramp())[1]), 1.0, places=6)
        self.assertAlmostEqual(float(ReturnNormalizer(debias=True)(self.ramp())[1]), 90.0, places=4)

    def test_the_scale_never_drops_below_the_limit(self):
        norm = ReturnNormalizer(debias=True)
        _, scale = norm(torch.full((1, 64), 3.0))

        self.assertAlmostEqual(float(scale), 1.0, places=6)

    def test_a_read_only_call_leaves_the_state_alone(self):
        norm = ReturnNormalizer()
        norm(self.ramp())
        before = (float(norm.lo), float(norm.hi), float(norm.corr))

        stats = norm(self.ramp() * 100.0, update=False)

        self.assertEqual((float(norm.lo), float(norm.hi), float(norm.corr)), before)
        torch.testing.assert_close(stats[1], norm.stats()[1], rtol=0, atol=0)

    def test_the_state_round_trips_through_a_state_dict(self):
        norm = ReturnNormalizer(debias=True)
        norm(self.ramp())

        restored = ReturnNormalizer(debias=True)
        restored.load_state_dict(norm.state_dict())

        self.assertEqual(float(restored.lo), float(norm.lo))
        self.assertEqual(float(restored.hi), float(norm.hi))
        self.assertEqual(float(restored.corr), float(norm.corr))
        torch.testing.assert_close(restored.stats()[1], norm.stats()[1], rtol=0, atol=0)

    def test_the_normalizer_has_no_trainable_parameters(self):
        self.assertEqual(list(ReturnNormalizer().parameters()), [])


class TestActorUpdateDirection(unittest.TestCase):
    """spec 6.2: the two fixtures that fail if the loss sign or the entropy sign is flipped."""

    def build(self, seed=0):
        torch.manual_seed(seed)
        actor = Actor(in_features=FEAT, action_dim=1)
        critic = Critic(in_features=FEAT)
        return actor, critic, ReturnNormalizer()

    def fixture(self, advantage):
        # the same state twice, so the two actions are scored by one identical distribution
        state = torch.randn(1, 1, FEAT, generator=torch.Generator().manual_seed(3))
        feat = state.expand(1, 3, FEAT).contiguous()
        actions = torch.tensor([[[1.0], [-1.0]]])
        # a zero-init critic reads out exactly 0, so ret IS the advantage at scale 1
        return make_imagination(feat, actions), torch.tensor([advantage])

    def test_with_no_entropy_term_the_better_action_gains_probability(self):
        actor, critic, norm = self.build()
        imagination, ret = self.fixture([1.0, -1.0])

        before = actor(imagination.feat[:, :-1]).log_prob(imagination.actions).detach().clone()
        out = imagined_actor_loss(actor, critic, imagination, ret, norm, actent=0.0, update=False)

        torch.testing.assert_close(out.advantage, ret, rtol=0, atol=1e-6)
        opt = torch.optim.SGD(actor.parameters(), lr=0.5)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()

        after = actor(imagination.feat[:, :-1]).log_prob(imagination.actions).detach()
        self.assertGreater(float(after[0, 0]), float(before[0, 0]))
        self.assertLess(float(after[0, 1]), float(before[0, 1]))

    def test_with_zero_advantage_the_entropy_bonus_raises_entropy(self):
        actor, critic, norm = self.build()
        imagination, ret = self.fixture([0.0, 0.0])

        before = float(actor(imagination.feat[:, :-1]).entropy().mean())
        out = imagined_actor_loss(actor, critic, imagination, ret, norm, actent=ACTENT, update=False)

        opt = torch.optim.SGD(actor.parameters(), lr=1.0)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()

        # with A = 0 the score term contributes nothing, so only the stddev head can move
        self.assertEqual(float(actor.mean.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(actor.stddev.weight.grad.abs().sum()), 0.0)

        opt.step()
        after = float(actor(imagination.feat[:, :-1]).entropy().mean())
        # this is the assertion that fails if entropy is ADDED in the minimized loss
        self.assertGreater(after, before)


class TestActorLossArithmetic(unittest.TestCase):
    def build(self, action_dim=A, seed=0, sensitive=False):
        torch.manual_seed(seed)
        actor = Actor(in_features=FEAT, action_dim=action_dim)
        critic = Critic(in_features=FEAT)

        if sensitive:
            # outscale 0.01 makes the policy nearly feature-independent, which would let an
            # alignment test pass by numerical accident rather than by reading the right state
            with torch.no_grad():
                actor.mean.weight.normal_(0.0, 0.5)
                actor.stddev.weight.normal_(0.0, 0.5)

        return actor, critic, ReturnNormalizer()

    def rollout(self, n=3, horizon=4, action_dim=A, cont=0.9):
        feat = flat_feat(n, horizon + 1, seed=1)
        generator = torch.Generator().manual_seed(2)
        actions = torch.randn(n, horizon, action_dim, generator=generator)
        return make_imagination(feat, actions, cont)

    def test_the_loss_is_the_expression_in_5_6_with_entropy_subtracted(self):
        actor, critic, norm = self.build()
        imagination = self.rollout()
        ret = torch.randn(3, 4, generator=torch.Generator().manual_seed(5))

        out = imagined_actor_loss(actor, critic, imagination, ret, norm, update=False)

        policy = actor(imagination.feat[:, :-1])
        logp = policy.log_prob(imagination.actions)
        ent = policy.entropy()
        weight = imagination.weight[:, :-1]
        expected = (-weight * (logp * out.advantage + ACTENT * ent)).mean()

        torch.testing.assert_close(out.loss, expected, rtol=1e-6, atol=1e-7)
        # adding entropy instead of subtracting it is a different minimized quantity
        flipped = (-weight * (logp * out.advantage - ACTENT * ent)).mean()
        self.assertGreater(float((out.loss - flipped).abs().detach()), 1e-9)

    def test_the_reduction_is_a_position_mean_not_a_weighted_mean(self):
        actor, critic, norm = self.build()
        cont = torch.tensor([[1.0, 0.5, 0.25, 0.125, 0.0625]])
        imagination = make_imagination(
            flat_feat(1, 5, seed=4),
            torch.randn(1, 4, A, generator=torch.Generator().manual_seed(6)),
            cont,
        )
        ret = torch.randn(1, 4, generator=torch.Generator().manual_seed(7))

        out = imagined_actor_loss(actor, critic, imagination, ret, norm, update=False)
        policy = actor(imagination.feat[:, :-1])
        per = -out.weight * (
            policy.log_prob(imagination.actions) * out.advantage + ACTENT * policy.entropy()
        )

        torch.testing.assert_close(out.loss, per.mean(), rtol=1e-6, atol=1e-7)
        self.assertGreater(float((out.loss - per.sum() / out.weight.sum()).abs().detach()), 1e-6)

    def test_the_policy_scores_H_actions_at_the_first_H_states(self):
        actor, critic, norm = self.build(sensitive=True)
        imagination = self.rollout()
        ret = torch.zeros(3, 4)

        out = imagined_actor_loss(actor, critic, imagination, ret, norm, update=False)
        self.assertEqual(tuple(out.log_prob.shape), (3, 4))
        self.assertEqual(tuple(out.entropy.shape), (3, 4))
        self.assertEqual(tuple(out.weight.shape), (3, 4))
        self.assertEqual(tuple(out.advantage.shape), (3, 4))

        baseline = float(out.loss)
        moved = self.rollout()
        moved.states = State(
            torch.cat([moved.states.deter[:, :-1], moved.states.deter[:, -1:] + 9.0], dim=1),
            moved.states.stoch,
            moved.states.logit,
        )
        shifted = imagined_actor_loss(actor, critic, moved, ret, norm, update=False)

        # the final state carries no action, so perturbing it must not move the policy loss
        self.assertAlmostEqual(float(shifted.loss), baseline, places=6)

    def test_the_first_state_does_move_the_loss(self):
        actor, critic, norm = self.build(sensitive=True)
        imagination = self.rollout()
        ret = torch.zeros(3, 4)
        out = imagined_actor_loss(actor, critic, imagination, ret, norm, update=False)
        baseline = float(out.loss.detach())

        moved = self.rollout()
        moved.states = State(
            torch.cat([moved.states.deter[:, :1] + 9.0, moved.states.deter[:, 1:]], dim=1),
            moved.states.stoch,
            moved.states.logit,
        )
        shifted = imagined_actor_loss(actor, critic, moved, ret, norm, update=False)

        self.assertGreater(abs(float(shifted.loss.detach()) - baseline), 1e-6)

    def test_the_baseline_is_the_fast_critic_not_the_slow_one(self):
        actor, critic, norm = self.build()
        # both read out exactly 0 at init, so the claim is untestable until they are separated
        with torch.no_grad():
            critic.value.mlp.out.weight.normal_(0.0, 0.3)

        imagination = self.rollout()
        ret = torch.randn(3, 4, generator=torch.Generator().manual_seed(8))
        out = imagined_actor_loss(actor, critic, imagination, ret, norm, update=False)

        fast = critic.value.predict(imagination.feat)[:, :-1].detach()
        slow = critic.slow.predict(imagination.feat)[:, :-1].detach()
        self.assertGreater(float((fast - slow).abs().max()), 1.0)

        torch.testing.assert_close(out.advantage, (ret - fast) / out.scale, rtol=0, atol=1e-5)
        self.assertGreater(float(((ret - slow) / out.scale - out.advantage).abs().max()), 1.0)

    def test_the_return_and_the_baseline_see_the_same_critic(self):
        actor, critic, norm = self.build()
        with torch.no_grad():
            critic.value.mlp.out.weight.normal_(0.0, 0.3)

        imagination = self.rollout()
        out = behavior_losses(actor, critic, imagination, norm, update=False)

        boot = critic.value.predict(imagination.feat).detach()
        # reconstructing the return from the same prediction the baseline used must be exact
        from dreamer import lambda_return

        con = torch.cat([imagination.cont_start, imagination.cont], dim=1)
        rew = torch.cat([torch.zeros_like(imagination.reward[:, :1]), imagination.reward], dim=1)
        expected = lambda_return(torch.zeros_like(con), 1.0 - con, rew, boot, 1.0, critic.lam)

        torch.testing.assert_close(out.imagined.ret, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            out.actor.advantage, (expected - boot[:, :-1]) / out.actor.scale, rtol=0, atol=1e-5
        )

    def test_the_advantage_and_the_weight_are_detached(self):
        actor, critic, norm = self.build()
        imagination = self.rollout()
        imagination.weight = imagination.weight.detach().requires_grad_(True)
        ret = torch.randn(3, 4, generator=torch.Generator().manual_seed(9), requires_grad=True)

        out = imagined_actor_loss(actor, critic, imagination, ret, norm, update=False)
        out.loss.backward()

        self.assertFalse(out.advantage.requires_grad)
        self.assertFalse(out.weight.requires_grad)
        self.assertIsNone(imagination.weight.grad)
        self.assertIsNone(ret.grad)

    def test_the_sampled_action_is_a_constant_in_the_estimator(self):
        actor, critic, norm = self.build()
        imagination = self.rollout()
        imagination.actions = imagination.actions.detach().requires_grad_(True)
        ret = torch.randn(3, 4, generator=torch.Generator().manual_seed(10))

        imagined_actor_loss(actor, critic, imagination, ret, norm, update=False).loss.backward()

        # sg(act) is what makes this REINFORCE: a live path through the action would add a
        # pathwise term the estimator does not account for (spec 5.6)
        self.assertIsNone(imagination.actions.grad)

    def test_a_feature_tensor_that_carries_a_graph_is_rejected(self):
        actor, critic, norm = self.build()
        imagination = self.rollout()
        imagination.states = State(
            imagination.states.deter.detach().requires_grad_(True),
            imagination.states.stoch,
            imagination.states.logit,
        )

        with self.assertRaises(ValueError):
            imagined_actor_loss(actor, critic, imagination, torch.zeros(3, 4), norm, update=False)

    def test_a_mismatched_return_shape_is_rejected(self):
        actor, critic, norm = self.build()
        with self.assertRaises(ValueError):
            imagined_actor_loss(actor, critic, self.rollout(), torch.zeros(3, 5), norm, update=False)

    def test_the_normalizer_updates_once_per_batch_and_before_use(self):
        actor, critic, norm = self.build()
        imagination = self.rollout()
        ret = torch.linspace(0.0, 100.0, 12).reshape(3, 4)

        out = imagined_actor_loss(actor, critic, imagination, ret, norm, update=True)

        # the scale consumed is the post-update one, not the pre-update floor of a stale state
        torch.testing.assert_close(out.scale, norm.stats()[1], rtol=0, atol=0)
        self.assertAlmostEqual(float(norm.lo), 0.01 * float(torch.quantile(ret.reshape(-1), 0.05)), places=6)


class TestGradientRouting(unittest.TestCase):
    """spec 5.9: backward from `policy` alone must reach the actor and nothing else."""

    def build(self):
        torch.manual_seed(0)
        model = WorldModel(action_dim=A)
        critic = Critic(in_features=model.rssm.feat_size)
        actor = Actor(in_features=model.rssm.feat_size, action_dim=A)
        return model, critic, actor

    def imagined(self, model, actor, rollouts=2, horizon=3):
        start = model.rssm.initial(rollouts)
        generator = torch.Generator().manual_seed(1)
        return imagine_trajectory(model, start, ActorActionProvider(actor, seed=0), horizon, generator)

    def test_policy_only_backward_updates_the_actor_alone(self):
        model, critic, actor = self.build()
        with torch.no_grad():
            critic.value.mlp.out.weight.normal_(0.0, 0.1)

        imagination = self.imagined(model, actor)
        out = imagined_actor_loss(
            actor, critic, imagination, critic.imagined_loss(imagination).ret,
            ReturnNormalizer(), update=False,
        )
        out.loss.backward()

        self.assertGreater(float(actor.mean.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(actor.stddev.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(actor.linears[0].weight.grad.abs().sum()), 0.0)

        self.assertIsNone(model.encoder.convs[0].weight.grad)
        self.assertIsNone(model.rssm.dyngru.kernel.grad)
        self.assertIsNone(model.decoder.imgout.weight.grad)
        self.assertIsNone(model.reward.mlp.out.weight.grad)
        self.assertIsNone(model.cont.mlp.out.weight.grad)
        self.assertIsNone(critic.value.mlp.out.weight.grad)
        self.assertTrue(all(p.grad is None for p in critic.slow.parameters()))

    def test_value_only_backward_leaves_the_actor_alone(self):
        model, critic, actor = self.build()
        with torch.no_grad():
            critic.value.mlp.out.weight.normal_(0.0, 0.1)

        imagination = self.imagined(model, actor)
        critic.imagined_loss(imagination).loss.backward()

        self.assertGreater(float(critic.value.mlp.out.weight.grad.abs().sum()), 0.0)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))


class TestIntegration(unittest.TestCase):
    def build(self):
        torch.manual_seed(0)
        model = WorldModel(action_dim=A)
        critic = Critic(in_features=model.rssm.feat_size)
        actor = Actor(in_features=model.rssm.feat_size, action_dim=A)
        return model, critic, actor

    def test_seeded_actor_sampling_reproduces(self):
        _, _, actor = self.build()
        feat = torch.randn(5, 640 if actor.linears[0].in_features == 640 else actor.linears[0].in_features)
        state = State(feat[:, :512], feat[:, 512:].reshape(5, 32, 4), feat[:, 512:].reshape(5, 32, 4))

        first = ActorActionProvider(actor, seed=11)(state)
        second = ActorActionProvider(actor, seed=11)(state)
        third = ActorActionProvider(actor, seed=12)(state)

        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertGreater(float((first - third).abs().max()), 0.0)

    def test_an_actor_driven_rollout_produces_finite_behavior_losses(self):
        model, critic, actor = self.build()
        start = model.rssm.initial(4)
        generator = torch.Generator().manual_seed(2)
        imagination = imagine_trajectory(
            model, start, ActorActionProvider(actor, seed=3), 5, generator
        )

        self.assertEqual(tuple(imagination.actions.shape), (4, 5, A))
        self.assertFalse(imagination.feat.requires_grad)

        out = behavior_losses(actor, critic, imagination, ReturnNormalizer())

        self.assertTrue(torch.isfinite(out.loss))
        self.assertTrue(torch.isfinite(out.actor.loss))
        self.assertTrue(torch.isfinite(out.imagined.loss))
        self.assertTrue(torch.isfinite(out.actor.entropy).all())
        torch.testing.assert_close(
            out.loss, out.actor.loss + out.imagined.loss, rtol=1e-6, atol=1e-7
        )

    def test_the_actor_scale_is_one_and_the_replay_critic_still_enters_at_0_3(self):
        model, critic, actor = self.build()
        start = model.rssm.initial(4)
        imagination = imagine_trajectory(
            model, start, ActorActionProvider(actor, seed=3), 5, torch.Generator().manual_seed(2)
        )
        feat = torch.randn(2, 4, model.rssm.feat_size)
        replay = critic.replay_loss(
            feat, torch.ones(2, 4), torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(2, 4)
        )

        out = behavior_losses(actor, critic, imagination, ReturnNormalizer(), replay=replay)

        torch.testing.assert_close(
            out.loss, out.actor.loss + out.imagined.loss + 0.3 * replay.loss, rtol=1e-6, atol=1e-7
        )


if __name__ == "__main__":
    unittest.main()
