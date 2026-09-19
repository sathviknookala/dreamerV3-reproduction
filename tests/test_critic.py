import unittest

import torch
from torch import Tensor

from dreamer import State, WorldModel
from dreamer.critic import (
    Critic,
    SlowCritic,
    ValueHead,
    check_bootstrap_holes,
    lambda_return,
    scatter_imagined_return,
)
from dreamer.imagine import Imagination
from dreamer.optim import LaProp
from dreamer.rssm import observe_sequence

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

A, N, H = 6, 2, 3
DETER, STOCH, CLASSES = 8, 2, 2
FEAT = DETER + STOCH * CLASSES
DISC, LAM = 0.9, 0.95
CONTDISC = 1.0 - 1.0 / 333


def row(values, n=1) -> Tensor:
    return torch.tensor([list(values)], dtype=torch.float32).expand(n, len(values)).contiguous()


def zeros(length, n=1) -> Tensor:
    return torch.zeros(n, length)


class TestLambdaReturnFixtures(unittest.TestCase):
    """spec 6.1: each fixture must fail if the bootstrap moves to v[t]."""

    L = 5

    def run_kernel(self, rew, boot, term=None, last=None, disc=DISC, lam=LAM):
        return lambda_return(
            zeros(self.L) if last is None else last,
            zeros(self.L) if term is None else term,
            rew,
            boot,
            disc,
            lam,
        )

    def test_output_length_is_L_minus_one(self):
        self.assertEqual(tuple(self.run_kernel(zeros(self.L), zeros(self.L)).shape), (1, self.L - 1))

    def test_a_zero_rewards_give_zero_returns(self):
        out = self.run_kernel(zeros(self.L), zeros(self.L))
        torch.testing.assert_close(out, zeros(self.L - 1), rtol=0, atol=0)

    def test_b_constant_reward_matches_the_closed_form(self):
        rew, boot = row([2.0] * self.L), row([7.0] * self.L)
        expected = [7.0]
        for _ in range(self.L - 1):
            expected.append(2.0 + DISC * ((1 - LAM) * 7.0 + LAM * expected[-1]))

        # expected was built backwards from the seed R_{L-1} = V_{L-1}
        torch.testing.assert_close(
            self.run_kernel(rew, boot), row(expected[-1:0:-1]), rtol=1e-6, atol=1e-6
        )

    def test_c_a_true_terminal_kills_the_bootstrap_exactly(self):
        term = zeros(self.L)
        term[0, 2] = 1.0
        out = self.run_kernel(row([2.0] * self.L), row([7.0] * self.L), term=term)

        # gamma_{t+1} = 0, so R_1 is the reward alone -- no bootstrap at all
        self.assertAlmostEqual(float(out[0, 1]), 2.0, places=6)

    def test_d_a_time_limit_cuts_the_trace_but_keeps_the_bootstrap(self):
        last = zeros(self.L)
        last[0, 2] = 1.0
        out = self.run_kernel(row([2.0] * self.L), row([7.0] * self.L), last=last)

        # lambda_{t+1} = 0, gamma_{t+1} = disc: bootstrap at FULL weight
        self.assertAlmostEqual(float(out[0, 1]), 2.0 + DISC * 7.0, places=5)

    def test_e_lambda_zero_is_one_step_td_everywhere(self):
        out = self.run_kernel(row([2.0] * self.L), row([7.0] * self.L), lam=0.0)
        torch.testing.assert_close(
            out, row([2.0 + DISC * 7.0] * (self.L - 1)), rtol=1e-6, atol=1e-5
        )

    def test_f_lambda_one_is_the_monte_carlo_sum_to_the_seed(self):
        expected = [7.0]
        for _ in range(self.L - 1):
            expected.append(2.0 + DISC * expected[-1])

        torch.testing.assert_close(
            self.run_kernel(row([2.0] * self.L), row([7.0] * self.L), lam=1.0),
            row(expected[-1:0:-1]),
            rtol=1e-6,
            atol=1e-5,
        )

    def test_the_seed_is_the_last_bootstrap_not_the_first(self):
        boot = row([0.0, 0.0, 0.0, 0.0, 5.0])
        out = self.run_kernel(zeros(self.L), boot, lam=1.0)

        # R_{L-2} = disc * boot[-1]; a v[t] bootstrap could not produce this column
        self.assertAlmostEqual(float(out[0, -1]), DISC * 5.0, places=6)
        self.assertAlmostEqual(float(out[0, 0]), DISC**4 * 5.0, places=6)

    def test_index_zero_of_rew_term_and_last_is_never_read(self):
        clean = self.run_kernel(row([0.0, 1.0, 1.0, 1.0, 1.0]), row([3.0] * self.L))
        poisoned_rew = self.run_kernel(row([-999.0, 1.0, 1.0, 1.0, 1.0]), row([3.0] * self.L))

        term = zeros(self.L)
        term[0, 0] = 1.0
        last = zeros(self.L)
        last[0, 0] = 1.0
        poisoned_flags = self.run_kernel(
            row([0.0, 1.0, 1.0, 1.0, 1.0]), row([3.0] * self.L), term=term, last=last
        )

        torch.testing.assert_close(clean, poisoned_rew, rtol=0, atol=0)
        torch.testing.assert_close(clean, poisoned_flags, rtol=0, atol=0)

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            lambda_return(zeros(4), zeros(5), zeros(5), zeros(5), DISC, LAM)


class TestGammaLocation(unittest.TestCase):
    """spec 5.4: gamma lives in the continuation head. Double-counting and dropping both train."""

    L, R = 16, 1.0

    def geometric(self, x):
        return sum(x**j for j in range(self.L - 1))

    def run_kernel(self, con, disc):
        term = 1.0 - row([con] * self.L)
        return lambda_return(zeros(self.L), term, row([self.R] * self.L), zeros(self.L), disc, 1.0)

    def test_the_three_readings_are_numerically_separated(self):
        correct = float(self.run_kernel(CONTDISC, 1.0)[0, 0])
        doubled = float(self.run_kernel(CONTDISC, CONTDISC)[0, 0])
        dropped = float(self.run_kernel(1.0, 1.0)[0, 0])

        self.assertAlmostEqual(correct, self.geometric(CONTDISC), places=4)
        self.assertAlmostEqual(doubled, self.geometric(CONTDISC * CONTDISC), places=4)
        self.assertAlmostEqual(dropped, float(self.L - 1), places=4)

        # H=15 separates them by ~0.3 and ~0.3 -- small, silent, and compounding
        self.assertGreater(correct - doubled, 0.2)
        self.assertGreater(dropped - correct, 0.2)


class TestValueHead(unittest.TestCase):
    def test_parameter_count_closes_the_derivation(self):
        self.assertEqual(sum(p.numel() for p in ValueHead().parameters()), 66_111)

    def test_the_zero_init_readout_is_exactly_zero(self):
        value = ValueHead()
        with torch.no_grad():
            # exactly 0, not merely small: a plain float32 sum over the bins returns -1.0
            self.assertEqual(float(value.predict(torch.randn(8, 640)).abs().max()), 0.0)

    def test_the_bin_support_spans_symexp_plus_minus_20(self):
        bins = ValueHead().bins
        lim = float(torch.expm1(torch.tensor(20.0, dtype=torch.float64)))

        self.assertEqual(bins.numel(), 255)
        # a narrower support saturates real targets silently; only the endpoints pin it
        self.assertAlmostEqual(float(bins[-1]) / lim, 1.0, places=6)
        self.assertAlmostEqual(float(bins[0]) / lim, -1.0, places=6)

    def test_the_readout_is_the_expectation_with_no_outer_symexp(self):
        value = ValueHead()
        probs = torch.zeros(1, 255)
        # mass 0.5 on 0 and 0.5 on the bin nearest 100; v2 reads the mean, v1 would read ~9
        bins = value.bins
        near = int((bins - 100.0).abs().argmin())
        probs[0, 127] = 0.5
        probs[0, near] = 0.5

        from dreamer.twohot import twohot_readout

        read = float(twohot_readout(probs, bins))
        self.assertAlmostEqual(read, 0.5 * float(bins[near]), places=3)
        self.assertGreater(read, 40.0)

    def test_the_head_is_invisible_upstream_until_its_kernel_is_perturbed(self):
        value = ValueHead()
        feat = torch.randn(4, 640, requires_grad=True)
        value.loss(feat, torch.full((4,), 3.0)).sum().backward()

        # outscale 0.0 zeroes dL/dfeat entirely: any upstream routing test is vacuous here
        self.assertEqual(float(feat.grad.abs().sum()), 0.0)

        value = ValueHead()
        with torch.no_grad():
            value.mlp.out.weight.normal_(0.0, 0.1)
        feat = torch.randn(4, 640, requires_grad=True)
        value.loss(feat, torch.full((4,), 3.0)).sum().backward()
        self.assertGreater(float(feat.grad.abs().sum()), 0.0)


class TestSlowCritic(unittest.TestCase):
    def test_the_mirror_starts_identical_to_the_fast_critic(self):
        value = ValueHead()
        with torch.no_grad():
            value.mlp.out.weight.normal_(0.0, 0.1)
        slow = SlowCritic(value)

        for a, b in zip(slow.mirror.parameters(), value.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_the_ema_arithmetic_is_hand_computable_and_not_inverted(self):
        value = ValueHead()
        slow = SlowCritic(value, rate=0.02)
        before = slow.mirror.mlp.out.weight.clone()

        with torch.no_grad():
            value.mlp.out.weight.fill_(1.0)
        slow.update(value)

        torch.testing.assert_close(
            slow.mirror.mlp.out.weight, 0.02 * torch.ones_like(before) + 0.98 * before,
            rtol=0, atol=1e-7,
        )
        # the inverted rate would land on 0.98, three orders of magnitude away
        self.assertAlmostEqual(float(slow.mirror.mlp.out.weight.abs().max()), 0.02, places=6)

    def test_the_mirror_carries_no_gradient_and_is_not_trainable(self):
        critic = Critic()
        self.assertTrue(all(not p.requires_grad for p in critic.slow.parameters()))

        trainable = {id(p) for p in critic.trainable_parameters()}
        self.assertFalse(trainable & {id(p) for p in critic.slow.parameters()})
        self.assertEqual(sum(p.numel() for p in critic.trainable_parameters()), 66_111)

    def test_the_mirror_doubles_the_measured_parameter_count_if_it_leaks_in(self):
        critic = Critic()
        # the detector named in spec 4.11: the mirror is 66,111 more, untrained
        self.assertEqual(sum(p.numel() for p in critic.parameters()), 2 * 66_111)


def imagination(rewards, cont, cont_start=None, n=N, deter=DETER, stoch_size=STOCH, classes=CLASSES):
    horizon = len(rewards)
    # nonzero features on purpose: with zero features every hidden activation is zero and
    # dL/d(out.weight) vanishes, which makes a routing assertion pass for the wrong reason
    generator = torch.Generator().manual_seed(7)
    deter = torch.randn(n, horizon + 1, deter, generator=generator)
    stoch = torch.randn(n, horizon + 1, stoch_size, classes, generator=generator)
    con = row(cont, n)
    start = row([cont[0]] if cont_start is None else [cont_start], n)
    weight = torch.cumprod(torch.cat([start, con], dim=1), dim=1)
    return Imagination(
        states=State(deter, stoch, stoch),
        actions=torch.zeros(n, horizon, A),
        reward=row(rewards, n),
        cont=con,
        cont_start=start,
        weight=weight,
    )


class TestImaginedLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.critic = Critic(in_features=FEAT)

    def test_the_return_is_hand_computable_against_the_zero_init_critic(self):
        # the zero-init readout makes boot exactly 0, so R_t is the discounted reward sum alone
        imag = imagination([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        out = self.critic.imagined_loss(imag)

        self.assertEqual(tuple(out.ret.shape), (N, H))
        torch.testing.assert_close(out.ret[0], torch.tensor([1 + LAM * (2 + LAM * 3), 2 + LAM * 3, 3.0]), rtol=0, atol=1e-5)

    def test_the_continuation_is_the_per_step_discount(self):
        c = 0.5
        imag = imagination([1.0, 1.0, 1.0], [c, c, c])
        out = self.critic.imagined_loss(imag)

        # R_t = r + c*lam*R_{t+1}; the discount enters only through con
        expected = [0.0]
        for _ in range(H):
            expected.append(1.0 + c * LAM * expected[-1])
        torch.testing.assert_close(out.ret[0], torch.tensor(expected[-1:0:-1]), rtol=0, atol=1e-5)

    def test_the_loss_uses_the_first_H_positions_and_the_matching_weight_slice(self):
        imag = imagination([1.0, 2.0, 3.0], [0.9, 0.9, 0.9])
        out = self.critic.imagined_loss(imag)

        self.assertEqual(tuple(out.weight.shape), (N, H))
        torch.testing.assert_close(out.weight, imag.weight[:, :-1], rtol=0, atol=0)
        # weight[0] is the START continuation, not 1
        self.assertAlmostEqual(float(out.weight[0, 0]), 0.9, places=6)

    def test_the_target_is_detached(self):
        imag = imagination([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        out = self.critic.imagined_loss(imag)

        self.assertFalse(out.ret.requires_grad)

    def test_the_continuation_weight_is_detached(self):
        imag = imagination([1.0, 2.0, 3.0], [0.9, 0.9, 0.9])
        imag.weight = imag.weight.detach().requires_grad_(True)
        with torch.no_grad():
            self.critic.value.mlp.out.weight.normal_(0.0, 0.3)

        out = self.critic.imagined_loss(imag)
        out.loss.backward()

        # spec 5.9 lists the weight among the value loss's detached inputs: without the detach
        # the critic loss backprops into the continuation head through cumprod(con)
        self.assertFalse(out.weight.requires_grad)
        self.assertIsNone(imag.weight.grad)

    def test_the_pad_value_on_rew_and_con_is_arbitrary(self):
        imag = imagination([1.0, 2.0, 3.0], [0.9, 0.9, 0.9], cont_start=0.9)
        baseline = self.critic.imagined_loss(imag).ret.clone()

        # cont_start reaches the weight, never the kernel: term[:, 1:] is exactly imag.cont
        imag.cont_start = torch.full_like(imag.cont_start, 0.1)
        torch.testing.assert_close(self.critic.imagined_loss(imag).ret, baseline, rtol=0, atol=0)


class TestSlowCriticIsARegularizerNotABootstrap(unittest.TestCase):
    """Both claims are invisible at init: zero-init makes fast and slow read out the same 0."""

    def build(self):
        torch.manual_seed(0)
        critic = Critic(in_features=FEAT)
        # the mirror was deep-copied at construction, so perturbing the fast head now separates them
        with torch.no_grad():
            critic.value.mlp.out.weight.normal_(0.0, 0.3)
        return critic

    def separated(self, critic, feat):
        fast = critic.value.predict(feat).detach()
        slow = critic.slow.predict(feat).detach()
        self.assertGreater(float((fast - slow).abs().max()), 1.0)
        return fast, slow

    def test_the_imagined_bootstrap_is_the_fast_critic_not_the_slow_one(self):
        critic = self.build()
        imag = imagination([1.0, 2.0, 3.0], [0.9, 0.9, 0.9])
        fast, slow = self.separated(critic, imag.feat)

        con = torch.cat([imag.cont_start, imag.cont], dim=1)
        rew = torch.cat([torch.zeros_like(imag.reward[:, :1]), imag.reward], dim=1)
        from_fast = lambda_return(torch.zeros_like(con), 1.0 - con, rew, fast, 1.0, critic.lam)
        from_slow = lambda_return(torch.zeros_like(con), 1.0 - con, rew, slow, 1.0, critic.lam)

        out = critic.imagined_loss(imag)
        torch.testing.assert_close(out.ret, from_fast, rtol=0, atol=0)
        self.assertGreater(float((from_fast - from_slow).abs().max()), 1.0)

    def test_the_slow_critic_adds_a_second_loss_term(self):
        critic = self.build()
        imag = imagination([1.0, 2.0, 3.0], [0.9, 0.9, 0.9])
        out = critic.imagined_loss(imag)
        feat = imag.feat[:, :-1]
        slow_target = critic.slow.predict(feat).detach()

        one_term = (critic.value.loss(feat, out.ret) * out.weight).mean()
        two_terms = (
            (critic.value.loss(feat, out.ret) + critic.value.loss(feat, slow_target)) * out.weight
        ).mean()

        torch.testing.assert_close(out.loss, two_terms, rtol=1e-6, atol=1e-6)
        # dropping slowreg is a real change in the minimized quantity, not a rounding difference
        self.assertGreater(float((out.loss - one_term).abs()), 1e-3)

    def test_the_slow_target_is_detached(self):
        critic = self.build()
        imag = imagination([1.0, 2.0, 3.0], [0.9, 0.9, 0.9])
        critic.imagined_loss(imag).loss.backward()

        # gradient into the mirror would stop it being an EMA, with no error raised
        self.assertTrue(all(p.grad is None for p in critic.slow.parameters()))


class TestReplayLoss(unittest.TestCase):
    B, T = 2, 4

    def setUp(self):
        torch.manual_seed(0)
        self.critic = Critic(in_features=FEAT)

    def inputs(self):
        return (
            torch.zeros(self.B, self.T, FEAT),
            torch.ones(self.B, self.T),
            torch.zeros(self.B, self.T),
            torch.zeros(self.B, self.T),
            torch.zeros(self.B, self.T),
        )

    def test_the_replay_discount_is_hard_coded_and_not_contdisc(self):
        feat, rewards, is_last, is_terminal, boot = self.inputs()
        out = self.critic.replay_loss(feat, rewards, is_last, is_terminal, boot)

        expected = [0.0]
        for _ in range(self.T - 1):
            expected.append(1.0 + CONTDISC * LAM * expected[-1])
        torch.testing.assert_close(out.ret[0], torch.tensor(expected[-1:0:-1]), rtol=0, atol=1e-5)

    def test_the_position_weight_masks_boundaries_and_sets_the_denominator(self):
        feat, rewards, is_last, is_terminal, boot = self.inputs()
        is_last[0, 1] = 1.0
        out = self.critic.replay_loss(feat, rewards, is_last, is_terminal, boot)

        self.assertEqual(tuple(out.weight.shape), (self.B, self.T - 1))
        self.assertEqual(out.metrics["repval_positions"], self.B * (self.T - 1))
        self.assertEqual(float(out.weight[0, 1]), 0.0)
        self.assertEqual(float(out.weight[1, 1]), 1.0)

    def perturbed(self):
        # at outscale 0.0 the loss is log(255) for every feature and every target, so an
        # alignment or reduction claim would hold for the wrong reason
        with torch.no_grad():
            self.critic.value.mlp.out.weight.normal_(0.0, 0.3)
        return self.critic

    def test_the_trained_features_are_the_ones_the_targets_align_with(self):
        critic = self.perturbed()
        feat, rewards, is_last, is_terminal, boot = self.inputs()
        feat = torch.randn(self.B, self.T, FEAT)
        rewards = torch.arange(1.0, self.B * self.T + 1).reshape(self.B, self.T)

        out = critic.replay_loss(feat, rewards, is_last, is_terminal, boot)
        aligned = critic._fit(feat[:, :-1], out.ret, out.weight, "x").loss
        shifted = critic._fit(feat[:, 1:], out.ret, out.weight, "x").loss

        torch.testing.assert_close(out.loss, aligned, rtol=0, atol=0)
        self.assertGreater(float((out.loss - shifted).abs().detach()), 1e-3)

    def test_the_reduction_is_a_position_mean_not_a_weighted_mean(self):
        critic = self.perturbed()
        feat, rewards, is_last, is_terminal, boot = self.inputs()
        feat = torch.randn(self.B, self.T, FEAT)
        is_last[0, 1] = 1.0

        out = critic.replay_loss(feat, rewards, is_last, is_terminal, boot)
        per = critic.value.loss(feat[:, :-1], out.ret) + critic.value.loss(
            feat[:, :-1], critic.slow.predict(feat[:, :-1]).detach()
        )

        torch.testing.assert_close(out.loss, (per * out.weight).mean(), rtol=1e-6, atol=1e-6)
        # dividing by the weight sum instead would rescale by numel/sum = 6/5 here
        self.assertGreater(
            float((out.loss - (per * out.weight).sum() / out.weight.sum()).abs().detach()), 1e-3
        )

    def test_a_non_terminal_hole_is_rejected_when_the_fill_mask_is_supplied(self):
        feat, rewards, is_last, is_terminal, boot = self.inputs()
        filled = torch.ones(self.B, self.T, dtype=torch.bool)
        self.critic.replay_loss(feat, rewards, is_last, is_terminal, boot, filled)

        filled[0, 2] = False
        with self.assertRaises(ValueError):
            self.critic.replay_loss(feat, rewards, is_last, is_terminal, boot, filled)

        is_terminal[0, 2] = 1.0
        self.critic.replay_loss(feat, rewards, is_last, is_terminal, boot, filled)

    def test_a_time_limit_is_not_a_terminal(self):
        feat, rewards, is_last, is_terminal, boot = self.inputs()
        boot = torch.full_like(boot, 5.0)
        is_last[:, 2] = 1.0
        limited = self.critic.replay_loss(feat, rewards, is_last, is_terminal, boot).ret

        is_terminal[:, 2] = 1.0
        terminated = self.critic.replay_loss(feat, rewards, is_last, is_terminal, boot).ret

        # time limit keeps the bootstrap; a true terminal removes it entirely
        self.assertAlmostEqual(float(limited[0, 1]), 1.0 + CONTDISC * 5.0, places=5)
        self.assertAlmostEqual(float(terminated[0, 1]), 1.0, places=6)


class TestLossScales(unittest.TestCase):
    def test_the_replay_critic_enters_at_weight_0_3(self):
        torch.manual_seed(0)
        critic = Critic(in_features=FEAT)
        imag = imagination([1.0, 2.0, 3.0], [0.9, 0.9, 0.9])
        b, t = 2, 4
        imagined = critic.imagined_loss(imag)
        replay = critic.replay_loss(
            torch.randn(b, t, FEAT), torch.ones(b, t), torch.zeros(b, t),
            torch.zeros(b, t), torch.zeros(b, t),
        )

        self.assertEqual(critic.scales, {"value": 1.0, "repval": 0.3})
        torch.testing.assert_close(
            critic.total(imagined, replay),
            imagined.loss + 0.3 * replay.loss,
            rtol=0,
            atol=0,
        )
        # the imagined loss alone must not silently carry the replay weight
        torch.testing.assert_close(critic.total(imagined), imagined.loss, rtol=0, atol=0)


class TestBootstrapScatter(unittest.TestCase):
    B, P, T = 2, 2, 3

    def masks(self):
        loss_mask = torch.ones(self.B, self.P + self.T, dtype=torch.bool)
        loss_mask[:, : self.P] = False
        return loss_mask, torch.zeros(self.B, self.P + self.T, dtype=torch.bool)

    def starts(self, loss_mask, is_terminal):
        keep = loss_mask & ~is_terminal
        return keep.reshape(-1).nonzero(as_tuple=True)[0]

    def test_the_index_addresses_the_full_P_plus_T_grid(self):
        loss_mask, is_terminal = self.masks()
        index = self.starts(loss_mask, is_terminal)
        ret = torch.arange(1.0, index.numel() + 1).unsqueeze(1)

        grid, filled = scatter_imagined_return(ret, index, self.B, self.P + self.T)

        self.assertEqual(tuple(grid.shape), (self.B, self.P + self.T))
        # burn-in columns are never starts and stay empty; the index is modulo P+T, not T
        self.assertFalse(bool(filled[:, : self.P].any()))
        self.assertTrue(bool(filled[:, self.P :].all()))
        torch.testing.assert_close(grid[0, self.P :], torch.tensor([1.0, 2.0, 3.0]), rtol=0, atol=0)

    def test_the_scatter_takes_the_first_imagined_return_not_the_last(self):
        loss_mask, is_terminal = self.masks()
        index = self.starts(loss_mask, is_terminal)
        n = index.numel()
        # R_0 is the return at the imagination START state, which is the replay position itself
        ret = torch.stack(
            [torch.arange(1.0, n + 1), torch.full((n,), -7.0), torch.full((n,), -9.0)], dim=1
        )

        grid, _ = scatter_imagined_return(ret, index, self.B, self.P + self.T)

        torch.testing.assert_close(
            grid[0, self.P :], torch.tensor([1.0, 2.0, 3.0]), rtol=0, atol=0
        )

    def test_a_terminal_hole_is_permitted_and_a_non_terminal_hole_is_not(self):
        loss_mask, is_terminal = self.masks()
        is_terminal[0, 3] = True
        index = self.starts(loss_mask, is_terminal)
        ret = torch.arange(1.0, index.numel() + 1).unsqueeze(1)
        _, filled = scatter_imagined_return(ret, index, self.B, self.P + self.T)

        check_bootstrap_holes(filled[:, self.P :], is_terminal[:, self.P :])

        with self.assertRaises(ValueError):
            check_bootstrap_holes(filled[:, self.P :], torch.zeros_like(is_terminal[:, self.P :]))

    def test_a_non_terminal_hole_would_corrupt_every_earlier_target(self):
        L = 6
        boot = torch.full((1, L), 4.0)
        rew = torch.ones(1, L)
        intact = lambda_return(torch.zeros(1, L), torch.zeros(1, L), rew, boot, CONTDISC, LAM)

        holed = boot.clone()
        holed[0, 3] = 0.0
        corrupted = lambda_return(torch.zeros(1, L), torch.zeros(1, L), rew, holed, CONTDISC, LAM)

        # the recursion runs backwards, so a hole at 3 moves positions 0..2 as well
        self.assertGreater(float((intact[0, :3] - corrupted[0, :3]).abs().min()), 0.1)

        term = torch.zeros(1, L)
        term[0, 3] = 1.0
        # at a true terminal, live[:, 2] is exactly 0, so the same hole changes nothing
        with_term = lambda_return(torch.zeros(1, L), term, rew, boot, CONTDISC, LAM)
        holed_term = lambda_return(torch.zeros(1, L), term, rew, holed, CONTDISC, LAM)
        torch.testing.assert_close(with_term, holed_term, rtol=0, atol=0)


class TestGradientRouting(unittest.TestCase):
    """spec 5.9 and 6.3. Every assertion here is vacuous until the critic kernel is perturbed."""

    def build(self):
        torch.manual_seed(0)
        wm = WorldModel(action_dim=A)
        critic = Critic(in_features=wm.rssm.feat_size)
        with torch.no_grad():
            # outscale 0.0 makes dL/dfeat exactly 0 and every routing claim below vacuous
            critic.value.mlp.out.weight.normal_(0.0, 0.1)
        return wm, critic

    def replay_feat(self, wm, batch=2, length=4):
        observations = torch.randint(0, 256, (batch, length, 64, 64, 3), dtype=torch.uint8)
        post, _ = observe_sequence(
            wm.encoder, wm.rssm, observations, torch.zeros(batch, length - 1, A)
        )
        return post.feat

    def test_the_imagined_loss_updates_the_critic_only(self):
        wm, critic = self.build()
        imag = imagination(
            [1.0, 2.0, 3.0],
            [0.9, 0.9, 0.9],
            deter=wm.rssm.deter,
            stoch_size=wm.rssm.stoch,
            classes=wm.rssm.classes,
        )

        critic.imagined_loss(imag).loss.backward()

        self.assertGreater(float(critic.value.mlp.out.weight.grad.abs().sum()), 0.0)
        self.assertIsNone(wm.encoder.convs[0].weight.grad)
        self.assertIsNone(wm.rssm.dyngru.kernel.grad)
        self.assertIsNone(wm.reward.mlp.out.weight.grad)
        self.assertTrue(all(p.grad is None for p in critic.slow.parameters()))

    def test_the_replay_loss_reaches_the_encoder_and_the_rssm(self):
        wm, critic = self.build()
        feat = self.replay_feat(wm)
        b, t = feat.shape[0], feat.shape[1]

        critic.replay_loss(
            feat, torch.ones(b, t), torch.zeros(b, t), torch.zeros(b, t), torch.zeros(b, t)
        ).loss.backward()

        # repval_grad: True -- value error shapes the representation (spec 6.3)
        self.assertGreater(float(wm.encoder.convs[0].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(wm.rssm.dyngru.kernel.grad.abs().sum()), 0.0)
        self.assertGreater(float(critic.value.mlp.out.weight.grad.abs().sum()), 0.0)
        self.assertIsNone(wm.decoder.imgout.weight.grad)
        self.assertTrue(all(p.grad is None for p in critic.slow.parameters()))

    def test_detaching_the_replay_features_would_sever_that_path(self):
        wm, critic = self.build()
        feat = self.replay_feat(wm)
        b, t = feat.shape[0], feat.shape[1]

        critic.replay_loss(
            feat.detach(), torch.ones(b, t), torch.zeros(b, t), torch.zeros(b, t), torch.zeros(b, t)
        ).loss.backward()

        self.assertIsNone(wm.encoder.convs[0].weight.grad)
        self.assertIsNone(wm.rssm.dyngru.kernel.grad)


class TestCriticFitting(unittest.TestCase):
    def test_laprop_reduces_value_error_without_touching_the_world_model(self):
        torch.manual_seed(0)
        wm = WorldModel(action_dim=A)
        critic = Critic(in_features=FEAT)
        snapshot = [p.detach().clone() for p in wm.parameters()]

        feat = torch.randn(32, 6, FEAT)
        target = torch.full((32, 5), 4.0)
        weight = torch.ones(32, 5)

        opt = LaProp(critic.trainable_parameters(), lr=3e-3, warmup=0)
        first = last = None

        for step in range(150):
            out = critic._fit(feat[:, :-1], target, weight, "value")
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            opt.step()
            critic.update_slow()
            first = first if first is not None else out.metrics["value_value_mae"]
            last = out.metrics["value_value_mae"]

        self.assertLess(last, 0.5 * first)
        for before, after in zip(snapshot, wm.parameters()):
            self.assertTrue(torch.equal(before, after))

    def test_the_slow_critic_trails_the_fast_one_rather_than_tracking_it(self):
        torch.manual_seed(0)
        critic = Critic(in_features=FEAT)
        with torch.no_grad():
            critic.value.mlp.out.weight.fill_(1.0)

        critic.update_slow()
        gap = (critic.value.mlp.out.weight - critic.slow.mirror.mlp.out.weight).abs().max().detach()

        self.assertAlmostEqual(float(gap), 0.98, places=6)


if __name__ == "__main__":
    unittest.main()
