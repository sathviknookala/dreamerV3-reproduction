import unittest

import numpy as np
import torch

from dreamer import LaProp, WorldModel
from dreamer.twohot import make_bins, twohot_encode, twohot_readout
from dreamer.world_model import batch_to_tensors, categorical_kl
from dreamer.types import SequenceBatch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

A, B, P, T = 6, 2, 5, 8
L = P + T

DERIVED = {"dec": 80_595, "rew": 57_663, "con": 41_153}
DERIVED_WORLD_MODEL = 570_419


def gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def build(seed=0):
    torch.manual_seed(seed)
    return WorldModel(action_dim=A)


def fake_batch(seed=0, terminal_at=None):
    rng = np.random.default_rng(seed)
    is_terminal = np.zeros((B, L), dtype=np.bool_)
    if terminal_at is not None:
        is_terminal[:, terminal_at] = True
    mask = np.ones((B, L), dtype=np.bool_)
    mask[:, :P] = False
    return SequenceBatch(
        observations=rng.integers(0, 256, (B, L + 1, 64, 64, 3), dtype=np.uint8),
        actions=rng.uniform(-1, 1, (B, L, A)).astype(np.float32),
        rewards=rng.uniform(0, 1, (B, L)).astype(np.float32),
        is_last=np.zeros((B, L), dtype=np.bool_),
        is_terminal=is_terminal,
        discounts=np.ones((B, L), dtype=np.float32),
        loss_mask=mask,
    )


def tensors(batch, device="cpu"):
    return batch_to_tensors(batch, torch.device(device))


class TestHeads(unittest.TestCase):
    def test_head_shapes(self):
        wm = build()
        deter, stoch = torch.randn(B, 3, 512), torch.randn(B, 3, 32, 4)
        feat = torch.cat([deter, stoch.reshape(B, 3, -1)], -1)

        self.assertEqual(tuple(wm.decoder(deter, stoch).shape), (B, 3, 64, 64, 3))
        self.assertEqual(tuple(wm.reward(feat).shape), (B, 3, 255))
        self.assertEqual(tuple(wm.cont(feat).shape), (B, 3))
        self.assertEqual(tuple(wm.reward.predict(feat).shape), (B, 3))
        self.assertEqual(tuple(wm.cont.predict(feat).shape), (B, 3))

    def test_decoder_target_scaling_has_no_shift(self):
        from dreamer.rssm import observe_sequence

        wm = build()
        batch = fake_batch(9)
        _, act, rew, term, mask = tensors(batch)
        # an all-black image makes the target exactly 0 under /255 and exactly
        # -0.5 under the encoder's /255 - 0.5, so the rec loss separates them
        obs = torch.zeros(B, L + 1, 64, 64, 3, dtype=torch.uint8)

        out = wm.loss(obs, act, rew, term, mask, gen(0))

        post, _ = observe_sequence(wm.encoder, wm.rssm, obs, act, generator=gen(0))
        recon = wm.decoder(post[:, 1:].deter, post[:, 1:].stoch)
        self.assertGreaterEqual(recon.min().item(), 0.0)
        self.assertLessEqual(recon.max().item(), 1.0)

        expected = (recon.square().sum((-1, -2, -3)) * mask).sum() / mask.sum()
        torch.testing.assert_close(out.losses["rec"], expected, rtol=1e-5, atol=1e-4)

        shifted = ((recon + 0.5).square().sum((-1, -2, -3)) * mask).sum() / mask.sum()
        self.assertFalse(torch.allclose(out.losses["rec"], shifted))

    def test_parameter_counts_match_the_derivation(self):
        wm = build()
        measured = {
            "dec": sum(p.numel() for p in wm.decoder.parameters()),
            "rew": sum(p.numel() for p in wm.reward.parameters()),
            "con": sum(p.numel() for p in wm.cont.parameters()),
        }
        self.assertEqual(measured, DERIVED)
        self.assertEqual(sum(p.numel() for p in wm.parameters()), DERIVED_WORLD_MODEL)


class TestTwoHot(unittest.TestCase):
    def test_encode_and_readout_match_hand_computed_values(self):
        bins = make_bins()
        self.assertEqual(bins.numel(), 255)
        self.assertEqual(bins[127].item(), 0.0)
        torch.testing.assert_close(bins[126], -bins[128])

        # exact zero readout at init is the property outscale=0.0 is there to give
        uniform = torch.full((1, 255), 1.0 / 255.0)
        self.assertEqual(twohot_readout(uniform, bins).item(), 0.0)

        for value in (0.0, 1.0, -3.5, 100.0, 1e4):
            target = twohot_encode(torch.tensor([value]), bins)
            self.assertLessEqual(int((target > 0).sum()), 2)
            torch.testing.assert_close(target.sum(-1), torch.ones(1))
            torch.testing.assert_close(
                twohot_readout(target, bins), torch.tensor([value]), rtol=1e-6, atol=1e-6
            )

        # hand-computable: 0.5 between adjacent bins reads out their midpoint
        mid = ((bins[128] + bins[129]) / 2).item()
        target = twohot_encode(torch.tensor([mid]), bins)
        torch.testing.assert_close(target[0, 128], torch.tensor(0.5), atol=1e-6, rtol=0)
        torch.testing.assert_close(target[0, 129], torch.tensor(0.5), atol=1e-6, rtol=0)

        saturated = twohot_encode(torch.tensor([1e12, -1e12]), bins)
        self.assertEqual(saturated[0, 254].item(), 1.0)
        self.assertEqual(saturated[1, 0].item(), 1.0)


class TestObjective(unittest.TestCase):
    def test_reward_and_continuation_align_to_the_resulting_state(self):
        wm = build()
        batch = fake_batch(1)
        obs, act, rew, term, mask = tensors(batch)

        # outscale=0.0 makes the reward loss exactly log(255) for ANY target, so
        # alignment is untestable until the output kernel is off zero
        torch.nn.init.normal_(wm.reward.mlp.out.weight, std=0.1)

        out = wm.loss(obs, act, rew, term, mask, gen(0))
        self.assertEqual(tuple(out.post.deter.shape), (B, L + 1, 512))

        # a one-step shift in the reward target must change the reward loss
        shifted = torch.roll(rew, 1, dims=1)
        out_shift = wm.loss(obs, act, shifted, term, mask, gen(0))
        self.assertFalse(torch.allclose(out.losses["rew"], out_shift.losses["rew"]))

        # the continuation target likewise reads off state j+1
        term_shift = term.clone()
        term_shift[:, L - 1] = True
        out_term = wm.loss(obs, act, rew, term_shift, mask, gen(0))
        self.assertFalse(torch.allclose(out.losses["con"], out_term.losses["con"]))

        # only s_L sees the final observation, and only transition L-1 uses s_L.
        # if the targets were read off s_j instead, obs[L] could reach neither the
        # reward nor the continuation loss at all
        final = obs.clone()
        final[:, L] = 0
        out_final = wm.loss(final, act, rew, term, mask, gen(0))
        self.assertFalse(torch.allclose(out.losses["rew"], out_final.losses["rew"]))
        self.assertFalse(torch.allclose(out.losses["con"], out_final.losses["con"]))

    def test_burn_in_and_reset_positions_are_excluded(self):
        wm = build()
        batch = fake_batch(2)
        obs, act, rew, term, mask = tensors(batch)
        out = wm.loss(obs, act, rew, term, mask, gen(0))

        self.assertEqual(out.metrics["train_positions"], float(B * T))

        # changing a burn-in reward target must not move any loss
        burned = rew.clone()
        burned[:, :P] += 17.0
        same = wm.loss(obs, act, burned, term, mask, gen(0))
        for key in out.losses:
            torch.testing.assert_close(out.losses[key], same.losses[key])

        # s_0 is the reset observation: it carries no transition loss, and the
        # reward that would accompany it does not exist
        self.assertEqual(out.post.deter.shape[1] - 1, mask.shape[1])

    def test_kl_stop_gradient_directions(self):
        wm = build()
        from dreamer.rssm import State

        # assert on the logits themselves: through the recurrence BOTH terms reach
        # both heads (the posterior sample at t-1 feeds deter at t), so a parameter
        # probe cannot isolate the stop-gradient direction the spec table fixes
        for name, index in (("dyn", 0), ("rep", 1)):
            post_logit = torch.randn(B, 4, 32, 4, generator=gen(1)).requires_grad_(True)
            prior_logit = torch.randn(B, 4, 32, 4, generator=gen(2)).requires_grad_(True)
            zeros = torch.zeros(B, 4, 512)

            term = wm.kl_losses(
                State(zeros, torch.zeros(B, 4, 32, 4), post_logit),
                State(zeros, torch.zeros(B, 4, 32, 4), prior_logit),
            )[index]
            term.sum().backward()

            if name == "dyn":
                # KL(sg(q) || p): trains the prior, never the posterior
                self.assertIsNone(post_logit.grad)
                self.assertGreater(prior_logit.grad.abs().sum().item(), 0.0)
            else:
                # KL(q || sg(p)): trains the posterior, never the prior
                self.assertIsNone(prior_logit.grad)
                self.assertGreater(post_logit.grad.abs().sum().item(), 0.0)

    def test_kl_aggregation_then_free_nats(self):
        q = torch.full((4, 32, 4), 0.25)
        p = torch.full((4, 32, 4), 0.25)
        kl = categorical_kl(q, p)
        self.assertEqual(tuple(kl.shape), (4,))
        torch.testing.assert_close(kl, torch.zeros(4), atol=1e-6, rtol=0)

        # per-factor KL of ~0.0625 sums to 2.0 over 32 factors: above a 1-nat
        # floor applied to the whole latent, below a 1-nat-per-factor floor
        skew = torch.tensor([0.4, 0.2, 0.2, 0.2]).expand(1, 32, 4)
        per_factor = (skew * (skew.log() - np.log(0.25)))[0, 0].sum().item()
        summed = categorical_kl(skew, torch.full((1, 32, 4), 0.25))
        torch.testing.assert_close(summed, torch.tensor([per_factor * 32]), rtol=1e-5, atol=1e-6)

        self.assertGreater(summed.item(), 1.0)
        torch.testing.assert_close(summed.clamp(min=1.0), summed)

        tiny = categorical_kl(
            torch.full((1, 32, 4), 0.25) + torch.tensor([1e-4, -1e-4, 0.0, 0.0]),
            torch.full((1, 32, 4), 0.25),
        )
        self.assertLess(tiny.item(), 1.0)
        self.assertEqual(tiny.clamp(min=1.0).item(), 1.0)
        # below the floor the term is constant, so its gradient is exactly zero
        live = (torch.full((1, 32, 4), 0.25) + torch.tensor([1e-4, -1e-4, 0.0, 0.0])).requires_grad_(True)
        categorical_kl(live, torch.full((1, 32, 4), 0.25)).clamp(min=1.0).sum().backward()
        self.assertEqual(live.grad.abs().sum().item(), 0.0)

    def test_isolated_losses_touch_only_their_own_heads(self):
        wm = build()
        batch = fake_batch(4)
        obs, act, rew, term, mask = tensors(batch)
        out = wm.loss(obs, act, rew, term, mask, gen(0))

        heads = {
            "rec": wm.decoder.imgout.weight,
            "rew": wm.reward.mlp.out.weight,
            "con": wm.cont.mlp.out.weight,
        }

        for active in heads:
            wm.zero_grad(set_to_none=True)
            out.losses[active].backward(retain_graph=True)
            for name, param in heads.items():
                if name == active:
                    self.assertIsNotNone(param.grad, f"{active} should reach {name}")
                    self.assertGreater(param.grad.abs().sum().item(), 0.0)
                else:
                    self.assertIsNone(param.grad, f"{active} must not reach {name}")

        # rec and con reach the encoder and RSSM; spec 5.3 and 5.9
        for name in ("rec", "con"):
            wm.zero_grad(set_to_none=True)
            out.losses[name].backward(retain_graph=True)
            self.assertGreater(
                wm.encoder.convs[0].weight.grad.abs().sum().item(), 0.0, name
            )
            self.assertGreater(wm.rssm.dyngru.kernel.grad.abs().sum().item(), 0.0, name)

        # rew does too, but only once outscale=0.0 lets go: at init the zero output
        # kernel makes the whole upstream gradient of the reward loss exactly zero
        wm.zero_grad(set_to_none=True)
        out.losses["rew"].backward(retain_graph=True)
        self.assertEqual(wm.encoder.convs[0].weight.grad.abs().sum().item(), 0.0)

        torch.nn.init.normal_(wm.reward.mlp.out.weight, std=0.1)
        live = wm.loss(obs, act, rew, term, mask, gen(0))
        wm.zero_grad(set_to_none=True)
        live.losses["rew"].backward()
        self.assertGreater(wm.encoder.convs[0].weight.grad.abs().sum().item(), 0.0)

    def test_total_objective_gradients_are_finite_and_nonzero(self):
        wm = build()
        batch = fake_batch(5)
        obs, act, rew, term, mask = tensors(batch)
        out = wm.loss(obs, act, rew, term, mask, gen(0))
        out.total.backward()

        grads = [p.grad for p in wm.parameters()]
        self.assertTrue(all(g is not None for g in grads))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(
            wm.encoder.convs[0].weight.grad.abs().sum().item(), 0.0
        )
        self.assertGreater(wm.rssm.dyngru.kernel.grad.abs().sum().item(), 0.0)
        self.assertTrue(np.isfinite(out.metrics["total"]))


class TestOverfit(unittest.TestCase):
    def test_small_fixed_subset_is_fit(self):
        torch.manual_seed(0)
        wm = WorldModel(action_dim=A)
        batch = fake_batch(6, terminal_at=L - 1)
        obs, act, rew, term, mask = tensors(batch)

        opt = LaProp(wm.parameters(), lr=3e-3, warmup=0)
        first, last = None, None

        for step in range(150):
            out = wm.loss(obs, act, rew, term, mask, gen(step))
            opt.zero_grad(set_to_none=True)
            out.total.backward()
            opt.step()
            if step == 0:
                first = out.metrics
            last = out.metrics

        for key in ("rec", "rew", "con"):
            self.assertLess(
                last[key], first[key] * 0.75, f"{key}: {first[key]:.4g} -> {last[key]:.4g}"
            )


if __name__ == "__main__":
    unittest.main()
