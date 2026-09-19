import inspect
import unittest

import torch

from dreamer import Encoder, RSSM, State, observe_sequence
from dreamer.distributions import straight_through_sample, unimix_probs

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DERIVED_ENCODER_PARAMS = 14_304
DERIVED_RSSM_PARAMS = 376_704

A = 6


def build(seed: int = 0):
    torch.manual_seed(seed)
    return Encoder(), RSSM(action_dim=A)


def gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def images(batch: int, length: int, seed: int = 7) -> torch.Tensor:
    g = gen(seed)
    return torch.randint(
        0, 256, (batch, length, 64, 64, 3), dtype=torch.uint8, generator=g
    )


class TestRecurrence(unittest.TestCase):
    def test_single_step_matches_sequence(self):
        encoder, rssm = build()
        batch, length = 3, 6
        obs = images(batch, length)
        actions = torch.randn(batch, length - 1, A, generator=gen(1))

        post_seq, prior_seq = observe_sequence(
            encoder, rssm, obs, actions, generator=gen(99)
        )

        embeds = encoder(obs)
        zero = torch.zeros(batch, 1, A)
        prev_actions = torch.cat([zero, actions], dim=1)
        is_first = torch.zeros(batch, length, dtype=torch.bool)
        is_first[:, 0] = True

        g = gen(99)
        carry = rssm.initial(batch)
        for t in range(length):
            post, prior = rssm.observe_step(
                carry, prev_actions[:, t], embeds[:, t], is_first[:, t], g
            )
            torch.testing.assert_close(post.deter, post_seq.deter[:, t])
            torch.testing.assert_close(post.stoch, post_seq.stoch[:, t])
            torch.testing.assert_close(post.logit, post_seq.logit[:, t])
            torch.testing.assert_close(prior.logit, prior_seq.logit[:, t])
            carry = post

    def test_gate_split_index_map(self):
        _, rssm = build()
        blocks, per_block = rssm.blocks, rssm.deter // rssm.blocks
        x = torch.arange(3 * rssm.deter, dtype=torch.float32).unsqueeze(0)

        gates = rssm._split_gates(x)
        self.assertEqual(len(gates), 3)

        for gate, values in enumerate(gates):
            self.assertEqual(tuple(values.shape), (1, rssm.deter))
            for b in range(blocks):
                for u in (0, per_block - 1):
                    self.assertEqual(
                        values[0, b * per_block + u].item(),
                        float(b * 3 * per_block + gate * per_block + u),
                    )

    def test_blocklinear_fan_in_is_the_full_input_width(self):
        _, rssm = build()
        for layer in (rssm.dynhid0, rssm.dyngru):
            expected = (1.0 / layer.in_features) ** 0.5
            per_block = (1.0 / layer.block_in) ** 0.5
            measured = layer.kernel.std().item()
            self.assertAlmostEqual(measured / expected, 1.0, delta=0.03)
            self.assertLess(measured / per_block, 0.5)

    def test_reset_mask_isolates_batch_elements(self):
        _, rssm = build()
        batch = 4
        state = State(
            deter=torch.randn(batch, 512, generator=gen(2)),
            stoch=torch.randn(batch, 32, 4, generator=gen(3)),
            logit=torch.zeros(batch, 32, 4),
        )
        action = torch.randn(batch, A, generator=gen(4))

        none_first = torch.zeros(batch, dtype=torch.bool)
        first_only = none_first.clone()
        first_only[0] = True

        base = rssm.imagine_step(state, action, none_first, gen(5)).deter
        mixed = rssm.imagine_step(state, action, first_only, gen(5)).deter

        zeroed = State(
            deter=torch.zeros(1, 512), stoch=torch.zeros(1, 32, 4), logit=torch.zeros(1, 32, 4)
        )
        from_zero = rssm.imagine_step(zeroed, torch.zeros(1, A), None, gen(6)).deter

        torch.testing.assert_close(mixed[0:1], from_zero)
        torch.testing.assert_close(mixed[1:], base[1:])
        self.assertFalse(torch.allclose(mixed[0], base[0]))

    def test_inputs_change_the_expected_downstream_state(self):
        encoder, rssm = build()
        state = State(
            deter=torch.randn(1, 512, generator=gen(8)),
            stoch=torch.randn(1, 32, 4, generator=gen(9)),
            logit=torch.zeros(1, 32, 4),
        )
        action = torch.zeros(1, A)

        deter_a = rssm.imagine_step(state, action, None, gen(1)).deter
        deter_action = rssm.imagine_step(
            state, action + 0.5, None, gen(1)
        ).deter
        self.assertFalse(torch.allclose(deter_a, deter_action))

        other = State(state.deter, state.stoch + 0.5, state.logit)
        deter_stoch = rssm.imagine_step(other, action, None, gen(1)).deter
        self.assertFalse(torch.allclose(deter_a, deter_stoch))

        embed_a = encoder(images(1, 1, seed=11)[:, 0])
        embed_b = encoder(images(1, 1, seed=12)[:, 0])
        post_a = rssm._posterior_logits(deter_a, embed_a)
        post_b = rssm._posterior_logits(deter_a, embed_b)
        self.assertFalse(torch.allclose(post_a, post_b))

    def test_prior_has_no_image_dependency(self):
        _, rssm = build()
        params = set(inspect.signature(rssm.imagine_step).parameters)
        self.assertEqual(params, {"state", "action", "is_first", "generator"})
        self.assertNotIn("embed", inspect.signature(rssm.imagine).parameters)

        source = inspect.getsource(rssm._prior_logits)
        self.assertNotIn("embed", source)

        deter = torch.randn(2, 512, generator=gen(13))
        torch.testing.assert_close(
            rssm._prior_logits(deter), rssm._prior_logits(deter)
        )


class TestCategorical(unittest.TestCase):
    def test_probs_normalize_and_samples_are_onehot(self):
        logits = torch.randn(5, 32, 4, generator=gen(14))
        probs = unimix_probs(logits, 0.01)
        torch.testing.assert_close(probs.sum(-1), torch.ones(5, 32))
        self.assertTrue((probs > 0).all())

        sample = straight_through_sample(logits, 0.01, gen(15))
        self.assertEqual(tuple(sample.shape), (5, 32, 4))
        torch.testing.assert_close(sample.sum(-1), torch.ones(5, 32))
        self.assertTrue(torch.isin(sample, torch.tensor([0.0, 1.0])).all())

    def test_straight_through_gradients_reach_logits(self):
        logits = torch.randn(4, 32, 4, generator=gen(16)).requires_grad_(True)
        sample = straight_through_sample(logits, 0.01, gen(17))
        sample.square().sum().backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_empirical_frequencies_match_known_distribution(self):
        logits = torch.log(torch.tensor([[[0.1, 0.2, 0.3, 0.4]]]))
        expected = unimix_probs(logits, 0.01)[0, 0]

        draws = logits.expand(200_000, 1, 4)
        sample = straight_through_sample(draws, 0.01, gen(18))
        empirical = sample.mean(0)[0]

        torch.testing.assert_close(empirical, expected, atol=5e-3, rtol=0)


class TestIntegration(unittest.TestCase):
    def test_gradients_finite_through_a_short_sequence(self):
        encoder, rssm = build()
        batch, length = 2, 5
        obs = images(batch, length, seed=19)
        actions = torch.randn(batch, length - 1, A, generator=gen(20))

        post, prior = observe_sequence(encoder, rssm, obs, actions, generator=gen(21))
        (post.feat.square().mean() + prior.logit.square().mean()).backward()

        grads = [p.grad for p in list(encoder.parameters()) + list(rssm.parameters())]
        self.assertTrue(all(g is not None for g in grads))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.0)

    def test_public_api_shapes(self):
        encoder, rssm = build()
        batch, length = 3, 4
        obs = images(batch, length, seed=22)
        actions = torch.randn(batch, length - 1, A, generator=gen(23))

        self.assertEqual(tuple(encoder(obs).shape), (batch, length, 256))
        self.assertEqual(tuple(encoder(obs[:, 0]).shape), (batch, 256))

        post, prior = observe_sequence(encoder, rssm, obs, actions, generator=gen(24))
        for state in (post, prior):
            self.assertEqual(tuple(state.deter.shape), (batch, length, 512))
            self.assertEqual(tuple(state.stoch.shape), (batch, length, 32, 4))
            self.assertEqual(tuple(state.logit.shape), (batch, length, 32, 4))
            self.assertEqual(tuple(state.feat.shape), (batch, length, 640))

        initial = rssm.initial(batch)
        self.assertEqual(tuple(initial.deter.shape), (batch, 512))
        self.assertEqual(tuple(initial.feat.shape), (batch, 640))

        step = rssm.imagine_step(initial, actions[:, 0], None, gen(25))
        self.assertEqual(tuple(step.deter.shape), (batch, 512))
        self.assertEqual(tuple(step.stoch.shape), (batch, 32, 4))

        rolled = rssm.imagine(initial, actions, gen(26))
        self.assertEqual(tuple(rolled.deter.shape), (batch, length - 1, 512))
        self.assertEqual(tuple(rolled.feat.shape), (batch, length - 1, 640))

    def test_parameter_counts_match_the_derivation(self):
        encoder, rssm = build()
        self.assertEqual(
            sum(p.numel() for p in encoder.parameters()), DERIVED_ENCODER_PARAMS
        )
        self.assertEqual(sum(p.numel() for p in rssm.parameters()), DERIVED_RSSM_PARAMS)


if __name__ == "__main__":
    unittest.main()
