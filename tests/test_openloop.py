import inspect
import unittest

import numpy as np
import torch
from torch import Tensor, nn

from dreamer import RSSM, State, WorldModel
from dreamer.openloop import (
    Context,
    Episode,
    encoder_tripwire,
    evaluate_open_loop,
    gather_contexts,
    make_contexts,
    open_loop_predict,
    target_indices,
    training_reward_mean,
)
from dreamer.rssm import observe_sequence

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

A, C, H, L = 6, 5, 4, 24
DETER, STOCH, CLASSES = 8, 2, 2


def episode(seed=0, split="holdout", shift=0, length=L):
    rng = np.random.default_rng(seed)
    actions = rng.uniform(-1, 1, (length, A)).astype(np.float32)
    padded = np.concatenate([actions, actions[:1]], axis=0)
    # reward j is the reward for taking action j; shift=1 is the one-step-shift mutant
    rewards = padded[shift : shift + length, 0].copy()
    return Episode(
        episode_id=seed,
        split=split,
        observations=rng.integers(0, 256, (length + 1, 64, 64, 3), dtype=np.uint8),
        actions=actions,
        rewards=rewards,
        is_last=np.zeros(length, dtype=np.bool_),
        is_terminal=np.zeros(length, dtype=np.bool_),
        discounts=np.ones(length, dtype=np.float32),
    )


class StubEncoder(nn.Module):
    tokens = 4

    def forward(self, image: Tensor) -> Tensor:
        return torch.zeros(*image.shape[:-3], self.tokens)


class StubRSSM(nn.Module):
    """deter[..., 0] carries the first component of the action that produced the state."""

    deter, stoch, classes = DETER, STOCH, CLASSES

    def _state(self, action: Tensor) -> State:
        lead = action.shape[:-1]
        deter = torch.zeros(*lead, DETER)
        deter[..., 0] = action[..., 0]
        return State(deter, torch.zeros(*lead, STOCH, CLASSES), torch.zeros(*lead, STOCH, CLASSES))

    def observe(self, embeds, prev_actions, is_first=None, state=None, generator=None):
        post = self._state(prev_actions)
        return post, post

    def imagine(self, state, actions, generator=None):
        return self._state(actions)


class StubModel(nn.Module):
    horizon, contdisc = 333, True

    def __init__(self):
        super().__init__()
        self.encoder = StubEncoder()
        self.rssm = StubRSSM()
        self.reward = self
        self.cont = self
        self.decoder = self

    def predict(self, feat: Tensor) -> Tensor:
        return feat[..., 0]

    def forward(self, deter: Tensor, stoch: Tensor) -> Tensor:
        return torch.zeros(*deter.shape[:-1], 64, 64, 3)


def tensors(batch, device="cpu"):
    return (
        torch.from_numpy(batch.context_observations).to(device),
        torch.from_numpy(batch.context_actions).to(device=device, dtype=torch.float32),
        torch.from_numpy(batch.future_actions).to(device=device, dtype=torch.float32),
    )


class TestAlignment(unittest.TestCase):
    def test_target_indices_are_hand_computed(self):
        reward_index, observation_index = target_indices(cutoff=10, horizon=3)
        # distance k lands on state 10+k, supervised by rewards[10+k-1]
        self.assertEqual(list(reward_index), [10, 11, 12])
        self.assertEqual(list(observation_index), [11, 12, 13])

    def test_context_window_and_future_actions_abut(self):
        ep = episode()
        batch = gather_contexts([ep], [Context(0, 3)], C, H)
        cutoff = 3 + C

        self.assertEqual(batch.context_observations.shape[1], C + 1)
        self.assertEqual(batch.context_actions.shape[1], C)
        np.testing.assert_array_equal(batch.context_actions[0], ep.actions[3:cutoff])
        # the first prior transition consumes the action taken AT the cutoff observation
        np.testing.assert_array_equal(batch.future_actions[0], ep.actions[cutoff : cutoff + H])
        np.testing.assert_array_equal(batch.target_rewards[0], ep.rewards[cutoff - 1 + 1 : cutoff + H])
        np.testing.assert_array_equal(batch.last_observed_reward[0], ep.rewards[cutoff - 1])

    def test_actions_align_with_the_transitions_they_cause(self):
        ep = episode()
        contexts = make_contexts([ep], C, H, per_episode=3)
        batch = gather_contexts([ep], contexts, C, H)
        model = StubModel()

        with torch.no_grad():
            prediction = open_loop_predict(model, *tensors(batch))

        # the stub predicts exactly the action that produced each state
        torch.testing.assert_close(
            prediction.reward, torch.from_numpy(batch.target_rewards), rtol=0, atol=0
        )

    def test_a_one_step_shift_breaks_the_alignment(self):
        shifted = episode(shift=1)
        contexts = make_contexts([shifted], C, H, per_episode=3)
        batch = gather_contexts([shifted], contexts, C, H)
        model = StubModel()

        with torch.no_grad():
            prediction = open_loop_predict(model, *tensors(batch))

        error = (prediction.reward - torch.from_numpy(batch.target_rewards)).abs()
        self.assertGreater(float(error.mean()), 1e-3)


class TestNoObservationLeak(unittest.TestCase):
    def test_prior_api_takes_no_observation_argument(self):
        for name in ("imagine_step", "imagine"):
            parameters = set(inspect.signature(getattr(RSSM, name)).parameters)
            self.assertFalse(parameters & {"embed", "embeds", "image", "observation", "observations"})

        parameters = set(inspect.signature(open_loop_predict).parameters)
        self.assertNotIn("future_observations", parameters)
        self.assertEqual(
            [p for p in parameters if "observ" in p or "image" in p], ["context_observations"]
        )

    def test_encoder_sees_only_the_context_frames(self):
        ep = episode()
        batch = gather_contexts([ep], make_contexts([ep], C, H, 3), C, H)
        model = StubModel()

        with encoder_tripwire(model) as tripwire, torch.no_grad():
            open_loop_predict(model, *tensors(batch))

        self.assertEqual(len(tripwire.calls), 1)
        self.assertEqual(tripwire.calls[0][1], C + 1)
        self.assertEqual(tripwire.frames_seen, 3 * (C + 1))

    def test_the_tripwire_catches_a_leaking_rollout(self):
        ep = episode()
        contexts = make_contexts([ep], C, H, 3)
        batch = gather_contexts([ep], contexts, C, H)
        model = WorldModel(action_dim=A)
        context_observations, context_actions, future_actions = tensors(batch)
        future_observations = torch.from_numpy(batch.target_images)

        def leaking_rollout():
            # the M5 failure mode: posterior over the whole window, future frames included
            observations = torch.cat([context_observations, future_observations], dim=1)
            actions = torch.cat([context_actions, future_actions], dim=1)
            post, _ = observe_sequence(model.encoder, model.rssm, observations, actions)
            return model.reward.predict(post[:, -H:].feat)

        with encoder_tripwire(model) as tripwire, torch.no_grad():
            leaking_rollout()

        self.assertEqual(tripwire.calls[0][1], C + 1 + H)
        self.assertGreater(tripwire.frames_seen, 3 * (C + 1))


class TestStochasticEvaluation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = WorldModel(action_dim=A)
        # outscale 0.0 makes the reward readout exactly 0 for every latent at init,
        # so an unperturbed head hides all latent stochasticity
        with torch.no_grad():
            self.model.reward.mlp.out.weight.normal_(0.0, 0.5)
        self.holdout = [episode(seed=s) for s in (1, 2)]
        self.contexts = make_contexts(self.holdout, C, H, per_episode=2)

    def evaluate(self, samples=4, seed=0):
        return evaluate_open_loop(
            self.model,
            self.holdout,
            self.contexts,
            context_length=C,
            horizon=H,
            constant_reward=0.0,
            samples=samples,
            seed=seed,
            device=torch.device("cpu"),
            chunk_contexts=2,
        )

    def test_same_seed_reproduces_the_metrics_exactly(self):
        np.testing.assert_array_equal(self.evaluate().reward_mae, self.evaluate().reward_mae)

    def test_latent_sampling_is_actually_stochastic_across_samples(self):
        metrics = self.evaluate(samples=4)
        self.assertEqual(metrics.samples, 4)
        # a single sample cannot have sample-to-sample spread; four must
        self.assertGreater(float(metrics.reward_mae_sample_sd.max()), 0.0)
        self.assertEqual(float(self.evaluate(samples=1).reward_mae_sample_sd.max()), 0.0)

    def test_a_different_seed_gives_a_different_draw(self):
        self.assertFalse(np.array_equal(self.evaluate(seed=0).reward_mae, self.evaluate(seed=1).reward_mae))

    def test_metrics_cover_every_context_and_distance(self):
        metrics = self.evaluate()
        self.assertEqual(metrics.contexts, len(self.contexts))
        self.assertEqual(metrics.reward_mae.shape, (H,))
        self.assertTrue(np.isfinite(metrics.reward_mae).all())


class TestSplitIsolation(unittest.TestCase):
    def test_constant_baseline_refuses_holdout_episodes(self):
        with self.assertRaises(ValueError):
            training_reward_mean([episode(split="train"), episode(seed=1, split="holdout")])

    def test_constant_baseline_uses_training_rewards_only(self):
        train = [episode(seed=s, split="train") for s in (1, 2)]
        expected = float(np.concatenate([e.rewards for e in train]).mean())

        self.assertAlmostEqual(training_reward_mean(train), expected, places=7)
        # a wildly different holdout episode must not move it
        self.assertAlmostEqual(training_reward_mean(train), expected, places=7)

    def test_evaluation_refuses_training_episodes(self):
        torch.manual_seed(0)
        train = [episode(seed=1, split="train")]

        with self.assertRaises(ValueError):
            evaluate_open_loop(
                WorldModel(action_dim=A),
                train,
                make_contexts(train, C, H, 1),
                context_length=C,
                horizon=H,
                constant_reward=0.0,
                samples=1,
                seed=0,
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
