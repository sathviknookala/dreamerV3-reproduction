import inspect
import unittest

import torch
from torch import Tensor, nn

from dreamer import RSSM, State, WorldModel
from dreamer.imagine import (
    ActionProvider,
    Imagination,
    RandomActionProvider,
    SequenceActionProvider,
    imagine_trajectory,
    select_start_states,
    trajectory_weight,
    valid_start_mask,
)
from dreamer.openloop import encoder_tripwire
from dreamer.rssm import observe_sequence

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

A, N, H = 6, 3, 4
DETER, STOCH, CLASSES = 8, 2, 2
CONT = 0.75


class StubRSSM(nn.Module):
    """deter[..., 0] carries the first component of the action that produced the state."""

    action_dim, deter, stoch, classes = A, DETER, STOCH, CLASSES

    def imagine_step(self, state, action, is_first=None, generator=None):
        lead = action.shape[:-1]
        deter = torch.zeros(*lead, DETER)
        deter[..., 0] = action[..., 0]
        # remember where we came from, so a dropped or repeated step is visible
        deter[..., 1] = state.deter[..., 0]
        zeros = torch.zeros(*lead, STOCH, CLASSES)
        return State(deter, zeros, zeros)


class StubHead(nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn
        self.calls = 0

    def predict(self, feat: Tensor) -> Tensor:
        self.calls += 1
        return self.fn(feat)


class StubDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, deter: Tensor, stoch: Tensor) -> Tensor:
        self.calls += 1
        return torch.zeros(*deter.shape[:-1], 64, 64, 3)


class StubModel(nn.Module):
    horizon, contdisc = 333, True

    def __init__(self, cont: float = CONT) -> None:
        super().__init__()
        self.rssm = StubRSSM()
        self.reward = StubHead(lambda feat: feat[..., 0])
        self.cont = StubHead(lambda feat: torch.full(feat.shape[:-1], cont))
        self.decoder = StubDecoder()


def start_state(batch: int = N) -> State:
    zeros = torch.zeros(batch, STOCH, CLASSES)
    return State(torch.zeros(batch, DETER), zeros, zeros)


def scripted(values, batch: int = N) -> SequenceActionProvider:
    actions = torch.zeros(batch, len(values), A)
    for step, value in enumerate(values):
        actions[:, step, 0] = value
    return SequenceActionProvider(actions)


class TestOutputContract(unittest.TestCase):
    def test_shapes_are_H_plus_one_states_and_H_of_everything_else(self):
        out = imagine_trajectory(StubModel(), start_state(), scripted(range(H)), H)

        self.assertEqual(tuple(out.states.deter.shape), (N, H + 1, DETER))
        self.assertEqual(tuple(out.states.stoch.shape), (N, H + 1, STOCH, CLASSES))
        self.assertEqual(tuple(out.feat.shape), (N, H + 1, DETER + STOCH * CLASSES))
        self.assertEqual(tuple(out.actions.shape), (N, H, A))
        self.assertEqual(tuple(out.reward.shape), (N, H))
        self.assertEqual(tuple(out.cont.shape), (N, H))
        self.assertEqual(tuple(out.cont_start.shape), (N, 1))
        self.assertEqual(tuple(out.weight.shape), (N, H + 1))
        self.assertEqual(out.horizon, H)
        self.assertEqual(out.rollouts, N)

    def test_the_start_state_is_returned_unchanged_as_state_zero(self):
        start = start_state()
        start.deter[:, 0] = 0.5
        out = imagine_trajectory(StubModel(), start, scripted(range(H)), H)

        torch.testing.assert_close(out.states.deter[:, 0], start.deter, rtol=0, atol=0)

    def test_image_is_absent_unless_decoding_is_requested(self):
        model = StubModel()
        out = imagine_trajectory(model, start_state(), scripted(range(H)), H)

        self.assertIsNone(out.image)
        self.assertEqual(model.decoder.calls, 0)

        decoded = imagine_trajectory(model, start_state(), scripted(range(H)), H, decode=True)
        self.assertEqual(tuple(decoded.image.shape), (N, H + 1, 64, 64, 3))
        self.assertEqual(model.decoder.calls, 1)

    def test_a_provider_returning_the_wrong_shape_is_rejected(self):
        bad = SequenceActionProvider(torch.zeros(N, H, A + 1))

        with self.assertRaises(ValueError):
            imagine_trajectory(StubModel(), start_state(), bad, H)

    def test_a_non_positive_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            imagine_trajectory(StubModel(), start_state(), scripted([0]), 0)


class TestAlignment(unittest.TestCase):
    def test_step_i_reward_comes_from_the_state_action_i_produced(self):
        values = [0.1, -0.4, 0.9, 0.25]
        out = imagine_trajectory(StubModel(), start_state(), scripted(values), H)

        # the stub reads back the action that produced each state, exactly
        torch.testing.assert_close(out.reward, out.actions[..., 0], rtol=0, atol=0)
        torch.testing.assert_close(
            out.reward[0], torch.tensor(values), rtol=0, atol=0
        )

    def test_the_chain_advances_one_step_per_action(self):
        values = [0.1, -0.4, 0.9, 0.25]
        out = imagine_trajectory(StubModel(), start_state(), scripted(values), H)

        # deter[1] of state t+1 holds deter[0] of state t
        torch.testing.assert_close(
            out.states.deter[:, 1:, 1], out.states.deter[:, :-1, 0], rtol=0, atol=0
        )

    def test_a_one_step_reward_shift_breaks_the_alignment(self):
        values = [0.1, -0.4, 0.9, 0.25]
        out = imagine_trajectory(StubModel(), start_state(), scripted(values), H)
        shifted = out.feat[:, :-1, 0]

        self.assertGreater(float((shifted - out.actions[..., 0]).abs().max()), 1e-3)

    def test_continuation_is_read_at_the_same_states_as_the_reward(self):
        model = StubModel()
        # a continuation head that reports the action that produced its state
        model.cont = StubHead(lambda feat: feat[..., 0])
        out = imagine_trajectory(model, start_state(), scripted([0.1, 0.2, 0.3, 0.4]), H)

        torch.testing.assert_close(out.cont, out.reward, rtol=0, atol=0)
        torch.testing.assert_close(
            out.cont_start[:, 0], out.states.deter[:, 0, 0], rtol=0, atol=0
        )


class TestContinuationWeights(unittest.TestCase):
    def test_weight_is_the_hand_computed_cumulative_product(self):
        out = imagine_trajectory(StubModel(CONT), start_state(), scripted(range(H)), H)
        expected = torch.tensor([CONT ** (t + 1) for t in range(H + 1)])

        torch.testing.assert_close(out.weight, expected.expand(N, H + 1), rtol=0, atol=1e-6)

    def test_weight_zero_equals_the_start_continuation_not_one(self):
        out = imagine_trajectory(StubModel(CONT), start_state(), scripted(range(H)), H)

        torch.testing.assert_close(out.weight[:, :1], out.cont_start, rtol=0, atol=0)
        # dropping the start continuation is the plausible mutant; it must differ
        self.assertNotAlmostEqual(float(out.weight[0, 0]), 1.0, places=3)

    def test_a_terminal_step_zeroes_every_later_weight(self):
        model = StubModel()
        # continuation collapses to 0 at the second imagined state
        model.cont = StubHead(
            lambda feat: torch.where(feat[..., 0] == 2.0, 0.0, 1.0).expand(feat.shape[:-1])
        )
        out = imagine_trajectory(model, start_state(), scripted([1.0, 2.0, 3.0, 4.0]), H)

        torch.testing.assert_close(
            out.weight[0], torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]), rtol=0, atol=0
        )

    def test_disc_is_one_only_because_contdisc_folds_it_into_the_label(self):
        model = StubModel(CONT)
        model.contdisc = False
        out = imagine_trajectory(model, start_state(), scripted(range(H)), H)
        disc = 1.0 - 1.0 / model.horizon
        expected = torch.tensor([disc**t * CONT ** (t + 1) for t in range(H + 1)])

        torch.testing.assert_close(out.weight[0], expected, rtol=0, atol=1e-6)

    def test_trajectory_weight_matches_the_reference_expression(self):
        cont = torch.rand(N, H + 1).clamp(min=0.1)

        torch.testing.assert_close(
            trajectory_weight(cont, 1.0), torch.cumprod(cont, dim=1), rtol=0, atol=0
        )


class TestStartSelection(unittest.TestCase):
    def states(self, batch=2, length=6):
        deter = torch.arange(batch * length * DETER, dtype=torch.float32)
        deter = deter.reshape(batch, length, DETER)
        zeros = torch.zeros(batch, length, STOCH, CLASSES)
        return State(deter, zeros, zeros)

    def test_burn_in_positions_are_not_rollout_starts(self):
        post = self.states()
        loss_mask = torch.ones(2, 5, dtype=torch.bool)
        loss_mask[:, :2] = False
        is_terminal = torch.zeros(2, 5, dtype=torch.bool)

        starts, index = select_start_states(post, loss_mask, is_terminal)

        self.assertEqual(starts.deter.shape[0], 6)
        # masked positions map onto states 3..5, never onto the burn-in prefix or the reset state
        torch.testing.assert_close(starts.deter[:3], post.deter[0, 3:], rtol=0, atol=0)
        self.assertEqual(list(index), [2, 3, 4, 7, 8, 9])

    def test_terminal_states_are_excluded(self):
        post = self.states()
        loss_mask = torch.ones(2, 5, dtype=torch.bool)
        is_terminal = torch.zeros(2, 5, dtype=torch.bool)
        is_terminal[0, 3] = True

        starts, index = select_start_states(post, loss_mask, is_terminal)

        self.assertEqual(starts.deter.shape[0], 9)
        self.assertNotIn(3, list(index))

    def test_a_time_limit_boundary_is_still_a_valid_start(self):
        # is_last with a nonzero discount is a time limit, not a terminal -- it must not be dropped
        loss_mask = torch.ones(1, 4, dtype=torch.bool)
        is_last = torch.zeros(1, 4, dtype=torch.bool)
        is_last[0, 3] = True
        is_terminal = torch.zeros(1, 4, dtype=torch.bool)

        keep = valid_start_mask(loss_mask, is_terminal)

        self.assertTrue(bool(keep[0, 3]))
        self.assertEqual(int(keep.sum()), 4)
        self.assertEqual(int(is_last.sum()), 1)

    def test_mask_length_must_match_the_posterior_sequence(self):
        post = self.states()

        with self.assertRaises(ValueError):
            select_start_states(post, torch.ones(2, 6, dtype=torch.bool), torch.zeros(2, 6, dtype=torch.bool))


class TestNoObservationLeak(unittest.TestCase):
    def test_no_imagination_api_takes_an_observation_argument(self):
        # substrings, not exact names: `future_observations` is the mutant a name set misses
        banned = ("observ", "image", "frame", "embed", "pixel")

        def offenders(function):
            return [
                p
                for p in inspect.signature(function).parameters
                if any(token in p for token in banned)
            ]

        for function in (imagine_trajectory, select_start_states, trajectory_weight):
            self.assertEqual(offenders(function), [], function.__name__)

        for name in ("imagine_step", "imagine"):
            self.assertEqual(offenders(getattr(RSSM, name)), [], name)

        self.assertEqual(
            set(inspect.signature(ActionProvider.__call__).parameters), {"self", "state"}
        )

    def test_the_encoder_is_never_called_during_imagination(self):
        model = WorldModel(action_dim=A)
        observations = torch.randint(0, 256, (2, 4, 64, 64, 3), dtype=torch.uint8)
        post, _ = observe_sequence(model.encoder, model.rssm, observations, torch.zeros(2, 3, A))

        with encoder_tripwire(model) as tripwire:
            imagine_trajectory(model, post[:, -1], RandomActionProvider(A, seed=0), 5)

        self.assertEqual(tripwire.calls, [])
        self.assertEqual(tripwire.frames_seen, 0)

    def test_the_tripwire_catches_a_posterior_call_during_the_rollout(self):
        model = WorldModel(action_dim=A)
        observations = torch.randint(0, 256, (2, 4, 64, 64, 3), dtype=torch.uint8)
        post, _ = observe_sequence(model.encoder, model.rssm, observations, torch.zeros(2, 3, A))
        future = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)

        class LeakingProvider:
            def __call__(self, state):
                # the failure mode: a "provider" that peeks at the next frame
                embed = model.encoder(future)
                model.rssm.observe_step(state, torch.zeros(2, A), embed)
                return torch.zeros(2, A)

        with encoder_tripwire(model) as tripwire:
            imagine_trajectory(model, post[:, -1], LeakingProvider(), 5)

        self.assertEqual(len(tripwire.calls), 5)
        self.assertEqual(tripwire.frames_seen, 10)

    def test_imagination_reaches_no_simulator(self):
        module = inspect.getmodule(imagine_trajectory)
        forbidden = {"DMCEnv", "Collector", "ReplayBuffer", "Environment", "EnvStep"}

        self.assertFalse(forbidden & set(vars(module)))
        source = inspect.getsource(module)
        for token in ("dm_control", "DMCEnv", "env.step", ".reset()"):
            self.assertNotIn(token, source)


class TestActionSensitivity(unittest.TestCase):
    def rollout(self, provider, seed=0, horizon=5):
        torch.manual_seed(0)
        model = WorldModel(action_dim=A)
        # outscale 0.0 makes the reward readout exactly 0 for every latent at init
        with torch.no_grad():
            model.reward.mlp.out.weight.normal_(0.0, 0.5)
        generator = torch.Generator()
        generator.manual_seed(seed)
        start = model.rssm.initial(2, device=torch.device("cpu"))
        return imagine_trajectory(model, start, provider, horizon, generator)

    def test_different_actions_give_different_states_and_rewards(self):
        low = self.rollout(SequenceActionProvider(torch.full((2, 5, A), -1.0)))
        high = self.rollout(SequenceActionProvider(torch.full((2, 5, A), 1.0)))

        self.assertGreater(float((low.states.deter - high.states.deter).abs().max()), 1e-4)
        self.assertGreater(float((low.reward - high.reward).abs().max().detach()), 1e-6)

    def test_the_random_provider_is_reproducible_and_seed_dependent(self):
        first = self.rollout(RandomActionProvider(A, seed=7))
        again = self.rollout(RandomActionProvider(A, seed=7))
        other = self.rollout(RandomActionProvider(A, seed=8))

        torch.testing.assert_close(first.actions, again.actions, rtol=0, atol=0)
        torch.testing.assert_close(first.states.deter, again.states.deter, rtol=0, atol=0)
        self.assertGreater(float((first.actions - other.actions).abs().max()), 1e-6)

    def test_the_latent_stream_is_reproducible_and_seed_dependent(self):
        actions = torch.full((2, 5, A), 0.3)
        first = self.rollout(SequenceActionProvider(actions), seed=1)
        again = self.rollout(SequenceActionProvider(actions), seed=1)
        other = self.rollout(SequenceActionProvider(actions), seed=2)

        torch.testing.assert_close(first.states.stoch, again.states.stoch, rtol=0, atol=0)
        self.assertGreater(float((first.states.stoch - other.states.stoch).abs().max()), 0.0)

    def test_the_random_provider_respects_its_bounds(self):
        provider = RandomActionProvider(A, seed=0, low=-0.5, high=0.5)
        out = self.rollout(provider)

        self.assertGreaterEqual(float(out.actions.min()), -0.5)
        self.assertLessEqual(float(out.actions.max()), 0.5)


class TestRealWorldModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = WorldModel(action_dim=A)
        self.generator = torch.Generator()

    def start(self, batch=2):
        return self.model.rssm.initial(batch, device=torch.device("cpu"))

    def test_every_project_horizon_gives_finite_correctly_shaped_output(self):
        for horizon in (5, 15, 30):
            self.generator.manual_seed(0)
            out = imagine_trajectory(
                self.model, self.start(), RandomActionProvider(A, seed=0), horizon, self.generator
            )

            self.assertEqual(tuple(out.states.deter.shape), (2, horizon + 1, 512))
            self.assertEqual(tuple(out.actions.shape), (2, horizon, A))
            self.assertEqual(tuple(out.reward.shape), (2, horizon))
            self.assertEqual(tuple(out.cont.shape), (2, horizon))
            self.assertEqual(tuple(out.weight.shape), (2, horizon + 1))
            for tensor in (out.feat, out.reward, out.cont, out.weight, out.actions):
                self.assertTrue(bool(torch.isfinite(tensor).all()), horizon)

    def test_a_scripted_provider_reproduces_the_M5_prior_only_path(self):
        actions = torch.rand(2, 15, A) * 2 - 1
        self.generator.manual_seed(3)
        out = imagine_trajectory(
            self.model, self.start(), SequenceActionProvider(actions), 15, self.generator
        )
        self.generator.manual_seed(3)
        reference = self.model.rssm.imagine(self.start(), actions, self.generator)

        torch.testing.assert_close(out.states.deter[:, 1:], reference.deter, rtol=0, atol=0)
        torch.testing.assert_close(out.states.stoch[:, 1:], reference.stoch, rtol=0, atol=0)

    def test_imagined_dynamics_carry_no_gradient_to_the_world_model(self):
        start = self.start()
        start.deter.requires_grad_(True)
        out = imagine_trajectory(self.model, start, RandomActionProvider(A, seed=0), 5)

        self.assertFalse(out.states.deter.requires_grad)
        self.assertFalse(out.feat.requires_grad)
        self.assertFalse(out.actions.requires_grad)

        out.reward.sum().backward()
        self.assertIsNotNone(self.model.reward.mlp.out.weight.grad)
        self.assertIsNone(self.model.rssm.dyngru.kernel.grad)
        self.assertIsNone(self.model.encoder.convs[0].weight.grad)
        self.assertIsNone(start.deter.grad)


if __name__ == "__main__":
    unittest.main()
