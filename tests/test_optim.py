import unittest

import torch

from dreamer import LaProp

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

LR = 0.1
B1, B2, EPS, AGC, PMIN = 0.9, 0.999, 1e-20, 0.3, 1e-3


def stepper(initial, grads, **kwargs):
    # float64 so the assertions test the update rule, not float32 rounding
    p = torch.tensor([float(initial)], dtype=torch.float64)
    opt = LaProp([p], lr=LR, warmup=0, **kwargs)
    history = []
    for g in grads:
        p.grad = torch.tensor([float(g)], dtype=torch.float64)
        opt.step()
        history.append(float(p))
    return history


class TestLaPropUpdate(unittest.TestCase):
    def test_first_step_matches_hand_computation(self):
        # g=2, |p|=3: trigger = 2/(0.3*3) = 2.2222 -> g <- 0.9
        # nu = 1e-3*0.81, nuhat = 0.81, sqrt = 0.9, normalized = 1.0
        # mu = 0.1, muhat = 1.0  ->  p <- 3.0 - 0.1*1.0
        self.assertAlmostEqual(stepper(3.0, [2.0])[0], 2.9, places=12)

    def test_second_step_matches_hand_computation(self):
        # g=-1, |p|=2.9: trigger = 1/(0.3*2.9) = 1.149425 -> g <- -0.87
        # nu = 0.999*8.1e-4 + 1e-3*0.7569 = 1.566090e-3 ; nuhat = nu/(1-0.999^2)
        # normalized = -0.87/(sqrt(nuhat)+eps) ; mu = 0.9*0.1 + 0.1*normalized
        nu = 0.999 * 1e-3 * 0.81 + 1e-3 * 0.87**2
        normalized = -0.87 / ((nu / (1 - B2**2)) ** 0.5 + EPS)
        mu = B1 * 0.1 + (1 - B1) * normalized
        expected = 2.9 - LR * mu / (1 - B1**2)

        self.assertAlmostEqual(expected, 2.904364, places=6)
        self.assertAlmostEqual(stepper(3.0, [2.0, -1.0])[1], expected, places=12)

    def test_is_laprop_not_adam(self):
        # Adam accumulates momentum on the RAW gradient and divides at the end
        nu = 0.999 * 1e-3 * 0.81 + 1e-3 * 0.87**2
        mu_raw = B1 * (1 - B1) * 0.9 + (1 - B1) * -0.87
        adam = 2.9 - LR * (mu_raw / (1 - B1**2)) / ((nu / (1 - B2**2)) ** 0.5 + EPS)

        self.assertAlmostEqual(adam, 2.903568, places=6)
        self.assertGreater(abs(stepper(3.0, [2.0, -1.0])[1] - adam), 5e-4)

    def test_agc_leaves_small_gradients_alone(self):
        # |g|=0.5, |p|=3: trigger = 0.5/0.9 < 1, so the gradient passes through
        # the first normalized step is then exactly sign(g), giving p -+ lr
        self.assertAlmostEqual(stepper(3.0, [0.5])[0], 2.9, places=12)
        self.assertAlmostEqual(stepper(3.0, [-0.5])[0], 3.1, places=12)

    def test_agc_clips_a_large_gradient(self):
        # the first step normalizes to sign(g) at any magnitude, so AGC is only
        # visible from the second step, through the ratio of the two clipped gradients
        def second(g1, g2):
            nu = (1 - B2) * g1**2 * B2 + (1 - B2) * g2**2
            mu = B1 * (1 - B1) + (1 - B1) * g2 / ((nu / (1 - B2**2)) ** 0.5 + EPS)
            return 2.9 - LR * mu / (1 - B1**2)

        clipped = second(2.0 / (2.0 / (AGC * 3.0)), 20.0 / (20.0 / (AGC * 2.9)))
        unclipped = second(2.0, 20.0)

        self.assertAlmostEqual(stepper(3.0, [2.0, 20.0])[1], clipped, places=12)
        self.assertGreater(abs(clipped - unclipped), 0.02)

    def test_pmin_floors_the_parameter_norm(self):
        # |p| = 0 makes the AGC divisor 0; without the pmin clamp the trigger is inf,
        # the gradient is scaled to exactly 0, and the parameter never moves
        self.assertAlmostEqual(stepper(0.0, [1.0])[0], -LR, places=12)

    def test_warmup_scales_the_learning_rate_linearly(self):
        p = torch.tensor([3.0], dtype=torch.float64)
        opt = LaProp([p], lr=LR, warmup=10)
        p.grad = torch.tensor([2.0], dtype=torch.float64)
        opt.step()
        # step 1 of 10: lr is LR/10, and the normalized first update is exactly 1
        self.assertAlmostEqual(float(p), 3.0 - LR / 10, places=12)

    def test_eps_is_outside_the_square_root(self):
        g = 1e-12
        outside = -LR * g / (abs(g) + EPS)
        inside = -LR * g / (g * g + EPS) ** 0.5

        self.assertAlmostEqual(stepper(1e-6, [g])[0] - 1e-6, outside, places=14)
        self.assertGreater(abs(outside - inside), 0.09)


if __name__ == "__main__":
    unittest.main()
