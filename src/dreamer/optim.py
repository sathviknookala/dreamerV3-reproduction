from __future__ import annotations

import torch
from torch import Tensor


class LaProp(torch.optim.Optimizer):
    """spec.md 5.8: clip_by_agc -> scale_by_rms -> scale_by_momentum -> lr. Not Adam."""

    def __init__(
        self,
        params,
        lr: float = 4e-5,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-20,
        agc: float = 0.3,
        pmin: float = 1e-3,
        warmup: int = 1000,
    ) -> None:
        super().__init__(
            params,
            dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, agc=agc, pmin=pmin, warmup=warmup),
        )
        self._step = 0

    def state_dict(self) -> dict:
        # _step drives the bias correction and the lr warm-up; torch.optim does not carry it
        state = super().state_dict()
        state["_step"] = self._step
        return state

    def load_state_dict(self, state: dict) -> None:
        state = dict(state)
        self._step = int(state.pop("_step", 0))
        super().load_state_dict(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self._step += 1
        t = self._step

        for group in self.param_groups:
            b1, b2, eps = group["beta1"], group["beta2"], group["eps"]
            warmup = group["warmup"]
            lr = group["lr"] * (min(1.0, t / warmup) if warmup else 1.0)

            for p in group["params"]:
                if p.grad is None:
                    continue

                g: Tensor = p.grad
                state = self.state[p]

                if not state:
                    state["nu"] = torch.zeros_like(p)
                    state["mu"] = torch.zeros_like(p)

                # AGC on the whole-tensor norm, not the per-row variant
                trigger = g.norm(2) / (group["agc"] * p.norm(2).clamp(min=group["pmin"]))
                g = g / trigger.clamp(min=1.0)

                nu, mu = state["nu"], state["mu"]
                nu.mul_(b2).addcmul_(g, g, value=1 - b2)
                # eps is OUTSIDE the sqrt
                normalized = g / (nu.div(1 - b2**t).sqrt() + eps)

                mu.mul_(b1).add_(normalized, alpha=1 - b1)
                p.add_(mu / (1 - b1**t), alpha=-lr)

        return loss
