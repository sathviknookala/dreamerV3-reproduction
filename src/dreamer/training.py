from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .actor import ACTENT, BehaviorOutput, behavior_losses
from .agent import Agent, EpisodicEnv
from .collector import Collector, UniformRandomPolicy
from .config import RunConfig
from .critic import CriticOutput, scatter_imagined_return
from .imagine import Imagination, imagine_trajectory, select_start_states
from .optim import LaProp
from .replay import ReplayBuffer
from .types import SequenceBatch, StepCounters
from .world_model import WorldModelOutput


@dataclass
class TrainingScheduler:
    """spec 7.7: `train_ratio` loss-bearing positions per collected transition.

    Credits are integer positions, so the remainder is exact and survives a checkpoint; a scheduler
    that rounded per-step would drift by thousands of updates over a 1e6-step run.
    """

    train_ratio: int = 64
    positions_per_update: int = 1024
    credits: int = 0

    def __post_init__(self) -> None:
        if self.train_ratio <= 0:
            raise ValueError("train_ratio must be positive")

        if self.positions_per_update <= 0:
            raise ValueError("positions_per_update must be positive")

    def credit(self, transitions: int = 1) -> None:
        if transitions < 0:
            raise ValueError("cannot credit a negative number of transitions")

        self.credits += int(transitions) * self.train_ratio

    def take(self) -> int:
        updates = self.credits // self.positions_per_update
        self.credits -= updates * self.positions_per_update
        return int(updates)

    def state_dict(self) -> dict:
        return {
            "train_ratio": self.train_ratio,
            "positions_per_update": self.positions_per_update,
            "credits": self.credits,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["train_ratio"]) != self.train_ratio:
            raise ValueError(
                f"checkpoint train_ratio {state['train_ratio']} does not match {self.train_ratio}"
            )

        if int(state["positions_per_update"]) != self.positions_per_update:
            raise ValueError("checkpoint positions_per_update does not match the configuration")

        self.credits = int(state["credits"])


@dataclass
class UpdateOutput:
    world: WorldModelOutput
    behavior: BehaviorOutput
    replay_critic: CriticOutput
    imagination: Imagination
    total: Tensor
    starts: int
    metrics: dict[str, float] = field(default_factory=dict)


def _batch_tensors(batch: SequenceBatch, device) -> dict[str, Tensor]:
    def to(array, dtype=None):
        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device)
        return tensor if dtype is None else tensor.to(dtype)

    return {
        "observations": to(batch.observations),
        "actions": to(batch.actions, torch.float32),
        "rewards": to(batch.rewards, torch.float32),
        "is_last": to(batch.is_last, torch.bool),
        "is_terminal": to(batch.is_terminal, torch.bool),
        "loss_mask": to(batch.loss_mask, torch.float32),
    }


def compute_losses(
    agent: Agent,
    batch: SequenceBatch,
    horizon: int | None = None,
    actent: float = ACTENT,
) -> UpdateOutput:
    """One world-model pass, one imagination, one behaviour call, one replay-critic fit."""
    config = agent.config
    horizon = int(config.horizon if horizon is None else horizon)
    burn_in = config.burn_in
    tensors = _batch_tensors(batch, agent.device)

    length = batch.transition_length

    if length != config.sequence_length:
        raise ValueError(
            f"batch carries {length} transitions, expected P+T = {config.sequence_length}"
        )

    # 1. the world-model loss and the posterior sequence, computed exactly once
    world = agent.world_model.loss(
        tensors["observations"],
        tensors["actions"],
        tensors["rewards"],
        tensors["is_terminal"],
        tensors["loss_mask"],
        generator=agent.generators["replay"],
    )

    # 2. loss-bearing, non-terminal starts only
    starts, index = select_start_states(
        world.post, tensors["loss_mask"], tensors["is_terminal"]
    )

    if starts.deter.shape[0] == 0:
        raise RuntimeError("no loss-bearing non-terminal position is available as a rollout start")

    # 3. actor-driven, prior-only rollout
    imagination = imagine_trajectory(
        agent.world_model, starts, agent.provider, horizon, agent.generators["imagine"]
    )

    # 4. one behaviour call; the replay critic is NOT attached here, it is formed below
    behavior = behavior_losses(
        agent.actor,
        agent.critic,
        imagination,
        agent.normalizer,
        replay=None,
        actent=actent,
        update=True,
    )

    # 5. the imagined return at the start state goes back to the transition it started from
    grid, filled = scatter_imagined_return(
        behavior.imagined.ret, index, batch.batch_size, length
    )

    # 6. burn-in is recomputed, never loss-bearing: the replay critic sees the T-position suffix
    replay_critic = agent.critic.replay_loss(
        world.post[:, burn_in + 1 :].feat,
        tensors["rewards"][:, burn_in:],
        tensors["is_last"][:, burn_in:].to(torch.float32),
        tensors["is_terminal"][:, burn_in:].to(torch.float32),
        grid[:, burn_in:],
        filled=filled[:, burn_in:],
    )

    # 7. the three scales: world model already carries its own, actor and imagined critic 1.0
    total = world.total + behavior.loss + agent.critic.scales["repval"] * replay_critic.loss

    metrics = dict(world.metrics)
    metrics.update(behavior.metrics)
    metrics.update(replay_critic.metrics)
    metrics.update(
        total=float(total.detach()),
        horizon=float(horizon),
        imagination_starts=float(starts.deter.shape[0]),
        imagined_transitions=float(starts.deter.shape[0] * horizon),
        retnorm_lo=float(agent.normalizer.lo),
        retnorm_hi=float(agent.normalizer.hi),
        retnorm_scale=float(agent.normalizer.scale),
        weight_sum=float(imagination.weight[:, :-1].detach().sum()),
    )

    return UpdateOutput(
        world=world,
        behavior=behavior,
        replay_critic=replay_critic,
        imagination=imagination,
        total=total,
        starts=int(starts.deter.shape[0]),
        metrics=metrics,
    )


def apply_update(agent: Agent, optimizer: LaProp, output: UpdateOutput) -> float:
    """8. one backward and one LaProp step; 9. the slow critic advances AFTER the step."""
    optimizer.zero_grad(set_to_none=True)
    output.total.backward()

    with torch.no_grad():
        grads = [p.grad for p in agent.trainable_parameters() if p.grad is not None]
        grad_norm = float(
            torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
        ) if grads else 0.0

    optimizer.step()
    # after the step: advancing before it would make the regularizer target the pre-update critic
    agent.critic.update_slow()

    output.metrics["grad_norm"] = grad_norm
    return grad_norm


def training_update(
    agent: Agent,
    optimizer: LaProp,
    batch: SequenceBatch,
    horizon: int | None = None,
    actent: float = ACTENT,
) -> UpdateOutput:
    output = compute_losses(agent, batch, horizon, actent)
    apply_update(agent, optimizer, output)
    return output


LOG_COLUMNS = (
    "segment", "env_step", "gradient_step", "elapsed_s", "eval_seconds", "total", "rec", "rew", "con", "dyn", "rep",
    "dyn_raw", "rep_raw", "reward_mae", "cont_mae", "policy", "policy_entropy",
    "policy_adv_mag", "policy_logp_mean", "policy_retnorm_scale", "policy_weight_mean",
    "value", "value_value_mean", "value_target_mean", "value_value_mae", "value_positions",
    "repval", "repval_value_mean", "repval_target_mean", "repval_value_mae", "repval_positions",
    "weight_sum", "imagination_starts", "train_positions", "grad_norm", "replay_occupancy",
    "realized_train_ratio", "peak_vram_mib",
)


class OnlineTrainer:
    """M9: alternate real collection, replay sampling, world-model and behaviour updates."""

    def __init__(
        self,
        agent: Agent,
        env: EpisodicEnv,
        replay: ReplayBuffer,
        config: RunConfig,
        out_dir: Path | str,
        evaluator=None,
        checkpointer=None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.replay = replay
        self.config = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = evaluator
        self.checkpointer = checkpointer

        self.warmup_policy = UniformRandomPolicy(
            env.action_low, env.action_high, seed=config.seed
        )
        self.collector = Collector(env, replay, self.warmup_policy)
        self.optimizer = LaProp(
            agent.trainable_parameters(),
            lr=config.lr,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
            agc=config.agc,
            warmup=config.lr_warmup,
        )
        self.scheduler = TrainingScheduler(
            train_ratio=config.train_ratio,
            positions_per_update=config.positions_per_update,
        )

        self._learned = False
        self._at_boundary = True
        # a resume replays from the checkpoint to wherever the previous segment died, so those
        # rows are written twice; the segment makes the later one identifiable rather than lost
        self.segment = 0
        self.next_eval = config.eval_every
        self.next_checkpoint = config.checkpoint_every
        self.evaluations: list[dict] = []
        self.elapsed = 0.0
        self._log = None

    @property
    def counters(self) -> StepCounters:
        return self.collector.counters

    def _maybe_switch_policy(self) -> None:
        if self._learned or self.counters.env_step < self.config.warmup_transitions:
            return

        # the recurrence restarts here; nothing recorded in replay depends on the latent carry
        self.collector.set_policy(self.agent.collection_policy())
        self._learned = True

    def _open_log(self) -> None:
        path = self.out_dir / "train-log.csv"
        exists = path.exists()
        self._log = path.open("a" if exists else "w")

        if not exists:
            self._log.write(",".join(LOG_COLUMNS) + "\n")
            self._log.flush()

    def _write_log(self, output: UpdateOutput) -> None:
        if self._log is None:
            return

        counters = self.counters
        trained = max(1, counters.env_step - self.config.warmup_transitions)
        row = {
            "segment": self.segment,
            "env_step": counters.env_step,
            "gradient_step": counters.gradient_step,
            "elapsed_s": round(self.elapsed, 3),
            "eval_seconds": round(counters.eval_seconds, 3),
            "replay_occupancy": len(self.replay),
            "realized_train_ratio": round(counters.train_position / trained, 4),
            "peak_vram_mib": (
                round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if self.agent.device.type == "cuda"
                else 0.0
            ),
        }

        for column in LOG_COLUMNS:
            if column not in row:
                row[column] = round(float(output.metrics.get(column, float("nan"))), 6)

        self._log.write(",".join(str(row[c]) for c in LOG_COLUMNS) + "\n")
        self._log.flush()

    def update(self) -> UpdateOutput:
        batch = self.replay.sample_sequences(
            self.config.batch_size, self.config.train_length, self.config.burn_in
        )
        output = training_update(self.agent, self.optimizer, batch, self.config.horizon)

        if not torch.isfinite(output.total):
            raise RuntimeError(f"non-finite loss at gradient step {self.counters.gradient_step}")

        self.counters.record_update(
            batch.train_positions, output.starts, self.config.horizon
        )
        return output

    def _evaluate(self) -> None:
        if self.evaluator is None:
            return

        result = self.evaluator(self.agent, self.counters.env_step, self.segment)
        self.counters.record_evaluation(result.steps, result.seconds)
        self.evaluations.append(result.summary())

    def _checkpoint(self, final: bool = False) -> None:
        if self.checkpointer is None:
            return

        # a resume checkpoint is only written at a boundary: replay holds a partial episode
        # otherwise, and the next add would not match its predecessor's next_observation
        self.checkpointer.save(self, final=final, resumable=self._at_boundary)

    def run(self, budget: int | None = None) -> StepCounters:
        budget = int(self.config.budget if budget is None else budget)
        self._open_log()
        start = time.perf_counter() - self.elapsed

        while self.counters.env_step < budget:
            self._maybe_switch_policy()
            transition = self.collector.collect_step()
            self._at_boundary = bool(transition.is_last)

            if self.counters.env_step > self.config.warmup_transitions:
                self.scheduler.credit(1)

            # credits accrue while replay still holds no complete episode of P+T transitions;
            # the debt is real and is paid once warm-up closes one
            updates = (
                self.scheduler.take()
                if self.replay.can_sample(self.config.sequence_length)
                else 0
            )

            for _ in range(updates):
                self.elapsed = time.perf_counter() - start
                output = self.update()

                if self.counters.gradient_step % self.config.log_every == 0:
                    self._write_log(output)

            self.elapsed = time.perf_counter() - start

            if self.counters.env_step >= self.next_eval:
                self.next_eval += self.config.eval_every
                self._evaluate()

            # resume is only exact at an episode boundary: mid-episode simulator state is not saved
            if transition.is_last and self.counters.env_step >= self.next_checkpoint:
                self.next_checkpoint += self.config.checkpoint_every
                self._checkpoint()

        self._checkpoint(final=True)

        if self._log is not None:
            self._log.close()
            self._log = None

        return self.counters

    def state_dict(self) -> dict:
        return {
            "counters": self.counters.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learned_policy": self._learned,
            "at_boundary": self._at_boundary,
            "segment": self.segment,
            "next_eval": self.next_eval,
            "next_checkpoint": self.next_checkpoint,
            "elapsed": self.elapsed,
            "evaluations": self.evaluations,
        }

    def load_state_dict(self, state: dict) -> None:
        self.counters.load_state_dict(state["counters"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._learned = bool(state["learned_policy"])
        self._at_boundary = bool(state.get("at_boundary", True))
        # every restore opens a new segment, so replayed rows are ordered, not ambiguous
        self.segment = int(state.get("segment", 0)) + 1
        self.next_eval = int(state["next_eval"])
        self.next_checkpoint = int(state["next_checkpoint"])
        self.elapsed = float(state["elapsed"])
        self.evaluations = list(state["evaluations"])

        if self._learned:
            self.collector.set_policy(self.agent.collection_policy())
