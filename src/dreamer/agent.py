from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .actor import Actor, ActorActionProvider, ReturnNormalizer
from .config import RunConfig, episode_seed, stream_seed
from .critic import Critic
from .env import DMCEnv
from .rssm import State
from .types import EnvStep, Image
from .world_model import WorldModel


class LatentPolicy:
    """spec 7.8: the posterior absorbs the CURRENT image before the action is selected.

    Nothing here touches replay or an optimizer. The previous action it conditions on is the one
    the simulator executed, which `Collector` feeds back through `observe_executed`.
    """

    def __init__(
        self,
        world_model: WorldModel,
        actor: Actor,
        generator: torch.Generator,
        mode: bool = False,
    ) -> None:
        self.world_model = world_model
        self.actor = actor
        self.generator = generator
        self.mode = bool(mode)

        self._state: State | None = None
        self._prev_action: Tensor | None = None
        self._is_first = True
        self._awaiting_feedback = False

    @property
    def device(self) -> torch.device:
        return self.world_model.device

    @property
    def state(self) -> State | None:
        return self._state

    def reset(self) -> None:
        """A time limit resets the recurrence exactly as a termination does (spec 7.4)."""
        self._state = None
        self._prev_action = None
        self._is_first = True
        self._awaiting_feedback = False

    @torch.no_grad()
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if self._awaiting_feedback:
            raise RuntimeError(
                "the previous action was never fed back through observe_executed: the recurrence "
                "would condition on a requested action the simulator never integrated"
            )

        rssm = self.world_model.rssm
        device = self.device

        image = torch.from_numpy(np.ascontiguousarray(observation)).to(device).unsqueeze(0)
        embed = self.world_model.encoder(image)

        carry = self._state if self._state is not None else rssm.initial(1, device)
        action = (
            self._prev_action
            if self._prev_action is not None
            else torch.zeros(1, rssm.action_dim, device=device)
        )
        is_first = torch.tensor([self._is_first], device=device, dtype=torch.bool)

        post, _ = rssm.observe_step(carry, action, embed, is_first, self.generator)
        policy = self.actor(post.feat)
        # evaluation acts on the mean; training samples (spec 7.8)
        sample = policy.mode if self.mode else policy.sample(self.generator)

        self._state = post
        self._is_first = False
        self._awaiting_feedback = True

        return sample.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def observe_executed(self, action: np.ndarray) -> None:
        executed = torch.from_numpy(np.ascontiguousarray(np.asarray(action, dtype=np.float32)))
        self._prev_action = executed.to(self.device).unsqueeze(0)
        self._awaiting_feedback = False


class EpisodicEnv:
    """One simulator per episode, seeded from the episode index, so a resume reproduces it.

    Mid-episode simulator state is not restorable, so checkpoints are taken at boundaries and this
    wrapper exists to make the *next* episode a pure function of a persisted integer.
    """

    def __init__(
        self,
        domain: str,
        task: str,
        base_seed: int,
        episode_index: int = 0,
        size: tuple[int, int] = (64, 64),
        recreate: bool = True,
    ) -> None:
        self.domain = domain
        self.task = task
        self.base_seed = int(base_seed)
        self.episode_index = int(episode_index)
        self.size = size
        self.recreate = bool(recreate)

        self._env = DMCEnv(domain, task, seed=self.next_seed, size=size)
        self.current_seed = self.next_seed

    @property
    def next_seed(self) -> int:
        return episode_seed(self.base_seed, self.episode_index)

    @property
    def physics_substeps(self) -> int:
        return self._env.physics_substeps

    @property
    def action_shape(self) -> tuple[int, ...]:
        return self._env.action_shape

    @property
    def action_low(self) -> np.ndarray:
        return self._env.action_low

    @property
    def action_high(self) -> np.ndarray:
        return self._env.action_high

    def reset(self) -> Image:
        if self.recreate:
            self._env.close()
            self._env = DMCEnv(self.domain, self.task, seed=self.next_seed, size=self.size)

        self.current_seed = self.next_seed
        self.episode_index += 1
        return self._env.reset()

    def step(self, action: np.ndarray) -> EnvStep:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()

    def state_dict(self) -> dict:
        return {
            "domain": self.domain,
            "task": self.task,
            "base_seed": self.base_seed,
            "episode_index": self.episode_index,
            "next_seed": self.next_seed,
        }

    def load_state_dict(self, state: dict) -> None:
        if (state["domain"], state["task"]) != (self.domain, self.task):
            raise ValueError(
                f"checkpoint holds {state['domain']}/{state['task']}, "
                f"not {self.domain}/{self.task}"
            )

        self.base_seed = int(state["base_seed"])
        self.episode_index = int(state["episode_index"])


@dataclass(frozen=True)
class ParameterCounts:
    world_model: int
    actor: int
    critic: int

    @property
    def total(self) -> int:
        return self.world_model + self.actor + self.critic


class Agent(nn.Module):
    """Owns the world model, the actor, the fast/slow critic, the normalizer and every stream."""

    def __init__(self, action_dim: int, config: RunConfig, device=None) -> None:
        super().__init__()

        self.config = config
        self.action_dim = int(action_dim)

        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # the initialization draw is its own stream, so changing the horizon does not move the init
        torch.manual_seed(stream_seed(config.seed, "init"))

        self.world_model = WorldModel(action_dim=action_dim)
        feat = self.world_model.rssm.feat_size
        self.critic = Critic(in_features=feat)
        self.actor = Actor(in_features=feat, action_dim=action_dim)
        self.normalizer = ReturnNormalizer(debias=False)
        self.to(device)

        self._device = device
        self.generators = {
            name: self._make_generator(stream_seed(config.seed, name))
            for name in ("collect", "replay", "imagine", "eval")
        }
        # the provider holds its OWN generator; recreating it each update would redraw the rollout
        self.provider = ActorActionProvider(
            self.actor, seed=stream_seed(config.seed, "provider")
        )

    def _make_generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator(device=self._device)
        generator.manual_seed(int(seed))
        return generator

    @property
    def device(self) -> torch.device:
        return self._device

    def trainable_parameters(self) -> list[nn.Parameter]:
        """[dyn, enc, dec, rew, con, pol, val] exactly once; slowval is excluded (spec 5.7)."""
        params: list[nn.Parameter] = []
        seen: set[int] = set()

        for module in (self.world_model, self.actor):
            params.extend(module.parameters())

        params.extend(self.critic.trainable_parameters())

        for parameter in params:
            if id(parameter) in seen:
                raise RuntimeError("a parameter reached the optimizer list twice")
            seen.add(id(parameter))

        slow = {id(p) for p in self.critic.slow.parameters()}

        if seen & slow:
            raise RuntimeError("the slow critic must not enter the optimizer")

        return params

    def parameter_counts(self) -> ParameterCounts:
        return ParameterCounts(
            world_model=sum(p.numel() for p in self.world_model.parameters()),
            actor=sum(p.numel() for p in self.actor.parameters()),
            critic=sum(p.numel() for p in self.critic.value.parameters()),
        )

    def collection_policy(self) -> LatentPolicy:
        """Training collection: samples the actor, on the collection stream."""
        return LatentPolicy(
            self.world_model, self.actor, self.generators["collect"], mode=False
        )

    def evaluation_policy(self, seed: int) -> LatentPolicy:
        """Acts on the mean; posterior sampling runs on a fresh generator no training step reads."""
        return LatentPolicy(
            self.world_model,
            self.actor,
            self._make_generator(stream_seed(seed, "eval")),
            mode=True,
        )

    def generator_state(self) -> dict:
        return {
            "generators": {
                name: generator.get_state() for name, generator in self.generators.items()
            },
            "provider": self.provider.state_dict(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }

    def load_generator_state(self, state: dict) -> None:
        for name, saved in state["generators"].items():
            self.generators[name].set_state(saved)

        self.provider.load_state_dict(state["provider"])
        torch.set_rng_state(state["torch_cpu"])

        if state["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def model_state(self) -> dict:
        return {
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "normalizer_settings": self.normalizer.settings(),
            "action_dim": self.action_dim,
        }

    def load_model_state(self, state: dict) -> None:
        if int(state["action_dim"]) != self.action_dim:
            raise ValueError(
                f"checkpoint action_dim {state['action_dim']} does not match {self.action_dim}"
            )

        self.world_model.load_state_dict(state["world_model"])
        self.actor.load_state_dict(state["actor"])
        # critic carries the slow mirror, which resume must restore even though it never trains
        self.critic.load_state_dict(state["critic"])
        self.normalizer.load_settings(state["normalizer_settings"])
        self.normalizer.load_state_dict(state["normalizer"])
