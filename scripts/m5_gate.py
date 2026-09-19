"""M5 gate: open-loop prediction on held-out Walker episodes, from the trained checkpoint."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from dreamer import RSSM, WorldModel
from dreamer.openloop import (
    Context,
    encoder_tripwire,
    evaluate_open_loop,
    gather_contexts,
    load_split,
    make_contexts,
    open_loop_predict,
    target_indices,
    training_reward_mean,
)
from dreamer.rssm import observe_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/m5-walker-random")
parser.add_argument("--checkpoint", default="runs/m5/world-model.pt")
parser.add_argument("--report", default="results/m5/openloop.json")
parser.add_argument("--context", type=int, default=5)
parser.add_argument("--horizon", type=int, default=30)
args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

C, H = args.context, args.horizon
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}  torch: {torch.__version__}")

checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
model = WorldModel(action_dim=checkpoint["action_dim"]).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

train_episodes = load_split(args.data, "train")
holdout_episodes = load_split(args.data, "holdout")
train_digests = {e.digest() for e in train_episodes}
holdout_digests = {e.digest() for e in holdout_episodes}
report = json.loads(Path(args.report).read_text())

print(f"checkpoint {args.checkpoint}  {checkpoint['steps']} gradient steps")
print(f"train {len(train_episodes)} episodes  holdout {len(holdout_episodes)} episodes")

print("\n## 1-2. observations reach the context only")
contexts = make_contexts(holdout_episodes, C, H, per_episode=2)
batch = gather_contexts(holdout_episodes, contexts, C, H)

context_observations = torch.from_numpy(batch.context_observations).to(device)
context_actions = torch.from_numpy(batch.context_actions).to(device=device, dtype=torch.float32)
future_actions = torch.from_numpy(batch.future_actions).to(device=device, dtype=torch.float32)
future_observations = torch.from_numpy(batch.target_images).to(device)

generator = torch.Generator(device=device)

with encoder_tripwire(model) as tripwire, torch.no_grad():
    generator.manual_seed(0)
    prediction = open_loop_predict(model, context_observations, context_actions, future_actions, generator)

check(
    "encoder is called exactly once, on the context prefix only",
    len(tripwire.calls) == 1 and tripwire.calls[0][1] == C + 1,
    f"calls={tripwire.calls}",
)
check(
    "frames encoded == contexts x (C+1)",
    tripwire.frames_seen == len(contexts) * (C + 1),
    f"{tripwire.frames_seen} vs {len(contexts) * (C + 1)}",
)

prior_parameters = set(inspect.signature(RSSM.imagine_step).parameters) | set(
    inspect.signature(RSSM.imagine).parameters
)
predict_parameters = set(inspect.signature(open_loop_predict).parameters)
check(
    "the prior API takes no image or embedding argument",
    not (prior_parameters & {"embed", "embeds", "image", "observation", "observations"}),
    str(sorted(prior_parameters)),
)
check(
    "open_loop_predict takes exactly one observation argument, the context",
    [p for p in sorted(predict_parameters) if "observ" in p or "image" in p] == ["context_observations"],
    str(sorted(predict_parameters)),
)

# corrupt every frame after the cutoff in the source data and re-run the whole
# gather -> predict path; if a future frame could reach the rollout, this moves it
rng = np.random.default_rng(0)
corrupted = []
for episode in holdout_episodes:
    observations = episode.observations.copy()
    observations[C + 1 :] = rng.integers(0, 256, observations[C + 1 :].shape, dtype=np.uint8)
    corrupted.append(replace(episode, observations=observations))

head = [Context(i, 0) for i in range(len(holdout_episodes))]
clean_batch = gather_contexts(holdout_episodes, head, C, H)
dirty_batch = gather_contexts(corrupted, head, C, H)


def predict_from(source, seed=0):
    with torch.no_grad():
        generator.manual_seed(seed)
        return open_loop_predict(
            model,
            torch.from_numpy(source.context_observations).to(device),
            torch.from_numpy(source.context_actions).to(device=device, dtype=torch.float32),
            torch.from_numpy(source.future_actions).to(device=device, dtype=torch.float32),
            generator,
        )


check(
    "randomizing every frame after the cutoff leaves the rollout bit-identical",
    bool(torch.equal(predict_from(clean_batch).reward, predict_from(dirty_batch).reward))
    and not np.array_equal(clean_batch.target_images, dirty_batch.target_images),
    "the corrupted frames are real targets, and they are not an input",
)

print("\n## 3-4. action and target alignment on real episodes")
aligned = True
for index, context in enumerate(contexts[:32]):
    episode = holdout_episodes[context.episode_index]
    cutoff = context.cutoff(C)
    reward_index, observation_index = target_indices(cutoff, H)
    aligned &= bool(
        np.array_equal(batch.future_actions[index], episode.actions[cutoff : cutoff + H])
        and np.array_equal(batch.target_rewards[index], episode.rewards[reward_index])
        and np.array_equal(batch.target_images[index], episode.observations[observation_index])
        and np.array_equal(batch.context_actions[index], episode.actions[context.start : cutoff])
    )
check("distance k pairs state cutoff+k with rewards[cutoff+k-1]", aligned)

with torch.no_grad():
    generator.manual_seed(0)
    rolled = open_loop_predict(
        model, context_observations, context_actions, future_actions.roll(1, dims=1), generator
    )
delta = float((rolled.reward - prediction.reward).abs().mean())
check("the rollout is action-conditioned: rolling the actions moves it", delta > 0.0, f"mean |delta| {delta:.6g}")

shifted_reward, _ = target_indices(contexts[0].cutoff(C) + 1, H)
check(
    "a one-step target shift selects different real rewards",
    not np.array_equal(
        holdout_episodes[contexts[0].episode_index].rewards[shifted_reward],
        batch.target_rewards[0],
    ),
)

print("\n## 5. reproducible stochastic evaluation")
constant = training_reward_mean(train_episodes)
small = contexts[:16]


def evaluate(seed: int, samples: int):
    return evaluate_open_loop(
        model, holdout_episodes, small, C, H, constant, samples, seed, device, chunk_contexts=8
    )


first, second = evaluate(0, 4), evaluate(0, 4)
check("same seed reproduces the metrics exactly", bool(np.array_equal(first.reward_mae, second.reward_mae)))
check(
    "four latent samples per context, with nonzero spread",
    first.samples == 4 and float(first.reward_mae_sample_sd.max()) > 0.0,
    f"max sample sd {first.reward_mae_sample_sd.max():.3g}",
)
check(
    "a different seed draws different latents",
    not np.array_equal(evaluate(1, 4).reward_mae, first.reward_mae),
)
check(
    "the reported evaluation averaged multiple samples",
    report["samples"] > 1 and max(report["reward_mae_sample_sd"]) > 0.0,
    f"samples={report['samples']}",
)

print("\n## 6. the held-out split cannot have been trained on")
check("train and holdout episode content is disjoint", not (train_digests & holdout_digests))
check(
    "the checkpoint records exactly the training split",
    set(checkpoint["train_episode_sha256"]) == train_digests,
    f"{len(checkpoint['train_episode_sha256'])} hashes",
)
check("no holdout episode is in the checkpoint's training set", not (set(checkpoint["train_episode_sha256"]) & holdout_digests))
check(
    "the report's holdout hashes match the holdout split",
    set(report["holdout_episode_sha256"]) == holdout_digests,
)

print("\n## 7. baselines are computed from training data only")
holdout_mean = float(np.concatenate([e.rewards for e in holdout_episodes]).mean())
check(
    "the constant baseline equals the training-split mean",
    abs(report["constant_reward_baseline"] - constant) < 1e-9,
    f"{report['constant_reward_baseline']:.6f}",
)
check(
    "it is not the holdout mean",
    abs(constant - holdout_mean) > 1e-9,
    f"train {constant:.6f}  holdout {holdout_mean:.6f}",
)

try:
    training_reward_mean(holdout_episodes)
    refused = False
except ValueError:
    refused = True
check("the baseline refuses held-out episodes", refused)

print("\n## 8. a leaking rollout is detected")
with encoder_tripwire(model) as leak, torch.no_grad():
    observe_sequence(
        model.encoder,
        model.rssm,
        torch.cat([context_observations, future_observations], dim=1),
        torch.cat([context_actions, future_actions], dim=1),
    )
check(
    "the tripwire catches a posterior pass over the future frames",
    leak.calls[0][1] == C + 1 + H and leak.frames_seen > len(contexts) * (C + 1),
    f"encoded {leak.frames_seen} frames instead of {len(contexts) * (C + 1)}",
)

print("\n## M5: open-loop reward prediction against the baselines")
mae = np.asarray(report["reward_mae"])
base = np.asarray(report["baseline_reward_mae_mean"])
persist = np.asarray(report["baseline_reward_mae_persistence"])
shuffled = np.asarray(report["reward_mae_shuffled"])

print(f"  {'k':>4}{'reward MAE':>14}{'mean base':>13}{'persist':>12}{'shuffled':>14}{'MAE/base':>10}")
for k in (1, 5, 15, 30):
    i = k - 1
    print(f"  {k:>4}{mae[i]:>14.6f}{base[i]:>13.6f}{persist[i]:>12.6f}{shuffled[i]:>14.6f}{mae[i]/base[i]:>10.3f}")

check("reward MAE beats the training-mean baseline at k=1", mae[0] < base[0], f"{mae[0]:.6f} < {base[0]:.6f}")
check("reward MAE beats the training-mean baseline at k=5", mae[4] < base[4], f"{mae[4]:.6f} < {base[4]:.6f}")
check(
    "predictions stay informative at every reported distance >= 5",
    bool((mae[4:] < base[4:]).all()),
    f"worst ratio {float((mae[4:] / base[4:]).max()):.3f} at k={int(np.argmax(mae[4:] / base[4:])) + 5}",
)
check("reward MAE beats last-reward persistence at k=5", mae[4] < persist[4], f"{mae[4]:.6f} < {persist[4]:.6f}")
check("all reported metrics finite", bool(np.isfinite(mae).all() and np.isfinite(base).all()))

print(f"\n  action-shuffling diagnostic: mean MAE {mae.mean():.6f} recorded vs "
      f"{shuffled.mean():.6f} with the future actions permuted in time "
      f"({shuffled.mean() / mae.mean():.3f}x)")
print("  (a sensitivity diagnostic, not a causal control result)")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
