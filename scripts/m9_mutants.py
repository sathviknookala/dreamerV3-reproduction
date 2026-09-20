"""M9 mutation harness: each new test must fail without its fix (CLAUDE.md testing rule)."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

MUTANTS = [
    ("executed action ignored; condition on the requested sample",
     "src/dreamer/agent.py",
     "        self._prev_action = executed.to(self.device).unsqueeze(0)",
     "        pass",
     "test_agent.TestLatentPolicyAlignment.test_the_conditioning_action_is_the_executed_one_not_the_requested_one"),

    ("missing-feedback guard removed",
     "src/dreamer/agent.py",
     "        if self._awaiting_feedback:\n            raise RuntimeError(",
     "        if False:\n            raise RuntimeError(",
     "test_agent.TestLatentPolicyAlignment.test_a_policy_call_without_executed_feedback_raises"),

    ("recurrence not reset at an episode boundary",
     "src/dreamer/collector.py",
     "    def reset(self) -> np.ndarray:\n        if self._reset_policy is not None:\n            self._reset_policy()",
     "    def reset(self) -> np.ndarray:\n        if False:\n            self._reset_policy()",
     "test_agent.TestLatentPolicyAlignment.test_a_time_limit_resets_the_recurrence_exactly_as_a_termination_does"),

    ("evaluation samples instead of taking the mean",
     "src/dreamer/agent.py",
     "        sample = policy.mode if self.mode else policy.sample(self.generator)",
     "        sample = policy.sample(self.generator)",
     "test_agent.TestPolicyModes.test_evaluation_acts_on_the_mean_and_training_samples"),

    ("the slow critic enters the optimizer",
     "src/dreamer/agent.py",
     "        params.extend(self.critic.trainable_parameters())",
     "        params.extend(self.critic.parameters())",
     "test_agent.TestAgentParameters.test_the_slow_critic_is_excluded"),

    ("scheduler discards the remainder",
     "src/dreamer/training.py",
     "        self.credits -= updates * self.positions_per_update",
     "        self.credits = 0",
     "test_training.TestTrainingScheduler.test_the_remainder_is_carried_not_discarded"),

    ("burn-in is not removed before the replay critic",
     "src/dreamer/training.py",
     "        world.post[:, burn_in + 1 :].feat,\n        tensors[\"rewards\"][:, burn_in:],",
     "        world.post[:, 1:].feat,\n        tensors[\"rewards\"][:, :],",
     "test_training.TestUpdateLayout.test_burn_in_positions_are_never_rollout_starts_or_critic_positions"),

    ("the bootstrap-hole mask is not passed",
     "src/dreamer/training.py",
     "        filled=filled[:, burn_in:],",
     "        filled=None,",
     "test_training.TestUpdateLayout.test_a_non_terminal_bootstrap_hole_is_rejected"),

    ("terminal positions are allowed as rollout starts",
     "src/dreamer/imagine.py",
     "    return loss_mask.to(torch.bool) & ~is_terminal.to(torch.bool)",
     "    return loss_mask.to(torch.bool)",
     "test_training.TestUpdateLayout.test_a_terminal_position_is_excluded_from_the_starts"),

    ("the actor reads the carry instead of the posterior that absorbed the image",
     "src/dreamer/agent.py",
     "        policy = self.actor(post.feat)",
     "        policy = self.actor(carry.feat)",
     "test_agent.TestLatentPolicyAlignment.test_the_actor_reads_the_state_that_absorbed_the_current_image"),

    ("the replay critic enters the total at weight 1.0 instead of 0.3",
     "src/dreamer/training.py",
     "    total = world.total + behavior.loss + agent.critic.scales[\"repval\"] * replay_critic.loss",
     "    total = world.total + behavior.loss + replay_critic.loss",
     "test_training.TestUpdateLayout.test_total_is_exactly_the_weighted_sum_of_the_three_parts"),

    ("the actor loss never reaches the total",
     "src/dreamer/actor.py",
     "    loss = actor_scale * policy.loss + critic.total(imagined, replay)",
     "    loss = critic.total(imagined, replay)",
     "test_training.TestGradientRouting.test_the_combined_loss_reaches_every_trainable_tensor_and_no_other"),

    ("imagination is built with a live graph, so the actor gets a pathwise gradient",
     "src/dreamer/imagine.py",
     "    with torch.no_grad():\n        carry = start.detach()",
     "    if True:\n        carry = start",
     "test_training.TestGradientRouting.test_the_imagined_features_carry_no_graph"),

    ("the replay critic is detached from the encoder",
     "src/dreamer/training.py",
     "        world.post[:, burn_in + 1 :].feat,",
     "        world.post[:, burn_in + 1 :].feat.detach(),",
     "test_training.TestGradientRouting.test_the_replay_critic_loss_reaches_the_encoder_and_the_rssm"),

    ("the slow critic advances BEFORE the optimizer step",
     "src/dreamer/training.py",
     "    optimizer.step()\n    # after the step: advancing before it would make the regularizer target the pre-update critic\n    agent.critic.update_slow()",
     "    agent.critic.update_slow()\n    optimizer.step()",
     "test_training.TestUpdateCardinality.test_the_slow_critic_advances_after_the_optimizer_step"),

    ("the normalizer is not advanced during the update",
     "src/dreamer/training.py",
     "        update=True,\n    )",
     "        update=False,\n    )",
     "test_training.TestUpdateCardinality.test_one_optimizer_step_and_one_normalizer_update_per_batch"),

    ("LaProp's private step counter is not checkpointed",
     "src/dreamer/optim.py",
     "        state[\"_step\"] = self._step",
     "        state[\"_step\"] = 0",
     "test_checkpoint.TestOptimizerRoundTrip.test_the_private_step_counter_survives"),

    ("the replay sampler stream is not checkpointed",
     "src/dreamer/replay.py",
     "        self._rng.bit_generator.state = state[\"rng\"]",
     "        pass",
     "test_checkpoint.TestReplayRoundTrip.test_contents_episode_metadata_and_the_sampler_stream_all_survive"),

    ("the provider's private generator is not checkpointed",
     "src/dreamer/actor.py",
     "            \"generator\": None if generator is None else generator.get_state(),",
     "            \"generator\": None,",
     "test_checkpoint.TestResumeRoundTrip.test_every_generator_including_the_providers_survives"),

    ("a mid-episode resume checkpoint is accepted",
     "src/dreamer/checkpoint.py",
     "    if payload[\"replay\"][\"current\"] is not None:",
     "    if False:",
     "test_checkpoint.TestResumeRoundTrip.test_a_mid_episode_resume_checkpoint_is_refused"),

    ("rotation evicts a live checkpoint",
     "src/dreamer/checkpoint.py",
     "            if path in self.written:\n                self.written.remove(path)",
     "            pass",
     "test_checkpoint.TestRotation.test_only_the_latest_two_resume_checkpoints_are_kept"),

    ("evaluation reads the collection stream",
     "src/dreamer/agent.py",
     "            self._make_generator(stream_seed(seed, \"eval\")),",
     "            self.generators[\"collect\"],",
     "test_evaluation.TestEvaluationIsolation.test_evaluation_changes_no_parameter_buffer_or_training_stream"),
]


def run(target):
    result = subprocess.run(
        [".venv/bin/python", "-m", "unittest", target],
        cwd=ROOT, capture_output=True, text=True,
        env={"PYTHONPATH": "src:tests", "PATH": "/usr/bin:/bin", "HOME": "/home/sathvik",
             "MUJOCO_GL": "egl"},
    )
    return result.returncode == 0, result.stderr[-300:]


survivors = []
for name, relpath, old, new, target in MUTANTS:
    path = ROOT / relpath
    original = path.read_text()
    if old not in original:
        print(f"SKIP   {name}  — anchor not found in {relpath}")
        survivors.append(name)
        continue
    path.write_text(original.replace(old, new, 1))
    try:
        passed, tail = run(target)
    finally:
        path.write_text(original)
    if passed:
        print(f"SURVIVED  {name}")
        survivors.append(name)
    else:
        print(f"killed    {name}")

print()
print(f"{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} mutants killed")
if survivors:
    print("SURVIVORS:", survivors)
    sys.exit(1)
