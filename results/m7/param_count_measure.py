"""MEASURED sum(p.numel()) over the instantiated critic (spec.md §10-7)."""

import sys

import torch

sys.path.insert(0, "src")

from dreamer import Critic, WorldModel

DERIVED_VAL = 66_111
DERIVED_WORLD_MODEL = 570_419

torch.manual_seed(0)
wm = WorldModel(action_dim=6)
critic = Critic(in_features=wm.rssm.feat_size)

val = sum(p.numel() for p in critic.trainable_parameters())
mirror = sum(p.numel() for p in critic.parameters()) - val
world_model = sum(p.numel() for p in wm.parameters())

print("# MEASURED parameter count, instantiated PyTorch critic")
print("# walker_walk, A=6, size1m + dmc_vision; torch", torch.__version__)
print("#")
print(f"{'module':<12}{'measured':>10}{'derived':>10}{'delta':>8}")
print(f"{'val':<12}{val:>10}{DERIVED_VAL:>10}{val - DERIVED_VAL:>8}")
print(f"{'slowval':<12}{mirror:>10}{DERIVED_VAL:>10}{mirror - DERIVED_VAL:>8}   (untrained, excluded from the optimizer)")
print()
print(f"world model   {world_model:>10}   (measured at M4)")
print(f"world + val   {world_model + val:>10}")
print(f"agent total   {world_model + val + 50_316:>10}   with pol 50,316 still DERIVED, owed at M8")
print()
print(f"# spec 4.11 total, excluding the slowval mirror: 686,846")
print(f"# still derived, not measured: pol (50,316)")

ok = val == DERIVED_VAL and mirror == DERIVED_VAL
print(f"\n{'PASS' if ok else 'FAIL'}  val and its mirror both equal the derivation")
sys.exit(0 if ok else 1)
