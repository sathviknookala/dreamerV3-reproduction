"""MEASURED sum(p.numel()) over the instantiated actor, closing spec.md §10-7."""

import sys

import torch

sys.path.insert(0, "src")

from dreamer import Actor, Critic, WorldModel

DERIVED_POL = 50_316
DERIVED_VAL = 66_111
DERIVED_WORLD_MODEL = 570_419
DERIVED_TOTAL = 686_846

torch.manual_seed(0)
wm = WorldModel(action_dim=6)
critic = Critic(in_features=wm.rssm.feat_size)
actor = Actor(in_features=wm.rssm.feat_size, action_dim=6)

pol = sum(p.numel() for p in actor.parameters())
val = sum(p.numel() for p in critic.trainable_parameters())
world = sum(p.numel() for p in wm.parameters())
cartpole = sum(p.numel() for p in Actor(in_features=wm.rssm.feat_size, action_dim=1).parameters())

print("# MEASURED parameter count, instantiated PyTorch actor")
print("# walker_walk, A=6, size1m + dmc_vision; torch", torch.__version__)
print("#")
print(f"{'module':<12}{'measured':>10}{'derived':>10}{'delta':>8}")
print(f"{'pol':<12}{pol:>10}{DERIVED_POL:>10}{pol - DERIVED_POL:>8}")
print()
print(f"world model   {world:>10}   (measured at M4)")
print(f"val           {val:>10}   (measured at M7)")
print(f"pol           {pol:>10}   (measured here)")
print(f"agent total   {world + val + pol:>10}   derived {DERIVED_TOTAL}, slowval mirror excluded")
print()
print(f"# cartpole actor (A=1): {cartpole}, which is {pol - cartpole} fewer -- 2 heads x 5 dims x (64 + 1)")
print("# every module in spec 4.11 is now MEASURED; nothing in that table is derived-only.")

ok = (
    pol == DERIVED_POL
    and world == DERIVED_WORLD_MODEL
    and val == DERIVED_VAL
    and world + val + pol == DERIVED_TOTAL
)
print(f"\n{'PASS' if ok else 'FAIL'}  pol and the agent total both equal the derivation")
sys.exit(0 if ok else 1)
