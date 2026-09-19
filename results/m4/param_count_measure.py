"""MEASURED sum(p.numel()) over the instantiated world model (spec.md §10-7)."""

import sys

import torch

sys.path.insert(0, "src")

from dreamer import WorldModel

DERIVED = {"dyn": 376_704, "enc": 14_304, "dec": 80_595, "rew": 57_663, "con": 41_153}

torch.manual_seed(0)
wm = WorldModel(action_dim=6)
modules = {
    "dyn": wm.rssm, "enc": wm.encoder, "dec": wm.decoder,
    "rew": wm.reward, "con": wm.cont,
}
measured = {k: sum(p.numel() for p in m.parameters()) for k, m in modules.items()}

print("# MEASURED parameter count, instantiated PyTorch world model")
print("# walker_walk, A=6, size1m + dmc_vision; torch", torch.__version__)
print("#")
print(f"{'module':<10}{'measured':>10}{'derived':>10}{'delta':>8}")

ok = True
for name in DERIVED:
    delta = measured[name] - DERIVED[name]
    ok &= delta == 0
    print(f"{name:<10}{measured[name]:>10}{DERIVED[name]:>10}{delta:>8}")

total, total_derived = sum(measured.values()), sum(DERIVED.values())
ok &= sum(p.numel() for p in wm.parameters()) == total
print(f"{'WORLD':<10}{total:>10}{total_derived:>10}{total - total_derived:>8}")
print()
print("# full agent is 686,846 derived; pol (50,316) and val (66,111) are owed at M7-M8")
print(f"# world model + pol + val = {total + 50_316 + 66_111}")
print()
print("PASS" if ok else "FAIL", "- measured == derived" if ok else "- SPEC AND CODE DISAGREE")
sys.exit(0 if ok else 1)
