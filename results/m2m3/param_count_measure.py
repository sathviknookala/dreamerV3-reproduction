"""MEASURED sum(p.numel()) over instantiated RSSM + encoder modules (spec.md §10-7)."""

import sys

import torch

sys.path.insert(0, "src")

from dreamer import Encoder, RSSM

DERIVED = {"dyn": 376_704, "enc": 14_304}

torch.manual_seed(0)
encoder = Encoder()
rssm = RSSM(action_dim=6)

measured = {
    "enc": sum(p.numel() for p in encoder.parameters()),
    "dyn": sum(p.numel() for p in rssm.parameters()),
}

print("# MEASURED parameter count, instantiated PyTorch modules")
print("# walker_walk, A=6, size1m + dmc_vision; torch", torch.__version__)
print("#")
print(f"{'module':<24}{'measured':>10}{'derived':>10}{'delta':>8}")

ok = True
for name in ("dyn", "enc"):
    delta = measured[name] - DERIVED[name]
    ok &= delta == 0
    print(f"{name:<24}{measured[name]:>10}{DERIVED[name]:>10}{delta:>8}")

total, total_derived = sum(measured.values()), sum(DERIVED.values())
print(f"{'TOTAL':<24}{total:>10}{total_derived:>10}{total - total_derived:>8}")
print()

for label, module in (("enc", encoder), ("dyn", rssm)):
    for name, p in module.named_parameters():
        print(f"{label}/{name:<28}{str(tuple(p.shape)):>22}{p.numel():>10}")

print()
print("PASS" if ok else "FAIL", "- measured == derived" if ok else "- SPEC AND CODE DISAGREE")
sys.exit(0 if ok else 1)
