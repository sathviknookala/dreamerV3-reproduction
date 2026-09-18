import json
import os
import platform
import subprocess
import sys

REQUIRED_ARCH = "sm_120"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


print("# M0 deferred item 1 — PyTorch compute compatibility")
print(f"# python {platform.python_version()}  {sys.executable}")
print()

try:
    import torch
except Exception as e:
    print(f"  [FAIL] import torch — {type(e).__name__}: {e}")
    sys.exit(1)

print("## Build identity")
wheel = "unknown"
for d in os.listdir(os.path.join(os.path.dirname(torch.__file__), "..")):
    if d.startswith("torch-") and d.endswith(".dist-info"):
        wheel = d[:-len(".dist-info")]
print(f"  torch            {torch.__version__}")
print(f"  dist-info        {wheel}")
print(f"  bundled CUDA     {torch.version.cuda}")
print(f"  cuDNN            {torch.backends.cudnn.version()}")
print(f"  arch list        {' '.join(torch.cuda.get_arch_list())}")
print()

print("## Device visibility")
check("torch.cuda.is_available()", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("\n  Cannot continue: no CUDA device visible.")
    sys.exit(1)
cap = torch.cuda.get_device_capability(0)
capstr = f"sm_{cap[0]}{cap[1]}"
print(f"  device           {torch.cuda.get_device_name(0)}")
print(f"  capability       {capstr} (compute {cap[0]}.{cap[1]})")
total = torch.cuda.get_device_properties(0).total_memory
print(f"  total VRAM       {total / 1024**2:.0f} MiB")
print()

print("## Architecture support — the Blackwell question")
arches = torch.cuda.get_arch_list()
check(f"{REQUIRED_ARCH} present in build arch list", REQUIRED_ARCH in arches)
check("device capability matches a compiled arch", capstr in arches,
      f"device is {capstr}")
print()

print("## Numerical agreement on the ops this project actually uses")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)
dev = torch.device("cuda:0")


def agree(name, fn, shapes, tol):
    xs = [torch.randn(*s, dtype=torch.float32) for s in shapes]
    out_cpu = fn(*xs)
    out_gpu = fn(*[x.to(dev) for x in xs]).cpu()
    err = (out_cpu - out_gpu).abs().max().item()
    return check(f"{name} (max|GPU-CPU| = {err:.3e} <= {tol:.0e})", err <= tol,
                 f"shape {tuple(out_cpu.shape)}")


agree("matmul 1024x1024", lambda a, b: a @ b, [(1024, 1024), (1024, 1024)], 1e-3)

enc0 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
agree("encoder stage-0 conv2d 3->8 k5 p2 on 64x64",
      lambda x: enc0(x) if x.is_cpu else enc0.to(dev)(x), [(16, 3, 64, 64)], 1e-4)

# the block-diagonal recurrent projection, spec section 4.1
agree("BlockLinear einsum '...ki,kio->...ko' (8 x 256 -> 8 x 64)",
      lambda x, w: torch.einsum("...ki,kio->...ko", x, w),
      [(16, 8, 256), (8, 256, 64)], 1e-3)


def rmsnorm(x, scale):
    w = x.float()
    return (w * torch.rsqrt(w.pow(2).mean(-1, keepdim=True) + 1e-4) * scale).to(x.dtype)


agree("RMSNorm in float32 (eps inside rsqrt)", rmsnorm, [(16, 64, 512), (512,)], 1e-5)


def blockdiag_equiv():
    x = torch.randn(4, 512, device=dev)
    w = torch.randn(8, 64, 192, device=dev)
    got = torch.einsum("...ki,kio->...ko", x.view(4, 8, 64), w).reshape(4, 1536)
    dense = torch.zeros(512, 1536, device=dev)
    for b in range(8):
        dense[b * 64:(b + 1) * 64, b * 192:(b + 1) * 192] = w[b]
    ref = x @ dense
    return (got - ref).abs().max().item()


e = blockdiag_equiv()
check(f"block einsum == block-diagonal dense matmul ({e:.3e})", e < 1e-3)
print()

print("## Autograd and optimizer step on device")
m = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 5, padding=2), torch.nn.Flatten(),
                        torch.nn.Linear(8 * 64 * 64, 16)).to(dev)
opt = torch.optim.SGD(m.parameters(), lr=1e-3)
before = m[0].weight.detach().clone()
loss = m(torch.randn(8, 3, 64, 64, device=dev)).square().mean()
loss.backward()
gnorm = m[0].weight.grad.norm().item()
opt.step()
moved = (m[0].weight.detach() - before).abs().max().item()
check(f"backward produces finite non-zero grad (norm {gnorm:.4g})",
      gnorm > 0 and torch.isfinite(m[0].weight.grad).all().item())
check(f"optimizer step changes parameters ({moved:.3e})", moved > 0)
print()

print("## Precision modes available")
for name, dt in [("float32", torch.float32), ("bfloat16", torch.bfloat16),
                 ("float16", torch.float16)]:
    try:
        a = torch.randn(256, 256, device=dev, dtype=dt)
        ok = torch.isfinite((a @ a).float()).all().item()
        check(f"{name} matmul runs and is finite", ok)
    except Exception as ex:
        check(f"{name} matmul runs and is finite", False, f"{type(ex).__name__}: {ex}")
print()

print("## TF32 (reported, not a pass/fail — it changes fp32 precision)")
torch.backends.cuda.matmul.allow_tf32 = True
a = torch.randn(2048, 2048)
err_tf32 = (a @ a - (a.to(dev) @ a.to(dev)).cpu()).abs().max().item()
torch.backends.cuda.matmul.allow_tf32 = False
err_fp32 = (a @ a - (a.to(dev) @ a.to(dev)).cpu()).abs().max().item()
print(f"  max|GPU-CPU| with TF32 on   {err_tf32:.3e}")
print(f"  max|GPU-CPU| with TF32 off  {err_fp32:.3e}")
print("  Correctness tests above ran with TF32 OFF. Leave it off while")
print("  validating against hand-computed fixtures (M2, M7).")
print()

print("## Shape feasibility at the specified batch")
print("  FLOORS, not measurements. Each excludes the rest of the model, the")
print("  optimizer state, and autograd activations it does not itself create.")
print("  Real peak VRAM at H=30 is deferred to M9 (spec.md section 10-8).")
print()

B, T, FEAT = 16, 64, 512 + 32 * 4
enc = torch.nn.Sequential(
    torch.nn.Conv2d(3, 8, 5, padding=2), torch.nn.MaxPool2d(2), torch.nn.SiLU(),
    torch.nn.Conv2d(8, 12, 5, padding=2), torch.nn.MaxPool2d(2), torch.nn.SiLU(),
    torch.nn.Conv2d(12, 16, 5, padding=2), torch.nn.MaxPool2d(2), torch.nn.SiLU(),
    torch.nn.Conv2d(16, 16, 5, padding=2), torch.nn.MaxPool2d(2), torch.nn.SiLU(),
).to(dev)
torch.cuda.reset_peak_memory_stats()
x = torch.randn(B * T, 3, 64, 64, device=dev)
tok = enc(x)
tok.square().mean().backward()
peak_enc = torch.cuda.max_memory_allocated() / 1024**2
check(f"encoder fwd+bwd on the full B*T={B * T} image batch, peak {peak_enc:.0f} MiB",
      tuple(tok.shape) == (B * T, 16, 4, 4), f"tokens {tuple(tok.shape)}")
del x, tok
torch.cuda.empty_cache()

print()
print(f"  {'H':>4}{'imagined states':>18}{'feat tensor MiB':>18}")
for H in (5, 15, 30):
    torch.cuda.reset_peak_memory_stats()
    buf = torch.empty(B * T, H + 1, FEAT, device=dev)
    mib = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  {H:>4}{f'({B * T}, {H + 1}, {FEAT})':>18}{mib:>18.1f}")
    del buf
    torch.cuda.empty_cache()
print("  Imagination starts from EVERY replay position (imag_last=0 => K=T), so the")
print(f"  leading dimension is B*T={B * T}, not B. This is what drives H=30 memory.")
print()
print(f"  free / total: {torch.cuda.mem_get_info()[0] / 1024**2:.0f} / {total / 1024**2:.0f} MiB")
print()

npass = sum(1 for _, ok, _ in results if ok)
print(f"## Verdict: {npass}/{len(results)} checks passed")
failed = [n for n, ok, _ in results if not ok]
if failed:
    print("   FAILED: " + "; ".join(failed))
print()
print(json.dumps({
    "torch": torch.__version__, "wheel": wheel, "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(), "arch_list": torch.cuda.get_arch_list(),
    "device": torch.cuda.get_device_name(0), "capability": capstr,
    "vram_mib": round(total / 1024**2), "python": platform.python_version(),
    "checks_passed": npass, "checks_total": len(results), "failed": failed,
}, indent=2))
sys.exit(0 if not failed else 1)
