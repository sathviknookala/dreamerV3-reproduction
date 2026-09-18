import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


PKGS = ["torch", "numpy", "mujoco", "dm-control", "absl-py", "dm-env", "dm-tree",
        "glfw", "labmaze", "lxml", "protobuf", "pyopengl", "scipy", "triton"]

env = {
    "captured": sh("date -Is"),
    "python": {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    },
    "os": {
        "distro": sh(". /etc/os-release && echo $PRETTY_NAME"),
        "kernel": platform.release(),
    },
    "cpu": sh("lscpu | awk -F: '/Model name/{gsub(/^ +/,\"\",$2); print $2; exit}'"),
    "ram_gib": int(sh("free -g | awk '/^Mem:/{print $2}'") or 0),
    "gpu": dict(zip(
        ["name", "compute_cap", "vram_mib", "driver"],
        [x.strip() for x in sh("nvidia-smi --query-gpu=name,compute_cap,"
                               "memory.total,driver_version --format=csv,noheader").split(",")]
    )),
    "cuda_toolkit_system": sh("nvcc --version | awk '/release/{print $6}'"),
    "packages": {},
}

for p in PKGS:
    try:
        env["packages"][p] = metadata.version(p)
    except metadata.PackageNotFoundError:
        env["packages"][p] = None

try:
    import torch
    env["torch_runtime"] = {
        "version": torch.__version__,
        "bundled_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "arch_list": torch.cuda.get_arch_list(),
        "cuda_available": torch.cuda.is_available(),
        "device_capability": (f"sm_{torch.cuda.get_device_capability(0)[0]}"
                              f"{torch.cuda.get_device_capability(0)[1]}"
                              if torch.cuda.is_available() else None),
    }
except Exception as e:
    env["torch_runtime"] = {"error": f"{type(e).__name__}: {e}"}

freeze = sh(f"{sys.executable} -m pip freeze")
env["pip_freeze_sha256"] = hashlib.sha256(freeze.encode()).hexdigest()
env["pip_freeze_lines"] = len(freeze.splitlines())

print(json.dumps(env, indent=2, sort_keys=True))
