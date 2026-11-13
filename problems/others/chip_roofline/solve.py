"""Timing on any device: synchronise (GPU work is asynchronous), warm up, then take the median of several runs."""

import statistics
import time
from collections.abc import Callable

import torch

from . import equation, tensors


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def median_seconds(fn: Callable[[], object], device: torch.device, repeats: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def measure_matmul(name: str, n: int, device: torch.device, repeats: int) -> dict:
    """Achieved FLOP/s of an n×n matrix product in one precision (or why it was skipped)."""
    ok, why = tensors.supported(name, device)
    if not ok:
        return {"precision": name, "skipped": why}
    dtype, mode = tensors.PRECISIONS[name]
    # CPUs without a BF16/FP16 matrix unit (AMX, AVX-512 BF16/FP16, Arm SME) emulate these formats very slowly:
    # a small probe first, and the full size only when the format is not far slower than FP32
    if device.type == "cpu" and dtype in (torch.bfloat16, torch.float16) and n > 256:
        probe, base = (measure_matmul(p, 256, device, 2) for p in (name, "fp32"))
        if probe["tflops"] < 0.1 * base["tflops"]:
            return {**probe, "slow": f"no fast {name.upper()} on this CPU: measured at n = 256 only"}
    before = torch.get_float32_matmul_precision()
    if mode:
        torch.set_float32_matmul_precision(mode)
    try:
        a, b = tensors.operands(n, dtype, device)
        s = median_seconds(lambda: a @ b, device, repeats)
    finally:
        torch.set_float32_matmul_precision(before)
    flops = equation.matmul_flops(n) / s
    return {
        "precision": name,
        "n": n,
        "seconds": s,
        "tflops": flops / 1e12,
        "intensity": equation.intensity(n, a.element_size()),
    }


def measure_bandwidth(megabytes: int, device: torch.device, repeats: int) -> dict:
    """Achieved memory bandwidth of a large copy, in GB/s (10^9 bytes per second)."""
    src, dst = tensors.buffers(megabytes, device)
    s = median_seconds(lambda: dst.copy_(src), device, repeats)
    return {"megabytes": megabytes, "seconds": s, "gbs": equation.copy_bytes(src.numel(), src.element_size()) / s / 1e9}


def cpu_name() -> str:
    """The processor's marketing name (e.g. "Apple M5 Pro", "AMD Ryzen 9 7950X"), else the architecture."""
    import platform
    import subprocess
    from pathlib import Path

    try:
        if platform.system() == "Darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text().splitlines():
                if line.lower().startswith(("model name", "hardware", "model")) and ":" in line:
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine() or "CPU"


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"{cpu_name()} GPU (MPS)"
    return f"{cpu_name()} CPU"
