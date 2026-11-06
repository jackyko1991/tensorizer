"""BLAS in practice: matrix-multiply throughput (GFLOP/s) against matrix size, CPU and GPU.

Feeds the "BLAS" tab of the Coding Skills page (frontend/src/content/blas_bench.json).

    pixi run python scripts/blas_benchmark.py            # CPU only (the default env's torch is CPU-only)
    pixi run -e cuda python scripts/blas_benchmark.py    # CPU + GPU (NVIDIA, CUDA build of torch)
    pixi run python scripts/blas_benchmark.py --note "measured under load"   # stored in the JSON

Every configuration times the same function, `gemm` (C = A @ B for n x n matrices), which PyTorch hands to
the BLAS library it was built with (MKL / OpenBLAS / Accelerate on the CPU, cuBLAS on NVIDIA GPUs). The
throughput is the conventional 2 n^3 floating-point operations divided by the time (n^2 outputs, each n
multiplies and n - 1 adds). Configurations: CPU float32 and float64, each with 1 thread and with all threads;
CUDA float32 and float64 when a GPU is present.

Before timing, the results are checked to agree: every configuration against a float64 CPU reference
(torch.allclose with a tolerance suited to the dtype). The JSON also records how far float32 results move
when only the thread count (or the device) changes: same maths, different summation order.

Timing: warm-up call, then the median of repeats (until a time budget), GPU calls bracketed by
torch.cuda.synchronize() with the inputs already on the device. The whole set is measured `--passes` times;
the JSON keeps the median GFLOP/s of the passes and their spread [min, max]. System CPU load is sampled
before and after: other jobs on the machine lower the numbers, especially the all-threads ones.

The dashed "peak" line is an estimate, not a measurement: cores x clock x FLOPs per cycle (CPU: 2 FMA
units x SIMD lanes x 2, from the CPU capability PyTorch reports; GPU: CUDA cores x 2 x max SM clock, with the
float64 ratio of the architecture). It is written only when every factor is known.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import inspect
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[1] / "frontend" / "src" / "content" / "blas_bench.json"
SIZES = [64, 128, 256, 512, 1024, 2048]
DTYPES = {"float32": torch.float32, "float64": torch.float64}
# agreement with the float64 reference, relative to the size of the entries (sums of n products of U(-1, 1))
RTOL = {"float32": 1e-4, "float64": 1e-10}


def gemm(A, B):
    return A @ B  # torch.matmul -> sgemm / dgemm of the BLAS library PyTorch was built with


def flops(n: int) -> int:
    """Floating-point operations of an n x n by n x n matrix product (the convention 2 n^3)."""
    return 2 * n**3


def gflops(n: int, seconds: float) -> float:
    return flops(n) / seconds / 1e9


# --- configurations ------------------------------------------------------------------------------------


def configs(cuda: bool, all_threads: int) -> list[dict]:
    out = []
    for name in DTYPES:
        out.append({"id": f"cpu_{name}_1t", "device": "cpu", "dtype": name, "threads": 1})
        out.append({"id": f"cpu_{name}_all", "device": "cpu", "dtype": name, "threads": all_threads})
    if cuda:
        for name in DTYPES:
            out.append({"id": f"cuda_{name}", "device": "cuda", "dtype": name, "threads": None})
    return out


def make_inputs(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Two n x n float64 matrices with entries in [-1, 1) (cast per configuration)."""
    return (
        2 * torch.rand(n, n, generator=gen, dtype=torch.float64) - 1,
        2 * torch.rand(n, n, generator=gen, dtype=torch.float64) - 1,
    )


def run(cfg: dict, A64: torch.Tensor, B64: torch.Tensor) -> torch.Tensor:
    """C = A @ B in the configuration's device / dtype / thread count, returned as float64 on the CPU."""
    dtype = DTYPES[cfg["dtype"]]
    A, B = A64.to(cfg["device"], dtype), B64.to(cfg["device"], dtype)
    prev = torch.get_num_threads()
    if cfg["threads"]:
        torch.set_num_threads(cfg["threads"])
    try:
        return gemm(A, B).to("cpu", torch.float64)
    finally:
        torch.set_num_threads(prev)


def agree(cfg: dict, A64: torch.Tensor, B64: torch.Tensor) -> tuple[bool, float]:
    """Compare against the float64 CPU reference; returns (ok, max |error| / max |C|)."""
    want = A64 @ B64
    got = run(cfg, A64, B64)
    scale = float(want.abs().max())
    err = float((got - want).abs().max()) / scale
    ok = torch.allclose(got, want, rtol=RTOL[cfg["dtype"]], atol=RTOL[cfg["dtype"]] * scale)
    return ok, err


# --- timing --------------------------------------------------------------------------------------------


def timeit(fn, args, sync=None, budget=0.5, max_reps=25) -> float:
    """Median seconds per call: one warm-up, then repeats until `budget` seconds or `max_reps`."""
    fn(*args)  # warm-up (thread pool start, GPU kernel selection)
    if sync:
        sync()
    times: list[float] = []
    total = 0.0
    while len(times) < max_reps and (len(times) < 3 or total < budget):
        t0 = time.perf_counter()
        fn(*args)
        if sync:
            sync()
        t = time.perf_counter() - t0
        times.append(t)
        total += t
    return statistics.median(times)


def time_config(cfg: dict, n: int, gen: torch.Generator) -> float:
    A64, B64 = make_inputs(n, gen)
    dtype = DTYPES[cfg["dtype"]]
    A, B = A64.to(cfg["device"], dtype), B64.to(cfg["device"], dtype)
    prev = torch.get_num_threads()
    if cfg["threads"]:
        torch.set_num_threads(cfg["threads"])
    try:
        sync = torch.cuda.synchronize if cfg["device"] == "cuda" else None
        return timeit(gemm, (A, B), sync=sync)
    finally:
        torch.set_num_threads(prev)


# --- machine description -------------------------------------------------------------------------------


def _powershell(expr: str) -> str:
    try:
        return subprocess.run(
            ["powershell", "-NoProfile", "-Command", expr], capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except Exception:
        return ""


def cpu_info() -> dict:
    """Name, physical cores and maximum clock (MHz) when the OS reports them."""
    info: dict = {"name": platform.processor() or platform.machine(), "cores": None, "max_mhz": None}
    if sys.platform == "win32":
        fields = '"$($p.Name)|$($p.NumberOfCores)|$($p.MaxClockSpeed)"'
        out = _powershell(f"$p = Get-CimInstance Win32_Processor; {fields}")
        parts = out.split("|")
        if len(parts) == 3:
            info["name"] = parts[0].strip() or info["name"]
            info["cores"] = int(parts[1]) if parts[1].strip().isdigit() else None
            info["max_mhz"] = int(parts[2]) if parts[2].strip().isdigit() else None
    elif Path("/proc/cpuinfo").is_file():
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.startswith("model name"):
                info["name"] = line.split(":", 1)[1].strip()
                break
        cores = {ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("core id")}
        info["cores"] = len(cores) or None
        mhz = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
        if mhz.is_file():
            info["max_mhz"] = int(mhz.read_text()) // 1000
    return info


def cpu_capability() -> str:
    try:
        return torch.backends.cpu.get_cpu_capability()  # "AVX2", "AVX512", "DEFAULT", ...
    except Exception:
        return "unknown"


def cpu_peak(info: dict, capability: str) -> dict | None:
    """Estimated peak GFLOP/s per dtype for 1 core and for all cores, or None when a factor is unknown.

    FLOPs per cycle per core = 2 FMA units x SIMD lanes x 2 (an FMA is a multiply and an add). Two FMA units
    is the common case for desktop/server x86 cores with AVX2 or AVX-512 (Zen 2+, Haswell+); it is an
    assumption, stated in the JSON.
    """
    lanes32 = {"AVX2": 8, "AVX512": 16}.get(capability)
    if not (lanes32 and info.get("cores") and info.get("max_mhz")):
        return None
    ghz = info["max_mhz"] / 1000
    out = {}
    for name, lanes in (("float32", lanes32), ("float64", lanes32 // 2)):
        per_core = 2 * lanes * 2 * ghz
        out[name] = {"1t": round(per_core, 1), "all": round(per_core * info["cores"], 1)}
    out["assumption"] = (
        f"{info['cores']} cores x {ghz:.2f} GHz (reported max clock; boost can be higher) x 2 FMA units x "
        f"{lanes32} float32 lanes ({capability}) x 2 FLOPs per FMA; float64 has half the lanes"
    )
    return out


# CUDA cores per SM and float64:float32 throughput ratio, by compute capability (NVIDIA CUDA programming guide).
CUDA_ARCH = {
    (6, 0): (64, 1 / 2),
    (6, 1): (128, 1 / 32),
    (7, 0): (64, 1 / 2),
    (7, 5): (64, 1 / 32),
    (8, 0): (64, 1 / 2),
    (8, 6): (128, 1 / 64),
    (8, 9): (128, 1 / 64),
    (9, 0): (128, 1 / 2),
}


def gpu_max_mhz() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return int(out.split()[0])
    except Exception:
        return None


def gpu_peak() -> dict | None:
    p = torch.cuda.get_device_properties(0)
    arch = CUDA_ARCH.get((p.major, p.minor))
    mhz = gpu_max_mhz()
    if not (arch and mhz):
        return None
    cores = p.multi_processor_count * arch[0]
    f32 = cores * 2 * mhz / 1000
    return {
        "float32": round(f32, 1),
        "float64": round(f32 * arch[1], 1),
        "assumption": (
            f"{p.multi_processor_count} SMs x {arch[0]} CUDA cores x 2 FLOPs per FMA x {mhz} MHz max SM clock; "
            f"float64 at 1/{round(1 / arch[1])} of float32 (compute capability {p.major}.{p.minor}); "
            "no Tensor Cores counted"
        ),
    }


def blas_in_use() -> dict:
    """What PyTorch reports about its BLAS / LAPACK (torch.__config__.show())."""
    cfg = torch.__config__.show()
    settings = {}
    for item in cfg.split("Build settings:")[-1].split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            settings[k.strip()] = v.strip()
    lines = [ln.strip(" -") for ln in cfg.splitlines()]
    return {
        "BLAS_INFO": settings.get("BLAS_INFO"),
        "LAPACK_INFO": settings.get("LAPACK_INFO"),
        "mkl_available": torch.backends.mkl.is_available(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "libraries": [
            ln
            for ln in lines
            if any(s in ln for s in ("Math Kernel", "MKL-DNN", "oneDNN", "OpenBLAS", "CuDNN", "CUDA Runtime"))
        ],
        "cpu_capability": cpu_capability(),
        "cuda": torch.version.cuda,
    }


def cpu_busy_percent(seconds: float = 1.0) -> float | None:
    """System-wide CPU use over `seconds` (Windows: GetSystemTimes; Linux: /proc/stat)."""
    try:
        if sys.platform == "win32":
            FT = ctypes.c_ulonglong

            def sample():
                idle, kern, user = FT(), FT(), FT()
                ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
                return idle.value, kern.value + user.value  # kernel time includes idle

            i0, t0 = sample()
            time.sleep(seconds)
            i1, t1 = sample()
            return round(100 * (1 - (i1 - i0) / max(t1 - t0, 1)), 1)
        if Path("/proc/stat").is_file():

            def sample():
                v = [int(x) for x in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
                return v[3] + v[4], sum(v)

            i0, t0 = sample()
            time.sleep(seconds)
            i1, t1 = sample()
            return round(100 * (1 - (i1 - i0) / max(t1 - t0, 1)), 1)
    except Exception:
        pass
    return None


def gpu_busy_percent() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return float(out.split()[0])
    except Exception:
        return None


# --- main ----------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--passes", type=int, default=3, help="measure everything this many times (spread)")
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--no-cuda", action="store_true", help="skip the GPU even if one is available")
    ap.add_argument("--note", default="", help="free text stored in machine.note (e.g. 'measured under load')")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    all_threads = torch.get_num_threads()  # default: one per physical core
    cuda = torch.cuda.is_available() and not args.no_cuda
    cfgs = configs(cuda, all_threads)
    gen = torch.Generator().manual_seed(0)

    # 1. every configuration computes the same product (to its dtype's precision)
    A64, B64 = make_inputs(256, gen)
    errors = {}
    for cfg in cfgs:
        ok, err = agree(cfg, A64, B64)
        assert ok, f"{cfg['id']}: disagrees with the float64 reference (max rel. error {err:.2e})"
        errors[cfg["id"]] = err
    # same dtype, only the thread count / device changes: bitwise equal or not?
    c1 = run({"device": "cpu", "dtype": "float32", "threads": 1}, A64, B64)
    cn = run({"device": "cpu", "dtype": "float32", "threads": all_threads}, A64, B64)
    repro = {
        "n": 256,
        "cpu_float32_1t_vs_all_bitwise_equal": bool(torch.equal(c1, cn)),
        "cpu_float32_1t_vs_all_max_abs_diff": float((c1 - cn).abs().max()),
    }
    if cuda:
        cg = run({"device": "cuda", "dtype": "float32", "threads": None}, A64, B64)
        repro["cpu_vs_cuda_float32_bitwise_equal"] = bool(torch.equal(c1, cg))
        repro["cpu_vs_cuda_float32_max_abs_diff"] = float((c1 - cg).abs().max())
    print("all configurations agree with the float64 reference:", {k: f"{v:.1e}" for k, v in errors.items()})
    print("reproducibility:", repro)

    cpu_load_before = cpu_busy_percent()
    gpu_load_before = gpu_busy_percent() if cuda else None
    print(f"load before: CPU {cpu_load_before}%  GPU {gpu_load_before}%")

    t0 = time.perf_counter()
    passes: list[dict[str, list[float]]] = []
    for p in range(args.passes):
        print(f"pass {p + 1}/{args.passes}", flush=True)
        res: dict[str, list[float]] = {}
        for cfg in cfgs:
            res[cfg["id"]] = []
            for n in args.sizes:
                s = time_config(cfg, n, gen)
                res[cfg["id"]].append(s)
                print(f"  {cfg['id']:18s} n={n:<5d} {s:9.2e} s  {gflops(n, s):8.1f} GFLOP/s", flush=True)
        passes.append(res)
    elapsed = time.perf_counter() - t0
    cpu_load_after = cpu_busy_percent()
    gpu_load_after = gpu_busy_percent() if cuda else None

    results = {}
    for cfg in cfgs:
        rows = []
        for i, n in enumerate(args.sizes):
            ts = [ps[cfg["id"]][i] for ps in passes]
            gs = [gflops(n, t) for t in ts]
            rows.append(
                {
                    "n": n,
                    "seconds": statistics.median(ts),
                    "gflops": round(statistics.median(gs), 2),
                    "gflops_range": [round(min(gs), 2), round(max(gs), 2)],
                }
            )
        results[cfg["id"]] = {**cfg, "rows": rows}

    info = cpu_info()
    cap = cpu_capability()
    out = {
        "machine": {
            "cpu": info["name"],
            "cores": info["cores"],
            "threads": all_threads,
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}",
            "date": dt.date.today().isoformat(),
            "passes": args.passes,
            "seconds": round(elapsed, 1),
            "cpu_load_percent": [cpu_load_before, cpu_load_after],
            "gpu_load_percent": [gpu_load_before, gpu_load_after] if cuda else None,
            "note": args.note,
        },
        "blas": blas_in_use(),
        "peak": {"cpu": cpu_peak(info, cap), "cuda": gpu_peak() if cuda else None},
        "agreement": {"reference": "float64 CPU", "rtol": RTOL, "max_rel_error": errors, "reproducibility": repro},
        "code": inspect.getsource(gemm).rstrip(),
        "flops": "2 n^3",
        "results": results,
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({elapsed:.0f} s)")


if __name__ == "__main__":
    main()
