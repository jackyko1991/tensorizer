"""For loop vs tensor: six pairs that compute the same thing, timed back to back.

Feeds the "For loop vs tensor" tab of the Coding Skills page (frontend/src/content/loop_bench.json).

    pixi run python scripts/loop_benchmark.py            # CPU only
    pixi run -e cuda python scripts/loop_benchmark.py    # CPU + GPU (NVIDIA, CUDA build of torch)
    pixi run python scripts/loop_benchmark.py --note "measured under load"   # stored in the JSON

Each case is a pair of functions: `<id>_loop` (plain Python over lists of floats, one element at a time) and
`<id>_tensor` (one PyTorch expression over whole tensors). The page shows the source of both, taken from this
file, so the code on the page is exactly the code that was timed. Every pair is checked to agree
(torch.allclose) before it is timed. Only measured sizes are reported (nothing is extrapolated).

Timing: warm-up call, then the median of repeats (fewer repeats for slow calls), GPU calls bracketed by
torch.cuda.synchronize() with the inputs already on the device (no host-device copies in the time). The
whole set of sizes is measured `--passes` times (default 2); the JSON keeps the median of the passes and
the spread of the speed-ups. Loop and tensor are timed back to back, so the ratios are meaningful even on a
busy machine; absolute times are not.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import inspect
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[1] / "frontend" / "src" / "content" / "loop_bench.json"
DTYPE = torch.float64  # float64 on both sides, so loop and tensor agree to rounding


# --- a. sum of squares / dot product -------------------------------------------------------------------


def dot_loop(x, y):
    s = 0.0
    for i in range(len(x)):
        s += x[i] * y[i]
    return s


def dot_tensor(x, y):
    return torch.dot(x, y)  # or torch.einsum("i,i->", x, y)


# --- b. pairwise distances of N particles --------------------------------------------------------------


def pairwise_loop(x):
    n = len(x)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = x[i][0] - x[j][0]
            dy = x[i][1] - x[j][1]
            dz = x[i][2] - x[j][2]
            d[i][j] = math.sqrt(dx * dx + dy * dy + dz * dz)
    return d


def pairwise_tensor(x):
    r = x[:, None, :] - x[None, :, :]  # (N, N, 3) all separations
    return r.norm(dim=-1)  # (N, N)


# --- c. matrix multiply --------------------------------------------------------------------------------


def matmul_loop(A, B):
    n, m, p = len(A), len(B), len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            s = 0.0
            for k in range(m):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def matmul_tensor(A, B):
    return A @ B  # or torch.einsum("ik,kj->ij", A, B)


# --- d. one RK4 step for B pendulums -------------------------------------------------------------------


def pendulum_loop(theta, omega, g=9.81, L=1.0, dt=0.01):
    def f(th, om):
        return om, -g / L * math.sin(th)

    out = []
    for th, om in zip(theta, omega):  # one pendulum at a time
        k1 = f(th, om)
        k2 = f(th + 0.5 * dt * k1[0], om + 0.5 * dt * k1[1])
        k3 = f(th + 0.5 * dt * k2[0], om + 0.5 * dt * k2[1])
        k4 = f(th + dt * k3[0], om + dt * k3[1])
        out.append(
            (
                th + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                om + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            )
        )
    return out


def pendulum_tensor(theta, omega, g=9.81, L=1.0, dt=0.01):
    y = torch.stack([theta, omega], dim=1)  # (B, 2): all pendulums at once

    def f(y):
        return torch.stack([y[:, 1], -g / L * torch.sin(y[:, 0])], dim=1)

    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# --- e. one explicit diffusion step on an n x n grid ---------------------------------------------------


def diffusion_loop(u, a=0.2):
    n = len(u)
    new = [row[:] for row in u]  # boundary stays fixed
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            lap = u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1] - 4 * u[i][j]
            new[i][j] = u[i][j] + a * lap
    return new


def diffusion_tensor(u, a=0.2):
    new = u.clone()
    c = u[1:-1, 1:-1]
    lap = u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * c
    new[1:-1, 1:-1] = c + a * lap
    return new


# --- f. Monte Carlo pi ---------------------------------------------------------------------------------


def mc_pi_loop(xs, ys):
    hits = 0
    for x, y in zip(xs, ys):  # one dart at a time
        if x * x + y * y <= 1.0:
            hits += 1
    return 4.0 * hits / len(xs)


def mc_pi_tensor(xs, ys):
    inside = xs**2 + ys**2 <= 1.0  # all darts at once: a (N,) bool tensor
    return 4.0 * inside.to(xs.dtype).mean()


# --- inputs: (tensor args, list args) for one size ----------------------------------------------------


def make_inputs(case: str, n: int, gen: torch.Generator) -> tuple:
    r = lambda *s: torch.rand(*s, generator=gen, dtype=DTYPE)  # noqa: E731
    if case == "dot":
        args = (r(n), r(n))
    elif case == "pairwise":
        args = (r(n, 3),)
    elif case == "matmul":
        args = (r(n, n), r(n, n))
    elif case == "pendulum":
        args = (math.pi * (r(n) - 0.5), r(n) - 0.5)
    elif case == "diffusion":
        args = (r(n, n),)
    elif case == "mc_pi":
        args = (r(n), r(n))
    else:
        raise KeyError(case)
    return args


CASES: dict[str, dict] = {
    "dot": {"sizes": [10_000, 100_000, 1_000_000], "unit": "N"},
    "pairwise": {"sizes": [100, 300, 1000], "unit": "N"},
    "matmul": {"sizes": [32, 64, 128], "unit": "n"},
    "pendulum": {"sizes": [1_000, 10_000, 100_000], "unit": "B"},
    "diffusion": {"sizes": [64, 256, 512], "unit": "n"},
    "mc_pi": {"sizes": [10_000, 100_000, 1_000_000], "unit": "N"},
}


def pair(case: str):
    mod = sys.modules[__name__]
    return getattr(mod, f"{case}_loop"), getattr(mod, f"{case}_tensor")


def agree(case: str, args: tuple, rtol: float = 1e-9, atol: float = 1e-9) -> bool:
    """Run both versions of `case` on the same input and compare."""
    loop, tensor = pair(case)
    want = torch.as_tensor(loop(*[a.tolist() for a in args]), dtype=DTYPE)
    got = tensor(*args).to("cpu", DTYPE)
    return want.shape == got.shape and torch.allclose(want, got, rtol=rtol, atol=atol)


# --- timing --------------------------------------------------------------------------------------------


def timeit(fn, args, sync=None, budget=1.0, max_reps=15) -> float:
    """Median seconds per call: one warm-up, then repeats until `budget` seconds or `max_reps`."""
    fn(*args)  # warm-up
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
        if t > budget:  # one slow call is enough
            break
    return statistics.median(times)


def timeit_loop(fn, args, budget=1.0) -> float:
    """Pure-Python loops: no JIT, so no real warm-up needed; median of up to 3 calls within the budget."""
    times: list[float] = []
    while len(times) < 3:
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
        if sum(times) > budget:
            break
    return statistics.median(times)


# --- machine description -------------------------------------------------------------------------------


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


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


def measure_pass(cuda: bool, gen: torch.Generator) -> dict[str, list[dict]]:
    res: dict[str, list[dict]] = {}
    for case, spec in CASES.items():
        loop, tensor = pair(case)
        rows = []
        for n in spec["sizes"]:
            args = make_inputs(case, n, gen)
            lists = [a.tolist() for a in args]
            loop_s = timeit_loop(loop, lists)
            cpu_s = timeit(tensor, args)
            gpu_s = None
            if cuda:
                dargs = [a.cuda() for a in args]
                gpu_s = timeit(tensor, dargs, sync=torch.cuda.synchronize)
            rows.append({"n": n, "loop_s": loop_s, "tensor_cpu_s": cpu_s, "tensor_gpu_s": gpu_s})
            print(
                f"  {case:10s} {spec['unit']}={n:<8d} loop {loop_s:9.2e}s  cpu {cpu_s:9.2e}s"
                f" (x{loop_s / cpu_s:8.0f})" + (f"  gpu {gpu_s:9.2e}s (x{loop_s / gpu_s:8.0f})" if gpu_s else ""),
                flush=True,
            )
        res[case] = rows
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--passes", type=int, default=2, help="measure everything this many times (spread)")
    ap.add_argument("--no-cuda", action="store_true", help="skip the GPU even if one is available")
    ap.add_argument("--note", default="", help="free text stored in machine.note (e.g. 'measured under load')")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    torch.set_num_threads(torch.get_num_threads())  # default: all cores
    cuda = torch.cuda.is_available() and not args.no_cuda
    gen = torch.Generator().manual_seed(0)

    # 1. every pair must compute the same result
    for case, spec in CASES.items():
        inputs = make_inputs(case, spec["sizes"][0], gen)
        assert agree(case, inputs), f"{case}: loop and tensor disagree"
        if cuda:
            loop, tensor = pair(case)
            want = tensor(*inputs)
            got = tensor(*[a.cuda() for a in inputs]).cpu()
            assert torch.allclose(want.to(DTYPE), got.to(DTYPE), rtol=1e-9, atol=1e-9), f"{case}: GPU disagrees"
    print("all pairs agree (torch.allclose, float64)")

    cpu_load_before = cpu_busy_percent()
    gpu_load_before = gpu_busy_percent() if cuda else None
    print(f"load before: CPU {cpu_load_before}%  GPU {gpu_load_before}%")

    t0 = time.perf_counter()
    passes = []
    for p in range(args.passes):
        print(f"pass {p + 1}/{args.passes}")
        passes.append(measure_pass(cuda, gen))
    elapsed = time.perf_counter() - t0
    cpu_load_after = cpu_busy_percent()
    gpu_load_after = gpu_busy_percent() if cuda else None

    def med(xs):
        xs = [x for x in xs if x is not None]
        return statistics.median(xs) if xs else None

    cases: dict[str, list[dict]] = {}
    for case in CASES:
        rows = []
        for i, n in enumerate(CASES[case]["sizes"]):
            ms = [ps[case][i] for ps in passes]
            loop_s = med([m["loop_s"] for m in ms])
            cpu_s = med([m["tensor_cpu_s"] for m in ms])
            gpu_s = med([m["tensor_gpu_s"] for m in ms])
            sp_cpu = [m["loop_s"] / m["tensor_cpu_s"] for m in ms]
            sp_gpu = [m["loop_s"] / m["tensor_gpu_s"] for m in ms if m["tensor_gpu_s"]]
            rows.append(
                {
                    "n": n,
                    "loop_s": loop_s,
                    "tensor_cpu_s": cpu_s,
                    "tensor_gpu_s": gpu_s,
                    "speedup_cpu": statistics.median(sp_cpu),
                    "speedup_gpu": statistics.median(sp_gpu) if sp_gpu else None,
                    "speedup_cpu_range": [min(sp_cpu), max(sp_cpu)],
                    "speedup_gpu_range": [min(sp_gpu), max(sp_gpu)] if sp_gpu else None,
                }
            )
        cases[case] = rows

    loop, tensor = zip(*(pair(c) for c in CASES))
    out = {
        "machine": {
            "cpu": cpu_name(),
            "threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "date": dt.date.today().isoformat(),
            "passes": args.passes,
            "seconds": round(elapsed, 1),
            "cpu_load_percent": [cpu_load_before, cpu_load_after],
            "gpu_load_percent": [gpu_load_before, gpu_load_after] if cuda else None,
            "note": args.note,
        },
        "units": {c: CASES[c]["unit"] for c in CASES},
        "code": {
            c: {"loop": inspect.getsource(lf).rstrip(), "tensor": inspect.getsource(tf).rstrip()}
            for c, lf, tf in zip(CASES, loop, tensor)
        },
        "cases": cases,
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({elapsed:.0f} s)")


if __name__ == "__main__":
    main()
