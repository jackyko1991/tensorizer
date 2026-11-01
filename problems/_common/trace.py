"""Verbose mode: trace the functions a problem's main loop calls, with tensor shapes and timings.

Enabled by `--verbose` on the command line (the Verbose checkbox in the web app). Every function
from the problem's own package that main.py uses — `tensors.gravity(...)`, `solve.solve(...)`,
a directly imported `probability(...)` — is wrapped so each call prints

    → tensors.initial_state(v0=30.0, angles=Tensor(6) float32 cpu, device=cpu)
      ← Tensor(6, 2, 2) float32 cpu   0.2 ms

and the run ends with a benchmark: time spent in each stage (file) and the slowest functions, counting each
call's own time only (time inside nested traced calls is attributed to those).
"""

from __future__ import annotations

import functools
import inspect
import time
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any

import torch

_depth = 0
# Benchmark: own (exclusive) time and call counts per traced function, and the child-time stack that makes the
# times exclusive: a call's own time = its wall time − the wall time of the traced calls made inside it.
_stats: dict[str, list[float]] = {}
_child: list[float] = []


def describe(value: Any, limit: int = 4) -> str:
    """Short, shape-first description of a value (never prints whole tensors)."""
    if isinstance(value, torch.Tensor):
        shape = ", ".join(str(s) for s in value.shape) or "scalar"
        dtype = str(value.dtype).removeprefix("torch.")
        return f"Tensor({shape}) {dtype} {value.device.type}"
    if isinstance(value, (tuple, list)):
        inner = ", ".join(describe(v, limit) for v in value[:limit])
        more = ", …" if len(value) > limit else ""
        return f"({inner}{more})" if isinstance(value, tuple) else f"[{inner}{more}]"
    if isinstance(value, dict):
        inner = ", ".join(f"{k}: {describe(v, limit)}" for k, v in list(value.items())[:limit])
        return "{" + inner + (", …" if len(value) > limit else "") + "}"
    if isinstance(value, torch.device):
        return value.type
    if isinstance(value, Path):
        return value.name
    if isinstance(value, float):
        return f"{value:.6g}"
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "…"


def traced(fn: FunctionType, label: str) -> FunctionType:
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _depth
        try:
            bound = signature.bind_partial(*args, **kwargs)
            shown = ", ".join(f"{k}={describe(v)}" for k, v in bound.arguments.items())
        except TypeError:
            shown = ", ".join(describe(a) for a in args)
        pad = "  " * _depth
        print(f"{pad}→ {label}({shown})", flush=True)
        _depth += 1
        _child.append(0.0)
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        finally:
            _depth -= 1
            elapsed = time.perf_counter() - start
            own = elapsed - _child.pop()
            if _child:
                _child[-1] += elapsed
            entry = _stats.setdefault(label, [0, 0.0])
            entry[0] += 1
            entry[1] += own
        print(f"{pad}  ← {describe(result)}   {elapsed * 1e3:.1f} ms", flush=True)
        return result

    return wrapper


def install(main: ModuleType) -> None:
    """Wrap the problem-package functions that `main` reaches, as module attributes or direct imports."""
    # Under `python -m problems.x.y.main` the module is called "__main__"; its spec keeps the real name.
    name = main.__spec__.name if main.__spec__ else main.__name__
    package = name.rsplit(".", 1)[0]

    def is_ours(obj: Any) -> bool:
        return isinstance(obj, FunctionType) and obj.__module__.startswith(package + ".")

    for name, obj in list(vars(main).items()):
        if isinstance(obj, ModuleType) and obj.__name__.startswith(package + "."):
            short = obj.__name__.rsplit(".", 1)[1]
            for fname, fn in list(vars(obj).items()):
                if is_ours(fn) and fn.__module__ == obj.__name__ and not fname.startswith("_"):
                    setattr(obj, fname, traced(fn, f"{short}.{fname}"))
        elif is_ours(obj) and obj.__module__ not in (main.__name__, name):
            setattr(main, name, traced(obj, f"{obj.__module__.rsplit('.', 1)[1]}.{name}"))


def report(total: float, top: int = 6) -> None:
    """Print the benchmark: time per stage (problem file) and the slowest functions, as tables."""
    if not _stats:
        return
    by_file: dict[str, list[float]] = {}
    for label, (calls, own) in _stats.items():
        f = by_file.setdefault(label.split(".", 1)[0] + ".py", [0, 0.0])
        f[0] += calls
        f[1] += own
    traced_total = sum(own for _, own in by_file.values())
    rows = sorted(by_file.items(), key=lambda kv: -kv[1][1])
    rows.append(("main.py (rest)", [0, max(total - traced_total, 0.0)]))
    width = 44
    print("\n── benchmark " + "─" * (width - 13), flush=True)
    print(f"{'stage':<18}{'calls':>7}{'time':>11}{'share':>8}")
    for name, (calls, own) in rows:
        share = 100 * own / total if total else 0.0
        print(f"{name:<18}{int(calls) or '':>7}{own:>10.3f}s{share:>7.1f}%")
    print(f"{'total':<18}{'':>7}{total:>10.3f}s")
    print("\nslowest functions (own time):")
    for label, (calls, own) in sorted(_stats.items(), key=lambda kv: -kv[1][1])[:top]:
        print(f"  {label:<34}{int(calls):>5}×{own:>9.3f}s")
    print("─" * width, flush=True)
