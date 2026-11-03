"""Defences for a shared server (the public demo), switched on with TENSORIZER_SHARED=1.

On a shared machine the hard limits below win over any other setting; on a person's own computer (the local runner)
they stay generous. Every limit is listed in docs/design/backend.md ("Shared-server defences").

- runs: at most RUN_TIMEOUT_S each, one queued or running job per visitor, a short queue (QUEUE_MAX), and a rate
  limit on fresh renders per visitor (cached results are free);
- processes: killed at the deadline even when nobody reads their output; on Linux also CPU-time, file-size and
  core-dump limits per run; console output capped;
- disk: the run cache is pruned to CACHE_MAX_BYTES, oldest first;
- code: user-written code (edits, Workspace problems, notebooks) is off unless TENSORIZER_SHARED_CODE=1;
- identity: the visitor is the address the trusted proxy saw (TRUSTED_PROXY_HOPS), not a header the client wrote.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

SHARED = os.environ.get("TENSORIZER_SHARED") == "1"
SHARED_CODE = os.environ.get("TENSORIZER_SHARED_CODE") == "1"


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


RUN_TIMEOUT_S = min(_num("TENSORIZER_TIMEOUT", 120), 120) if SHARED else _num("TENSORIZER_TIMEOUT", 120)
PER_CLIENT = 1 if SHARED else int(_num("TENSORIZER_PER_CLIENT", 0))
QUEUE_MAX = int(min(_num("TENSORIZER_MAX_QUEUE", 20), 20)) if SHARED else int(_num("TENSORIZER_MAX_QUEUE", 20))
RATE_RUNS = int(_num("TENSORIZER_RATE_RUNS", 12 if SHARED else 0))  # fresh renders per window per visitor; 0 = off
RATE_WINDOW_S = _num("TENSORIZER_RATE_WINDOW", 600)
MAX_CONSOLE_BYTES = int(_num("TENSORIZER_MAX_CONSOLE", 1_000_000 if SHARED else 20_000_000))
CACHE_MAX_BYTES = int(_num("TENSORIZER_CACHE_MAX_MB", 1500 if SHARED else 20_000) * 1e6)
TRUSTED_PROXY_HOPS = int(_num("TRUSTED_PROXY_HOPS", 1 if SHARED else 0))


def user_code_allowed() -> bool:
    """Running code the visitor wrote: always on a person's own runner, opt-in on a shared one."""
    return not SHARED or SHARED_CODE


def client_address(forwarded: str, peer: str | None) -> str:
    """Who is asking. Each proxy appends the address it saw to X-Forwarded-For, so behind trusted proxies the real
    visitor is the rightmost *public* address (internal proxy hops are private); entries further left were written by
    the client and can be faked. Without trusted proxies the TCP peer is the visitor."""
    import ipaddress

    if TRUSTED_PROXY_HOPS:
        hops = [h.strip() for h in forwarded.split(",") if h.strip()]
        for h in reversed(hops[-(TRUSTED_PROXY_HOPS + 3) :]):  # look past a few internal hops at most
            try:
                if ipaddress.ip_address(h).is_global:
                    return h
            except ValueError:
                continue
        if hops:
            return hops[-1]
    return peer or "unknown"


def limit_process() -> None:
    """preexec_fn for a run on POSIX: CPU seconds, largest file and no core dumps (Windows has no rlimits)."""
    import resource

    cpu = int(RUN_TIMEOUT_S * 4) + 30  # CPU time adds up over threads; the wall-clock deadline is the main limit
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (200_000_000, 200_000_000))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.nice(5)


PREEXEC = limit_process if SHARED and sys.platform != "win32" else None


def prune_cache(cache: Path, keep: Path | None = None) -> int:
    """Delete the oldest cached runs until the cache fits CACHE_MAX_BYTES; returns how many were removed."""
    if not cache.is_dir():
        return 0
    runs = []
    for d in cache.iterdir():
        if d.is_dir() and d != keep:
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            runs.append((d.stat().st_mtime, size, d))
    total = sum(size for _, size, _ in runs)
    removed = 0
    for _, size, d in sorted(runs):
        if total <= CACHE_MAX_BYTES:
            break
        shutil.rmtree(d, ignore_errors=True)
        total -= size
        removed += 1
    return removed


class RateLimit:
    """At most `runs` fresh renders per `window` seconds for each visitor (a sliding window)."""

    def __init__(self, runs: int, window: float):
        self.runs, self.window = runs, window
        self._seen: dict[str, list[float]] = {}

    def check(self, client: str) -> float:
        """0 if the client may start a run now (and counts it), else the seconds to wait."""
        if not self.runs:
            return 0.0
        now = time.monotonic()
        recent = [t for t in self._seen.get(client, []) if now - t < self.window]
        if len(recent) >= self.runs:
            self._seen[client] = recent
            return self.window - (now - recent[0])
        recent.append(now)
        self._seen[client] = recent
        if len(self._seen) > 10_000:  # forget idle visitors
            self._seen = {c: ts for c, ts in self._seen.items() if ts and now - ts[-1] < self.window}
        return 0.0


RATE = RateLimit(RATE_RUNS, RATE_WINDOW_S)
