"""Detached jobs: runs that keep going after the visitor closes the page, and can be watched again later.

job = JOBS.start(visitor, client, "mechanics/projectile", params)   # → Job, running in the background
async for event in JOBS.events(job, start=0): ...                   # replay from event N, then follow live
JOBS.cancel(job)

A job is the same run as /api/run-stream (the run queue, the limits, the secret-free environment, the result cache)
but its events go to an append-only log on disk instead of to one HTTP response, so nothing is killed when the
page closes. Each job lives in <cache>/_jobs/<id>/: job.json (status, times, result summary) and events.ndjson
(one event per line, the same events as the stream: line, queued, progress, tests, result, error).

Ownership: the browser keeps a random visitor id and sends it as X-Visitor; only its hash is stored, and a visitor
sees only their own jobs. Finished jobs are deleted JOB_TTL_S after they end (0 = kept). A server restart marks
jobs that were queued or running as failed ("server restarted") instead of losing them silently.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import secrets
import shutil
import time
import weakref
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from . import limits
from .auth import output_url
from .runner import RunError, stream_problem

ACTIVE = ("queued", "running")
FINISHED = ("done", "failed", "cancelled")
# what job.json holds: no address and no visitor id, only its hash
RECORD = ("id", "visitor", "category", "slug", "params", "device", "status", "created", "started", "finished")
RECORD += ("error", "result", "events")
PROGRESS_EVERY_S = 0.25  # progress events are thinned in the log (a render can send thousands)


def visitor_hash(visitor: str) -> str:
    """What is stored for a visitor id (never the id itself)."""
    return hashlib.sha256(f"tensorizer-visitor:{visitor}".encode()).hexdigest()[:32]


@dataclass(eq=False)
class Job:
    id: str
    visitor: str  # visitor_hash(X-Visitor)
    category: str
    slug: str
    params: dict = field(default_factory=dict)
    device: str = "auto"
    status: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    result: dict | None = None  # {"output", "cached", "tests": {"passed", "failed", "ok"}}
    position: int | None = None  # place in the run queue while queued
    events: int = 0  # how many events the log holds
    # in memory only (live jobs)
    client: str = ""  # the address, for the per-visitor limit (not written to disk)
    ticket: object = None
    task: asyncio.Task | None = None
    cancel_requested: bool = False
    _wake: asyncio.Event | None = None
    _log: object = None
    _last_progress: float = 0.0

    @property
    def problem(self) -> str:
        return f"{self.category}/{self.slug}"

    def refresh(self) -> None:
        """queued → running once the run queue lets the run start."""
        ticket = self.ticket
        if self.status == "queued" and ticket is not None and getattr(ticket, "started", None) is not None:
            self.status = "running"
            self.started = time.time() - (time.monotonic() - ticket.started)
            self.position = None

    def summary(self) -> dict:
        self.refresh()
        end = self.finished or time.time()
        return {
            "id": self.id,
            "problem": self.problem,
            "category": self.category,
            "slug": self.slug,
            "status": self.status,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "elapsed": round(end - self.created, 1),
            "position": self.position,
            "events": self.events,
            "error": self.error,
            "result": self.result,
            "params": self.params,
            "device": self.device,
        }

    def record(self) -> dict:
        return {k: getattr(self, k) for k in RECORD}


class JobLimit(Exception):
    """The visitor already has a job in progress, or the server's line is full (→ HTTP 429 / 503)."""

    def __init__(self, message: str, status: int = 429):
        super().__init__(message)
        self.status = status


class JobManager:
    def __init__(self, root: Path):
        self.root = root
        self.jobs: dict[str, Job] = {}
        self._load()
        _MANAGERS.add(self)

    # --- storage ------------------------------------------------------------------------------------------------

    def _dir(self, job: Job) -> Path:
        return self.root / job.id

    def _save(self, job: Job) -> None:
        d = self._dir(job)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "job.json.tmp"
        tmp.write_text(json.dumps(job.record()), encoding="utf-8")
        tmp.replace(d / "job.json")

    def _load(self) -> None:
        """Read every job from disk; ones that were queued or running when the server stopped have failed."""
        if not self.root.is_dir():
            return
        for d in self.root.iterdir():
            try:
                rec = json.loads((d / "job.json").read_text(encoding="utf-8"))
                job = Job(**{k: rec[k] for k in RECORD if k in rec})
            except (OSError, ValueError, TypeError, KeyError, IndexError):
                shutil.rmtree(d, ignore_errors=True)  # a half-written job folder
                continue
            job.events = _count_lines(d / "events.ndjson")
            if job.status not in FINISHED:
                job.status, job.error, job.finished = "failed", "server restarted", time.time()
                self._append(job, {"type": "error", "message": "server restarted", "console": ""})
                self._close_log(job)
                self._save(job)
            self.jobs[job.id] = job

    def _append(self, job: Job, event: dict) -> None:
        if job._log is None:
            self._dir(job).mkdir(parents=True, exist_ok=True)
            job._log = open(self._dir(job) / "events.ndjson", "a", encoding="utf-8")  # noqa: SIM115
        job._log.write(json.dumps({**event, "seq": job.events}) + "\n")
        job._log.flush()
        job.events += 1
        self._wake(job)

    @staticmethod
    def _wake(job: Job) -> None:
        """Wake every follower: each waits on the event it saw before reading, so none misses an append."""
        if job._wake is not None:
            old, job._wake = job._wake, asyncio.Event()
            old.set()

    @staticmethod
    def _close_log(job: Job) -> None:
        if job._log is not None:
            job._log.close()
            job._log = None

    # --- lifecycle ----------------------------------------------------------------------------------------------

    def start(
        self,
        visitor: str,
        client: str,
        category: str,
        slug: str,
        params: dict,
        files: dict[str, str] | None = None,
        verbose: bool = False,
        device: str = "auto",
    ) -> Job:
        """Create a job and start it on the running event loop; raises JobLimit before anything starts."""
        self.prune()
        active = [j for j in self.jobs.values() if j.status in ACTIVE]
        if limits.PER_CLIENT and sum(j.client == client or j.visitor == visitor for j in active) >= limits.PER_CLIENT:
            raise JobLimit("you already have a run in progress — wait for it to finish")
        if sum(j.status == "queued" for j in active) >= limits.QUEUE_MAX:
            raise JobLimit("the server is busy — try again in a minute", 503)
        job = Job(secrets.token_hex(8), visitor, category, slug, params, device, client=client)
        job._wake = asyncio.Event()
        self.jobs[job.id] = job
        self._save(job)
        job.task = asyncio.get_running_loop().create_task(self._run(job, files, verbose))
        return job

    async def _run(self, job: Job, files: dict[str, str] | None, verbose: bool) -> None:
        def keep(ticket) -> None:
            job.ticket = ticket

        try:
            events = stream_problem(
                job.category,
                job.slug,
                job.params,
                files,
                verbose,
                job.client,
                device=job.device,
                timeout=limits.JOB_TIMEOUT_S,
                on_ticket=keep,
            )
            async for event in events:
                job.refresh()
                kind = event["type"]
                if kind == "queued":
                    job.position = event["position"]
                elif kind == "progress":
                    now = time.monotonic()
                    if now - job._last_progress < PROGRESS_EVERY_S and event["done"] < event["total"]:
                        continue
                    job._last_progress = now
                elif kind == "result":
                    event = {
                        **event,
                        "output": output_url(event["run_id"], event["output"]),
                        "params": job.params,
                    }
                    tests = event.get("tests") or None
                    job.result = {
                        "output": event["output"],
                        "run_id": event["run_id"],
                        "cached": event["cached"],
                        "tests": tests and {k: tests[k] for k in ("passed", "failed", "ok")},
                    }
                self._append(job, event)
            if job.result is None:  # every job's log ends with a result or an error
                raise RunError("the run produced no result")
            self._end(job, "done", None)
        except (RunError, ValueError, TypeError) as e:
            message = str(e)
            cancelled = job.cancel_requested or getattr(job.ticket, "cancelled", False)
            self._append(job, {"type": "error", "message": message, "console": getattr(e, "console", "")})
            self._end(job, "cancelled" if cancelled else "failed", message)
        except asyncio.CancelledError:  # cancel(), or the server shutting down
            message = "cancelled" if job.cancel_requested else "server stopped"
            self._append(job, {"type": "error", "message": message, "console": ""})
            self._end(job, "cancelled" if job.cancel_requested else "failed", message)
        except Exception as e:  # noqa: BLE001 - a job must always end with a status
            self._append(job, {"type": "error", "message": f"internal error: {e}", "console": ""})
            self._end(job, "failed", f"internal error: {e}")

    def _end(self, job: Job, status: str, error: str | None) -> None:
        job.refresh()
        job.status, job.error, job.finished, job.position = status, error, time.time(), None
        job.ticket = None
        self._close_log(job)
        self._save(job)
        self._wake(job)
        self.prune()

    def cancel(self, job: Job) -> bool:
        """Stop a queued or running job (its process is killed); False when it had already ended."""
        if job.status not in ACTIVE:
            return False
        job.cancel_requested = True
        ticket = job.ticket
        if ticket is not None:
            ticket.cancelled = True  # the runner's stop() check kills the process
        if job.task is not None and not job.task.done():
            job.task.cancel()  # and the task ends at once, also while queued or reading a cached result
        return True

    # --- reading ------------------------------------------------------------------------------------------------

    def get(self, job_id: str, visitor: str) -> Job | None:
        job = self.jobs.get(job_id)
        return job if job is not None and job.visitor == visitor else None

    def mine(self, visitor: str) -> list[Job]:
        self.prune()
        return sorted((j for j in self.jobs.values() if j.visitor == visitor), key=lambda j: -j.created)

    async def events(self, job: Job, start: int = 0, poll_s: float = 5.0) -> AsyncIterator[dict]:
        """The job's events from number `start`, then new ones as they come, until the job has ended."""
        path = self._dir(job) / "events.ndjson"
        index, offset, buffer = 0, 0, b""
        while True:
            wake = job._wake
            ended = job.status in FINISHED  # read before the file, so the last events are never missed
            if path.is_file():
                with open(path, "rb") as f:
                    f.seek(offset)
                    chunk = f.read()
                offset += len(chunk)
                buffer += chunk
                *lines, buffer = buffer.split(b"\n")
                for line in lines:
                    if index >= start and line:
                        yield json.loads(line)
                    index += 1
            if ended or wake is None:
                return
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(wake.wait(), poll_s)

    def protected(self) -> set[str]:
        """Run folders that unexpired jobs show (kept by the cache pruning)."""
        return {j.result["run_id"] for j in self.jobs.values() if j.result and j.result.get("run_id")}

    def prune(self, now: float | None = None) -> int:
        """Delete jobs that ended more than JOB_TTL_S ago; returns how many."""
        ttl = limits.JOB_TTL_S
        if not ttl:
            return 0
        now = time.time() if now is None else now
        old = [j for j in self.jobs.values() if j.status in FINISHED and j.finished and now - j.finished > ttl]
        for job in old:
            self.jobs.pop(job.id, None)
            self._close_log(job)
            shutil.rmtree(self._dir(job), ignore_errors=True)
        return len(old)


def _count_lines(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


_MANAGERS: weakref.WeakSet[JobManager] = weakref.WeakSet()


def _protected_runs() -> set[str]:
    return {name for manager in list(_MANAGERS) for name in manager.protected()}


limits.PRUNE_PROTECT.append(_protected_runs)
