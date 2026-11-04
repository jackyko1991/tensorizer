"""A first-come-first-served queue for renders, so a shared server never runs more than a few at once.

ticket = QUEUE.join(client, "mechanics/projectile")   # raises QueueFull when the line is too long
try:
    async for position in QUEUE.wait(ticket):
        ...                          # tell the user their place in line (1 = next)
    ...                              # run
finally:
    QUEUE.leave(ticket)

A request can vanish without reaching its `finally` (the page closed while its stream was paused at a `yield`), so
tickets also expire: a waiter checks in every few seconds while it waits and is dropped when it stops, and a run
is dropped once it has been running longer than the run timeout allows. Otherwise one abandoned ticket would hold
the line, or the only run slot, forever.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from . import limits

CHECK_IN_S = 3.0  # a waiter wakes this often, even when nothing changed
STALE_S = 15.0  # a waiter that hasn't checked in for this long is gone


class QueueFull(Exception):
    pass


@dataclass(eq=False)
class Ticket:
    client: str
    label: str = ""
    joined: float = field(default_factory=time.monotonic)
    seen: float = field(default_factory=time.monotonic)
    started: float | None = None
    cancelled: bool = False  # set by the queue window's Cancel; a running job's process is then stopped


class RunQueue:
    def __init__(self, concurrent: int = 1, limit: int = 20, per_client: int = 0, max_run_s: float = 180.0):
        self.concurrent, self.limit, self.per_client = concurrent, limit, per_client  # per_client 0 = no limit
        self.max_run_s = max_run_s  # a run older than this is abandoned (its process was killed by the timeout)
        self.waiting: list[Ticket] = []
        self.running: set[Ticket] = set()
        self._version = 0  # bumped on every change, so waiters never miss a notification
        self._changed: asyncio.Condition | None = None

    def _condition(self) -> asyncio.Condition:
        if self._changed is None:
            self._changed = asyncio.Condition()
        return self._changed

    def join(self, client: str, label: str = "") -> Ticket:
        self._reap()
        mine = sum(t.client == client for t in (*self.waiting, *self.running))
        if self.per_client and mine >= self.per_client:
            raise QueueFull("you already have a run in progress — wait for it to finish")
        if len(self.waiting) >= self.limit:
            raise QueueFull(f"the server is busy ({len(self.waiting)} runs waiting) — try again in a minute")
        ticket = Ticket(client, label)
        self.waiting.append(ticket)
        self._bump()
        return ticket

    async def wait(self, ticket: Ticket) -> AsyncIterator[int]:
        """Yield the ticket's place in line whenever it changes; return once it may run."""
        last = None
        while True:
            ticket.seen = time.monotonic()
            self._reap()
            if ticket not in self.waiting:  # cancelled from the queue window
                raise QueueFull("this run was cancelled")
            if len(self.running) < self.concurrent and self.waiting[0] is ticket:
                self.waiting.pop(0)
                ticket.started = time.monotonic()
                self.running.add(ticket)
                self._bump()
                return
            position, version = self.waiting.index(ticket) + 1, self._version
            if position != last:
                last = position
                yield position
            try:
                async with self._condition():
                    await asyncio.wait_for(
                        self._condition().wait_for(lambda v=version: self._version != v), timeout=CHECK_IN_S
                    )
            except TimeoutError:
                pass  # check in, and look for abandoned tickets ahead

    def leave(self, ticket: Ticket) -> None:
        """Remove a finished, failed or abandoned ticket (safe to call more than once)."""
        if ticket in self.waiting:
            self.waiting.remove(ticket)
        self.running.discard(ticket)
        self._bump()

    def cancel(self, client: str) -> int:
        """Drop this client's tickets (the queue window's Cancel); returns how many."""
        mine = [t for t in (*self.waiting, *self.running) if t.client == client]
        for t in mine:
            t.cancelled = True
            self.leave(t)
        return len(mine)

    def _reap(self) -> None:
        now = time.monotonic()
        gone = [t for t in self.waiting if now - t.seen > STALE_S]
        gone += [t for t in self.running if t.started is not None and now - t.started > self.max_run_s]
        for t in gone:
            self.leave(t)

    def status(self) -> dict:
        self._reap()
        return {"running": len(self.running), "waiting": len(self.waiting), "concurrent": self.concurrent}

    def details(self, client: str) -> dict:
        """The queue window: every run's problem, state and time, with yours marked; no one else's identity."""
        self._reap()
        now = time.monotonic()

        def row(t: Ticket, state: str, since: float, **extra) -> dict:
            mine = t.client == client
            return {"problem": t.label, "state": state, **extra, "seconds": round(now - since), "mine": mine}

        runs = [row(t, "running", t.started or now) for t in sorted(self.running, key=lambda t: t.started or 0)]
        runs += [row(t, "waiting", t.joined, position=i + 1) for i, t in enumerate(self.waiting)]
        return {**self.status(), "limit": self.limit, "max_run_s": self.max_run_s, "runs": runs}

    def _bump(self) -> None:
        self._version += 1
        try:
            asyncio.get_running_loop().create_task(self._notify())
        except RuntimeError:  # no event loop (e.g. called from a plain test)
            pass

    async def _notify(self) -> None:
        async with self._condition():
            self._condition().notify_all()


QUEUE = RunQueue(
    concurrent=int(os.environ.get("TENSORIZER_MAX_CONCURRENT", "1")),
    limit=limits.QUEUE_MAX,  # at most 20 on a shared server
    per_client=limits.PER_CLIENT,  # one job per visitor on a shared server
    max_run_s=limits.RUN_TIMEOUT_S + 60,  # the run timeout plus tests and saving
)
