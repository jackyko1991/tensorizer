"""A first-come-first-served queue for renders, so a shared server never runs more than a few at once.

ticket = QUEUE.join(client)          # raises QueueFull when the line is too long
try:
    async for position in QUEUE.wait(ticket):
        ...                          # tell the user their place in line (1 = next)
    ...                              # run
finally:
    QUEUE.leave(ticket)
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass


class QueueFull(Exception):
    pass


@dataclass(eq=False)
class Ticket:
    client: str


class RunQueue:
    def __init__(self, concurrent: int = 1, limit: int = 20, per_client: int = 0):
        self.concurrent, self.limit, self.per_client = concurrent, limit, per_client  # per_client 0 = no limit
        self.waiting: list[Ticket] = []
        self.running: set[Ticket] = set()
        self._version = 0  # bumped on every change, so waiters never miss a notification
        self._changed: asyncio.Condition | None = None

    def _condition(self) -> asyncio.Condition:
        if self._changed is None:
            self._changed = asyncio.Condition()
        return self._changed

    def join(self, client: str) -> Ticket:
        mine = sum(t.client == client for t in (*self.waiting, *self.running))
        if self.per_client and mine >= self.per_client:
            raise QueueFull("you already have a run in progress — wait for it to finish")
        if len(self.waiting) >= self.limit:
            raise QueueFull(f"the server is busy ({len(self.waiting)} runs waiting) — try again in a minute")
        ticket = Ticket(client)
        self.waiting.append(ticket)
        self._bump()
        return ticket

    async def wait(self, ticket: Ticket) -> AsyncIterator[int]:
        """Yield the ticket's place in line whenever it changes; return once it may run."""
        last = None
        while True:
            if len(self.running) < self.concurrent and self.waiting and self.waiting[0] is ticket:
                self.waiting.pop(0)
                self.running.add(ticket)
                self._bump()
                return
            position, version = self.waiting.index(ticket) + 1, self._version
            if position != last:
                last = position
                yield position
            async with self._condition():
                await self._condition().wait_for(lambda v=version: self._version != v)

    def leave(self, ticket: Ticket) -> None:
        """Remove a finished, failed or abandoned ticket (safe to call more than once)."""
        if ticket in self.waiting:
            self.waiting.remove(ticket)
        self.running.discard(ticket)
        self._bump()

    def status(self) -> dict:
        return {"running": len(self.running), "waiting": len(self.waiting), "concurrent": self.concurrent}

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
    limit=int(os.environ.get("TENSORIZER_MAX_QUEUE", "20")),
    per_client=int(os.environ.get("TENSORIZER_PER_CLIENT", "0")),
)
