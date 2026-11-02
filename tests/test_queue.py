"""The render queue: first come first served, a concurrency limit, a length limit and one run per visitor."""

import asyncio

import pytest

from backend.app.run_queue import QueueFull, RunQueue


async def _run(queue: RunQueue, client: str, log: list, hold: asyncio.Event):
    ticket = queue.join(client)
    positions = []
    try:
        async for p in queue.wait(ticket):
            positions.append(p)
        log.append(("start", client))
        await hold.wait()
        log.append(("end", client))
    finally:
        queue.leave(ticket)
    return positions


def test_runs_one_at_a_time_in_arrival_order_with_positions():
    async def main():
        q, log, hold = RunQueue(concurrent=1, limit=10), [], asyncio.Event()
        tasks = [asyncio.create_task(_run(q, c, log, hold)) for c in "abc"]
        await asyncio.sleep(0.05)
        assert log == [("start", "a")] and q.status() == {"running": 1, "waiting": 2, "concurrent": 1}
        hold.set()
        positions = await asyncio.gather(*tasks)
        assert [e for e in log if e[0] == "start"] == [("start", "a"), ("start", "b"), ("start", "c")]
        assert positions[0] == [] and positions[1] == [1] and positions[2][0] == 2  # c starts second in line
        assert positions[2] == sorted(positions[2], reverse=True)  # and only moves up
        assert q.status()["running"] == 0 and q.status()["waiting"] == 0

    asyncio.run(main())


def test_concurrency_two_starts_two():
    async def main():
        q, log, hold = RunQueue(concurrent=2, limit=10), [], asyncio.Event()
        tasks = [asyncio.create_task(_run(q, c, log, hold)) for c in "abc"]
        await asyncio.sleep(0.05)
        assert sorted(log) == [("start", "a"), ("start", "b")]
        hold.set()
        await asyncio.gather(*tasks)

    asyncio.run(main())


def test_full_queue_and_per_client_limit():
    q = RunQueue(concurrent=1, limit=2, per_client=1)
    q.join("a")
    q.join("b")
    with pytest.raises(QueueFull, match="busy"):
        q.join("c")
    q2 = RunQueue(concurrent=1, limit=5, per_client=1)
    q2.join("a")
    with pytest.raises(QueueFull, match="already"):
        q2.join("a")


def test_abandoned_waiter_leaves_the_line():
    async def main():
        q, log, hold = RunQueue(concurrent=1, limit=10), [], asyncio.Event()
        first = asyncio.create_task(_run(q, "a", log, hold))
        second = asyncio.create_task(_run(q, "b", log, hold))
        await asyncio.sleep(0.05)
        second.cancel()  # the browser tab was closed
        await asyncio.sleep(0.05)
        assert q.status()["waiting"] == 0
        hold.set()
        await first

    asyncio.run(main())


def test_abandoned_waiter_is_dropped_so_the_line_moves(monkeypatch):
    """A request that vanished while waiting (its stream never resumed) must not hold the line forever."""
    from backend.app import run_queue

    monkeypatch.setattr(run_queue, "STALE_S", 0.2)
    monkeypatch.setattr(run_queue, "CHECK_IN_S", 0.05)

    async def main():
        q, log, hold = RunQueue(concurrent=1, limit=10), [], asyncio.Event()
        runner = asyncio.create_task(_run(q, "a", log, hold))
        await asyncio.sleep(0.05)
        ghost = q.join("ghost")  # joined, then its page closed: nobody ever waits on it
        late = asyncio.create_task(_run(q, "b", log, hold))
        await asyncio.sleep(0.05)
        hold.set()
        await asyncio.wait_for(asyncio.gather(runner, late), timeout=2)  # b runs although the ghost was ahead
        assert ghost not in q.waiting and ("start", "b") in log

    asyncio.run(main())


def test_stuck_run_expires_and_details_hide_identities():
    async def main():
        q = RunQueue(concurrent=1, limit=10, max_run_s=0.1)
        stuck = q.join("a", "mechanics/projectile")
        async for _ in q.wait(stuck):
            pass
        details = q.details("b")
        assert details["runs"] == [{"problem": "mechanics/projectile", "state": "running", "seconds": 0, "mine": False}]
        await asyncio.sleep(0.15)
        assert q.status()["running"] == 0  # the abandoned run no longer holds the only slot
        t = q.join("b", "x")
        assert q.cancel("b") == 1 and t not in q.waiting

    asyncio.run(main())
