"""Shared-server defences: visitor identity, rate limit, cache pruning, killing abandoned runs, user code gating."""

import asyncio
import sys
import time

from backend.app import limits, runner


def test_visitor_is_the_rightmost_public_address(monkeypatch):
    monkeypatch.setattr(limits, "TRUSTED_PROXY_HOPS", 1)
    # the client wrote a fake first hop; the proxy appended the real address; an internal hop follows
    assert limits.client_address("1.2.3.4, 81.2.69.160, 10.0.0.7", "10.0.0.8") == "81.2.69.160"
    assert limits.client_address("", "127.0.0.1") == "127.0.0.1"
    monkeypatch.setattr(limits, "TRUSTED_PROXY_HOPS", 0)  # own computer: the header is ignored
    assert limits.client_address("1.2.3.4", "127.0.0.1") == "127.0.0.1"


def test_rate_limit_is_per_visitor_and_slides():
    rate = limits.RateLimit(runs=2, window=0.2)
    assert rate.check("a") == 0 and rate.check("a") == 0
    assert rate.check("a") > 0 and rate.check("b") == 0  # a is limited, b is not
    time.sleep(0.25)
    assert rate.check("a") == 0  # the window moved on


def test_cache_is_pruned_oldest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(limits, "CACHE_MAX_BYTES", 2500)
    for i, name in enumerate(["old", "mid", "new"]):
        d = tmp_path / name
        d.mkdir()
        (d / "out.gif").write_bytes(b"x" * 1000)
        t = time.time() - 100 + i
        import os

        os.utime(d, (t, t))
    assert limits.prune_cache(tmp_path) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mid", "new"]


def test_closing_a_run_early_kills_its_process(tmp_path):
    """A page that closes mid-run must not leave the process running (on a shared server it would eat the CPU)."""
    beat = tmp_path / "beat.txt"
    code = "\n".join(
        [
            "import time",
            "print('start', flush=True)",
            "while True:",
            f"    open({str(beat)!r}, 'w').write(str(time.time()))",
            "    time.sleep(0.05)",
        ]
    )

    async def main():
        stream = runner._stream_process([sys.executable, "-c", code], tmp_path, runner._env(tmp_path))
        assert (await stream.__anext__())["text"] == "start"
        await stream.aclose()  # the client went away

    asyncio.run(main())
    time.sleep(0.5)
    last = beat.read_text()
    time.sleep(0.5)
    assert beat.read_text() == last  # no more heartbeats: the process was killed


def test_user_code_is_off_on_a_shared_server_unless_opted_in(monkeypatch):
    monkeypatch.setattr(limits, "SHARED", True)
    monkeypatch.setattr(limits, "SHARED_CODE", False)
    assert not limits.user_code_allowed()
    monkeypatch.setattr(limits, "SHARED_CODE", True)
    assert limits.user_code_allowed()
    monkeypatch.setattr(limits, "SHARED", False)
    assert limits.user_code_allowed()
