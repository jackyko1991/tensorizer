"""Run a problem in a subprocess with a timeout, streaming its output and caching results by a hash of the inputs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from .run_queue import QUEUE, QueueFull

ROOT = Path(__file__).resolve().parents[2]
PROBLEMS = ROOT / "problems"
CACHE = Path(os.environ.get("TENSORIZER_CACHE", ROOT / ".cache" / "runs"))
TIMEOUT_S = float(os.environ.get("TENSORIZER_TIMEOUT", "120"))
CODE_FILES = ["test.py", "main.py", "equation.py", "tensors.py", "solve.py", "plot.py"]
DEVICES = ("auto", "cpu", "cuda", "mps")
TEST_SUMMARY = re.compile(r"(\d+) (passed|failed|error|errors)\b")


@dataclass
class RunResult:
    run_id: str
    output: str
    console: str
    cached: bool


class RunError(Exception):
    def __init__(self, message: str, console: str = ""):
        super().__init__(message)
        self.console = console


def problem_dir(category: str, slug: str) -> Path:
    d = PROBLEMS / category / slug
    if not (d / "meta.yaml").is_file() or category.startswith("_"):
        raise FileNotFoundError(f"{category}/{slug}")
    return d


def _run_id(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def _source_hash(problem: Path) -> str:
    """Hash of the problem's code and _common, so editing a file on disk invalidates cached runs."""
    h = hashlib.sha256()
    for f in sorted([*problem.glob("*.py"), problem / "meta.yaml", *(PROBLEMS / "_common").glob("*.py")]):
        h.update(f.read_bytes())
    for f in sorted((problem / "data").glob("*")):  # a Workspace problem's uploaded data: name, size and time
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _find_output(out_dir: Path) -> str | None:
    found = sorted(p.name for p in out_dir.glob("*") if p.suffix in {".gif", ".png"} and not p.name.startswith("."))
    return found[0] if found else None


def _clean_console(text: str) -> str:
    """Progress bars redraw in place with carriage returns; keep only the final state of each line."""
    lines = text.replace("\r\n", "\n").split("\n")
    return "\n".join(line.rstrip("\r").split("\r")[-1] for line in lines)


def _tally(lines: list[str], code: int | None) -> dict:
    """{"passed", "failed", "ok"} from pytest's summary line ("3 passed, 1 failed in 0.4s")."""
    counts = {"passed": 0, "failed": 0}
    for line in lines[-5:]:
        for n, word in TEST_SUMMARY.findall(line):
            counts["passed" if word == "passed" else "failed"] += int(n)
    return {**counts, "ok": code == 0 and counts["failed"] == 0 and counts["passed"] > 0}


async def _stream_tests(test_file: Path, work: Path, env: dict[str, str]) -> AsyncIterator[dict]:
    """Run the problem's test.py (the TDD step 0) after a successful render: line events, then {"type": "tests"}."""
    cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short", "--no-header", "-p", "no:cacheprovider"]
    cmd += ["-m", "not slow", "-W", "ignore::pytest.PytestUnknownMarkWarning"]  # quick checks only (CI runs all)
    yield {"type": "line", "text": "", "replace": False}
    yield {"type": "line", "text": "── tests: test.py ──", "replace": False}
    lines: list[str] = []
    code = None
    try:
        async for event in _stream_process(cmd, work, env):
            if event["type"] == "exit":
                code = event["code"]
                break
            text = event["text"].replace(str(work) + os.sep, "")
            lines.append(text)
            yield {**event, "text": text}
    except RunError as e:
        lines.append(f"tests {e}")
        yield {"type": "line", "text": f"tests {e}", "replace": False}
    yield {"type": "tests", **_tally(lines, code), "log": "\n".join(lines)}


def _effective_device(device: str, src: Path, files: dict[str, str] | None) -> tuple[str, str | None]:
    """The device to run on, and a console note when it differs from the choice: Apple's MPS has no float64, so a
    problem whose code uses float64 runs on CPU."""
    if device not in DEVICES:
        raise ValueError(f"unknown device: {device} (use {', '.join(DEVICES)})")
    if device != "mps":
        return device, None
    code = {f.name: f.read_text(encoding="utf-8") for f in src.glob("*.py")} | (files or {})
    if any("float64" in c for name, c in code.items() if name != "test.py"):
        return "cpu", "device: cpu (this problem uses float64, which MPS does not support)"
    return "mps", None


def _env(out_dir: Path, device: str = "auto") -> dict[str, str]:
    # Keep the environment (conda/pixi needs it to find DLLs) but never hand secrets to user code.
    env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in ("TOKEN", "SECRET", "KEY"))}
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", MPLCONFIGDIR=str(out_dir / ".mpl"))
    # problems._common.runtime.device() reads the choice; ops MPS lacks fall back to CPU instead of failing
    env.update(TENSORIZER_DEVICE=device, PYTORCH_ENABLE_MPS_FALLBACK="1")
    env.update(TENSORIZER_PROGRESS="1")  # "@@progress done total stage" lines → progress events (not console)
    return env


async def _stream_process(cmd: list[str], cwd: Path, env: dict[str, str]) -> AsyncIterator[dict]:
    """Yield {"type": "line", "text", "replace"} events as the process prints, then {"type": "exit", "code"}.

    A segment ending in \\r (a tqdm redraw) is sent with replace=True so the terminal overwrites it in place.
    A plain Popen read by a thread is used because asyncio subprocesses are unsupported by the event loop
    uvicorn uses on Windows with --reload.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # Windows: no console window flashing up per run
    )

    def reader() -> None:
        buffer = b""
        while chunk := proc.stdout.read1(4096):
            buffer += chunk
            while True:
                cut = min((i for i in (buffer.find(b"\n"), buffer.find(b"\r")) if i >= 0), default=-1)
                if cut < 0:
                    break
                text, sep, buffer = buffer[:cut], buffer[cut : cut + 1], buffer[cut + 1 :]
                if sep == b"\r" and buffer.startswith(b"\n"):  # Windows line ending, not a redraw
                    sep, buffer = b"\n", buffer[1:]
                event = {"type": "line", "text": text.decode("utf-8", "replace"), "replace": sep == b"\r"}
                loop.call_soon_threadsafe(queue.put_nowait, event)
        if buffer:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "line", "text": buffer.decode("utf-8", "replace")})
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "exit", "code": proc.wait()})

    threading.Thread(target=reader, daemon=True).start()
    deadline = time.monotonic() + TIMEOUT_S
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            proc.kill()
            raise RunError(f"timed out after {TIMEOUT_S:.0f} s") from None
        yield event
        if event["type"] == "exit":
            return


async def stream_problem(
    category: str,
    slug: str,
    params: dict,
    files: dict[str, str] | None = None,
    verbose: bool = False,
    client: str = "local",
    root: Path | None = None,
    device: str = "auto",
) -> AsyncIterator[dict]:
    """Render one problem, yielding line events and finally {"type": "result", ...} (or raising RunError).

    With `root`, the problem is taken from <root>/problems/<category>/<slug> (an agent's workspace) instead of the
    repository.

    With `files`, the edited code replaces the originals in a scratch copy of the package. Renders are CPU-heavy,
    so new ones wait in the shared queue and get {"type": "queued", "position": n} events meanwhile.
    """
    src = root / "problems" / category / slug if root else problem_dir(category, slug)
    if files:
        unknown = set(files) - set(CODE_FILES)
        if unknown:
            raise ValueError(f"unknown file(s): {', '.join(sorted(unknown))}")
    device, device_note = _effective_device(device, src, files)
    note = f"{device_note}\n" if device_note else ""  # shown first, also for a cached result run on that device
    if device_note:
        yield {"type": "line", "text": device_note, "replace": False}
    run_id = _run_id(category, slug, params, files or {}, _source_hash(src), verbose, bool(root), "tests", device)
    out_dir = CACHE / run_id

    # A finished run writes console.txt last; a folder without it is a run cut short (the page closed mid-run, a
    # crash), so it is rendered again instead of being served half-done.
    if (out_dir / "console.txt").exists() and (name := _find_output(out_dir)) is not None:
        console = (out_dir / "console.txt").read_text(encoding="utf-8")
        for line in console.splitlines():
            yield {"type": "line", "text": line, "replace": False}
        tests_file = out_dir / "tests.json"
        tests = json.loads(tests_file.read_text(encoding="utf-8")) if tests_file.exists() else None
        console = note + console
        yield {"type": "result", "run_id": run_id, "output": name, "console": console, "cached": True, "tests": tests}
        return

    try:
        ticket = QUEUE.join(client, f"{category}/{slug}")
    except QueueFull as e:
        raise RunError(str(e)) from None
    try:
        try:
            async for position in QUEUE.wait(ticket):
                yield {"type": "queued", "position": position}
        except QueueFull as e:  # cancelled from the queue window
            raise RunError(str(e)) from None
        out_dir.mkdir(parents=True, exist_ok=True)
        module = f"problems.{category}.{slug}.main"
        cmd = [sys.executable, "-m", module, "--out", str(out_dir), "--params", json.dumps(params)]
        if verbose:
            cmd.append("--verbose")
        work = ROOT
        scratch = None
        if root:
            for package in (root / "problems", root / "problems" / category):
                (package / "__init__.py").touch()
            work = root
        if files:
            scratch = Path(tempfile.mkdtemp())
            shutil.copytree(PROBLEMS / "_common", scratch / "problems" / "_common")
            shutil.copytree(src, scratch / "problems" / category / slug)
            (scratch / "problems" / "__init__.py").touch()
            (scratch / "problems" / category / "__init__.py").touch()
            for fname, code in files.items():
                (scratch / "problems" / category / slug / fname).write_text(code, encoding="utf-8")
            work = scratch

        raw: list[str] = []
        code = None
        tests: dict | None = None
        test_log = ""
        try:
            async for event in _stream_process(cmd, work, _env(out_dir, device)):
                if event["type"] == "exit":
                    code = event["code"]
                    break
                if event["text"].startswith("@@progress "):  # problems._common.progress.report
                    done, total, stage = (event["text"].split(" ", 3)[1:] + [""])[:3]
                    yield {"type": "progress", "done": int(done), "total": int(total), "stage": stage}
                    continue
                raw.append(event["text"] + ("\r" if event.get("replace") else "\n"))
                if not event["text"].startswith('{"output"'):  # the CLI's final machine-readable line
                    yield event
            test_file = work / "problems" / category / slug / "test.py"  # the edited copy when there are edits
            if code == 0 and test_file.is_file():
                async for event in _stream_tests(test_file, work, _env(out_dir, device)):
                    if event["type"] == "tests":
                        tests = {k: event[k] for k in ("passed", "failed", "ok")}
                        test_log = event["log"]
                    else:
                        yield event
        except RunError as e:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise RunError(str(e), _clean_console("".join(raw))) from None
        finally:
            if scratch:
                shutil.rmtree(scratch, ignore_errors=True)

        console = _clean_console("".join(raw)).rstrip("\n")
        lines = console.splitlines()
        if lines and lines[-1].startswith('{"output"'):
            console = "\n".join(lines[:-1])
        if code != 0:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise RunError("the run failed", console)
        if tests is not None:
            console += "\n\n── tests: test.py ──\n" + test_log
            (out_dir / "tests.json").write_text(json.dumps(tests), encoding="utf-8")
        (out_dir / "console.txt").write_text(console, encoding="utf-8")
    finally:
        QUEUE.leave(ticket)

    name = _find_output(out_dir)
    if name is None:
        raise RunError("the run produced no GIF/PNG", console)
    console = note + console
    yield {"type": "result", "run_id": run_id, "output": name, "console": console, "cached": False, "tests": tests}


async def run_problem(
    category: str,
    slug: str,
    params: dict,
    files: dict[str, str] | None = None,
    verbose: bool = False,
    client: str = "local",
) -> RunResult:
    """Non-streaming wrapper: run to completion and return the result."""
    async for event in stream_problem(category, slug, params, files, verbose, client):
        if event["type"] == "result":
            return RunResult(event["run_id"], event["output"], event["console"], event["cached"])
    raise RunError("the run produced no result")
