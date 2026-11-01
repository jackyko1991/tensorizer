"""Run a coding-agent CLI as a subprocess and turn its JSON-lines output into the panel's events.

Shared by the Codex, OpenCode and Gemini adapters. The prompt goes in on stdin (all three read it there when it
isn't a terminal), so no user text passes through a shell or a Windows .cmd shim's argument parsing. The process
runs with cwd = the problem's workspace; events from Tensorizer's MCP tools (run_problem) arrive on `sink`
while it runs and are interleaved with the CLI's own.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import AsyncIterator, Callable
from pathlib import Path

STDERR_TAIL = 2000
TOOL_LABELS = {  # each CLI's tool names → the names the panel already labels
    "read": "Read",
    "read_file": "Read",
    "write": "Write",
    "write_file": "Write",
    "edit": "Edit",
    "replace": "Edit",
    "patch": "Edit",
    "glob": "Glob",
    "grep": "Grep",
    "grep_search": "Grep",
    "search_file_content": "Grep",
    "list": "List",
    "list_directory": "List",
    "bash": "Bash",
    "run_shell_command": "Bash",
}
RUN_TOOL = "mcp__tensorizer__run_problem"


def tool_name(name: str) -> str:
    """A CLI's tool name as the panel shows it (Tensorizer's run tool under any prefix → Run)."""
    if name.endswith("run_problem"):
        return RUN_TOOL
    return TOOL_LABELS.get(name, name)


def tool_input(data: object) -> dict:
    """Tool arguments for display: a `file_path` key whatever the CLI calls it; long strings trimmed."""
    if not isinstance(data, dict):
        return {}
    out = {k: (v[:300] + "…" if isinstance(v, str) and len(v) > 300 else v) for k, v in data.items()}
    for key in ("filePath", "absolute_path", "path", "dir_path"):
        if "file_path" not in out and isinstance(out.get(key), str):
            out["file_path"] = out[key]
    return out


def scrub(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return text


async def run(
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    prompt: str,
    mapper: Callable[[dict], list[dict]],
    sink: asyncio.Queue,
    secrets: list[str] | None = None,
    label: str = "agent",
) -> AsyncIterator[dict]:
    """Yield mapped events until the process exits; ends with `done`, or `error` if it failed."""
    secrets = secrets or []
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env={**os.environ, **env},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=16 * 1024 * 1024,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, NotImplementedError) as e:
        yield {"type": "error", "message": scrub(f"couldn't start {label}: {e}", secrets)[:500]}
        return
    assert proc.stdin and proc.stdout and proc.stderr
    proc.stdin.write(prompt.encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()

    stderr: list[str] = []
    failed = False

    async def read_stdout() -> None:
        async for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # banners / progress text: not events
            if isinstance(obj, dict):
                for event in mapper(obj):
                    await sink.put(event)

    async def read_stderr() -> None:
        async for raw in proc.stderr:
            stderr.append(raw.decode("utf-8", "replace"))
            del stderr[:-40]

    async def pump() -> None:
        try:
            await asyncio.gather(read_stdout(), read_stderr())
            await proc.wait()
        finally:
            await sink.put(None)

    task = asyncio.create_task(pump())
    try:
        while (event := await sink.get()) is not None:
            if event.get("type") == "error":
                failed = True
                event = {**event, "message": scrub(str(event.get("message", "")), secrets)[:1000]}
            yield event
        await task
        if proc.returncode not in (0, None) and not failed:
            tail = scrub("".join(stderr)[-STDERR_TAIL:], secrets).strip()
            yield {
                "type": "error",
                "message": f"{label} exited with code {proc.returncode}" + (f":\n{tail}" if tail else ""),
            }
            failed = True
        if not failed:
            yield {"type": "done", "cost_usd": None}
    finally:
        task.cancel()
        if proc.returncode is None:  # the visitor pressed stop or closed the tab
            try:
                proc.kill()
            except ProcessLookupError:
                pass
