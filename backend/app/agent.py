"""The "Ask Claude" agent: Claude Code (via the Claude Agent SDK) working on a scratch copy of one problem.

Nothing new is built here: the agent loop, tools and context handling are Claude Code's. This module only
  * gives each chat session its own workspace (a copy of one problem folder, deleted after 30 idle minutes),
  * limits Claude to Read / Write / Edit / Glob / Grep inside that folder, plus one tool `run_problem` that renders
    the edited problem through the shared run queue,
  * passes the visitor's API key to that session's Claude Code process only (never stored or logged), or, on a
    personal machine (AGENT_AUTH=local), can let Claude Code use the machine's own login,
  * turns the SDK's messages into the newline-delimited JSON events the web page draws.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from .runner import CODE_FILES, PROBLEMS, RunError, problem_dir, stream_problem

ENABLED = os.environ.get("AGENT_ENABLED") == "1"
# On a personal machine (AGENT_AUTH=local) the agent may use this machine's Claude Code login; a shared server
# only ever uses the visitor's own API key.
LOCAL_ALLOWED = os.environ.get("AGENT_AUTH") == "local"
AGENTS = ("claude-code", "claude-api")
MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "24"))
IDLE_S = 30 * 60
WORK_ROOT = Path(os.environ.get("AGENT_WORKSPACES", Path(tempfile.gettempdir()) / "tensorizer-agent"))
_active = asyncio.Semaphore(int(os.environ.get("MAX_AGENT_SESSIONS", "2")))

SYSTEM = """You are the agent inside Tensorizer, an educational website where physics problems are written in
vector form and index (tensor) form and solved with PyTorch. You are working on ONE problem, whose files are in your
working directory: meta.yaml (title, equations, translation table, parameters), main.py (the main loop),
equation.py (the physics written as tensor operations), tensors.py (building the tensors), solve.py and plot.py.
Shared helpers live in ../../_common (read-only for you).

- Explain physics and code clearly for a university student; keep answers short and concrete.
- When asked to change something, edit the files, then call the run_problem tool to check that it runs and
  produces its animation. If it fails, read the error, fix it and run again.
- Keep the index-notation style: einsum and broadcasting, with shapes in comments.
- You can only touch files in this folder. This is a free trial: one problem, nothing is saved after the visit."""

NEW_PROBLEM = """
This is a NEW problem that the user wants to create. The folder starts as a copy of the simple pendulum, as a
template: rewrite meta.yaml and all five .py files for the user's problem, keeping the same structure (main.py has
run(params, out_dir) returning the saved GIF/PNG path and ends with cli(run, __file__); meta.yaml has title,
category, level, order, summary, animated, vector_form, index_form, translation, notes, params). ADDING_PROBLEMS.md in
this folder is the full convention. Delete or replace the pendulum's diagram.svg. Run it with run_problem until it
works and the animation is clear. Keep the run under about a minute on a CPU."""

NEW_ID = "new/my_problem"  # where a brand-new problem lives in the workspace
TEMPLATE = "mechanics/simple_pendulum"


@dataclass
class Session:
    id: str
    problem: str  # "category/slug"
    workspace: Path  # <root>/<id>/problems/<category>/<slug>
    claude_session: str | None = None
    last_used: float = field(default_factory=time.monotonic)
    busy: bool = False
    output: str | None = None  # latest rendered GIF/PNG (API path) and its terminal output
    console: str = ""


_sessions: dict[str, Session] = {}


def local_claude() -> dict:
    """Is Claude Code usable here without any setup? The SDK ships the CLI; a login leaves a credentials file
    (on macOS the login sits in the Keychain, so it is reported as unknown and simply tried)."""
    if not (ENABLED and LOCAL_ALLOWED):
        return {"available": False, "logged_in": False}
    import shutil
    import sys

    try:
        import claude_agent_sdk

        bundled = any((Path(claude_agent_sdk.__file__).parent / "_bundled").glob("claude*"))
    except ImportError:
        bundled = False
    cli = bundled or shutil.which("claude") is not None
    creds = (Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / ".credentials.json").is_file()
    logged_in = True if creds or os.environ.get("ANTHROPIC_API_KEY") else (None if sys.platform == "darwin" else False)
    return {"available": cli, "logged_in": logged_in}


def status() -> dict:
    """What the AI Assistant can use on this server, for the panel's Agent form."""
    local = local_claude()
    return {
        "enabled": ENABLED,
        "agents": [
            {"id": "claude-code", "label": "Claude Code (local)", **local},
            {
                "id": "claude-api",
                "label": "API",
                "available": ENABLED,
                "providers": [
                    {"id": "anthropic", "label": "Anthropic (Claude)", "available": True},
                    {"id": "openrouter", "label": "OpenRouter", "available": False},
                ],
            },
        ],
        "default": "claude-code" if local["available"] and local["logged_in"] is not False else "claude-api",
        "models": MODELS,
        "workspace": ENABLED,
    }


def create_workspace() -> dict:
    """A blank new problem (a copy of the template) the user edits and runs by hand, without the AI."""
    s = _session(None, "new")
    entry = workspace_entry(s)
    assert entry is not None
    return entry


def _cleanup() -> None:
    now = time.monotonic()
    for sid, s in list(_sessions.items()):
        if not s.busy and now - s.last_used > IDLE_S:
            shutil.rmtree(WORK_ROOT / sid, ignore_errors=True)
            del _sessions[sid]


def _session(session_id: str | None, problem: str) -> Session:
    """The existing session, or a new one with a fresh copy of the problem."""
    _cleanup()
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    if problem == "new":
        problem = NEW_ID
        src = problem_dir(*TEMPLATE.split("/"))
    else:
        src = problem_dir(*problem.split("/", 1))  # raises FileNotFoundError for unknown problems
    category, slug = problem.split("/", 1)
    sid = uuid.uuid4().hex
    root = WORK_ROOT / sid / "problems"
    shutil.copytree(src, root / category / slug, ignore=shutil.ignore_patterns("__pycache__", "i18n"))
    shutil.copytree(PROBLEMS / "_common", root / "_common", ignore=shutil.ignore_patterns("__pycache__"))
    s = Session(sid, problem, root / category / slug)
    if problem == NEW_ID:
        shutil.copy(PROBLEMS.parent / "docs" / "adding-problems.md", s.workspace / "ADDING_PROBLEMS.md")
    _sessions[sid] = s
    return s


def _edited_files(s: Session) -> dict[str, str]:
    """Code files that differ from the original problem (a new problem: all its files)."""
    if s.problem == NEW_ID:
        names = ["meta.yaml", *CODE_FILES, "diagram.svg"]
        return {n: (s.workspace / n).read_text(encoding="utf-8") for n in names if (s.workspace / n).is_file()}
    category, slug = s.problem.split("/", 1)
    original = problem_dir(category, slug)
    out = {}
    for name in CODE_FILES:
        new = (s.workspace / name).read_text(encoding="utf-8")
        if new != (original / name).read_text(encoding="utf-8"):
            out[name] = new
    return out


def _with_context(s: Session, message: str, context: dict) -> str:
    """Mirror the user's editor into the workspace and tell Claude what they are looking at."""
    notes = []
    if s.problem != NEW_ID and isinstance(context.get("files"), dict):
        original = problem_dir(*s.problem.split("/", 1))
        for name in CODE_FILES:  # the editor is the truth: edited files as typed, the others as shipped
            code = context["files"].get(name)
            text = code if isinstance(code, str) else (original / name).read_text(encoding="utf-8")
            (s.workspace / name).write_text(text[:200_000], encoding="utf-8")
        title = str(context.get("title") or s.problem)[:200]
        edited = ", ".join(sorted(n for n in context["files"] if n in CODE_FILES)) or "none"
        notes.append(
            f'The user is looking at the problem "{title}" ({s.problem}). The files in your folder are exactly '
            f"what their editor shows (files they edited: {edited})."
        )
        if isinstance(context.get("params"), dict) and context["params"]:
            notes.append(f"Their current parameters: {json.dumps(context['params'])[:1000]}")
    sel = context.get("selection")
    if isinstance(sel, dict) and isinstance(sel.get("text"), str) and sel.get("file") in CODE_FILES:
        lines = f"{sel.get('from_line')}–{sel.get('to_line')}"
        notes.append(f"They highlighted {sel['file']} lines {lines}:\n```python\n{sel['text'][:4000]}\n```")
    return "\n".join(f"[{n}]" for n in notes) + ("\n\n" if notes else "") + message


def workspace_entry(s: Session) -> dict | None:
    """A brand-new problem as a library entry (same shape as the manifest's), for the Workspace section."""
    import base64

    import yaml

    try:
        meta = yaml.safe_load((s.workspace / "meta.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    files = [{"name": n, "code": (s.workspace / n).read_text(encoding="utf-8")} for n in CODE_FILES]
    svg = s.workspace / "diagram.svg"
    diagram = "data:image/svg+xml;base64," + base64.b64encode(svg.read_bytes()).decode() if svg.is_file() else None
    return {
        **{k: meta.get(k) for k in ("title", "level", "summary", "animated", "vector_form", "index_form", "notes")},
        "translation": meta.get("translation") or [],
        "params": meta.get("params") or {},
        "id": f"workspace/{s.id[:8]}",
        "category": "workspace",
        "slug": s.id[:8],
        "session": s.id,
        "files": files,
        "output": s.output,
        "console": s.console,
        "diagram": diagram,
    }


async def run_workspace(session_id: str, params: dict, files: dict[str, str] | None, verbose: bool, client: str):
    """Run a Workspace problem (with the visitor's edits written into the sandbox first)."""
    from problems._common.runtime import load_meta, resolve_params

    s = _sessions.get(session_id)
    if s is None:
        raise RunError("this workspace has expired — ask Claude to create the problem again")
    s.last_used = time.monotonic()
    for name, code in (files or {}).items():
        if name not in CODE_FILES:
            raise ValueError(f"unknown file: {name}")
        (s.workspace / name).write_text(code, encoding="utf-8")
    category, slug = s.problem.split("/", 1)
    params = resolve_params(load_meta(s.workspace), params)
    async for event in stream_problem(category, slug, params, None, verbose, client, s.workspace.parents[2]):
        if event["type"] == "result":
            s.output, s.console = f"/api/outputs/{event['run_id']}/{event['output']}", event["console"]
        yield event


def _inside(path: str, root: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


async def chat(
    session_id: str | None,
    problem: str,
    message: str,
    model: str | None,
    api_key: str | None,
    client_id: str,
    context: dict | None = None,
    agent: str = "claude-code",
) -> AsyncIterator[dict]:
    """One user turn. Yields events: session, text, tool, tool_result, run, queued, files, done | error."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ResultMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
        create_sdk_mcp_server,
        tool,
    )
    from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny, StreamEvent

    if agent not in AGENTS:
        agent = "claude-api" if api_key else "claude-code"
    if agent == "claude-code" and not LOCAL_ALLOWED:
        yield {"type": "error", "message": "Claude Code (local) only runs on your own computer; use the API option"}
        return
    if agent == "claude-api" and not api_key:
        yield {"type": "error", "message": "enter your Anthropic API key in the Agent settings first"}
        return
    s = _session(session_id, problem)
    message = _with_context(s, message, context or {})
    if s.busy:
        yield {"type": "error", "message": "this chat is still answering the previous message"}
        return
    yield {"type": "session", "id": s.id}

    events: asyncio.Queue = asyncio.Queue()  # run_problem pushes its own events here while the SDK runs

    @tool(
        "run_problem",
        "Render the problem with the current files (and optional parameter overrides). "
        "Returns the terminal output and whether an animation/plot was produced.",
        {"type": "object", "properties": {"params": {"type": "object"}}},
    )
    async def run_problem(args: dict) -> dict:
        category, slug = s.problem.split("/", 1)
        from problems._common.runtime import load_meta, resolve_params

        try:
            params = resolve_params(load_meta(s.workspace), args.get("params") or {})
            result = None
            async for e in stream_problem(category, slug, params, None, False, client_id, s.workspace.parents[2]):
                if e["type"] == "queued":
                    await events.put(e)
                elif e["type"] == "result":
                    result = e
            out = f"/api/outputs/{result['run_id']}/{result['output']}"
            s.output, s.console = out, result["console"]
            await events.put({"type": "run", "output": out, "console": result["console"], "ok": True})
            return {"content": [{"type": "text", "text": f"Run succeeded.\n{result['console'][-3000:]}"}]}
        except (RunError, ValueError, TypeError) as e:
            console = getattr(e, "console", "")
            await events.put({"type": "run", "output": None, "console": f"{e}\n{console}", "ok": False})
            return {"content": [{"type": "text", "text": f"Run failed: {e}\n{console[-3000:]}"}], "is_error": True}

    async def can_use_tool(name: str, data: dict, _context) -> PermissionResultAllow | PermissionResultDeny:
        path = data.get("file_path") or data.get("path") or data.get("notebook_path")
        if path is not None and not _inside(path, s.workspace):
            return PermissionResultDeny(message="Only the files of this problem can be read or changed.")
        return PermissionResultAllow(updated_input=data)

    env = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "CLAUDE_CONFIG_DIR": str(WORK_ROOT / s.id / ".claude")}
    if agent == "claude-code":
        env.pop("CLAUDE_CONFIG_DIR")  # use this machine's own Claude Code login
    else:
        env["ANTHROPIC_API_KEY"] = api_key  # the visitor's key, for this chat's Claude Code process only
    prompt = SYSTEM + (NEW_PROBLEM if s.problem == NEW_ID else "")
    options = ClaudeAgentOptions(
        cwd=str(s.workspace),
        tools=["Read", "Write", "Edit", "Glob", "Grep"],
        disallowed_tools=["Bash", "WebFetch", "WebSearch", "Task"],
        allowed_tools=["mcp__tensorizer__run_problem"],
        mcp_servers={"tensorizer": create_sdk_mcp_server("tensorizer", tools=[run_problem])},
        can_use_tool=can_use_tool,
        permission_mode="default",
        system_prompt={"type": "preset", "preset": "claude_code", "append": prompt},
        model=model if model in MODELS else MODELS[0],
        max_turns=MAX_TURNS,
        include_partial_messages=True,
        setting_sources=[],
        resume=s.claude_session,
        env=env,
    )

    async def drive() -> None:
        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(message)
                async for m in client.receive_response():
                    await events.put(("sdk", m))
        except Exception as e:  # noqa: BLE001 - shown to the user, never contains the key
            text = str(e).replace(api_key, "***") if api_key else str(e)
            if "logged in" in text.lower() or "auth" in text.lower():
                text += " — Claude Code isn't logged in on this computer; switch the Agent to API and enter a key."
            await events.put({"type": "error", "message": text[:500]})
        finally:
            await events.put(None)

    s.busy = True
    async with _active:
        task = asyncio.create_task(drive())
        try:
            while (item := await events.get()) is not None:
                if isinstance(item, dict):
                    yield item
                    continue
                m = item[1]
                if isinstance(m, StreamEvent):
                    ev = m.event
                    if ev.get("type") == "content_block_delta" and ev.get("delta", {}).get("type") == "text_delta":
                        yield {"type": "text", "delta": ev["delta"]["text"]}
                elif isinstance(m, AssistantMessage):
                    for b in m.content:
                        if isinstance(b, ToolUseBlock):
                            yield {"type": "tool", "id": b.id, "name": b.name, "input": _short(b.input)}
                        elif isinstance(b, TextBlock) and not options.include_partial_messages:
                            yield {"type": "text", "delta": b.text}
                elif isinstance(m, UserMessage) and isinstance(m.content, list):
                    for b in m.content:
                        if isinstance(b, ToolResultBlock):
                            yield {
                                "type": "tool_result",
                                "id": b.tool_use_id,
                                "ok": not b.is_error,
                                "text": _text(b.content)[:2000],
                            }
                elif isinstance(m, ResultMessage):
                    s.claude_session = m.session_id
                    edited = _edited_files(s)
                    if edited:
                        yield {"type": "files", "problem": s.problem, "files": edited}
                    if s.problem == NEW_ID and (entry := workspace_entry(s)):
                        yield {"type": "problem", "entry": entry}
                    yield {"type": "done", "cost_usd": m.total_cost_usd}
        finally:
            task.cancel()  # the visitor pressed stop or closed the tab: end Claude Code too
            s.busy = False
            s.last_used = time.monotonic()


def _short(data: dict) -> dict:
    """Tool inputs for display: long strings (whole files) trimmed."""
    return {k: (v[:300] + "…" if isinstance(v, str) and len(v) > 300 else v) for k, v in data.items()}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return json.dumps(content)[:2000]
