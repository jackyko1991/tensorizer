"""Tensorizer as a remote MCP server (Streamable HTTP at /mcp), so an agent the user already has can drive it.

CLOUD mode: any agent that supports custom MCP connectors adds `<backend>/mcp/<token>` and gets tools to list and
read problems, edit the files of a private workspace copy, and render it through the shared run queue. The token is
a per-browser "connector token" (POST /api/agents/connector); what the agent does is mirrored to that browser,
which polls GET /api/agents/connector/events.

LOCAL mode reuses the same server: a CLI agent (Codex / OpenCode / Gemini) gets a short-lived token bound to its
chat's workspace (`session_token`), so its run_problem renders go straight into that chat.

Built on the official MCP Python SDK (mcp 2.x: MCPServer + StreamableHTTPSessionManager, stateless, JSON
responses). FastAPI routers can't mount sub-applications, so routes_agents.py forwards /mcp requests here (`handle`).

Notebooks live in the browser. When a LOCAL chat is about an open notebook, the page sends its cells, agent.py writes
them to <workspace>/_context/notebook.json, and read_notebook / edit_notebook work on that file; after the turn
agent.chat hands the edited cells back to the page (the same mechanism as an agent editing the file directly).
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

from mcp.server.mcpserver.exceptions import ToolError

from . import agent
from .runner import CODE_FILES, PROBLEMS, problem_dir

MAX_EVENTS = 300
logging.getLogger("mcp").setLevel(logging.WARNING)  # the SDK logs every request at INFO
MAX_FILE_BYTES = 200_000
SESSION_TOKEN_TTL_S = 60 * 60


@dataclass
class Connector:
    id: str
    client: str
    session: str | None = None  # a LOCAL chat's token: bound to that one workspace
    expires: float | None = None
    sessions: dict[str, str] = field(default_factory=dict)  # CLOUD: problem id → agent session id
    events: list[dict] = field(default_factory=list)
    seq: int = 0
    base: str = ""  # where this backend was reached, for absolute output links

    def emit(self, event: dict) -> None:
        self.seq += 1
        self.events.append({**event, "seq": self.seq, "time": time.time()})
        del self.events[:-MAX_EVENTS]


_connectors: dict[str, Connector] = {}  # sha256(token) → connector
_current: contextvars.ContextVar[Connector | None] = contextvars.ContextVar("tensorizer_mcp_connector", default=None)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _store() -> Path:
    return agent.WORK_ROOT / "connectors.json"


def _save() -> None:
    """Keep cloud connector tokens (hashed) across API restarts, so a connected agent keeps working."""
    keep = {h: {"id": c.id, "client": c.client} for h, c in _connectors.items() if c.session is None}
    try:
        _store().parent.mkdir(parents=True, exist_ok=True)
        _store().write_text(json.dumps(keep), encoding="utf-8")
    except OSError:
        pass


def _load() -> None:
    try:
        saved = json.loads(_store().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for h, c in saved.items() if isinstance(saved, dict) else ():
        if h not in _connectors and isinstance(c, dict):
            _connectors[h] = Connector(id=str(c.get("id", "")), client=str(c.get("client", "")))


_load()


def create_connector(client: str, previous: str | None = None) -> tuple[str, Connector]:
    """A new cloud connector token (shown to the user once). `previous`: a token it replaces."""
    if previous:
        revoke(previous)
    token = "tzr_" + secrets.token_urlsafe(32)
    c = Connector(id=secrets.token_hex(6), client=client)
    _connectors[_hash(token)] = c
    _save()
    return token, c


def session_token(session_id: str, client: str) -> str:
    """A token for one LOCAL chat's workspace (revoked when the turn ends)."""
    token = "tzs_" + secrets.token_urlsafe(32)
    _connectors[_hash(token)] = Connector(
        id=secrets.token_hex(6), client=client, session=session_id, expires=time.time() + SESSION_TOKEN_TTL_S
    )
    return token


def revoke(token: str) -> None:
    c = _connectors.pop(_hash(token), None)
    if c is not None and c.session is None:
        _save()


def lookup(token: str | None) -> Connector | None:
    if not token:
        return None
    c = _connectors.get(_hash(token))
    if c is not None and c.expires is not None and time.time() > c.expires:
        _connectors.pop(_hash(token), None)
        return None
    return c


def events_after(c: Connector, after: int) -> list[dict]:
    return [e for e in c.events if e["seq"] > after]


# --- the problems an agent can work on ----------------------------------------------------------------------------

_problem_list: list[dict] | None = None


def problems() -> list[dict]:
    global _problem_list
    if _problem_list is None:
        import yaml

        out = []
        for meta in sorted(PROBLEMS.glob("*/*/meta.yaml")):
            category, slug = meta.parent.parent.name, meta.parent.name
            if category.startswith("_"):
                continue
            try:
                data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            out.append({"id": f"{category}/{slug}", "title": data.get("title", slug), "category": category})
        _problem_list = out
    return _problem_list


def _session_for(c: Connector, problem: str) -> agent.Session:
    """This connector's private workspace copy of a problem (made on first use)."""
    if c.session is not None:  # a LOCAL chat: only its own workspace
        s = agent._sessions.get(c.session)
        if s is None:
            raise ToolError("this chat has ended")
        if problem and problem not in (s.problem, "new" if s.problem == agent.NEW_ID else s.problem):
            raise ToolError(f"this chat works on {s.problem} only")
        return s
    problem = (problem or "").strip()
    if problem != "new":
        try:
            problem_dir(*problem.split("/", 1))
        except (TypeError, ValueError, FileNotFoundError):
            raise ToolError(f"unknown problem: {problem!r} (see list_problems)") from None
    sid = c.sessions.get(problem)
    if sid and sid in agent._sessions:
        return agent._sessions[sid]
    s = agent._session(None, problem)
    c.sessions[problem] = s.id
    return s


def _file(s: agent.Session, path: str) -> Path:
    """A file inside the workspace (relative path); anything that would leave it is refused."""
    if not path or Path(path).is_absolute() or path[0] in "/\\" or ":" in path:
        raise ToolError("give a path relative to the problem folder, e.g. equation.py")
    target = s.workspace / path
    if not agent._inside(str(target), s.workspace):
        raise ToolError("only files inside this problem's folder can be read or written")
    return target


def _connector() -> Connector:
    c = _current.get()
    if c is None:
        raise ToolError("not authorised")
    return c


def _mirror(c: Connector, event: dict) -> None:
    if c.session is None:  # LOCAL chats show their own CLI's tool calls; cloud ones are mirrored here
        c.emit(event)


def _tool_call(c: Connector, name: str, args: dict) -> str:
    call = secrets.token_hex(4)
    _mirror(c, {"type": "tool", "id": call, "name": name, "input": agent._short(args)})
    return call


# --- the MCP server -----------------------------------------------------------------------------------------------


def _build():
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        name="tensorizer",
        title="Tensorizer",
        instructions=(
            "Tensorizer: physics problems written in index (tensor) notation and solved with PyTorch. "
            "Use list_problems / get_problem to find one, read_file / write_file to change its files (meta.yaml, "
            "test.py, main.py, equation.py, tensors.py, solve.py, plot.py) in a private copy, and run_problem to "
            "render it (it then runs test.py: write the physics tests first, TDD). "
            "Problem id 'new' starts a brand-new problem from a template (ADDING_PROBLEMS.md explains the format)."
        ),
    )

    @server.tool(description="List the problems in the Tensorizer library (id, title, category).")
    def list_problems(category: str = "") -> str:
        _connector()
        rows = [p for p in problems() if not category or p["category"] == category]
        return json.dumps(rows, ensure_ascii=False)

    @server.tool(description="A problem's meta.yaml and code files (from your working copy once you have one).")
    def get_problem(problem: str = "") -> str:
        c = _connector()
        call = _tool_call(c, "get_problem", {"problem": problem})
        s = _session_for(c, problem)
        names = ["meta.yaml", *CODE_FILES]
        files = {n: (s.workspace / n).read_text(encoding="utf-8") for n in names if (s.workspace / n).is_file()}
        others = sorted(p.name for p in s.workspace.iterdir() if p.is_file() and p.name not in files)
        _mirror(c, {"type": "tool_result", "id": call, "ok": True, "text": ", ".join(files)})
        return json.dumps({"id": s.problem, "files": files, "other_files": others}, ensure_ascii=False)

    @server.tool(description="Read one file of a problem (path relative to the problem folder).")
    def read_file(path: str, problem: str = "") -> str:
        c = _connector()
        call = _tool_call(c, "Read", {"file_path": path})
        target = _file(_session_for(c, problem), path)
        if not target.is_file():
            _mirror(c, {"type": "tool_result", "id": call, "ok": False, "text": "no such file"})
            raise ToolError(f"no such file: {path}")
        text = target.read_text(encoding="utf-8", errors="replace")
        _mirror(c, {"type": "tool_result", "id": call, "ok": True, "text": text[:2000]})
        return text

    @server.tool(description="Write one file of a problem in your private working copy (creates or replaces it).")
    def write_file(path: str, content: str, problem: str = "") -> str:
        c = _connector()
        call = _tool_call(c, "Write", {"file_path": path, "content": content})
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ToolError("file too large (limit 200 kB)")
        s = _session_for(c, problem)
        target = _file(s, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        _mirror(c, {"type": "tool_result", "id": call, "ok": True, "text": f"wrote {path}"})
        if c.session is None:
            if edited := agent._edited_files(s):
                c.emit({"type": "files", "problem": s.problem, "files": edited})
            if s.problem == agent.NEW_ID and (entry := agent.workspace_entry(s)):
                c.emit({"type": "problem", "entry": entry})
        return f"Wrote {path} ({len(content)} characters)."

    @server.tool(
        description="Render the problem with its current files (and optional parameter overrides) on the shared "
        "run queue. Returns the terminal output and a link to the animation/plot."
    )
    async def run_problem(problem: str = "", params: dict | None = None) -> str:
        c = _connector()
        call = _tool_call(c, "mcp__tensorizer__run_problem", {"problem": problem, "params": params or {}})
        s = _session_for(c, problem)
        own_sink = s.sink is None  # a cloud run: collect the render events to mirror them to the browser
        if own_sink:
            s.sink = asyncio.Queue()
        try:
            text, is_error = await agent.run_in_session(s, params or {}, c.client)
        finally:
            if own_sink:
                sink, s.sink = s.sink, None
                while not sink.empty():
                    if (e := sink.get_nowait()) and e.get("type") == "run":
                        _mirror(c, e)
        _mirror(c, {"type": "tool_result", "id": call, "ok": not is_error, "text": text[:2000]})
        if c.base and s.output:
            text = text.replace(s.output, c.base.rstrip("/") + s.output)
        if is_error:
            raise ToolError(text)
        return text

    @server.tool(
        description="The notebook the user has open in this chat (title and cells: id, type markdown|code, source, "
        "last output). Only in a chat started from an open notebook."
    )
    def read_notebook(problem: str = "") -> str:
        c = _connector()
        call = _tool_call(c, "Read", {"file_path": f"{agent.CONTEXT_DIR}/{agent.NOTEBOOK_FILE}"})
        target = _notebook_file(_session_for(c, problem))
        text = target.read_text(encoding="utf-8")
        _mirror(c, {"type": "tool_result", "id": call, "ok": True, "text": text[:2000]})
        return text

    @server.tool(
        description="Replace the open notebook's cells: a list of {id (keep it for cells you keep; omit for new "
        "ones), type: 'markdown' | 'code', source}. The user's page shows them after this turn."
    )
    def edit_notebook(cells: list[dict], problem: str = "") -> str:
        c = _connector()
        call = _tool_call(c, "Edit", {"file_path": f"{agent.CONTEXT_DIR}/{agent.NOTEBOOK_FILE}"})
        target = _notebook_file(_session_for(c, problem))
        try:
            clean = agent.validate_notebook({"cells": cells})
        except ValueError as e:
            raise ToolError(str(e)) from None
        data = json.loads(target.read_text(encoding="utf-8"))
        target.write_text(json.dumps({**data, "cells": clean}, ensure_ascii=False, indent=1), encoding="utf-8")
        _mirror(c, {"type": "tool_result", "id": call, "ok": True, "text": f"{len(clean)} cells"})
        return f"The notebook now has {len(clean)} cells; the page applies them after this turn."

    return server


def _notebook_file(s: agent.Session) -> Path:
    target = s.workspace / agent.CONTEXT_DIR / agent.NOTEBOOK_FILE
    if s.notebook is None or not target.is_file():
        raise ToolError("no notebook is open in this chat (ask the user to open one and send the message from there)")
    return target


_server = None
_manager = None
_loop: asyncio.AbstractEventLoop | None = None
_tasks: set[asyncio.Task] = set()
_lock: asyncio.Lock | None = None


def server():
    global _server
    if _server is None:
        _server = _build()
    return _server


async def _session_manager():
    """The SDK's session manager, running (stateless, JSON responses) on the current event loop."""
    global _manager, _loop, _lock
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    loop = asyncio.get_running_loop()
    if _manager is not None and _loop is loop:
        return _manager
    if _lock is None or _loop is not loop:
        _lock, _loop, _manager = asyncio.Lock(), loop, None
    async with _lock:
        if _manager is None:
            # mcp 2.x keeps the low-level server on MCPServer._lowlevel_server; the manager needs that object.
            manager = StreamableHTTPSessionManager(app=server()._lowlevel_server, stateless=True, json_response=True)
            started = asyncio.Event()

            async def serve() -> None:
                async with manager.run():
                    started.set()
                    await asyncio.Event().wait()

            task = loop.create_task(serve())
            _tasks.add(task)
            task.add_done_callback(_tasks.discard)
            await started.wait()
            _manager = manager
    return _manager


async def handle(scope, receive, send, connector: Connector) -> None:
    """Serve one MCP request (already authorised) for `connector`."""
    manager = await _session_manager()
    token = _current.set(connector)
    try:
        await manager.handle_request(scope, receive, send)
    finally:
        _current.reset(token)
