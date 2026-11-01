"""The AI Assistant: a coding agent working on a scratch copy of one problem.

Nothing new is built here: the agent loop, tools and context handling are the agent's own. This module only
  * gives each chat session its own workspace (a copy of one problem folder, deleted after 30 idle minutes),
  * picks the agent (backend/app/agents/): Claude Code via the Claude Agent SDK, or, on a personal machine, the
    Codex / OpenCode / Gemini CLI found there; each works only inside that folder and renders the problem through
    one tool, `run_problem`, which uses the shared run queue,
  * passes the visitor's API key (Anthropic or OpenRouter) to that session's agent process only (never stored or
    logged), or, on a personal machine (AGENT_AUTH=local), lets the agent use the machine's own login,
  * turns the agent's output into the newline-delimited JSON events the web page draws.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
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
AGENTS = ("claude-code", "claude-api", "codex", "opencode", "gemini")
LOCAL_CLIS = ("codex", "opencode", "gemini")  # experimental adapters, offered only after "Search this computer"
MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
# OpenRouter through its Anthropic-compatible endpoint, as documented for Claude Code:
# https://openrouter.ai/docs/guides/guides/claude-code-integration (ids from GET https://openrouter.ai/api/v1/models)
OPENROUTER_BASE = "https://openrouter.ai/api"
OPENROUTER_MODELS = ["anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5"]
PROVIDERS = {"anthropic": MODELS, "openrouter": OPENROUTER_MODELS}
_CLI_MODEL = re.compile(r"^[\w./:~@-]{1,100}$")  # a model name a CLI may get on its command line
MAX_SCREENSHOT_BYTES = 3 * 1024 * 1024
CONTEXT_DIR = "_context"  # screenshots etc. the user attached, inside the workspace so the agent can read them
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
    cli_session: str | None = None  # a CLI agent's own session/thread id, to resume the conversation
    cli_agent: str | None = None  # which CLI that id belongs to
    sink: asyncio.Queue | None = None  # while a turn runs: where run_problem reports its renders
    last_used: float = field(default_factory=time.monotonic)
    busy: bool = False
    output: str | None = None  # latest rendered GIF/PNG (API path) and its terminal output
    console: str = ""
    sandbox: str | None = None  # a Workspace problem's sandbox (workspace.py) this session works in
    notebook: dict | None = None  # the notebook written to _context/notebook.json this turn: {id, cells}


_sessions: dict[str, Session] = {}


def local_claude() -> dict:
    """Is Claude Code usable here without any setup? The SDK ships the CLI; a login leaves a credentials file
    (on macOS the login sits in the Keychain, so it is reported as unknown and simply tried)."""
    if not (ENABLED and LOCAL_ALLOWED):
        return {"available": False, "logged_in": False}
    from .agents import scan

    return {"available": scan.executable("claude-code") is not None, "logged_in": scan.logged_in("claude-code")}


def status() -> dict:
    """What the AI Assistant can use on this server, for the panel's Agent form. Nothing is run here: the other
    local agents are only looked for when the user clicks "Search this computer" (POST /api/agents/scan)."""
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
                    {"id": "anthropic", "label": "Anthropic (Claude)", "available": True, "models": MODELS},
                    {"id": "openrouter", "label": "OpenRouter", "available": True, "models": OPENROUTER_MODELS},
                ],
            },
        ],
        "default": "claude-code" if local["available"] and local["logged_in"] is not False else "claude-api",
        "models": MODELS,
        "workspace": ENABLED,
        # Which of the three modes this server offers: LOCAL only on the user's own machine.
        "modes": {"local": ENABLED and LOCAL_ALLOWED, "cloud": ENABLED, "api": ENABLED},
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


def _workspace_session(session_id: str | None, sandbox: str | None) -> Session:
    """The agent session for a Workspace problem: the AI-created problem's own session, or one working directly in
    the problem's workspace.py sandbox (made on first use, kept per sandbox). LookupError when the sandbox is gone."""
    from . import workspace

    _cleanup()
    if sandbox and sandbox in _sessions:  # a problem the AI created: keep using its session
        return _sessions[sandbox]
    box = workspace._sandboxes.get(sandbox or "")
    if box is None:
        raise LookupError(sandbox)
    if session_id and (s := _sessions.get(session_id)) is not None and s.sandbox == sandbox:
        return s
    for s in _sessions.values():
        if s.sandbox == sandbox:
            return s
    s = Session(uuid.uuid4().hex, f"{workspace.CATEGORY}/{box.slug}", box.folder, sandbox=sandbox)
    _sessions[s.id] = s
    return s


WS_FILES = ["meta.yaml", *CODE_FILES]


def _snapshot(s: Session) -> dict[str, str]:
    return {n: (s.workspace / n).read_text(encoding="utf-8") for n in WS_FILES if (s.workspace / n).is_file()}


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
    ws = context.get("workspace") if isinstance(context.get("workspace"), dict) else None
    if ws is not None or s.problem.startswith("workspace/"):
        # A Workspace problem: the browser's files were synced into the folder just before this turn.
        notes.extend(_workspace_notes(s, context, ws or {}))
    elif s.problem != NEW_ID and isinstance(context.get("files"), dict):
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
    has_text = isinstance(sel, dict) and isinstance(sel.get("text"), str) and sel["text"].strip()
    if has_text and sel.get("file") in WS_FILES:
        lines = f"{sel.get('from_line')}–{sel.get('to_line')}"
        lang = "yaml" if sel["file"] == "meta.yaml" else "python"
        notes.append(f"They highlighted {sel['file']} lines {lines}:\n```{lang}\n{sel['text'][:4000]}\n```")
    elif has_text and sel.get("from_line") and _label(sel.get("region")):
        # Highlighted in another editor of the page: the Markdown document or a notebook cell.
        lang = sel.get("lang") if sel.get("lang") in ("python", "markdown", "yaml") else ""
        lines = f"{sel.get('from_line')}–{sel.get('to_line')}"
        region = _label(sel.get("region"))
        notes.append(f"They highlighted lines {lines} of the {region}:\n```{lang}\n{sel['text'][:4000]}\n```")
    elif has_text:
        # Text highlighted elsewhere on the page: the summary, equations, translation table, terminal, ...
        region = _label(sel.get("region"))[:60] or "page"
        notes.append(f"They highlighted this in the problem's {region}: «{sel['text'].strip()[:2000]}»")
    if isinstance(context.get("screenshot"), str):
        notes.append(_save_screenshot(s, context["screenshot"]))
    return "\n".join(f"[{n}]" for n in notes) + ("\n\n" if notes else "") + message


def _label(value) -> str:
    """A short label the page gave a region, e.g. "notebook: Notebook 1, cell 3" (no markup or newlines)."""
    return re.sub(r"[^\w :,.()·-]", "", str(value or ""))[:80].strip()


# --- Workspace problems: what the user sees (Source in Inline / Markdown / meta.yaml, or a notebook) ----------------

NOTEBOOK_FILE = "notebook.json"  # in CONTEXT_DIR: the open notebook, which the agent may edit
NOTE_BUDGET = 8000  # characters of notebook cells quoted in the message (the focused cell always in full)
MAX_CELLS = 60
MAX_NOTEBOOK_BYTES = 200_000
SOURCE_MODES = {
    "inline": "the Inline view (title, summary, equations, translation table, notes and parameters, rendered and "
    "editable in place)",
    "markdown": "the Markdown view: the problem as one Markdown document (# title, summary, ## Vector form, "
    "## Index form, ## Translation, ## Notes), generated from meta.yaml and parsed back into it",
    "yaml": "the meta.yaml editor",
}


def _workspace_notes(s: Session, context: dict, ws: dict) -> list[str]:
    title = str(context.get("title") or "Untitled problem")[:200]
    notes = [
        f'The user is working on their own Workspace problem "{title}". The files in your folder (meta.yaml and '
        "main.py, equation.py, tensors.py, solve.py, plot.py) are exactly what their browser holds. Edit them "
        "directly: after your turn the page shows your changes in the Inline view, the Markdown view, the meta.yaml "
        "editor and the code editor. Title, summary, equations (vector_form, index_form), translation table, notes "
        "and parameters all live in meta.yaml."
    ]
    nb_file = s.workspace / CONTEXT_DIR / NOTEBOOK_FILE
    s.notebook = None
    if ws.get("kind") == "notebook":
        notes.append(_notebook_note(s, ws, nb_file))
    elif ws.get("kind") == "data":
        nb_file.unlink(missing_ok=True)
        notes.append("They have the problem's Data page open: the files they uploaded (listed below).")
    else:
        nb_file.unlink(missing_ok=True)
        mode = ws.get("mode") if ws.get("mode") in SOURCE_MODES else "inline"
        notes.append(f"They have the problem's Source open, in {SOURCE_MODES[mode]}.")
        if mode == "markdown" and isinstance(ws.get("markdown"), str) and ws["markdown"].strip():
            notes.append(
                "The Markdown document on their screen (change it by editing meta.yaml):\n```markdown\n"
                f"{ws['markdown'][:6000]}\n```"
            )
    if data := data_note(s.workspace):
        notes.append(data)
    if isinstance(context.get("params"), dict) and context["params"]:
        notes.append(f"Their current parameters: {json.dumps(context['params'])[:1000]}")
    return notes


def data_note(folder: Path) -> str:
    """The problem's uploaded data files (data/): name, size and a peek inside (columns and first rows of tables,
    array shapes, image sizes), so the agent can load and fit them."""
    files = sorted(f for f in (folder / "data").glob("*") if f.is_file())
    if not files:
        return ""
    rows = []
    for f in files[:20]:
        size = f.stat().st_size
        peek = _peek(f)
        rows.append(f"- data/{f.name} ({size / 1024:.1f} kB){': ' + peek if peek else ''}")
    head = (
        "The user uploaded data files into data/ (next to meta.yaml). Load them in notebook cells as "
        'Path("data/<name>") and in the problem\'s code as Path(__file__).parent / "data" / "<name>":'
    )
    return NL.join([head, *rows])


NL = chr(10)
FENCE = "```"


def _peek(f: Path) -> str:
    ext = f.suffix.lower()
    try:
        if ext in (".csv", ".tsv", ".txt", ".dat"):
            with f.open(encoding="utf-8", errors="replace") as fh:
                lines = [line.rstrip(NL)[:300] for _, line in zip(range(6), fh)]
            return NL.join(["first lines (header + 5 rows):", FENCE, *lines, FENCE])
        if ext == ".json":
            return "starts with: " + f.read_text(encoding="utf-8", errors="replace")[:400].replace(NL, " ")
        if ext == ".npy":
            import numpy as np

            with f.open("rb") as fh:
                version = np.lib.format.read_magic(fh)
                shape, _, dtype = np.lib.format._read_array_header(fh, version)
            return f"array shape {shape}, dtype {dtype}"
        if ext == ".npz":
            import zipfile

            with zipfile.ZipFile(f) as z:
                return "arrays: " + ", ".join(n.removesuffix(".npy") for n in z.namelist()[:20])
        if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            from PIL import Image

            with Image.open(f) as im:
                return f"image {im.width}×{im.height}, mode {im.mode}"
    except Exception:  # noqa: BLE001 - a peek is optional
        return ""
    return ""


def _clean_cells(cells) -> list[dict]:
    out = []
    for c in cells if isinstance(cells, list) else []:
        if not isinstance(c, dict) or c.get("type") not in ("markdown", "code") or not isinstance(c.get("source"), str):
            continue
        cell = {"id": str(c.get("id") or "")[:40], "type": c["type"], "source": c["source"][:50_000]}
        if c["type"] == "code" and isinstance(c.get("output"), str) and c["output"].strip():
            cell["output"] = c["output"][-1000:]
        out.append(cell)
    return out[:MAX_CELLS]


def _notebook_note(s: Session, ws: dict, nb_file: Path) -> str:
    """Write the open notebook to _context/notebook.json and describe it, quoting its cells within NOTE_BUDGET."""
    cells = _clean_cells(ws.get("cells"))
    nb_title = str(ws.get("notebook_title") or "Notebook")[:120]
    focus = ws.get("focused_cell")
    focus = focus if isinstance(focus, int) and 1 <= focus <= len(cells) else None
    nb_file.parent.mkdir(exist_ok=True)
    nb_file.write_text(json.dumps({"title": nb_title, "cells": cells}, ensure_ascii=False, indent=1), encoding="utf-8")
    s.notebook = {"id": str(ws.get("notebook_id") or ""), "cells": [_bare(c) for c in cells]}

    def quote(i: int, c: dict, full: bool) -> str:
        src = c["source"] if full else _cut(c["source"], 600)
        head = f"cell {i} ({c['type']}{', focused' if i == focus else ''})"
        body = f"```{'python' if c['type'] == 'code' else 'markdown'}\n{src}\n```"
        out = f"\noutput: {_cut(c['output'], 1000 if full else 300)}" if c.get("output") else ""
        return f"{head}:\n{body}{out}"

    parts: list[tuple[int, str]] = []
    used = 0
    if focus:
        parts.append((focus, quote(focus, cells[focus - 1], True)))
        used = len(parts[0][1])
    skipped = 0
    for i, c in enumerate(cells, 1):
        if i == focus:
            continue
        q = quote(i, c, False)
        if used + len(q) > NOTE_BUDGET:
            skipped += 1
            continue
        parts.append((i, q))
        used += len(q)
    quoted = "\n\n".join(q for _, q in sorted(parts))
    more = f"\n\n({skipped} more cell(s) not shown: read {CONTEXT_DIR}/{NOTEBOOK_FILE})" if skipped else ""
    where = f"; cell {focus} is focused" if focus else ""
    return (
        f'They have the notebook "{nb_title}" of this problem open ({len(cells)} cells{where}). Code cells run in '
        "order in a fresh Python process with cwd in this folder, the problem importable as `problem` (e.g. `from "
        "problem import equation`) and its default parameters as `params`; figures are shown with plt.show(). The "
        "outputs are from the user's last run ([figure] = a plot).\n\n"
        f"{quoted or '(no cells)'}{more}\n\n"
        f"The whole notebook is {CONTEXT_DIR}/{NOTEBOOK_FILE}. To change its cells, edit that file and keep its "
        'structure: {"title": ..., "cells": [{"id": ..., "type": "markdown" | "code", "source": "..."}]} (keep the '
        "ids of the cells you keep; new cells need no id; outputs are ignored). The page applies it after your turn."
    )


def _cut(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + f"… [{len(text) - n} more characters]"


def _bare(c: dict) -> dict:
    return {"id": c.get("id", ""), "type": c["type"], "source": c["source"]}


def validate_notebook(data) -> list[dict]:
    """Cells from an edited notebook.json; ValueError when the structure is wrong."""
    if not isinstance(data, dict) or not isinstance(data.get("cells"), list):
        raise ValueError('expected {"cells": [...]}')
    cells = []
    for i, c in enumerate(data["cells"], 1):
        if not isinstance(c, dict) or c.get("type") not in ("markdown", "code") or not isinstance(c.get("source"), str):
            raise ValueError(f'cell {i} needs "type" ("markdown" or "code") and a "source" string')
        cells.append({"id": str(c.get("id") or "")[:40], "type": c["type"], "source": c["source"]})
    if len(cells) > MAX_CELLS or sum(len(c["source"]) for c in cells) > MAX_NOTEBOOK_BYTES:
        raise ValueError("notebook too large")
    return cells


def notebook_event(s: Session, problem: str) -> dict | None:
    """After a turn: the notebook's cells if the agent changed _context/notebook.json (an error event if it broke
    it, leaving the notebook as it was)."""
    if s.notebook is None:
        return None
    nb_file = s.workspace / CONTEXT_DIR / NOTEBOOK_FILE
    try:
        cells = validate_notebook(json.loads(nb_file.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None
    except (ValueError, OSError) as e:  # json.JSONDecodeError is a ValueError
        message = f"{CONTEXT_DIR}/{NOTEBOOK_FILE} is not valid ({e}); the notebook was left unchanged"
        return {"type": "error", "message": message[:500]}
    if cells == s.notebook["cells"]:
        return None
    s.notebook["cells"] = cells
    return {"type": "notebook", "problem": problem, "notebook": s.notebook["id"], "cells": cells}


class ContextError(ValueError):
    """Context the page sent that can't be used (e.g. a screenshot that is too large)."""


def _save_screenshot(s: Session, data: str) -> str:
    """Write a PNG screenshot (base64, optionally a data: URL) to <workspace>/_context/ and describe it."""
    raw = data.split(",", 1)[1] if data.startswith("data:") else data
    if len(raw) * 3 // 4 > MAX_SCREENSHOT_BYTES:
        raise ContextError("the screenshot is too large (limit 3 MB)")
    try:
        png = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise ContextError("the screenshot is not valid base64") from None
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ContextError("the screenshot must be a PNG image")
    folder = s.workspace / CONTEXT_DIR
    folder.mkdir(exist_ok=True)
    n = len(list(folder.glob("screenshot-*.png"))) + 1
    (folder / f"screenshot-{n}.png").write_bytes(png)
    rel = f"{CONTEXT_DIR}/screenshot-{n}.png"
    return f"The user attached a screenshot of the app: {rel}. Look at it with the Read tool."


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


async def run_in_session(s: Session, params: dict, client_id: str) -> tuple[str, bool]:
    """The `run_problem` tool, for every agent (in-process for Claude Code, over MCP for the others): render the
    workspace through the shared run queue. Returns (text for the agent, is_error); the render itself goes to the
    chat that is running (s.sink) as a `run` event."""
    from problems._common.runtime import load_meta, resolve_params

    category, slug = s.problem.split("/", 1)
    s.last_used = time.monotonic()

    async def report(event: dict) -> None:
        if s.sink is not None:
            await s.sink.put(event)

    try:
        resolved = resolve_params(load_meta(s.workspace), params or {})
        result = None
        async for e in stream_problem(category, slug, resolved, None, False, client_id, s.workspace.parents[2]):
            if e["type"] == "queued":
                await report(e)
            elif e["type"] == "result":
                result = e
        if result is None:
            raise RunError("the run produced no result")
        out = f"/api/outputs/{result['run_id']}/{result['output']}"
        s.output, s.console = out, result["console"]
        await report({"type": "run", "output": out, "console": result["console"], "ok": True})
        return f"Run succeeded. Output: {out}\n{result['console'][-3000:]}", False
    except (RunError, ValueError, TypeError) as e:
        console = getattr(e, "console", "")
        await report({"type": "run", "output": None, "console": f"{e}\n{console}", "ok": False})
        return f"Run failed: {e}\n{console[-3000:]}", True


def _model_for(agent: str, provider: str, model: str | None) -> str | None:
    if agent == "claude-api":
        models = PROVIDERS.get(provider, MODELS)
        return model if model in models else models[0]
    if agent == "claude-code":
        return model if model in MODELS else MODELS[0]
    return model if model and _CLI_MODEL.match(model) else None  # a CLI: its own default unless a name is given


async def chat(
    session_id: str | None,
    problem: str,
    message: str,
    model: str | None,
    api_key: str | None,
    client_id: str,
    context: dict | None = None,
    agent: str = "claude-code",
    provider: str = "anthropic",
    mcp_base: str | None = None,
    sandbox: str | None = None,
) -> AsyncIterator[dict]:
    """One user turn. Yields events: session, text, tool, tool_result, run, queued, files, problem, done | error.

    agent: "claude-code" (this machine's Claude Code login), "claude-api" (Claude Code with the visitor's key for
    `provider`: anthropic or openrouter), or a local CLI ("codex", "opencode", "gemini"; needs `mcp_base`, the URL
    this backend is reachable at from the same machine, so the CLI can call run_problem over MCP).

    problem "workspace/<id>": a Workspace problem, worked on in place in `sandbox` (the workspace.py sandbox the
    page synced just before, or the agent session of a problem the AI created); edited files (meta.yaml included)
    come back as a `files` event with the re-parsed `entry`, an edited notebook as a `notebook` event.
    """
    from .agents import claude_code, scan

    if agent not in AGENTS:
        agent = "claude-api" if api_key else "claude-code"
    if provider not in PROVIDERS:
        provider = "anthropic"
    if agent != "claude-api" and not LOCAL_ALLOWED:
        yield {"type": "error", "message": "local agents only run on your own computer; use the API option"}
        return
    if agent == "claude-api" and not api_key:
        name = "OpenRouter" if provider == "openrouter" else "Anthropic"
        yield {"type": "error", "message": f"enter your {name} API key in the Agent settings first"}
        return
    exe = scan.executable(agent) if agent in LOCAL_CLIS else None
    if agent in LOCAL_CLIS and (exe is None or not mcp_base):
        yield {"type": "error", "message": f"{agent} was not found on this computer — search again in Agent ▸ Local"}
        return
    in_ws = problem.startswith("workspace/")
    if in_ws:
        try:
            s = _workspace_session(session_id, sandbox)
        except LookupError:
            yield {"type": "error", "message": "this problem's sandbox has expired — send the message again"}
            return
    else:
        s = _session(session_id, problem)
    if s.busy:
        yield {"type": "error", "message": "this chat is still answering the previous message"}
        return
    try:
        message = _with_context(s, message, context or {})
    except ContextError as e:
        yield {"type": "error", "message": str(e)}
        return
    yield {"type": "session", "id": s.id}
    before = _snapshot(s) if in_ws else {}
    box = _hold_sandbox(s.sandbox, +1)

    secrets = [api_key] if api_key else []
    s.busy = True
    s.sink = asyncio.Queue()
    chosen = _model_for(agent, provider, model)
    try:
        async with _active:
            if agent in ("claude-code", "claude-api"):
                env = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1", "CLAUDE_CONFIG_DIR": str(WORK_ROOT / s.id / ".claude")}
                if agent == "claude-code":
                    env.pop("CLAUDE_CONFIG_DIR")  # use this machine's own Claude Code login
                elif provider == "openrouter":  # exactly the variables OpenRouter documents for Claude Code
                    env.update(ANTHROPIC_BASE_URL=OPENROUTER_BASE, ANTHROPIC_AUTH_TOKEN=api_key, ANTHROPIC_API_KEY="")
                else:
                    env["ANTHROPIC_API_KEY"] = api_key  # the visitor's key, for this chat's Claude Code process only
                hint = (
                    " — Claude Code isn't logged in on this computer; switch the Agent to API and enter a key."
                    if agent == "claude-code"
                    else " — check the API key in the Agent settings."
                )
                stream = claude_code.run(
                    s,
                    message,
                    model=chosen,
                    env=env,
                    system_prompt=SYSTEM + (NEW_PROBLEM if s.problem == NEW_ID and not in_ws else ""),
                    max_turns=MAX_TURNS,
                    run_tool=lambda params: run_in_session(s, params, client_id),
                    sink=s.sink,
                    secrets=secrets,
                    login_hint=hint,
                )
            else:
                stream = _cli_stream(s, agent, exe, message, chosen, mcp_base, client_id, in_ws)
            delivered = False
            async for event in stream:
                if event["type"] == "done" and in_ws:
                    delivered = True
                    for e in _workspace_results(s, problem, before):
                        yield e
                elif event["type"] == "done" and not in_ws:
                    edited = _edited_files(s)
                    if edited:
                        yield {"type": "files", "problem": s.problem, "files": edited}
                    if s.problem == NEW_ID and (entry := workspace_entry(s)):
                        yield {"type": "problem", "entry": entry}
                yield event
            if in_ws and not delivered:  # ended with an error: what the agent already changed still reaches the page
                for e in _workspace_results(s, problem, before):
                    yield e
    finally:
        s.busy = False
        s.sink = None
        s.last_used = time.monotonic()
        _hold_sandbox(box, -1)


def _hold_sandbox(sandbox: str | None, by: int) -> str | None:
    """Keep a workspace.py sandbox from being cleaned up while a turn works in it."""
    from . import workspace

    if sandbox and (box := workspace._sandboxes.get(sandbox)) is not None:
        box.busy += by
        box.last_used = time.monotonic()
    return sandbox


def _workspace_results(s: Session, problem: str, before: dict[str, str]) -> list[dict]:
    """A Workspace turn's results for the page: the files the agent changed (with meta.yaml re-parsed) and the
    notebook, if it edited _context/notebook.json."""
    from . import workspace

    out = []
    changed = {n: text for n, text in _snapshot(s).items() if before.get(n) != text}
    if changed:
        event = {"type": "files", "problem": problem, "files": changed}
        try:
            event["entry"] = workspace.entry(s.sandbox or s.id)
        except (KeyError, ValueError, OSError):
            pass  # broken meta.yaml: the page shows the YAML error when it next syncs
        out.append(event)
    if (nb := notebook_event(s, problem)) is not None:
        out.append(nb)
    return out


def _cli_stream(
    s: Session, agent: str, exe: str, message: str, model: str | None, mcp_base: str, client_id: str, in_ws=False
):
    """A local CLI agent (Codex / OpenCode / Gemini) on the workspace, with run_problem over Tensorizer's MCP."""
    from . import mcp_server
    from .agents import cli, codex, gemini, opencode

    if s.cli_agent != agent:  # a different CLI can't resume another one's conversation
        s.cli_session, s.cli_agent = None, agent
    token = mcp_server.session_token(s.id, client_id)
    url = f"{mcp_base.rstrip('/')}/mcp/{token}"
    first = s.cli_session is None
    extra = NEW_PROBLEM if s.problem == NEW_ID and not in_ws else ""
    rules = (
        f"{SYSTEM}{extra}\n\nTo render the problem, call the MCP tool "
        "`run_problem` of the `tensorizer` server (shell commands are disabled).\n\n---\n\n"
    )
    prompt = (rules if first else "") + message
    ws = str(s.workspace)
    if agent == "codex":
        mapper, argv, env, label = codex.Mapper(), codex.command(exe, ws, model, s.cli_session, url), {}, codex.LABEL
    elif agent == "opencode":
        argv, env = opencode.command(exe, ws, model, s.cli_session), opencode.env(url)
        mapper, label = opencode.Mapper(), opencode.LABEL
    else:
        settings_file = WORK_ROOT / s.id / "gemini-settings.json"  # outside the problem folder the agent edits
        settings_file.parent.mkdir(parents=True, exist_ok=True)  # a Workspace sandbox's session has no folder yet
        argv, env = gemini.command(exe, model, not first), gemini.env(settings_file, url)
        mapper, label = gemini.Mapper(), gemini.LABEL

    async def stream():
        try:
            async for event in cli.run(argv, s.workspace, env, prompt, mapper, s.sink, [token], label):
                yield event
        finally:
            s.cli_session = mapper.session or s.cli_session
            mcp_server.revoke(token)

    return stream()


def _short(data: dict) -> dict:
    """Tool inputs for display (kept here for older callers)."""
    from .agents.claude_code import short

    return short(data)


def _text(content) -> str:
    from .agents.claude_code import text_of

    return text_of(content)
