"""Sandboxes for Workspace problems: blank create, copy from the library, restore from the browser's files, delete,
and stateless notebook execution.

The browser (IndexedDB) is the source of truth; a sandbox is a disposable copy on the server:

    <WS_ROOT>/<id>/problems/_common/...              shared helpers (copied)
    <WS_ROOT>/<id>/problems/workspace/p<id12>/...    meta.yaml, main.py, equation.py, tensors.py, solve.py, plot.py

Sandboxes are deleted after IDLE_S idle seconds; the browser re-creates one by POSTing its files (sync). Problems the
AI created live in the agent's own sessions (agent._sessions) and are resolved here too.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from problems._common.runtime import load_meta, resolve_params

from . import agent
from .run_queue import QUEUE, QueueFull
from .runner import CODE_FILES, PROBLEMS, RunError, _env, _stream_process, problem_dir, stream_problem

WS_ROOT = Path(os.environ.get("TENSORIZER_WORKSPACES", Path(tempfile.gettempdir()) / "tensorizer-ws"))
IDLE_S = 60 * 60
FILES = ["meta.yaml", *CODE_FILES]
MAX_CODE_BYTES = 200_000
MAX_CELLS = 60
MAX_FIGURES = 8
MAX_STDOUT = 50_000
CATEGORY = "workspace"
# Uploaded data (CSV, images, arrays, …) for real-data problems: <problem>/data/<name>
DATA_DIR = "data"
MAX_DATA_FILE = 20 * 1024 * 1024
MAX_DATA_TOTAL = 100 * 1024 * 1024
DATA_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".npy",
    ".npz",
    ".txt",
    ".dat",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".h5",
    ".hdf5",
}
_DATA_NAME = re.compile(r"^\w[\w .()-]{0,99}$")
EDITED_KEYS = {"title", "level", "summary", "animated", "vector_form", "index_form", "translation", "notes", "params"}

BLANK_META = """title: Untitled problem
level: beginner
summary: ""
animated: false
# Equations in LaTeX, e.g. vector_form: '\\ddot{\\mathbf{r}} = -\\frac{GM}{r^3}\\mathbf{r}'
vector_form: ""
index_form: ""
# One row per symbol: {symbol, code, shape, meaning}
translation: []
notes: ""
# name: {default, min, max, step, label, unit}  (type: int | choice for integers or options)
params: {}
"""

BLANK_MAIN = '''"""Main loop: parameters → tensors → solve → plot. Start writing your problem here."""

from pathlib import Path

import matplotlib.pyplot as plt

from problems._common.plotting import save_png
from problems._common.runtime import cli

# from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    # 1. tensors.py  — build the initial state as tensors, e.g. y0 = tensors.initial_state(params)
    # 2. equation.py — write the physics as tensor operations (einsum, broadcasting)
    # 3. solve.py    — step the equation in time, e.g. times, traj = solve.solve(y0, params)
    # 4. plot.py     — draw the result and return the saved PNG/GIF path
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.text(0.5, 0.5, "Empty canvas: write your problem in main.py", ha="center", va="center", fontsize=12)
    ax.set_axis_off()
    return save_png(fig, out_dir / "result.png")


if __name__ == "__main__":
    cli(run, __file__)
'''

BLANK_DOCS = {
    "equation.py": '"""The physics written as tensor operations (einsum, broadcasting), with shapes in comments."""\n',
    "tensors.py": '"""Build the PyTorch tensors: initial state, grids and constants."""\n',
    "solve.py": '"""The solver: step the equation in time (or solve it directly)."""\n',
    "plot.py": '"""Draw the result and save it as a PNG or GIF (see problems._common.plotting)."""\n',
}

# Runs the notebook's code cells in one fresh process (see run_notebook). argv: cells.json, result.json, package.
HARNESS = r"""
import ast, base64, contextlib, importlib, io, json, sys, traceback
from pathlib import Path

cells_path, result_path, package = sys.argv[1:4]
spec = json.loads(Path(cells_path).read_text(encoding="utf-8"))
sys.path.insert(0, str(Path.cwd().parents[2]))  # the sandbox root, holding problems/
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

problem = importlib.import_module(package)
sys.modules["problem"] = problem  # `from problem import equation` works whatever the sandbox is called
ns = {"__name__": "__main__", "problem": problem}
try:
    from problems._common.runtime import load_meta, resolve_params
    ns["params"] = resolve_params(load_meta(Path.cwd()))
except Exception:
    ns["params"] = {}

figures = []
def grab():
    for num in plt.get_fignums()[: max(0, MAX_FIGURES - len(figures))]:
        buf = io.BytesIO()
        plt.figure(num).savefig(buf, format="png", dpi=90, bbox_inches="tight")
        figures.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    plt.close("all")

def quiet(value):
    items = value if isinstance(value, (list, tuple)) and value else [value]
    return value is None or type(items[0]).__module__.startswith("matplotlib")

MAX_FIGURES = spec["max_figures"]
out, n = [], len(spec["cells"])
for i, src in enumerate(spec["cells"]):
    capture = spec["all"] or i == n - 1
    figures = []
    plt.close("all")
    plt.show = grab if capture else (lambda *a, **k: plt.close("all"))
    buf = io.StringIO()
    error = None
    try:
        tree = ast.parse(src, f"<cell {i + 1}>")
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(tree, f"<cell {i + 1}>", "exec"), ns)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), f"<cell {i + 1}>", "eval"), ns)
                if not quiet(value):
                    print(repr(value))
    except BaseException:
        kind, exc, tb = sys.exc_info()
        while tb is not None and not tb.tb_frame.f_code.co_filename.startswith("<cell"):
            tb = tb.tb_next  # hide this harness's own frames
        error = "".join(traceback.format_exception(kind, exc, tb))
    if capture or error:
        if capture:
            grab()
        out.append({"index": i, "stdout": buf.getvalue()[-MAX_STDOUT:], "figures": figures, "error": error})
    if error:
        break
Path(result_path).write_text(json.dumps({"cells": out}), encoding="utf-8")
""".replace("MAX_STDOUT", str(MAX_STDOUT))


@dataclass
class Sandbox:
    id: str
    root: Path  # <WS_ROOT>/<id>
    last_used: float = field(default_factory=time.monotonic)
    busy: int = 0

    @property
    def slug(self) -> str:
        return f"p{self.id[:12]}"

    @property
    def folder(self) -> Path:
        return self.root / "problems" / CATEGORY / self.slug


_sandboxes: dict[str, Sandbox] = {}


def _cleanup() -> None:
    now = time.monotonic()
    for sid, s in list(_sandboxes.items()):
        if not s.busy and now - s.last_used > IDLE_S:
            shutil.rmtree(s.root, ignore_errors=True)
            del _sandboxes[sid]


def _check_files(files: dict[str, str]) -> None:
    if set(files) - set(FILES):
        raise ValueError(f"unknown file(s): {', '.join(sorted(set(files) - set(FILES)))}")
    if sum(len(c) for c in files.values()) > MAX_CODE_BYTES:
        raise ValueError("code too large")


def _new(files: dict[str, str], diagram: Path | None = None) -> Sandbox:
    """A fresh sandbox holding these files (missing ones blank)."""
    _cleanup()
    _check_files(files)
    sid = uuid.uuid4().hex
    s = Sandbox(sid, WS_ROOT / sid)
    shutil.copytree(PROBLEMS / "_common", s.root / "problems" / "_common", ignore=shutil.ignore_patterns("__pycache__"))
    s.folder.mkdir(parents=True)
    for package in (s.root / "problems", s.root / "problems" / CATEGORY, s.folder):
        (package / "__init__.py").touch()
    for name, code in blank_files().items():
        (s.folder / name).write_text(files.get(name, code), encoding="utf-8")
    if diagram and diagram.is_file():
        shutil.copy(diagram, s.folder / "diagram.svg")
    _sandboxes[sid] = s
    return s


def blank_files() -> dict[str, str]:
    return {"meta.yaml": BLANK_META, "main.py": BLANK_MAIN, **BLANK_DOCS}


def create_blank() -> dict:
    return entry(_new({}).id)


def copy_library(problem: str) -> dict:
    """A copy of a library problem ("category/slug") to edit freely."""
    src = problem_dir(*problem.split("/", 1))  # raises FileNotFoundError / TypeError for unknown problems
    files = {n: (src / n).read_text(encoding="utf-8") for n in FILES}
    return entry(_new(files, src / "diagram.svg").id)


def _resolve(session: str) -> tuple[Path, str, str]:
    """The folder of a live sandbox or agent session, with its category and slug; KeyError when expired."""
    if (s := _sandboxes.get(session)) is not None:
        s.last_used = time.monotonic()
        return s.folder, CATEGORY, s.slug
    if (a := agent._sessions.get(session)) is not None:
        a.last_used = time.monotonic()
        category, slug = a.problem.split("/", 1)
        return a.workspace, category, slug
    raise KeyError(session)


def alive(session: str | None) -> bool:
    try:
        return bool(session) and _resolve(session) is not None
    except KeyError:
        return False


def sync(session: str | None, files: dict[str, str], data: list[str] | None = None) -> dict:
    """Write the browser's files into the problem's sandbox, re-creating it when it has expired. `data`: the names
    of the data files the browser has; any other file in data/ is removed (the page uploads missing ones after)."""
    _check_files(files)
    if not (session and alive(session)):
        session = _new(files).id
    folder = _resolve(session)[0]
    for name, code in files.items():
        (folder / name).write_text(code, encoding="utf-8")
    if data is not None:
        keep = set(data)
        for f in _data_files(folder):
            if f.name not in keep:
                f.unlink()
    return entry(session)


def data_name(name: str) -> str:
    """A safe data file name (no folders, a known extension); ValueError otherwise."""
    if not _DATA_NAME.match(name or "") or ".." in name or name.strip() != name:
        raise ValueError(f"invalid file name: {name!r}")
    if Path(name).suffix.lower() not in DATA_EXTENSIONS:
        raise ValueError(
            f"unsupported file type: {Path(name).suffix or name} (use {', '.join(sorted(DATA_EXTENSIONS))})"
        )
    return name


def _data_files(folder: Path) -> list[Path]:
    d = folder / DATA_DIR
    return sorted(f for f in d.iterdir() if f.is_file()) if d.is_dir() else []


def data_list(folder: Path) -> list[dict]:
    return [{"name": f.name, "size": f.stat().st_size} for f in _data_files(folder)]


def put_data(session: str, name: str, raw: bytes) -> dict:
    """Store one uploaded data file in the problem's data/ folder; KeyError when expired, ValueError for a bad name
    or a file over the limits (MAX_DATA_FILE each, MAX_DATA_TOTAL per problem)."""
    folder = _resolve(session)[0]
    name = data_name(name)
    if len(raw) > MAX_DATA_FILE:
        raise ValueError(f"{name} is larger than {MAX_DATA_FILE // 2**20} MB")
    others = sum(f.stat().st_size for f in _data_files(folder) if f.name != name)
    if others + len(raw) > MAX_DATA_TOTAL:
        raise ValueError(f"the problem's data would exceed {MAX_DATA_TOTAL // 2**20} MB")
    target = folder / DATA_DIR / name
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(raw)
    return {"name": name, "size": len(raw)}


def delete_data(session: str, name: str) -> None:
    folder = _resolve(session)[0]
    (folder / DATA_DIR / data_name(name)).unlink(missing_ok=True)


def delete(session: str) -> None:
    """Remove one of our sandboxes (an agent's session is left to the agent's own cleanup)."""
    if (s := _sandboxes.pop(session, None)) is not None:
        shutil.rmtree(s.root, ignore_errors=True)


def files_of(session: str) -> dict[str, str]:
    folder = _resolve(session)[0]
    return {n: (folder / n).read_text(encoding="utf-8") for n in FILES if (folder / n).is_file()}


def entry(session: str) -> dict:
    """The problem as a library entry (the manifest's shape) plus its raw meta.yaml; ValueError for broken YAML."""
    folder = _resolve(session)[0]
    text = (folder / "meta.yaml").read_text(encoding="utf-8")
    try:
        meta = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"meta.yaml: {e}") from None
    if not isinstance(meta, dict):
        raise ValueError("meta.yaml must be a mapping")
    svg = folder / "diagram.svg"
    diagram = "data:image/svg+xml;base64," + base64.b64encode(svg.read_bytes()).decode() if svg.is_file() else None
    text_of = lambda k: str(meta.get(k) or "")  # noqa: E731
    return {
        "id": f"{CATEGORY}/{session[:8]}",
        "category": CATEGORY,
        "slug": session[:8],
        "session": session,
        "title": text_of("title") or "Untitled problem",
        "level": meta.get("level") if meta.get("level") in ("beginner", "intermediate", "advanced") else "beginner",
        "summary": text_of("summary"),
        "animated": bool(meta.get("animated")),
        "vector_form": text_of("vector_form"),
        "index_form": text_of("index_form"),
        "translation": meta.get("translation") or [],
        "notes": text_of("notes"),
        "params": meta.get("params") or {},
        "order": 0,
        "files": [{"name": n, "code": (folder / n).read_text(encoding="utf-8")} for n in CODE_FILES],
        "meta": text,
        # keys the page doesn't edit (category, order, …), kept when it regenerates meta.yaml from the view
        "meta_extra": {k: v for k, v in meta.items() if k not in EDITED_KEYS},
        "output": None,
        "console": "",
        "diagram": diagram,
        "data": data_list(folder),
    }


async def run(session: str, params: dict, verbose: bool, client: str) -> AsyncIterator[dict]:
    """Render the problem in its sandbox; the same events as runner.stream_problem."""
    try:
        folder, category, slug = _resolve(session)
    except KeyError:
        raise RunError("this problem's sandbox has expired — sync its files first") from None
    params = resolve_params(load_meta(folder), params)
    async for event in stream_problem(category, slug, params, None, verbose, client, folder.parents[2]):
        yield event


async def run_notebook(session: str, cells: list[str], capture_all: bool, client: str) -> dict:
    """Run code cells 1..k in a fresh process with cwd in the problem's folder and the problem importable as
    `problem`. Returns {"cells": [{index, stdout, figures (PNG data URLs), error}]}: the last cell only (or every
    cell with capture_all), plus the cell that failed, if any."""
    try:
        folder, category, slug = _resolve(session)
    except KeyError:
        raise RunError("this problem's sandbox has expired — sync its files first") from None
    if len(cells) > MAX_CELLS or sum(len(c) for c in cells) > MAX_CODE_BYTES:
        raise ValueError("notebook too large")
    if not cells:
        return {"cells": []}
    s = _sandboxes.get(session)
    run_dir = folder.parents[2] / ".notebook" / uuid.uuid4().hex
    run_dir.mkdir(parents=True)
    (run_dir / "harness.py").write_text(HARNESS, encoding="utf-8")
    spec = {"cells": cells, "all": capture_all, "max_figures": MAX_FIGURES}
    (run_dir / "cells.json").write_text(json.dumps(spec), encoding="utf-8")
    result = run_dir / "result.json"
    cmd = [sys.executable, str(run_dir / "harness.py"), str(run_dir / "cells.json"), str(result)]
    cmd.append(f"problems.{category}.{slug}")

    try:
        ticket = QUEUE.join(client)
    except QueueFull as e:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise RunError(str(e)) from None
    if s:
        s.busy += 1
    raw: list[str] = []
    try:
        async for _ in QUEUE.wait(ticket):
            pass
        async for event in _stream_process(cmd, folder, _env(run_dir)):
            if event["type"] == "line":
                raw.append(event["text"])
        if not result.is_file():  # crashed before writing (e.g. killed, or a broken problem package)
            raise RunError("the notebook failed to run", "\n".join(raw)[-MAX_STDOUT:])
        return json.loads(result.read_text(encoding="utf-8"))
    except RunError as e:
        raise RunError(str(e), e.console or "\n".join(raw)[-MAX_STDOUT:]) from None
    finally:
        QUEUE.leave(ticket)
        if s:
            s.busy -= 1
        shutil.rmtree(run_dir, ignore_errors=True)
