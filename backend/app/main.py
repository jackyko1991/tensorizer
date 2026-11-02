"""Tensorizer API: re-render problems with custom parameters (and edited code when allowed)."""

from __future__ import annotations

import functools
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from problems._common.runtime import load_meta, resolve_params

from . import agent
from .routes_agents import router as agents_router
from .routes_workspace import router as workspace_router
from .run_queue import QUEUE
from .runner import CACHE, CODE_FILES, ROOT, RunError, problem_dir, run_problem, stream_problem

# Running edited code executes arbitrary Python: only enable on a private machine.
ALLOW_CODE_EXEC = os.environ.get("ALLOW_CODE_EXEC") == "1"
ORIGINS = [o for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o]
MAX_CODE_BYTES = 200_000

# Windows can map .js to text/plain in the registry, and browsers refuse module scripts served that way.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

app = FastAPI(title="Tensorizer API")
# allow_private_network: lets the public github.io page call a backend on the visitor's own machine
# (Chrome's Private Network Access preflight), for "run on this computer".
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_private_network=True,
)
app.include_router(agents_router)
app.include_router(workspace_router)


class RunRequest(BaseModel):
    params: dict = Field(default_factory=dict)
    files: dict[str, str] | None = None
    verbose: bool = False
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


RUNNER_VERSION = "0.1.0"


@functools.cache
def _machine() -> dict:
    """What this runner can compute on, for the Backend panel (torch is imported once, on the first health check)."""
    import platform

    import torch

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    gpu = torch.cuda.get_device_name(0) if "cuda" in devices else ("Apple GPU" if "mps" in devices else None)
    # sys.platform, not platform.system(): on Windows that runs `cmd /c ver`, flashing a console window
    os_name = {"win32": "Windows", "darwin": "macOS"}.get(sys.platform, "Linux")
    return {
        "runner": RUNNER_VERSION,
        "torch": torch.__version__.split("+")[0],
        "python": platform.python_version(),
        "os": os_name,
        "devices": devices,
        "gpu": gpu,
        # an NVIDIA GPU the installed PyTorch can't use (a CPU-only build): the panel says how to get CUDA
        "nvidia": None if "cuda" in devices else _nvidia_gpu(),
    }


def _nvidia_gpu() -> str | None:
    """The NVIDIA GPU's name from nvidia-smi (installed with the driver), without opening a console window."""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    name = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
    return name[:80] or None


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "allow_code_exec": ALLOW_CODE_EXEC, "queue": QUEUE.status(), **_machine()}


def _client(request: Request) -> str:
    """Who is asking, for the one-run-per-visitor limit; behind a proxy (Hugging Face) the first forwarded hop."""
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _validate(category: str, slug: str, req: RunRequest) -> dict:
    """Resolve parameters and check the request; raises HTTPException before any work starts."""
    try:
        meta = load_meta(problem_dir(category, slug))
        params = resolve_params(meta, req.params)
    except FileNotFoundError:
        raise HTTPException(404, "unknown problem") from None
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e)) from None

    if req.files is not None:
        if not ALLOW_CODE_EXEC:
            raise HTTPException(403, "running edited code is disabled on this server")
        if sum(len(c) for c in req.files.values()) > MAX_CODE_BYTES:
            raise HTTPException(413, "code too large")
        if set(req.files) - set(CODE_FILES):
            raise HTTPException(422, "unknown file(s)")
    return params


@app.post("/api/run-stream/{category}/{slug}")
async def run_stream(category: str, slug: str, req: RunRequest, request: Request) -> StreamingResponse:
    """Like /api/run, but streams newline-delimited JSON events while the problem runs:
    {"type": "line", "text", "replace"} for each printed line (replace=True: a progress-bar redraw),
    and {"type": "queued", "position"} while waiting for a free slot,
    then {"type": "result", "output", "console", "cached"} or {"type": "error", "message", "console"}.
    """
    params = _validate(category, slug, req)

    async def events():
        try:
            async for event in stream_problem(
                category, slug, params, req.files, req.verbose, _client(request), device=req.device
            ):
                if event["type"] == "result":
                    event = {**event, "output": f"/api/outputs/{event['run_id']}/{event['output']}", "params": params}
                yield json.dumps(event) + "\n"
        except RunError as e:
            yield json.dumps({"type": "error", "message": str(e), "console": e.console}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


@app.post("/api/run/{category}/{slug}")
async def run(category: str, slug: str, req: RunRequest, request: Request) -> dict:
    params = _validate(category, slug, req)
    try:
        result = await run_problem(category, slug, params, req.files, req.verbose, _client(request))
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    except RunError as e:
        raise HTTPException(400, {"message": str(e), "console": e.console}) from None
    return {
        "output": f"/api/outputs/{result.run_id}/{result.output}",
        "console": result.console,
        "cached": result.cached,
        "params": params,
    }


class AgentRequest(BaseModel):
    session: str | None = None
    problem: str
    message: str = Field(max_length=8000)
    model: str | None = None
    # What is on screen: {title, files (the editor's edited files), params, selection {file, from_line, to_line, text}}
    context: dict | None = None
    agent: str = "claude-code"  # "claude-code" (this machine's login) or "claude-api" (the X-Anthropic-Key)


@app.get("/api/agent/status")
def agent_status() -> dict:
    return agent.status()


@app.post("/api/agent/chat")
async def agent_chat(req: AgentRequest, request: Request) -> StreamingResponse:
    """One chat turn with the Claude Code agent, streamed as newline-delimited JSON events (see agent.chat).
    The visitor's Anthropic API key comes in the X-Anthropic-Key header and is only handed to Claude Code."""
    if not agent.ENABLED:
        raise HTTPException(403, "the agent is not enabled on this server")
    try:
        if req.problem != "new":  # "new": the agent writes a brand-new problem from a template
            problem_dir(*req.problem.split("/", 1))
    except (TypeError, ValueError, FileNotFoundError):
        raise HTTPException(404, "unknown problem") from None
    key = request.headers.get("x-anthropic-key") or None

    async def events():
        events = agent.chat(
            req.session, req.problem, req.message, req.model, key, _client(request), req.context, req.agent
        )
        async for event in events:
            yield json.dumps(event) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


@app.post("/api/workspace/new")
def workspace_new() -> dict:
    """Start a blank problem in the Workspace (from the template), to edit and run by hand."""
    if not agent.ENABLED:
        raise HTTPException(403, "workspaces are not enabled on this server")
    return agent.create_workspace()


@app.post("/api/workspace/{session}/run-stream")
async def workspace_run_stream(session: str, req: RunRequest, request: Request) -> StreamingResponse:
    """Run a problem the agent created (Workspace), inside its sandbox; same events as /api/run-stream."""
    if not agent.ENABLED:
        raise HTTPException(403, "the agent is not enabled on this server")

    async def events():
        try:
            async for event in agent.run_workspace(session, req.params, req.files, req.verbose, _client(request)):
                if event["type"] == "result":
                    event = {**event, "output": f"/api/outputs/{event['run_id']}/{event['output']}"}
                yield json.dumps(event) + "\n"
        except (RunError, ValueError, TypeError) as e:
            yield json.dumps({"type": "error", "message": str(e), "console": getattr(e, "console", "")}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


@app.get("/api/outputs/{run_id}/{name}")
def output(run_id: str, name: str) -> FileResponse:
    path = (CACHE / run_id / name).resolve()
    if path.parent != (CACHE / run_id).resolve() or path.suffix not in {".gif", ".png"} or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


# In `pixi run dev` the live, hot-reloading site is Vite's; send browsers there instead of a stale build.
DEV_SITE = os.environ.get("TENSORIZER_DEV_SITE")
if DEV_SITE:

    @app.get("/", include_in_schema=False)
    def dev_site() -> RedirectResponse:
        return RedirectResponse(DEV_SITE)


# Otherwise serve the built frontend from the same process when it exists (`pixi run serve`).
DIST = Path(os.environ.get("FRONTEND_DIST", ROOT / "frontend" / "dist"))
if DIST.is_dir() and not DEV_SITE:
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
