"""HTTP routes for the Workspace → Problem → Notebook hierarchy (blank problems, restore, notebook cells).

The browser keeps every problem (files, meta.yaml, notebooks) in IndexedDB; these routes keep a disposable sandbox
per problem on the server. Before running, the page POSTs its files to /sync, which re-creates an expired sandbox.
Like the agent's workspace routes, everything here needs AGENT_ENABLED=1.
"""

import json
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import agent, limits, workspace
from .runner import RunError

router = APIRouter(prefix="/api/ws")


def _enabled() -> None:
    if not agent.ENABLED:
        raise HTTPException(403, "workspaces are not enabled on this server")


def _may_run() -> None:
    """Running a Workspace problem or notebook executes the visitor's own code: off on a shared server by default."""
    _enabled()
    if not limits.user_code_allowed():
        raise HTTPException(403, "running your own code is off on this shared server: use Backend > Local")


def _client(request: Request) -> str:
    peer = request.client.host if request.client else None
    return limits.client_address(request.headers.get("x-forwarded-for", ""), peer)


class CopyRequest(BaseModel):
    problem: str  # library id, "category/slug"


class SyncRequest(BaseModel):
    session: str | None = None
    files: dict[str, str] = Field(default_factory=dict)
    data: list[str] | None = None  # names of the data files the browser holds (others are removed from data/)


class RunRequest(BaseModel):
    params: dict = Field(default_factory=dict)
    verbose: bool = False
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


class NotebookRequest(BaseModel):
    cells: list[str]  # code cells 1..k, in order
    all: bool = False  # capture every cell's output (Run all), not only the last one


@router.post("/blank")
def blank() -> dict:
    """A new, empty problem: a skeleton that runs and draws a placeholder figure."""
    _enabled()
    return workspace.create_blank()


@router.post("/copy")
def copy(req: CopyRequest) -> dict:
    """A new problem copied from the library."""
    _enabled()
    try:
        return workspace.copy_library(req.problem)
    except (FileNotFoundError, TypeError, ValueError):
        raise HTTPException(404, "unknown problem") from None


@router.post("/sync")
def sync(req: SyncRequest) -> dict:
    """Write the browser's files into the sandbox (re-created when expired); returns the parsed entry."""
    _enabled()
    try:
        return workspace.sync(req.session, req.files, req.data)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.post("/{session}/files")
def files(session: str) -> dict:
    """The problem's files including meta.yaml (for problems the AI created)."""
    _enabled()
    try:
        return {"files": workspace.files_of(session)}
    except KeyError:
        raise HTTPException(410, "expired") from None


@router.put("/{session}/data/{name}")
async def put_data(session: str, name: str, request: Request) -> dict:
    """Upload one data file (the raw bytes as the body) into the problem's data/ folder."""
    _enabled()
    if not workspace.alive(session):
        raise HTTPException(410, "this problem's sandbox has expired")
    declared = int(request.headers.get("content-length") or 0)
    if declared > workspace.MAX_DATA_FILE:
        raise HTTPException(413, f"larger than {workspace.MAX_DATA_FILE // 2**20} MB")
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > workspace.MAX_DATA_FILE:
            raise HTTPException(413, f"larger than {workspace.MAX_DATA_FILE // 2**20} MB")
    try:
        return workspace.put_data(session, name, bytes(raw))
    except KeyError:
        raise HTTPException(410, "expired") from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None


@router.delete("/{session}/data/{name}")
def delete_data(session: str, name: str) -> dict:
    _enabled()
    try:
        workspace.delete_data(session, name)
    except KeyError:
        raise HTTPException(410, "expired") from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return {"ok": True}


@router.post("/{session}/delete")
def delete(session: str) -> dict:
    _enabled()
    workspace.delete(session)
    return {"ok": True}


@router.post("/{session}/run-stream")
async def run_stream(session: str, req: RunRequest, request: Request) -> StreamingResponse:
    """Render the problem in its sandbox; the same newline-delimited JSON events as /api/run-stream."""
    _may_run()
    if not workspace.alive(session):
        raise HTTPException(410, "this problem's sandbox has expired")

    async def events():
        try:
            async for event in workspace.run(session, req.params, req.verbose, _client(request), req.device):
                if event["type"] == "result":
                    event = {**event, "output": f"/api/outputs/{event['run_id']}/{event['output']}"}
                yield json.dumps(event) + "\n"
        except (RunError, ValueError, TypeError) as e:
            yield json.dumps({"type": "error", "message": str(e), "console": getattr(e, "console", "")}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


@router.post("/{session}/notebook")
async def notebook(session: str, req: NotebookRequest, request: Request) -> dict:
    """Run code cells 1..k in a fresh process (stateless); returns the last cell's (or every cell's) output."""
    _may_run()
    if not workspace.alive(session):
        raise HTTPException(410, "this problem's sandbox has expired")
    try:
        return await workspace.run_notebook(session, req.cells, req.all, _client(request))
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    except RunError as e:
        raise HTTPException(400, {"message": str(e), "console": e.console}) from None
