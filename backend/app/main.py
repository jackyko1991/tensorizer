"""Tensorizer API: re-render problems with custom parameters (and edited code when allowed)."""

from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from problems._common.runtime import load_meta, resolve_params

from .runner import CACHE, CODE_FILES, ROOT, RunError, problem_dir, run_problem, stream_problem

# Running edited code executes arbitrary Python: only enable on a private machine.
ALLOW_CODE_EXEC = os.environ.get("ALLOW_CODE_EXEC") == "1"
ORIGINS = [o for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o]
MAX_CODE_BYTES = 200_000

# Windows can map .js to text/plain in the registry, and browsers refuse module scripts served that way.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

app = FastAPI(title="Tensorizer API")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


class RunRequest(BaseModel):
    params: dict = Field(default_factory=dict)
    files: dict[str, str] | None = None
    verbose: bool = False


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "allow_code_exec": ALLOW_CODE_EXEC}


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
async def run_stream(category: str, slug: str, req: RunRequest) -> StreamingResponse:
    """Like /api/run, but streams newline-delimited JSON events while the problem runs:
    {"type": "line", "text", "replace"} for each printed line (replace=True: a progress-bar redraw),
    then {"type": "result", "output", "console", "cached"} or {"type": "error", "message", "console"}.
    """
    params = _validate(category, slug, req)

    async def events():
        try:
            async for event in stream_problem(category, slug, params, req.files, req.verbose):
                if event["type"] == "result":
                    event = {**event, "output": f"/api/outputs/{event['run_id']}/{event['output']}", "params": params}
                yield json.dumps(event) + "\n"
        except RunError as e:
            yield json.dumps({"type": "error", "message": str(e), "console": e.console}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


@app.post("/api/run/{category}/{slug}")
async def run(category: str, slug: str, req: RunRequest) -> dict:
    params = _validate(category, slug, req)
    try:
        result = await run_problem(category, slug, params, req.files, req.verbose)
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
