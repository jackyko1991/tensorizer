"""HTTP routes for the AI Assistant's three modes.

LOCAL  POST /api/agents/scan                 which agent CLIs are on this machine (only when the user asks)
       POST /api/agents/chat                 one chat turn with any agent (Claude Code, Codex, OpenCode, Gemini, API)
CLOUD  POST /api/agents/connector            a connector token + MCP URL for the user's own cloud agent
       GET  /api/agents/connector/events     what that agent did, mirrored to the browser (polled)
       GET|POST|DELETE /mcp[/<token>]        Tensorizer's MCP server (Streamable HTTP; token in path, bearer or ?token=)
API    the same /api/agents/chat with agent "claude-api", provider anthropic | openrouter and the visitor's key in the
       X-Agent-Key header (never stored or logged).
"""

from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import agent, mcp_server
from .runner import problem_dir

router = APIRouter()
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _client(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _enabled() -> None:
    if not agent.ENABLED:
        raise HTTPException(403, "the agent is not enabled on this server")


def _local_base(request: Request) -> str:
    """This backend as seen from the same machine (for a local CLI calling our MCP server)."""
    server = request.scope.get("server") or ("127.0.0.1", 8000)
    return f"http://127.0.0.1:{server[1]}"


def _public_base(request: Request) -> str:
    """Where an agent elsewhere reaches this backend: TENSORIZER_PUBLIC_URL, a proxy's forwarded host (Hugging Face),
    or, for a backend on this computer, localhost (the page itself may be on Vite's port, which doesn't proxy /mcp)."""
    if url := os.environ.get("TENSORIZER_PUBLIC_URL"):
        return url.rstrip("/")
    server = request.scope.get("server")
    if server and server[0] in LOOPBACK:
        return f"http://localhost:{server[1]}"
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    return f"{proto}://{host}" if host else str(request.base_url).rstrip("/")


# --- LOCAL ----------------------------------------------------------------------------------------------------------


@router.post("/api/agents/scan")
async def scan_agents() -> dict:
    """Look for agent CLIs on this machine. Only booleans, version numbers and labels come back."""
    _enabled()
    if not agent.LOCAL_ALLOWED:
        raise HTTPException(403, "searching for local agents only works on your own computer")
    from .agents import scan

    return {"agents": await run_in_threadpool(scan.scan)}


class ChatRequest(BaseModel):
    session: str | None = None
    problem: str
    message: str = Field(max_length=8000)
    model: str | None = None
    # What is on screen: {title, files, params, selection {file, from_line, to_line, text, region?, lang?} |
    # {region, text}, screenshot (PNG base64), workspace {kind: source | notebook, mode, markdown, notebook_id,
    # notebook_title, cells [{id, type, source, output}], focused_cell}}
    context: dict | None = None
    agent: str = "claude-code"  # claude-code | codex | opencode | gemini (LOCAL) or claude-api (API)
    provider: str = "anthropic"  # API: anthropic | openrouter
    # A Workspace problem (problem "workspace/<id>"): its sandbox, synced by the page just before (POST /api/ws/sync)
    sandbox: str | None = None


@router.post("/api/agents/chat")
async def agents_chat(req: ChatRequest, request: Request) -> StreamingResponse:
    """One chat turn with the chosen agent, streamed as newline-delimited JSON events (see agent.chat)."""
    _enabled()
    try:
        if req.problem != "new" and not req.problem.startswith("workspace/"):
            problem_dir(*req.problem.split("/", 1))
    except (TypeError, ValueError, FileNotFoundError):
        raise HTTPException(404, "unknown problem") from None
    key = request.headers.get("x-agent-key") or request.headers.get("x-anthropic-key") or None
    client, base = _client(request), _local_base(request)

    async def events():
        stream = agent.chat(
            req.session,
            req.problem,
            req.message,
            req.model,
            key,
            client,
            req.context,
            req.agent,
            req.provider,
            base,
            sandbox=req.sandbox,
        )
        async for event in stream:
            yield json.dumps(event) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


# --- CLOUD ----------------------------------------------------------------------------------------------------------


class ConnectorRequest(BaseModel):
    previous: str | None = None  # a token this one replaces (revoked)


@router.post("/api/agents/connector")
def new_connector(req: ConnectorRequest, request: Request) -> dict:
    """A connector token for this browser and the MCP URL to give the cloud agent (shown once)."""
    _enabled()
    token, _ = mcp_server.create_connector(_client(request), req.previous)
    base = _public_base(request)
    return {
        "token": token,
        "url": f"{base}/mcp",
        "url_with_token": f"{base}/mcp/{token}",
        "local": base.startswith(("http://localhost", "http://127.0.0.1")),
    }


def _bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else None


@router.get("/api/agents/connector/events")
def connector_events(request: Request, after: int = 0) -> dict:
    c = mcp_server.lookup(_bearer(request))
    if c is None or c.session is not None:
        raise HTTPException(401, "unknown or expired connector token")
    return {"events": mcp_server.events_after(c, after), "seq": c.seq}


class _ASGI(Response):
    """Hands the raw request to the MCP SDK's ASGI handler (FastAPI routers can't mount sub-apps)."""

    def __init__(self, connector: mcp_server.Connector) -> None:
        super().__init__()
        self.connector = connector

    async def __call__(self, scope, receive, send) -> None:
        await mcp_server.handle(scope, receive, send, self.connector)


async def _mcp(request: Request, token: str | None) -> Response:
    if not agent.ENABLED:
        return JSONResponse({"error": "the agent is not enabled on this server"}, status_code=403)
    c = mcp_server.lookup(token or _bearer(request) or request.query_params.get("token"))
    if c is None:
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32001, "message": "missing or invalid connector token"}},
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="tensorizer"'},
        )
    c.base = _public_base(request)
    return _ASGI(c)


@router.api_route("/mcp", methods=["GET", "POST", "DELETE"], include_in_schema=False)
async def mcp_endpoint(request: Request) -> Response:
    return await _mcp(request, None)


@router.api_route("/mcp/{token}", methods=["GET", "POST", "DELETE"], include_in_schema=False)
async def mcp_endpoint_token(token: str, request: Request) -> Response:
    return await _mcp(request, token)
