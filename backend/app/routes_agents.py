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

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from tensorizer_physics.jobs import visitor_hash

from . import agent, mcp_server, turns
from .runner import CACHE, problem_dir

router = APIRouter()
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
TURNS = turns.TurnManager(CACHE / "_turns")
VISITOR_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


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
    # The History entry this message continues (GET /api/agents/conversations): kept together even when the agent
    # session behind it has to start again
    conversation: str | None = Field(None, max_length=80)


def _visitor(request: Request, required: bool = True) -> str:
    """The hash of the browser's X-Visitor id (as for jobs); "" when absent and not required."""
    v = (request.headers.get("x-visitor") or "").strip()
    if not v and not required:
        return ""
    if not 16 <= len(v) <= 128 or set(v) - VISITOR_CHARS:
        raise HTTPException(400, "missing or malformed X-Visitor header")
    return visitor_hash(v)


def _own_turn(turn_id: str, request: Request) -> turns.Turn:
    turn = TURNS.get(turn_id, _visitor(request))
    if turn is None:  # someone else's turn looks the same as no turn
        raise HTTPException(404, "unknown turn")
    return turn


def _follow(turn: turns.Turn, start: int) -> StreamingResponse:
    async def events():
        async for event in TURNS.events(turn, max(0, start)):
            yield json.dumps(event) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


@router.post("/api/agents/chat")
async def agents_chat(req: ChatRequest, request: Request) -> StreamingResponse:
    """One chat turn with the chosen agent, streamed as newline-delimited JSON events (see agent.chat).

    The turn runs detached (turns.py): this response only follows its log, so closing or reloading the page does
    not stop the agent. The first event is {"type": "turn", "id"}; GET /api/agents/turns/<id>/events?from=N
    follows it again, POST /api/agents/turns/<id>/cancel stops it."""
    _enabled()
    try:
        if req.problem != "new" and not req.problem.startswith("workspace/"):
            problem_dir(*req.problem.split("/", 1))
    except (TypeError, ValueError, FileNotFoundError):
        raise HTTPException(404, "unknown problem") from None
    visitor = _visitor(request, required=False)
    key = request.headers.get("x-agent-key") or request.headers.get("x-anthropic-key") or None
    client, base = _client(request), _local_base(request)
    message = req.message
    if req.conversation and visitor and not agent.known_session(req.session):
        # Continuing an earlier chat whose agent session is gone: the new session starts with what was said.
        if recap := TURNS.recap(visitor, req.conversation):
            message = (
                "[This continues an earlier conversation with the user. What was said so far (oldest first, "
                f"possibly shortened):\n{recap}\n]\n\n{req.message}"
            )
    stream = agent.chat(
        req.session,
        req.problem,
        message,
        req.model,
        key,
        client,
        req.context,
        req.agent,
        req.provider,
        base,
        sandbox=req.sandbox,
    )
    turn = TURNS.start(visitor, req.problem, req.agent, req.message, stream, req.conversation)
    return _follow(turn, 0)


@router.get("/api/agents/turns/{turn_id}/events")
async def turn_events(turn_id: str, request: Request, start: int = Query(0, alias="from")) -> StreamingResponse:
    """A turn's events from number ?from=N (each with its "seq"), then live until it ends."""
    return _follow(_own_turn(turn_id, request), start)


@router.get("/api/agents/turns/{turn_id}")
def turn_status(turn_id: str, request: Request) -> dict:
    return _own_turn(turn_id, request).summary()


@router.post("/api/agents/turns/{turn_id}/cancel")
def turn_cancel(turn_id: str, request: Request) -> dict:
    """The Stop button: ends the agent (and the render it started)."""
    turn = _own_turn(turn_id, request)
    return {"cancelled": TURNS.cancel(turn), **turn.summary()}


@router.get("/api/agents/conversations")
def conversations(request: Request, problem: str | None = None) -> dict:
    """This browser's agent conversations (newest first), optionally for one problem: the panel's History."""
    return {"conversations": TURNS.conversations(_visitor(request), problem)}


@router.delete("/api/agents/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, request: Request) -> dict:
    visitor = _visitor(request)
    found = [c for c in TURNS.conversations(visitor) if c["id"] == conversation_id]
    if not found:
        raise HTTPException(404, "unknown conversation")
    for turn_id in found[0]["turns"]:
        if (turn := TURNS.get(turn_id, visitor)) is not None:
            TURNS.delete(turn)
    return {"deleted": len(found[0]["turns"])}


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


def _cloud_connector(request: Request) -> mcp_server.Connector:
    c = mcp_server.lookup(_bearer(request))
    if c is None or c.session is not None:
        raise HTTPException(401, "unknown or expired connector token")
    return c


@router.get("/api/agents/connector")
def connection_status(request: Request) -> dict:
    c = _cloud_connector(request)
    return {"connected": c.last_seen is not None, "last_seen": c.last_seen}


@router.delete("/api/agents/connector", status_code=204)
def disconnect_connector(request: Request) -> Response:
    _cloud_connector(request)
    mcp_server.revoke(_bearer(request))
    return Response(status_code=204)


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
