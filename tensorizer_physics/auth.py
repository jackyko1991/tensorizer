"""Pairing token for a person's own runner (docs/design/backend.md §5, step 3).

Set TENSORIZER_TOKEN (or TENSORIZER_TOKEN_SHA256, the hex SHA-256 of it) and every endpoint except GET /api/health
(and CORS preflights) needs `Authorization: Bearer <token>`. Only the hash is kept: the plain token is removed from
the environment at import, compared in constant time (hmac.compare_digest on the SHA-256), and never logged.

Exempt routes carry their own credential:
- /api/outputs/<run>/<file>: run results are shown by <img> tags, which cannot send a header, so the server signs
  each output URL it hands out (`?sig=` = HMAC of the path with a key derived from the token hash). A signature opens
  that one file only, cannot be made without the token, and stops working when the token changes.
- /mcp and /api/agents/connector/events: the connector token, which is issued by an authenticated call.

Failed attempts (a wrong token, not a missing one) are limited per client: FAIL_MAX in FAIL_WINDOW_S seconds, then
429 for every protected route until the window has passed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time

REALM = 'Bearer realm="tensorizer"'
FAIL_MAX = 10
FAIL_WINDOW_S = 60.0

# Routes open without the pairing token (they are public, or check a credential of their own).
PUBLIC = {"/api/health"}
OWN_CREDENTIAL = ("/mcp", "/api/agents/connector/events")
PROTECTED_PREFIXES = ("/api/", "/mcp", "/docs", "/redoc", "/openapi.json")

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


class Auth:
    """The configured token hash (or none), the bearer check and the failed-attempt limit."""

    def __init__(self, token: str | None = None, token_sha256: str | None = None):
        self.configure(token, token_sha256)

    def configure(self, token: str | None = None, token_sha256: str | None = None) -> None:
        digest: bytes | None = None
        if token:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
        elif token_sha256:
            if not _HEX64.match(token_sha256.strip()):
                raise ValueError("TENSORIZER_TOKEN_SHA256 must be 64 hex characters (the SHA-256 of the token)")
            digest = bytes.fromhex(token_sha256.strip())
        self._digest = digest
        self._sign_key = hashlib.sha256(b"tensorizer-outputs\0" + digest).digest() if digest else None
        self._fails: dict[str, list[float]] = {}

    @classmethod
    def from_env(cls, env=os.environ) -> Auth:
        """Read the token from the environment and remove the plain one (run processes never see it)."""
        token = env.pop("TENSORIZER_TOKEN", None) or None
        return cls(token, env.get("TENSORIZER_TOKEN_SHA256") or None)

    @property
    def enabled(self) -> bool:
        return self._digest is not None

    def check(self, presented: str | None) -> bool:
        """True when `presented` is the token (constant time; always True when no token is configured)."""
        if self._digest is None:
            return True
        candidate = hashlib.sha256((presented or "").encode("utf-8")).digest()
        return hmac.compare_digest(candidate, self._digest) and bool(presented)

    # --- signed output URLs ---------------------------------------------------------------------------------------

    def _sig(self, run_id: str, name: str) -> str:
        assert self._sign_key is not None
        return hmac.new(self._sign_key, f"{run_id}/{name}".encode(), hashlib.sha256).hexdigest()[:32]

    def output_url(self, run_id: str, name: str) -> str:
        url = f"/api/outputs/{run_id}/{name}"
        return f"{url}?sig={self._sig(run_id, name)}" if self.enabled else url

    def output_sig_ok(self, path: str, sig: str | None) -> bool:
        parts = path.split("/")  # "", "api", "outputs", run_id, name
        if not self.enabled or not sig or len(parts) != 5:
            return False
        return hmac.compare_digest(self._sig(parts[3], parts[4]), sig)

    # --- failed attempts ------------------------------------------------------------------------------------------

    def blocked(self, client: str) -> float:
        """Seconds until `client` may try again (0 = not blocked)."""
        now = time.monotonic()
        recent = [t for t in self._fails.get(client, []) if now - t < FAIL_WINDOW_S]
        if recent:
            self._fails[client] = recent
        else:
            self._fails.pop(client, None)
        return FAIL_WINDOW_S - (now - recent[0]) if len(recent) >= FAIL_MAX else 0.0

    def failed(self, client: str) -> None:
        now = time.monotonic()
        self._fails.setdefault(client, []).append(now)
        if len(self._fails) > 10_000:  # forget idle clients
            self._fails = {c: ts for c, ts in self._fails.items() if ts and now - ts[-1] < FAIL_WINDOW_S}


AUTH = Auth.from_env()


def output_url(run_id: str, name: str) -> str:
    """The URL of a run's output file as handed to the browser (signed when a token is configured)."""
    return AUTH.output_url(run_id, name)


def needs_token(method: str, path: str) -> bool:
    if method == "OPTIONS" or path in PUBLIC:
        return False
    if path == "/mcp" or path.startswith("/mcp/") or path in OWN_CREDENTIAL:
        return False
    return path.startswith(PROTECTED_PREFIXES)


def _bearer(headers: list[tuple[bytes, bytes]]) -> str | None:
    for k, v in headers:
        if k == b"authorization":
            value = v.decode("latin-1")
            return value[7:].strip() if value[:7].lower() == "bearer " else ""
    return None


class TokenMiddleware:
    """Pure ASGI middleware (streams pass through untouched). Add it *before* the CORS middleware, so CORS wraps it:
    preflights are answered by CORS, and a 401/429 still carries the CORS headers the page needs to read it."""

    def __init__(self, app, auth: Auth | None = None, client=None):
        self.app = app
        self.auth = auth
        self.client = client  # (scope) -> str, who is asking (behind trusted proxies: see limits.client_address)

    async def __call__(self, scope, receive, send):
        auth = self.auth or AUTH
        if scope["type"] != "http" or not auth.enabled or not needs_token(scope["method"], scope["path"]):
            return await self.app(scope, receive, send)
        path = scope["path"]
        if path.startswith("/api/outputs/"):
            query = scope.get("query_string", b"").decode("latin-1")
            sig = next((p[4:] for p in query.split("&") if p.startswith("sig=")), None)
            if auth.output_sig_ok(path, sig):
                return await self.app(scope, receive, send)
        client = self.client(scope) if self.client else (scope.get("client") or ("unknown",))[0]
        wait = auth.blocked(client)
        if wait:
            return await _reply(
                send, 429, "too many wrong tokens; try again later", {"retry-after": str(int(wait) + 1)}
            )
        presented = _bearer(scope.get("headers", []))
        if presented is None:
            return await _reply(send, 401, "this server needs its pairing token", {"www-authenticate": REALM})
        if not auth.check(presented):
            auth.failed(client)
            return await _reply(send, 401, "wrong pairing token", {"www-authenticate": REALM})
        return await self.app(scope, receive, send)


async def _reply(send, status: int, detail: str, headers: dict[str, str]) -> None:
    body = json.dumps({"detail": detail}).encode()
    raw = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    raw += [(k.encode(), v.encode()) for k, v in headers.items()]
    await send({"type": "http.response.start", "status": status, "headers": raw})
    await send({"type": "http.response.body", "body": body})
