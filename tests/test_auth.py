"""Pairing token (tensorizer_physics.auth): health stays public, everything else needs `Authorization: Bearer`,
the token is compared in constant time, guessing is rate-limited, the token never reaches a log, and the
code-execution policy follows shared / token / bind address."""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import main as api
from tensorizer_physics import auth, limits
from tensorizer_physics.jobs import JobManager

TOKEN = "s3cret-pairing-token-0123456789abcdef"
GOOD = {"Authorization": f"Bearer {TOKEN}"}
BAD = {"Authorization": "Bearer not-the-token"}
VISITOR = {"X-Visitor": "me-0123456789abcdef"}
QUICK = {"params": {"n_angles": 2, "t_max": 1}}
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def secured(monkeypatch, tmp_path):
    a = auth.Auth(TOKEN)
    monkeypatch.setattr(auth, "AUTH", a)
    monkeypatch.setattr(api, "AUTH", a)
    monkeypatch.setattr("backend.app.runner.CACHE", tmp_path)
    monkeypatch.setattr(api, "CACHE", tmp_path)
    monkeypatch.setattr(api, "JOBS", JobManager(tmp_path / "_jobs"))
    with TestClient(api.app) as c:
        yield c


def test_health_is_public_and_reports_auth(secured):
    r = secured.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["auth"] == "token" and isinstance(body["shared"], bool) and isinstance(body["code_exec"], bool)


def test_health_without_a_token_says_none():
    body = TestClient(api.app).get("/api/health").json()
    assert body["auth"] == "none"


@pytest.mark.parametrize(
    "method,path,extra",
    [
        ("GET", "/api/queue", {}),
        ("POST", "/api/queue/cancel", {}),
        ("GET", "/api/jobs", VISITOR),
        ("GET", "/api/agent/status", {}),
        ("POST", "/api/run-stream/mechanics/projectile", {}),
        ("GET", "/openapi.json", {}),
    ],
)
def test_protected_routes_need_the_token(secured, method, path, extra):
    for headers in ({}, BAD, {"Authorization": "Basic Zm9vOmJhcg=="}, {"Authorization": TOKEN}):
        r = secured.request(method, path, headers={**extra, **headers}, json=QUICK if method == "POST" else None)
        assert r.status_code == 401, (path, headers, r.text)
        assert r.headers["www-authenticate"] == 'Bearer realm="tensorizer"'


def test_right_token_opens_the_routes(secured):
    assert secured.get("/api/queue", headers=GOOD).status_code == 200
    assert secured.get("/api/jobs", headers={**GOOD, **VISITOR}).json() == {"jobs": []}
    assert secured.get("/api/agent/status", headers=GOOD).status_code == 200


def test_run_stream_with_the_token_and_a_signed_output(secured):
    with secured.stream("POST", "/api/run-stream/mechanics/projectile", json=QUICK, headers=GOOD) as r:
        assert r.status_code == 200
        events = [json.loads(line) for line in r.iter_lines() if line]
    result = events[-1]
    assert result["type"] == "result", events[-1]
    url = result["output"]
    assert "?sig=" in url  # <img> tags cannot send a header: the URL carries a per-file signature
    assert secured.get(url).status_code == 200
    path, sig = url.split("?sig=")
    assert secured.get(path).status_code == 401  # no signature, no header
    assert secured.get(f"{path}?sig={'0' * len(sig)}").status_code == 401
    assert secured.get(path, headers=GOOD).status_code == 200  # the header works too
    other = path.rsplit("/", 1)[0] + "/other.png"
    assert secured.get(f"{other}?sig={sig}").status_code == 401  # a signature opens its own file only


def test_signature_changes_with_the_token():
    a, b = auth.Auth("one"), auth.Auth("two")
    assert a.output_url("r1", "out.gif") != b.output_url("r1", "out.gif")
    assert auth.Auth().output_url("r1", "out.gif") == "/api/outputs/r1/out.gif"  # no token: unchanged


def test_cors_preflight_needs_no_token_and_401_keeps_cors_headers(secured):
    origin = "https://jackyko1991.github.io"
    pre = secured.options(
        "/api/queue",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization,x-visitor",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert pre.status_code == 200
    assert "authorization" in pre.headers["access-control-allow-headers"].lower()
    assert pre.headers.get("access-control-allow-private-network") == "true"
    r = secured.get("/api/queue", headers={"Origin": origin})
    assert r.status_code == 401 and r.headers.get("access-control-allow-origin") in (origin, "*")


def test_routes_with_their_own_credential_are_not_double_locked(secured):
    # /mcp checks the connector token itself (issued by an authenticated call); the pairing check stays out of its way
    r = secured.post("/mcp", json={})
    assert "pairing token" not in r.text


def test_constant_time_compare_is_used(monkeypatch):
    calls = []
    real = auth.hmac.compare_digest

    def spy(a, b):
        calls.append((len(a), len(b)))
        return real(a, b)

    monkeypatch.setattr(auth.hmac, "compare_digest", spy)
    a = auth.Auth(TOKEN)
    assert a.check(TOKEN) and not a.check("x") and not a.check(None) and not a.check("")
    assert calls and all(n == (32, 32) for n in calls)  # SHA-256 digests: same length whatever was sent


def test_sha256_form_and_env_handling():
    import hashlib

    env = {"TENSORIZER_TOKEN": TOKEN}
    a = auth.Auth.from_env(env)
    assert "TENSORIZER_TOKEN" not in env  # removed: run processes and later code never see it
    assert a.check(TOKEN) and TOKEN not in repr(vars(a))  # only the hash is kept
    b = auth.Auth.from_env({"TENSORIZER_TOKEN_SHA256": hashlib.sha256(TOKEN.encode()).hexdigest()})
    assert b.enabled and b.check(TOKEN) and not b.check("nope")
    with pytest.raises(ValueError):
        auth.Auth(token_sha256="not-hex")


def test_guessing_is_rate_limited(secured, monkeypatch):
    for _ in range(auth.FAIL_MAX):
        assert secured.get("/api/queue", headers=BAD).status_code == 401
    blocked = secured.get("/api/queue", headers=GOOD)  # even the right token waits now
    assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) > 0
    assert secured.get("/api/health").status_code == 200  # health stays open
    # a missing header is not a guess: it never counts
    a = auth.Auth(TOKEN)
    for _ in range(50):
        assert not a.blocked("x")
    # the window passes
    a.failed("y")
    now = auth.time.monotonic()
    for _ in range(auth.FAIL_MAX - 1):
        a.failed("y")
    assert a.blocked("y") > 0
    monkeypatch.setattr(auth.time, "monotonic", lambda: now + auth.FAIL_WINDOW_S + 1)
    assert a.blocked("y") == 0


def test_token_never_logged(secured, caplog, capsys):
    caplog.set_level(logging.DEBUG)
    secured.get("/api/queue", headers=GOOD)
    secured.get("/api/queue", headers={"Authorization": f"Bearer {TOKEN}x"})
    secured.get("/api/jobs", headers={**GOOD, **VISITOR})
    out = capsys.readouterr()
    assert TOKEN not in caplog.text and TOKEN not in out.out and TOKEN not in out.err


# --- code-execution policy ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shared,shared_code,token,allow,bind,user_code,edited",
    [
        (True, False, True, None, "0.0.0.0", False, False),  # shared + token: still off
        (True, False, True, "1", "0.0.0.0", False, False),
        (True, True, True, "1", "0.0.0.0", True, True),  # shared: explicit opt-in only
        (False, False, True, None, "0.0.0.0", True, True),  # own server + token: on by default
        (False, False, True, "0", "0.0.0.0", False, False),  # ...unless switched off
        (False, False, False, "1", "0.0.0.0", False, False),  # no token on a network address: never
        (False, False, False, "1", "127.0.0.1", True, True),  # `pixi run api`: as before
        (False, False, False, None, "127.0.0.1", True, False),  # Workspace as before; edited code needs ALLOW=1
        (False, False, False, "1", None, True, True),  # started another way (uvicorn, tests): as before
    ],
)
def test_code_exec_policy(shared, shared_code, token, allow, bind, user_code, edited):
    assert limits.code_exec_policy(shared, shared_code, token, allow, bind) is user_code
    assert limits.edited_code_policy(shared, shared_code, token, allow, bind) is edited


def _server_flags(**env) -> dict:
    """Import the app in a fresh interpreter with this environment and report its policy (no request is made)."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("TENSORIZER_", "ALLOW_CODE", "AGENT_"))}
    code = (
        "import json, os; from backend.app import main, agent; from tensorizer_physics import limits; "
        "print(json.dumps({'edited': main.ALLOW_CODE_EXEC, 'user': limits.user_code_allowed(), "
        "'agent': agent.ENABLED, 'auth': main.AUTH.enabled, 'leak': 'TENSORIZER_TOKEN' in os.environ, "
        "'origins': main.ORIGINS}))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env={**clean, **env}, capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_private_server_with_token_turns_code_on():
    f = _server_flags(TENSORIZER_TOKEN=TOKEN, TENSORIZER_SHARED="0", TENSORIZER_BIND="0.0.0.0")
    assert f == {**f, "edited": True, "user": True, "agent": True, "auth": True, "leak": False}
    assert "https://jackyko1991.github.io" in f["origins"] and "*" not in f["origins"]


def test_shared_server_with_token_keeps_code_off():
    f = _server_flags(TENSORIZER_TOKEN=TOKEN, TENSORIZER_SHARED="1", TENSORIZER_BIND="0.0.0.0")
    assert f["edited"] is False and f["user"] is False and f["agent"] is False and f["auth"] is True


def test_no_token_on_localhost_keeps_todays_behaviour():
    f = _server_flags(ALLOW_CODE_EXEC="1", AGENT_ENABLED="1", TENSORIZER_BIND="127.0.0.1")
    assert f["edited"] is True and f["user"] is True and f["agent"] is True and f["auth"] is False
    assert f["origins"] == ["*"]
    exposed = _server_flags(ALLOW_CODE_EXEC="1", AGENT_ENABLED="1", TENSORIZER_BIND="0.0.0.0")
    assert exposed["edited"] is False and exposed["user"] is False
