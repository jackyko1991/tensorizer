"""The AI Assistant's three modes without calling any model: the local agent scan, the CLI adapters' event mapping
(from recorded/documented sample output), and Tensorizer's MCP server over HTTP with its connector tokens."""

# ruff: noqa: E501  (sample CLI output lines are kept verbatim)

import json
import sys

import pytest
from fastapi.testclient import TestClient

from backend.app import agent, mcp_server
from backend.app import main as api
from backend.app.agents import cli, codex, gemini, opencode, scan

MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ENABLED", True)
    monkeypatch.setattr(agent, "WORK_ROOT", tmp_path)
    return TestClient(api.app)


# --- LOCAL: scan ----------------------------------------------------------------------------------------------------


def test_scan_returns_booleans_and_versions_only(monkeypatch, tmp_path):
    secret = "sk-ant-very-secret-key"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    (tmp_path / "codex").mkdir()
    (tmp_path / "codex" / "auth.json").write_text(json.dumps({"token": secret}), encoding="utf-8")
    fake = str(tmp_path / "secret-dir" / "bin")
    monkeypatch.setattr(scan.shutil, "which", lambda name: f"{fake}/{name}" if name in ("claude", "codex") else None)

    class Done:
        stdout = f"codex-cli 0.98.1 (installed at {fake}, key {secret})\n"
        stderr = ""

    monkeypatch.setattr(scan.subprocess, "run", lambda *a, **k: Done())
    result = scan.scan()
    dumped = json.dumps(result)
    assert secret not in dumped and fake not in dumped and "secret-dir" not in dumped
    by_id = {a["id"]: a for a in result}
    assert set(by_id) == {"claude-code", "codex", "opencode", "gemini"}
    for a in result:
        assert set(a) == {"id", "label", "installed", "version", "logged_in", "experimental", "docs"}
        assert isinstance(a["installed"], bool) and isinstance(a["experimental"], bool)
        assert a["logged_in"] in (True, False, None)
    assert by_id["codex"]["installed"] and by_id["codex"]["version"] == "0.98.1" and by_id["codex"]["logged_in"]
    assert by_id["claude-code"]["logged_in"] is True  # ANTHROPIC_API_KEY is set (its value is never read out)
    assert not by_id["gemini"]["installed"] and by_id["gemini"]["logged_in"] is False
    assert not by_id["codex"]["experimental"] and not by_id["claude-code"]["experimental"]
    assert by_id["opencode"]["experimental"] and by_id["gemini"]["experimental"]
    assert scan.executable("codex") == f"{fake}/codex"  # the path stays on the server


def test_scan_route_only_on_a_local_backend(client, monkeypatch):
    monkeypatch.setattr(agent, "LOCAL_ALLOWED", False)
    assert client.post("/api/agents/scan").status_code == 403
    monkeypatch.setattr(agent, "LOCAL_ALLOWED", True)
    monkeypatch.setattr(scan, "scan", lambda: [{"id": "codex", "installed": False}])
    assert client.post("/api/agents/scan").json() == {"agents": [{"id": "codex", "installed": False}]}


def test_local_agents_are_refused_on_a_shared_server(client, monkeypatch):
    monkeypatch.setattr(agent, "LOCAL_ALLOWED", False)
    body = {"problem": "mechanics/simple_pendulum", "message": "hi", "agent": "codex"}
    lines = _turn_events(client.post("/api/agents/chat", json=body).text)
    assert lines == [{"type": "error", "message": "local agents only run on your own computer; use the API option"}]
    body = {**body, "agent": "claude-api", "provider": "openrouter"}
    lines = _turn_events(client.post("/api/agents/chat", json=body).text)
    assert lines[0]["type"] == "error" and "OpenRouter API key" in lines[0]["message"]


def _turn_events(text: str) -> list[dict]:
    """A chat response's agent events: without the leading {"type": "turn"} event and each event's "seq"."""
    events = [json.loads(x) for x in text.splitlines()]
    assert events[0]["type"] == "turn"
    return [{k: v for k, v in e.items() if k != "seq"} for e in events[1:]]


# --- LOCAL: adapters, from sample output ----------------------------------------------------------------------------

# The first, fourth and fifth lines are verbatim from https://learn.chatgpt.com/docs/non-interactive-mode; the
# others follow codex-rs/exec/src/exec_events.rs.
CODEX_SAMPLE = """
{"type":"thread.started","thread_id":"0199a213-81c0-7800-8aa1-bbab2a035a53"}
{"type":"turn.started"}
{"type":"item.started","item":{"id":"item_1","type":"command_execution","command":"bash -lc ls","status":"in_progress"}}
{"type":"item.completed","item":{"id":"item_1","type":"command_execution","command":"bash -lc ls","aggregated_output":"main.py\\n","exit_code":0,"status":"completed"}}
{"type":"item.completed","item":{"id":"item_2","type":"reasoning","text":"thinking"}}
{"type":"item.completed","item":{"id":"item_4","type":"file_change","changes":[{"path":"plot.py","kind":"update"}],"status":"completed"}}
{"type":"item.started","item":{"id":"item_5","type":"mcp_tool_call","server":"tensorizer","tool":"run_problem","arguments":{"params":{}},"status":"in_progress"}}
{"type":"item.completed","item":{"id":"item_5","type":"mcp_tool_call","server":"tensorizer","tool":"run_problem","arguments":{},"result":{"content":[{"type":"text","text":"Run succeeded."}]},"status":"completed"}}
{"type":"item.completed","item":{"id":"item_3","type":"agent_message","text":"Repo contains docs, sdk, and examples directories."}}
{"type":"turn.completed","usage":{"input_tokens":24763,"cached_input_tokens":24448,"output_tokens":122,"reasoning_output_tokens":0}}
"""


def _map(mapper, sample: str) -> list[dict]:
    return [e for line in sample.strip().splitlines() for e in mapper(json.loads(line))]


def test_codex_events_map_to_the_panel_events():
    m = codex.Mapper()
    events = _map(m, CODEX_SAMPLE)
    assert m.session == "0199a213-81c0-7800-8aa1-bbab2a035a53"
    assert events == [
        {"type": "tool", "id": "item_1", "name": "Bash", "input": {"command": "bash -lc ls"}},
        {"type": "tool_result", "id": "item_1", "ok": True, "text": "main.py\n"},
        {"type": "tool", "id": "item_4", "name": "Edit", "input": {"file_path": "plot.py"}},
        {"type": "tool_result", "id": "item_4", "ok": True, "text": ""},
        {"type": "tool", "id": "item_5", "name": "mcp__tensorizer__run_problem", "input": {"params": {}}},
        {"type": "tool_result", "id": "item_5", "ok": True, "text": "Run succeeded."},
        {"type": "text", "delta": "Repo contains docs, sdk, and examples directories."},
    ]
    assert m({"type": "turn.failed", "error": {"message": "quota"}}) == [{"type": "error", "message": "quota"}]
    argv = codex.command("codex", "/ws", "gpt-5", "abc", "http://127.0.0.1:8000/mcp/tok")
    assert argv[:3] == ["codex", "exec", "--json"] and argv[-3:] == ["resume", "abc", "-"]
    assert ["--sandbox", "workspace-write"] == argv[argv.index("--sandbox") : argv.index("--sandbox") + 2]
    assert "sandbox_workspace_write.network_access=false" in argv and 'approval_policy="never"' in argv
    assert 'mcp_servers.tensorizer.url="http://127.0.0.1:8000/mcp/tok"' in argv


# Shapes from packages/cli/src/run/noninteractive.ts (opencode run --format json).
OPENCODE_SAMPLE = """
{"type":"step_start","timestamp":1,"sessionID":"ses_1","part":{"type":"step-start"}}
{"type":"tool_use","timestamp":2,"sessionID":"ses_1","part":{"partID":"p1","type":"tool","id":"call_1","tool":"read","state":{"status":"completed","input":{"filePath":"/ws/equation.py"},"output":"import torch","title":"read"}}}
{"type":"tool_use","timestamp":3,"sessionID":"ses_1","part":{"partID":"p2","type":"tool","id":"call_2","tool":"tensorizer_run_problem","state":{"status":"error","input":{},"error":"Run failed: boom"}}}
{"type":"text","timestamp":4,"sessionID":"ses_1","part":{"type":"text","text":"Done."}}
{"type":"step_finish","timestamp":5,"sessionID":"ses_1","part":{"type":"step-finish","cost":0.002,"tokens":{}}}
"""


def test_opencode_events_map_to_the_panel_events():
    m = opencode.Mapper()
    events = _map(m, OPENCODE_SAMPLE)
    assert m.session == "ses_1" and m.cost == pytest.approx(0.002)
    assert events == [
        {
            "type": "tool",
            "id": "call_1",
            "name": "Read",
            "input": {"filePath": "/ws/equation.py", "file_path": "/ws/equation.py"},
        },
        {"type": "tool_result", "id": "call_1", "ok": True, "text": "import torch"},
        {"type": "tool", "id": "call_2", "name": "mcp__tensorizer__run_problem", "input": {}},
        {"type": "tool_result", "id": "call_2", "ok": False, "text": "Run failed: boom"},
        {"type": "text", "delta": "Done."},
    ]
    assert m({"type": "error", "error": {"type": "unknown", "message": "no model"}})[0]["message"] == "no model"
    config = json.loads(opencode.env("http://x/mcp/t")["OPENCODE_CONFIG_CONTENT"])
    assert config["permission"]["bash"] == "deny" and config["mcp"]["tensorizer"]["url"] == "http://x/mcp/t"
    assert opencode.command("opencode", "/ws", None, "ses_1") == [
        *("opencode", "run", "--format", "json", "--dir", "/ws", "-s", "ses_1")
    ]


# Event shapes from packages/core/src/output/types.ts (gemini --output-format stream-json).
GEMINI_SAMPLE = """
{"type":"init","timestamp":"t","session_id":"g-1","model":"gemini-2.5-pro"}
{"type":"message","timestamp":"t","role":"user","content":"hi"}
{"type":"message","timestamp":"t","role":"assistant","content":"Let me ","delta":true}
{"type":"message","timestamp":"t","role":"assistant","content":"look.","delta":true}
{"type":"tool_use","timestamp":"t","tool_name":"read_file","tool_id":"r1","parameters":{"file_path":"main.py"}}
{"type":"tool_result","timestamp":"t","tool_id":"r1","status":"success","output":"def run"}
{"type":"tool_use","timestamp":"t","tool_name":"run_problem","tool_id":"r2","parameters":{}}
{"type":"tool_result","timestamp":"t","tool_id":"r2","status":"error","error":{"type":"x","message":"failed"}}
{"type":"message","timestamp":"t","role":"assistant","content":"Fixed.","delta":true}
{"type":"error","timestamp":"t","severity":"warning","message":"loop detected"}
{"type":"result","timestamp":"t","status":"success","stats":{"total_tokens":10}}
"""


def test_gemini_events_map_to_the_panel_events(tmp_path):
    m = gemini.Mapper()
    events = _map(m, GEMINI_SAMPLE)
    assert m.session == "g-1"
    assert events == [
        {"type": "text", "delta": "Let me "},
        {"type": "text", "delta": "look."},
        {"type": "tool", "id": "r1", "name": "Read", "input": {"file_path": "main.py"}},
        {"type": "tool_result", "id": "r1", "ok": True, "text": "def run"},
        {"type": "tool", "id": "r2", "name": "mcp__tensorizer__run_problem", "input": {}},
        {"type": "tool_result", "id": "r2", "ok": False, "text": "failed"},
        {"type": "text", "delta": "\n\nFixed."},
    ]
    env = gemini.env(tmp_path / "settings.json", "http://x/mcp/t")
    settings = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert env == {"GEMINI_CLI_SYSTEM_SETTINGS_PATH": str(tmp_path / "settings.json")}
    assert "run_shell_command" in settings["tools"]["exclude"] and "run_shell_command" not in settings["tools"]["core"]
    assert settings["mcpServers"]["tensorizer"] == {"httpUrl": "http://x/mcp/t", "trust": True}
    assert gemini.command("gemini", None, True) == [
        *("gemini", "--output-format", "stream-json", "--approval-mode", "auto_edit", "--resume", "latest")
    ]


def test_a_cli_runs_in_the_workspace_with_the_prompt_on_stdin(tmp_path):
    """The subprocess plumbing, with a stand-in CLI that echoes stdin and cwd as Gemini-style events."""
    import asyncio

    script = (
        "import json, os, sys; p = sys.stdin.read(); "
        "print('banner text'); "
        "print(json.dumps({'type': 'init', 'session_id': 's9'})); "
        "print(json.dumps({'type': 'message', 'role': 'assistant', 'content': p + ' @ ' + os.getcwd()})); "
        "print('secret-token', file=sys.stderr); sys.exit(int(p == 'fail'))"
    )

    async def go(prompt):
        mapper = gemini.Mapper()
        out = [
            e
            async for e in cli.run(
                [sys.executable, "-c", script],
                tmp_path,
                {"PYTHONIOENCODING": "utf-8"},
                prompt,
                mapper,
                asyncio.Queue(),
                ["secret-token"],
            )
        ]
        return mapper, out

    mapper, events = asyncio.run(go("héllo"))
    assert mapper.session == "s9"
    assert events[0] == {"type": "text", "delta": f"héllo @ {tmp_path}"} and events[-1]["type"] == "done"
    _, events = asyncio.run(go("fail"))
    assert events[-1]["type"] == "error" and "code 1" in events[-1]["message"]
    assert "secret-token" not in events[-1]["message"] and "***" in events[-1]["message"]


# --- CLOUD: the MCP server ------------------------------------------------------------------------------------------


def _rpc(client, url, method, params=None, token=None, rid=1):
    headers = {**MCP_HEADERS, "MCP-Protocol-Version": "2025-06-18"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {"jsonrpc": "2.0", "id": rid, "method": method, **({"params": params} if params is not None else {})}
    return client.post(url, json=body, headers=headers)


def _call(client, token, name, **arguments):
    r = _rpc(client, f"/mcp/{token}", "tools/call", {"name": name, "arguments": arguments})
    assert r.status_code == 200, r.text
    return r.json()["result"]


def test_mcp_initialize_list_and_call_with_the_token(client):
    connector = client.post("/api/agents/connector", json={}).json()
    token = connector["token"]
    assert connector["url_with_token"].endswith(f"/mcp/{token}") and connector["url"].endswith("/mcp")

    init = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}
    r = _rpc(client, f"/mcp/{token}", "initialize", init)
    assert r.status_code == 200 and r.json()["result"]["serverInfo"]["name"] == "tensorizer"

    r = _rpc(client, "/mcp", "tools/list", token=token)  # bearer token instead of the URL
    names = {t["name"] for t in r.json()["result"]["tools"]}
    assert names == {
        "list_problems",
        "get_problem",
        "read_file",
        "write_file",
        "run_problem",
        "wait_render",
        "cancel_render",
        "ask_user",
        "read_notebook",
        "edit_notebook",
    }
    assert _rpc(client, f"/mcp?token={token}", "tools/list").status_code == 200

    listed = json.loads(_call(client, token, "list_problems", category="mechanics")["content"][0]["text"])
    assert {"id": "mechanics/simple_pendulum", "category": "mechanics"}.items() <= next(
        p for p in listed if p["id"] == "mechanics/simple_pendulum"
    ).items()
    got = json.loads(_call(client, token, "get_problem", problem="mechanics/simple_pendulum")["content"][0]["text"])
    assert set(got["files"]) >= {"meta.yaml", "equation.py", "main.py"}

    result = _call(client, token, "write_file", problem="mechanics/simple_pendulum", path="plot.py", content="# hi\n")
    assert not result.get("isError")
    read = _call(client, token, "read_file", problem="mechanics/simple_pendulum", path="plot.py")
    assert read["content"][0]["text"] == "# hi\n"
    original = agent.problem_dir("mechanics", "simple_pendulum") / "plot.py"
    assert original.read_text(encoding="utf-8") != "# hi\n"  # only the private copy changed

    # What the agent did is mirrored to the browser holding the token.
    assert client.get("/api/agents/connector/events").status_code == 401
    events = client.get("/api/agents/connector/events", headers={"Authorization": f"Bearer {token}"}).json()
    kinds = [e["type"] for e in events["events"]]
    assert "tool" in kinds and "tool_result" in kinds and "files" in kinds
    files = next(e for e in events["events"] if e["type"] == "files")
    assert files == {**files, "problem": "mechanics/simple_pendulum", "files": {"plot.py": "# hi\n"}}
    later = client.get(
        f"/api/agents/connector/events?after={events['seq']}", headers={"Authorization": f"Bearer {token}"}
    )
    assert later.json()["events"] == []


def test_mcp_rejects_calls_without_a_valid_token(client):
    r = _rpc(client, "/mcp", "tools/list")
    assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Bearer")
    assert _rpc(client, "/mcp/not-a-token", "tools/list").status_code == 401
    assert _rpc(client, "/mcp", "tools/list", token="tzr_wrong").status_code == 401
    token = client.post("/api/agents/connector", json={}).json()["token"]
    newer = client.post("/api/agents/connector", json={"previous": token}).json()["token"]
    assert _rpc(client, f"/mcp/{token}", "tools/list").status_code == 401  # replaced → revoked
    assert _rpc(client, f"/mcp/{newer}", "tools/list").status_code == 200


def test_mcp_file_tools_stay_inside_the_workspace(client, tmp_path):
    token = client.post("/api/agents/connector", json={}).json()["token"]
    p = "mechanics/simple_pendulum"
    outside = tmp_path / "outside.txt"
    for path in ["../../_common/runtime.py", "../../../../outside.txt", str(outside), "/etc/passwd", "C:/x.txt"]:
        assert _call(client, token, "read_file", problem=p, path=path)["isError"], path
        assert _call(client, token, "write_file", problem=p, path=path, content="x")["isError"], path
    assert not outside.exists()
    assert _call(client, token, "read_file", problem="../../etc", path="main.py")["isError"]
    assert "unknown problem" in _call(client, token, "get_problem", problem="nope/nope")["content"][0]["text"]


def test_a_chat_token_only_reaches_its_own_workspace(client):
    s = agent._session(None, "mechanics/simple_pendulum")
    token = mcp_server.session_token(s.id, "test")
    assert _call(client, token, "read_file", path="main.py")["content"][0]["text"].startswith('"""')
    other = _call(client, token, "read_file", problem="waves/wave_1d", path="main.py")
    assert other["isError"] and "only" in other["content"][0]["text"]
    mcp_server.revoke(token)
    assert _rpc(client, f"/mcp/{token}", "tools/list").status_code == 401


def test_mcp_notebook_tools_edit_the_open_notebook_of_a_chat(client):
    s = agent._session(None, "mechanics/simple_pendulum")
    token = mcp_server.session_token(s.id, "test")
    assert "no notebook" in _call(client, token, "read_notebook")["content"][0]["text"]
    ws = {"kind": "notebook", "notebook_id": "n1", "cells": [{"id": "a", "type": "code", "source": "print(1)"}]}
    agent._with_context(s, "hi", {"workspace": ws})
    assert json.loads(_call(client, token, "read_notebook")["content"][0]["text"])["cells"][0]["source"] == "print(1)"
    bad = _call(client, token, "edit_notebook", cells=[{"type": "shell", "source": "x"}])
    assert bad["isError"]
    cells = [{"id": "a", "type": "code", "source": "print(1)"}, {"type": "markdown", "source": "# note"}]
    assert not _call(client, token, "edit_notebook", cells=cells).get("isError")
    event = agent.notebook_event(s, "workspace/x")
    assert event["notebook"] == "n1" and [c["type"] for c in event["cells"]] == ["code", "markdown"]
    mcp_server.revoke(token)


def test_a_claude_code_subscription_login_is_detected(monkeypatch, tmp_path):
    from backend.app.agents import scan

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(scan, "_keychain_item", lambda service: False)
    monkeypatch.setattr(scan.sys, "platform", "linux")
    assert scan.logged_in("claude-code") is False
    (tmp_path / ".claude.json").write_text('{"oauthAccount": {"emailAddress": "x"}}', encoding="utf-8")
    assert scan.logged_in("claude-code") is True  # `claude` /login, token kept elsewhere (macOS: the Keychain)
    (tmp_path / ".claude.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(scan.sys, "platform", "darwin")
    assert scan.logged_in("claude-code") is None  # nothing seen: unknown, simply tried
    monkeypatch.setattr(scan, "_keychain_item", lambda service: service == "Claude Code-credentials")
    assert scan.logged_in("claude-code") is True
