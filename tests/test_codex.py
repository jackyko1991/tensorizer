"""Codex's contract with the same workspace and render workflow as Claude Code."""

import asyncio
import json
import os
import socket

import pytest

from backend.app import agent, mcp_server
from backend.app.agents import cli, codex, scan


def test_codex_disables_shell_and_web_and_requires_its_render_tool():
    argv = codex.command("codex", "/ws", None, None, "http://localhost/mcp/token")
    assert "features.shell_tool=false" in argv
    assert 'web_search="disabled"' in argv
    assert "mcp_servers.tensorizer.required=true" in argv
    assert 'mcp_servers.tensorizer.default_tools_approval_mode="approve"' in argv
    assert "--ignore-user-config" in argv
    assert "--ignore-rules" in argv


@pytest.mark.parametrize("tool,label", [("read_file", "Read"), ("write_file", "Write")])
def test_workspace_mcp_tools_use_the_same_panel_labels_as_claude(tool, label):
    event = codex.Mapper()(
        {
            "type": "item.started",
            "item": {
                "id": "call",
                "type": "mcp_tool_call",
                "server": "tensorizer",
                "tool": tool,
                "arguments": {"path": "test.py"},
            },
        }
    )[0]
    assert event == {"type": "tool", "id": "call", "name": label, "input": {"path": "test.py", "file_path": "test.py"}}


def test_mcp_application_errors_are_failed_tool_results():
    event = codex.Mapper()(
        {
            "type": "item.completed",
            "item": {
                "id": "call",
                "type": "mcp_tool_call",
                "server": "tensorizer",
                "tool": "run_problem",
                "status": "completed",
                "result": {
                    "isError": True,
                    "content": [
                        {"type": "text", "text": "test.py: 1 failed (RED)"},
                    ],
                },
            },
        }
    )[0]
    assert event == {"type": "tool_result", "id": "call", "ok": False, "text": "test.py: 1 failed (RED)"}


def test_codex_chat_edits_renders_and_resumes_with_a_fresh_scoped_token(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "WORK_ROOT", tmp_path)
    monkeypatch.setattr(agent, "LOCAL_ALLOWED", True)
    monkeypatch.setattr(agent, "_sessions", {})
    monkeypatch.setattr(scan, "executable", lambda _: "codex")
    calls = []
    tokens = []

    async def stand_in(argv, cwd, env, prompt, mapper, sink, secrets, label):
        calls.append((argv, prompt))
        tokens.append(secrets[0])
        assert mcp_server._hash(secrets[0]) in mcp_server._connectors
        if len(calls) == 1:
            assert "test.py" in prompt and "run_problem" in prompt
        (cwd / "plot.py").write_text("# edited by Codex\n")
        mapper({"type": "thread.started", "thread_id": "codex-thread"})
        await sink.put({"type": "run", "ok": True, "output": "/render.gif"})
        yield await sink.get()
        yield {"type": "done", "cost_usd": None}

    monkeypatch.setattr(cli, "run", stand_in)

    async def go():
        first = [
            e
            async for e in agent.chat(
                None,
                "mechanics/simple_pendulum",
                "edit and render",
                None,
                None,
                "test",
                agent="codex",
                mcp_base="http://localhost",
            )
        ]
        sid = first[0]["id"]
        second = [
            e
            async for e in agent.chat(
                sid,
                "mechanics/simple_pendulum",
                "continue",
                None,
                None,
                "test",
                agent="codex",
                mcp_base="http://localhost",
            )
        ]
        return first, second, sid

    first, second, sid = asyncio.run(go())
    assert first[-1]["type"] == second[-1]["type"] == "done"
    assert next(e for e in first if e["type"] == "files")["files"]["plot.py"] == "# edited by Codex\n"
    assert any(e["type"] == "run" for e in first)
    assert calls[1][0][-3:] == ["resume", "codex-thread", "-"]
    assert tokens[0] != tokens[1]
    assert all(mcp_server._hash(t) not in mcp_server._connectors for t in tokens)
    assert agent._sessions[sid].cli_session == "codex-thread"
    assert json.loads((tmp_path / sid / "session.json").read_text())["cli_session"] == "codex-thread"


def test_stopping_codex_revokes_its_tool_token_and_keeps_the_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "WORK_ROOT", tmp_path)
    s = agent._session(None, "mechanics/simple_pendulum")
    s.sink = asyncio.Queue()
    tokens = []

    async def stand_in(argv, cwd, env, prompt, mapper, sink, secrets, label):
        tokens.extend(secrets)
        mapper({"type": "thread.started", "thread_id": "interrupted-thread"})
        yield {"type": "text", "delta": "Working"}
        pytest.fail("the cancelled stream must not continue")

    monkeypatch.setattr(cli, "run", stand_in)

    async def go():
        stream = agent._cli_stream(s, "codex", "codex", "hi", None, "http://localhost", "test")
        await anext(stream)
        await stream.aclose()

    asyncio.run(go())
    assert s.cli_session == "interrupted-thread"
    assert tokens and all(mcp_server._hash(t) not in mcp_server._connectors for t in tokens)


@pytest.mark.skipif(os.environ.get("TENSORIZER_TEST_CODEX") != "1", reason="opt-in: uses local Codex login")
def test_live_codex_edit_render_and_resume(tmp_path, monkeypatch):
    """Real CLI, model, HTTP MCP and renderer; run with TENSORIZER_TEST_CODEX=1 ALLOW_CODE_EXEC=1."""
    import uvicorn

    from backend.app.main import app

    monkeypatch.setattr(agent, "ENABLED", True)
    monkeypatch.setattr(agent, "WORK_ROOT", tmp_path)
    monkeypatch.setattr(agent, "LOCAL_ALLOWED", True)
    monkeypatch.setattr(agent, "_sessions", {})
    original = agent.problem_dir("mechanics", "simple_pendulum") / "test.py"
    source = original.read_text()

    async def go():
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        base = f"http://127.0.0.1:{sock.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(180):
                while not server.started:
                    if task.done():
                        await task
                    await asyncio.sleep(0.05)
                first = [
                    e
                    async for e in agent.chat(
                        None,
                        "mechanics/simple_pendulum",
                        "Use tensorizer's read_file to read test.py, then write_file to append exactly "
                        "'\n# Codex integration verified\n' without changing any tests. Then call run_problem "
                        "with default parameters. Remember the word violet-pendulum for my next message. "
                        "Do not use shell commands or other tools. Report the test verdict.",
                        None,
                        None,
                        "codex-live",
                        agent="codex",
                        mcp_base=base,
                    )
                ]
                assert not [e for e in first if e["type"] == "error"], first
                sid = first[0]["id"]
                s = agent._sessions[sid]
                assert s.cli_session
                thread = s.cli_session
                assert (s.workspace / "test.py").read_text() == source + "\n# Codex integration verified\n", first
                assert any(e["type"] == "run" and e["ok"] for e in first), first
                assert any(e["type"] == "files" and "test.py" in e["files"] for e in first)
                assert any(e["type"] == "tool_result" and e["ok"] and "all green" in e["text"] for e in first)
                second = [
                    e
                    async for e in agent.chat(
                        sid,
                        "mechanics/simple_pendulum",
                        "What word did I ask you to remember? Use tensorizer's read_file to check test.py "
                        "still ends in the verification comment. Reply with the word and confirmation.",
                        None,
                        None,
                        "codex-live",
                        agent="codex",
                        mcp_base=base,
                    )
                ]
                assert not [e for e in second if e["type"] == "error"], second
                assert "violet-pendulum" in "".join(e.get("delta", "") for e in second)
                assert any(e["type"] == "tool" and e["name"] == "Read" for e in second), second
                assert s.cli_session == thread
                assert all(e.get("name") != "Bash" for e in first + second)
                assert first[-1]["type"] == second[-1]["type"] == "done"
                assert original.read_text() == source
                assert not any(c.session == sid for c in mcp_server._connectors.values())
        finally:
            server.should_exit = True
            await task
            sock.close()

    asyncio.run(go())
