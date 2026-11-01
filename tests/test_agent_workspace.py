"""The AI Assistant on Workspace problems: sandbox resolution, the view/notebook context, and edits flowing back."""

import asyncio
import json

import pytest

from backend.app import agent, workspace


@pytest.fixture(autouse=True)
def roots(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "WORK_ROOT", tmp_path / "agent")
    monkeypatch.setattr(agent, "_sessions", {})
    monkeypatch.setattr(workspace, "WS_ROOT", tmp_path / "ws")
    monkeypatch.setattr(workspace, "_sandboxes", {})


def test_a_workspace_problem_resolves_to_its_sandbox_and_keeps_one_session_per_sandbox():
    box = workspace.create_blank()["session"]
    s = agent._workspace_session(None, box)
    assert s.workspace == workspace._sandboxes[box].folder and s.sandbox == box
    assert s.problem == f"workspace/{workspace._sandboxes[box].slug}"
    assert agent._workspace_session(s.id, box) is s
    assert agent._workspace_session(None, box) is s  # a new chat in the same browser problem: same session
    other = workspace.create_blank()["session"]
    assert agent._workspace_session(s.id, other) is not s  # a session never crosses into another sandbox
    with pytest.raises(LookupError):
        agent._workspace_session(None, "expired")


def test_an_ai_created_problem_keeps_its_own_agent_session():
    created = agent._session(None, "new")
    assert agent._workspace_session(None, created.id) is created


def test_source_context_describes_the_mode_and_the_markdown_without_touching_files():
    box = workspace.create_blank()["session"]
    s = agent._workspace_session(None, box)
    before = agent._snapshot(s)
    ctx = {
        "title": "Untitled problem",
        "files": {"main.py": "# would overwrite"},
        "params": {"k": 2},
        "workspace": {"kind": "source", "mode": "markdown", "markdown": "# Untitled problem\n\nA summary"},
        "selection": {
            "region": "markdown",
            "file": "markdown",
            "lang": "markdown",
            "from_line": 3,
            "to_line": 3,
            "text": "A summary",
        },
    }
    msg = agent._with_context(s, "what is this?", ctx)
    assert agent._snapshot(s) == before  # the page synced the files already: the context never overwrites them
    assert '"Untitled problem"' in msg and "Markdown view" in msg and "# Untitled problem\n\nA summary" in msg
    assert "lines 3–3 of the markdown:\n```markdown\nA summary" in msg and '"k": 2' in msg
    yaml_msg = agent._with_context(
        s,
        "?",
        {
            "workspace": {"kind": "source", "mode": "yaml"},
            "selection": {"file": "meta.yaml", "from_line": 1, "to_line": 1, "text": "title: X"},
        },
    )
    assert "meta.yaml editor" in yaml_msg and "meta.yaml lines 1–1:\n```yaml\ntitle: X" in yaml_msg


def test_notebook_context_is_written_quoted_within_budget_and_edits_come_back():
    box = workspace.create_blank()["session"]
    s = agent._workspace_session(None, box)
    cells = [{"id": f"c{i}", "type": "code", "source": f"x{i} = {i}\n" + "#" * 900, "output": "ok"} for i in range(30)]
    cells[3] = {
        "id": "c3",
        "type": "code",
        "source": "print(params)  # FOCUSED\n" + "y" * 5000,
        "output": "{}\n[figure]",
    }
    ws = {"kind": "notebook", "notebook_id": "nb1", "notebook_title": "Notebook 1", "cells": cells, "focused_cell": 4}
    msg = agent._with_context(
        s,
        "which cell?",
        {
            "title": "T",
            "workspace": ws,
            "selection": {
                "region": "notebook: Notebook 1, cell 4",
                "lang": "python",
                "from_line": 1,
                "to_line": 1,
                "text": "print(params)",
            },
        },
    )
    assert 'notebook "Notebook 1"' in msg and "cell 4 is focused" in msg
    assert "y" * 5000 in msg  # the focused cell in full
    assert "more cell(s) not shown" in msg and len(msg) < agent.NOTE_BUDGET + 9000
    assert "lines 1–1 of the notebook: Notebook 1, cell 4:" in msg
    nb_file = s.workspace / "_context" / "notebook.json"
    data = json.loads(nb_file.read_text(encoding="utf-8"))
    assert len(data["cells"]) == 30 and data["cells"][3]["output"] == "{}\n[figure]"

    assert agent.notebook_event(s, "workspace/abc") is None  # untouched
    data["cells"] = [{"id": "c0", "type": "code", "source": "x0 = 0"}, {"type": "markdown", "source": "# new"}]
    nb_file.write_text(json.dumps(data), encoding="utf-8")
    event = agent.notebook_event(s, "workspace/abc")
    assert event["type"] == "notebook" and event["notebook"] == "nb1" and event["problem"] == "workspace/abc"
    assert event["cells"] == [
        {"id": "c0", "type": "code", "source": "x0 = 0"},
        {"id": "", "type": "markdown", "source": "# new"},
    ]
    nb_file.write_text("{not json", encoding="utf-8")
    assert agent.notebook_event(s, "workspace/abc")["type"] == "error"
    nb_file.write_text(json.dumps({"cells": [{"type": "shell", "source": "rm"}]}), encoding="utf-8")
    assert "cell 1" in agent.notebook_event(s, "workspace/abc")["message"]

    agent._with_context(s, "now source", {"workspace": {"kind": "source", "mode": "inline"}})
    assert not nb_file.exists() and s.notebook is None  # a stale notebook is never applied


def test_a_chat_turn_edits_the_sandbox_and_sends_files_with_the_parsed_entry(monkeypatch):
    """The agent itself is faked: it edits meta.yaml and the notebook file as Claude Code would."""
    from backend.app.agents import claude_code

    monkeypatch.setattr(agent, "LOCAL_ALLOWED", True)
    box = workspace.create_blank()["session"]
    workspace.sync(box, {"main.py": "# from the browser\n"})

    async def fake_run(s, message, **kw):
        assert "notebook" in message and (s.workspace / "main.py").read_text(encoding="utf-8") == "# from the browser\n"
        assert "NEW problem" not in kw["system_prompt"]
        meta = (s.workspace / "meta.yaml").read_text(encoding="utf-8").replace("Untitled problem", "Pipe flow")
        (s.workspace / "meta.yaml").write_text(meta, encoding="utf-8")
        nb = s.workspace / "_context" / "notebook.json"
        data = json.loads(nb.read_text(encoding="utf-8"))
        data["cells"].append({"type": "code", "source": "print(2)"})
        nb.write_text(json.dumps(data), encoding="utf-8")
        yield {"type": "text", "delta": "done"}
        yield {"type": "done", "cost_usd": 0}

    monkeypatch.setattr(claude_code, "run", fake_run)
    ws = {"kind": "notebook", "notebook_id": "n", "cells": [{"id": "a", "type": "code", "source": "print(1)"}]}

    async def collect():
        return [
            e
            async for e in agent.chat(
                None, "workspace/loc1", "rename it", None, None, "t", {"workspace": ws}, "claude-code", sandbox=box
            )
        ]

    events = asyncio.run(collect())
    kinds = [e["type"] for e in events]
    assert kinds[0] == "session" and kinds[-1] == "done" and "problem" not in kinds
    files = next(e for e in events if e["type"] == "files")
    assert files["problem"] == "workspace/loc1" and list(files["files"]) == ["meta.yaml"]
    assert files["entry"]["title"] == "Pipe flow"
    nb = next(e for e in events if e["type"] == "notebook")
    assert [c["source"] for c in nb["cells"]] == ["print(1)", "print(2)"]
    assert workspace._sandboxes[box].busy == 0

    async def expired():
        return [
            e async for e in agent.chat(None, "workspace/x", "hi", None, None, "t", {}, "claude-code", sandbox="gone")
        ]

    assert asyncio.run(expired())[0]["type"] == "error"
