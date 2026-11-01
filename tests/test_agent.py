"""The agent's sandboxing helpers (no Claude calls): workspaces, path confinement and edited-file detection."""

from backend.app import agent


def test_workspace_is_a_private_copy_and_paths_are_confined(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "WORK_ROOT", tmp_path)
    s = agent._session(None, "mechanics/simple_pendulum")
    assert (s.workspace / "equation.py").is_file() and (s.workspace.parents[1] / "_common" / "runtime.py").is_file()
    assert agent._session(s.id, "mechanics/simple_pendulum") is s  # same chat → same workspace

    assert agent._inside(str(s.workspace / "equation.py"), s.workspace)
    assert not agent._inside(str(s.workspace / ".." / ".." / ".." / "secret.txt"), s.workspace)
    assert not agent._inside(str(agent.PROBLEMS / "mechanics" / "simple_pendulum" / "main.py"), s.workspace)

    assert agent._edited_files(s) == {}
    (s.workspace / "plot.py").write_text("# changed\n", encoding="utf-8")
    assert agent._edited_files(s) == {"plot.py": "# changed\n"}


def test_long_tool_inputs_are_trimmed_for_display():
    out = agent._short({"file_path": "a.py", "content": "x" * 1000})
    assert out["file_path"] == "a.py" and len(out["content"]) == 301


def test_new_problem_starts_from_the_template_and_becomes_a_workspace_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "WORK_ROOT", tmp_path)
    s = agent._session(None, "new")
    assert s.problem == agent.NEW_ID and (s.workspace / "ADDING_PROBLEMS.md").is_file()
    entry = agent.workspace_entry(s)
    assert entry["category"] == "workspace" and entry["session"] == s.id
    assert [f["name"] for f in entry["files"]] == agent.CODE_FILES and entry["params"]
    assert set(agent._edited_files(s)) >= {"meta.yaml", "main.py"}  # a new problem hands back all its files


def test_running_an_expired_workspace_fails_cleanly():
    import asyncio

    import pytest

    async def run():
        async for _ in agent.run_workspace("no-such-session", {}, None, False, "test"):
            pass

    with pytest.raises(agent.RunError, match="expired"):
        asyncio.run(run())


def test_context_mirrors_the_editor_and_describes_the_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "WORK_ROOT", tmp_path)
    s = agent._session(None, "mechanics/simple_pendulum")
    (s.workspace / "solve.py").write_text("# an old agent edit the user reset\n", encoding="utf-8")
    context = {
        "title": "Simple pendulum",
        "files": {"plot.py": "# edited in the browser\n"},
        "params": {"g": 3.7},
        "selection": {"file": "equation.py", "from_line": 25, "to_line": 27, "text": "def angular_acceleration"},
    }
    message = agent._with_context(s, "why sin?", context)
    assert (s.workspace / "plot.py").read_text(encoding="utf-8") == "# edited in the browser\n"
    original = agent.problem_dir("mechanics", "simple_pendulum") / "solve.py"
    assert (s.workspace / "solve.py").read_text(encoding="utf-8") == original.read_text(encoding="utf-8")
    assert '"Simple pendulum"' in message and '"g": 3.7' in message and "equation.py lines 25–27" in message
    assert message.endswith("why sin?")
    assert agent._with_context(s, "hi", {}) == "hi"


def test_a_highlight_elsewhere_on_the_page_is_described_by_its_region(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "WORK_ROOT", tmp_path)
    s = agent._session(None, "mechanics/simple_pendulum")
    text = "以 (V, p) 取樣張量，功是 torch.trapezoid 求得的各曲線下面積"
    message = agent._with_context(s, "what does this mean?", {"selection": {"region": "summary", "text": text}})
    assert f"They highlighted this in the problem's summary: «{text}»" in message
    assert message.endswith("what does this mean?")
    long = agent._with_context(s, "?", {"selection": {"region": "<b>notes</b>", "text": "x" * 5000}})
    assert "problem's bnotesb:" in long and "x" * 2000 + "»" in long and "x" * 2001 not in long


def test_a_screenshot_lands_in_the_workspace_context_folder(tmp_path, monkeypatch):
    import base64

    import pytest

    monkeypatch.setattr(agent, "WORK_ROOT", tmp_path)
    s = agent._session(None, "mechanics/simple_pendulum")
    png = b"\x89PNG\r\n\x1a\n" + b"\0" * 64
    data = "data:image/png;base64," + base64.b64encode(png).decode()
    message = agent._with_context(s, "look", {"screenshot": data})
    shot = s.workspace / "_context" / "screenshot-1.png"
    assert shot.read_bytes() == png and agent._inside(str(shot), s.workspace)
    assert "_context/screenshot-1.png" in message and "Read tool" in message
    agent._with_context(s, "again", {"screenshot": data})
    assert (s.workspace / "_context" / "screenshot-2.png").is_file()
    assert agent._edited_files(s) == {}  # attachments are not problem files
    with pytest.raises(agent.ContextError, match="PNG"):
        agent._with_context(s, "x", {"screenshot": base64.b64encode(b"GIF89a").decode()})
    with pytest.raises(agent.ContextError, match="too large"):
        agent._with_context(s, "x", {"screenshot": "A" * (5 * 1024 * 1024)})
