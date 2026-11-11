"""Workspace problems: blank create, restore from files, delete, notebook execution and sandbox confinement."""

import json

import pytest
from fastapi.testclient import TestClient

from backend.app import agent, runner, workspace
from backend.app import main as api

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def sandboxes(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ENABLED", True)
    monkeypatch.setattr(workspace, "WS_ROOT", tmp_path / "ws")
    monkeypatch.setattr("backend.app.runner.CACHE", tmp_path / "runs")
    monkeypatch.setattr(api, "CACHE", tmp_path / "runs")
    monkeypatch.setattr(workspace, "_sandboxes", {})
    return tmp_path


def run_events(session: str, params: dict | None = None) -> list[dict]:
    r = client.post(f"/api/ws/{session}/run-stream", json={"params": params or {}})
    assert r.status_code == 200, r.text
    return [json.loads(line) for line in r.text.splitlines() if line]


def files_of(entry: dict) -> dict[str, str]:
    return {**{f["name"]: f["code"] for f in entry["files"]}, "meta.yaml": entry["meta"]}


def test_disabled_without_agent(monkeypatch):
    monkeypatch.setattr(agent, "ENABLED", False)
    assert client.post("/api/ws/blank").status_code == 403


def test_blank_problem_is_a_runnable_boilerplate_with_tests():
    entry = client.post("/api/ws/blank").json()
    assert entry["category"] == "workspace" and entry["title"] == "Untitled problem"
    assert set(entry["params"]) == {"half_life", "t_max"} and entry["vector_form"] and entry["translation"]
    assert [f["name"] for f in entry["files"]] == workspace.CODE_FILES
    assert "def run(params" in files_of(entry)["main.py"]
    assert files_of(entry)["equation.py"].startswith('"""') and "def test_" in files_of(entry)["test.py"]
    result = run_events(entry["session"])[-1]
    assert result["type"] == "result", result
    assert result["tests"]["ok"] and result["tests"]["passed"] == 3  # the boilerplate's own test.py is green
    assert client.get(result["output"]).status_code == 200


def test_copy_from_library_and_unknown_problem():
    entry = client.post("/api/ws/copy", json={"problem": "mechanics/simple_pendulum"}).json()
    assert entry["params"] and "pendulum" in entry["title"].lower() and entry["diagram"].startswith("data:image/svg")
    assert client.post("/api/ws/copy", json={"problem": "mechanics/nope"}).status_code == 404
    assert client.post("/api/ws/copy", json={"problem": "_common/x"}).status_code == 404


def test_sync_writes_files_and_restores_an_expired_sandbox():
    entry = client.post("/api/ws/blank").json()
    files = files_of(entry)
    files["meta.yaml"] = files["meta.yaml"].replace("title: Untitled problem", 'title: "Falling ball"')
    files["main.py"] = files["main.py"].replace("Boilerplate example", "Falling ball")
    same = client.post("/api/ws/sync", json={"session": entry["session"], "files": files}).json()
    assert same["session"] == entry["session"] and same["title"] == "Falling ball"

    # The server forgot it (idle cleanup, restart): the same files come back in a new sandbox.
    workspace._sandboxes.clear()
    assert client.post(f"/api/ws/{entry['session']}/run-stream", json={}).status_code == 410
    restored = client.post("/api/ws/sync", json={"session": entry["session"], "files": files}).json()
    assert restored["session"] != entry["session"] and restored["title"] == "Falling ball"
    assert files_of(restored)["main.py"] == files["main.py"]
    assert run_events(restored["session"])[-1]["type"] == "result"


def test_sync_rejects_bad_input():
    bad = client.post("/api/ws/sync", json={"files": {"../evil.py": "x"}})
    assert bad.status_code == 422
    assert client.post("/api/ws/sync", json={"files": {"meta.yaml": "title: [unclosed"}}).status_code == 422


def test_delete_removes_the_sandbox():
    entry = client.post("/api/ws/blank").json()
    folder = workspace._sandboxes[entry["session"]].root
    assert client.post(f"/api/ws/{entry['session']}/delete").json() == {"ok": True}
    assert not folder.exists() and not workspace.alive(entry["session"])
    assert client.post(f"/api/ws/{entry['session']}/notebook", json={"cells": ["1"]}).status_code == 410


def test_notebook_runs_cells_and_returns_the_last_output_and_figure():
    session = client.post("/api/ws/blank").json()["session"]
    cells = [
        "import numpy as np\nx = np.linspace(0, 1, 5)\nprint('cell one')",
        "import matplotlib.pyplot as plt\nplt.plot(x, x**2)\nprint('sum', x.sum())\nplt.show()",
    ]
    r = client.post(f"/api/ws/{session}/notebook", json={"cells": cells})
    assert r.status_code == 200, r.text
    (out,) = r.json()["cells"]
    assert out["index"] == 1 and out["error"] is None
    assert out["stdout"].strip() == "sum 2.5"  # cell 1's print is not repeated
    assert len(out["figures"]) == 1 and out["figures"][0].startswith("data:image/png;base64,")


def test_notebook_run_all_errors_and_the_problem_package():
    session = client.post("/api/ws/blank").json()["session"]
    cells = ["from problem import main\nmain.__name__", "params", "1 / 0", "print('never')"]
    out = client.post(f"/api/ws/{session}/notebook", json={"cells": cells, "all": True}).json()["cells"]
    assert [c["index"] for c in out] == [0, 1, 2]
    assert out[0]["stdout"].strip().endswith(".main'")
    assert out[1]["stdout"].strip() == "{'half_life': 2.0, 't_max': 10.0}"
    assert "ZeroDivisionError" in out[2]["error"] and "harness" not in out[2]["error"]


def test_notebook_code_runs_inside_its_sandbox(sandboxes):
    entry = client.post("/api/ws/blank").json()
    folder = workspace._sandboxes[entry["session"]].folder
    cells = ["import os\nopen('notes.txt', 'w').write('hi')\nprint(os.getcwd())"]
    out = client.post(f"/api/ws/{entry['session']}/notebook", json={"cells": cells}).json()["cells"][0]
    assert out["stdout"].strip() == str(folder)
    assert (folder / "notes.txt").read_text() == "hi"
    assert folder.resolve().is_relative_to((sandboxes / "ws").resolve())
    assert not (runner.PROBLEMS.parent / "notes.txt").exists()


def test_problems_the_ai_created_resolve_through_the_agent_session(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "WORK_ROOT", tmp_path / "agent")
    s = agent._session(None, "new")
    try:
        files = client.post(f"/api/ws/{s.id}/files").json()["files"]
        assert set(files) == set(workspace.FILES)
        out = client.post(f"/api/ws/{s.id}/notebook", json={"cells": ["print(1 + 1)"]}).json()["cells"][0]
        assert out["stdout"].strip() == "2"
    finally:
        agent._sessions.pop(s.id, None)


def test_data_uploads_are_confined_limited_and_synced(monkeypatch):
    entry = client.post("/api/ws/blank").json()
    sid = entry["session"]
    csv = b"t,x\n0,1.0\n1,0.5\n2,0.25\n"
    r = client.put(f"/api/ws/{sid}/data/measurements.csv", content=csv)
    assert r.status_code == 200 and r.json() == {"name": "measurements.csv", "size": len(csv)}
    folder = workspace._resolve(sid)[0]
    assert (folder / "data" / "measurements.csv").read_bytes() == csv
    for bad in ["..%2Fmeta.yaml", "evil.exe", ".hidden.csv", "a..b.csv"]:
        assert client.put(f"/api/ws/{sid}/data/{bad}", content=b"x").status_code in (404, 405, 422), bad
    assert not (folder.parent / "meta.yaml").exists()
    monkeypatch.setattr(workspace, "MAX_DATA_FILE", 10)
    assert client.put(f"/api/ws/{sid}/data/big.csv", content=b"x" * 11).status_code == 413
    monkeypatch.setattr(workspace, "MAX_DATA_FILE", 1000)
    monkeypatch.setattr(workspace, "MAX_DATA_TOTAL", len(csv) + 5)
    assert client.put(f"/api/ws/{sid}/data/more.csv", content=b"y" * 10).status_code == 422
    # sync lists the data and removes files the browser no longer has
    synced = client.post("/api/ws/sync", json={"session": sid, "files": {}, "data": ["measurements.csv"]}).json()
    assert synced["data"] == [{"name": "measurements.csv", "size": len(csv)}]
    assert client.post("/api/ws/sync", json={"session": sid, "files": {}, "data": []}).json()["data"] == []
    # notebook cells read data/ relative to the problem folder
    client.put(f"/api/ws/{sid}/data/m.csv", content=csv)
    out = client.post(f"/api/ws/{sid}/notebook", json={"cells": ["print(open('data/m.csv').readline().strip())"]})
    assert out.json()["cells"][-1]["stdout"].strip() == "t,x"
    from backend.app import agent as agent_mod

    note = agent_mod.data_note(folder)
    assert "data/m.csv" in note and "t,x" in note and "0,1.0" in note
    assert client.delete(f"/api/ws/{sid}/data/m.csv").status_code == 200
    assert not (folder / "data" / "m.csv").exists()


def test_tags_round_trip_through_meta_yaml():
    entry = client.post("/api/ws/blank").json()
    files = files_of(entry)
    files["meta.yaml"] += "tags: [homework 3, orbits, '']\n"
    synced = client.post("/api/ws/sync", json={"session": entry["session"], "files": files}).json()
    assert synced["tags"] == ["homework 3", "orbits"]  # blank tags dropped
    assert "tags" not in synced["meta_extra"]  # edited by the page, not carried as an unknown key


def test_uploaded_picture_is_kept_and_unsafe_svg_rejected():
    entry = client.post("/api/ws/blank").json()
    files = files_of(entry)
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><circle cx="5" cy="5" r="4"/></svg>'
    synced = client.post("/api/ws/sync", json={"session": entry["session"], "files": {**files, "diagram.svg": svg}})
    assert synced.json()["diagram"].startswith("data:image/svg+xml;base64,")
    workspace._sandboxes.clear()  # a re-created sandbox gets the picture back from the browser's files
    restored = client.post("/api/ws/sync", json={"session": entry["session"], "files": {**files, "diagram.svg": svg}})
    assert restored.json()["diagram"]
    for bad in (
        '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"/>',
        '<svg xmlns="http://www.w3.org/2000/svg"><image href="https://evil.example/x.png"/></svg>',
        "<html/>",
    ):
        r = client.post("/api/ws/sync", json={"files": {**files, "diagram.svg": bad}})
        assert r.status_code == 422, bad


def test_a_workspace_run_is_a_detached_job_listed_under_the_page_id(tmp_path, monkeypatch):
    import time as _time

    from tensorizer_physics.jobs import JobManager

    me = {"X-Visitor": "me-0123456789abcdef"}
    monkeypatch.setattr(api, "JOBS", JobManager(tmp_path / "runs" / "_jobs"))
    with TestClient(api.app) as c:  # one event loop, so the job keeps running between requests
        entry = c.post("/api/ws/blank").json()
        body = {"params": {}, "problem": "workspace/mine-1"}
        started = c.post(f"/api/ws/{entry['session']}/jobs", json=body, headers=me)
        assert started.status_code == 200, started.text
        job_id = started.json()["job_id"]
        assert workspace._sandboxes[entry["session"]].busy == 1  # kept while the job runs
        deadline = _time.monotonic() + 120
        while (job := c.get(f"/api/jobs/{job_id}", headers=me).json())["status"] in ("queued", "running"):
            assert _time.monotonic() < deadline, job
            _time.sleep(0.2)
        assert job["status"] == "done" and job["problem"] == "workspace/mine-1", job
        assert job["result"]["tests"]["ok"] and c.get(job["result"]["output"]).status_code == 200
        assert [j["problem"] for j in c.get("/api/jobs", headers=me).json()["jobs"]] == ["workspace/mine-1"]
        _time.sleep(0.1)
        assert workspace._sandboxes[entry["session"]].busy == 0
        assert c.post("/api/ws/nope/jobs", json=body, headers=me).status_code == 410
        assert c.post(f"/api/ws/{entry['session']}/jobs", json=body).status_code == 400  # no visitor id


def test_export_folder_for_vscode_then_pull_back(tmp_path, monkeypatch):
    """Open in VS Code: a complete pixi project in a stable folder; edits made there come back with pull, and an
    export never silently overwrites edits that were not pulled."""
    monkeypatch.setattr(workspace, "EXPORT_ROOT", tmp_path / "export")
    entry = client.post("/api/ws/blank").json()
    s = entry["session"]
    r = client.post(f"/api/ws/{s}/export-folder", json={"name": "my-decay"})
    assert r.status_code == 200, r.text
    root = tmp_path / "export" / "my-decay"
    problem = root / "problems" / "workspace" / "p_my_decay"
    assert r.json()["path"] == str(root) and (problem / "main.py").is_file() and (problem / "meta.yaml").is_file()
    assert (root / "pixi.toml").is_file() and "p_my_decay/test.py" in (root / "pixi.toml").read_text()
    assert (root / "problems" / "_common").is_dir() and (root / "tensorizer_physics").is_dir()
    assert all((d / "__init__.py").is_file() for d in (root / "problems", problem.parent, problem))

    # the user edits main.py in VS Code: a second export refuses, naming the file; pull brings it back
    (problem / "main.py").write_text("# edited in VS Code\n", encoding="utf-8")
    r = client.post(f"/api/ws/{s}/export-folder", json={"name": "my-decay"})
    assert r.status_code == 409 and r.json()["detail"]["changed"] == ["main.py"]
    pulled = client.post("/api/ws/pull-folder", json={"name": "my-decay"}).json()["files"]
    assert pulled["main.py"] == "# edited in VS Code\n" and "meta.yaml" in pulled
    # once pulled (the page writes the files into the Workspace and syncs), exporting works again
    client.post("/api/ws/sync", json={"session": s, "files": pulled})
    assert client.post(f"/api/ws/{s}/export-folder", json={"name": "my-decay"}).status_code == 200
    # or force overwrites
    (problem / "main.py").write_text("# again\n", encoding="utf-8")
    assert client.post(f"/api/ws/{s}/export-folder", json={"name": "my-decay", "force": True}).status_code == 200
    assert (problem / "main.py").read_text() == pulled["main.py"]


def test_export_folder_rejects_unsafe_names_and_unknown_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "EXPORT_ROOT", tmp_path / "export")
    s = client.post("/api/ws/blank").json()["session"]
    for bad in ["../etc", "a/b", "", "A", ".hidden", "x" * 65]:
        assert client.post(f"/api/ws/{s}/export-folder", json={"name": bad}).status_code == 422, bad
    assert client.post("/api/ws/pull-folder", json={"name": "never-exported"}).status_code == 404
    assert not (tmp_path / "export").exists() or not any((tmp_path / "export").iterdir())
