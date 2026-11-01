from fastapi.testclient import TestClient

from backend.app import main as api

client = TestClient(api.app)


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_run_renders_and_serves_output(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.app.runner.CACHE", tmp_path)
    monkeypatch.setattr(api, "CACHE", tmp_path)
    r = client.post("/api/run/mechanics/projectile", json={"params": {"n_angles": 2, "t_max": 1}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["params"]["n_angles"] == 2
    assert client.get(body["output"]).status_code == 200
    again = client.post("/api/run/mechanics/projectile", json={"params": {"n_angles": 2, "t_max": 1}})
    assert again.json()["cached"] is True


def test_unknown_problem_and_param():
    assert client.post("/api/run/mechanics/nope", json={}).status_code == 404
    assert client.post("/api/run/_common/x", json={}).status_code == 404
    assert client.post("/api/run/mechanics/projectile", json={"params": {"bogus": 1}}).status_code == 422


def test_edited_code_blocked_unless_enabled(monkeypatch):
    monkeypatch.setattr(api, "ALLOW_CODE_EXEC", False)
    r = client.post("/api/run/mechanics/projectile", json={"files": {"main.py": "print(1)"}})
    assert r.status_code == 403


def test_path_traversal_rejected():
    assert client.get("/api/outputs/abc/..%2F..%2Fpyproject.toml").status_code == 404


def test_static_js_served_as_javascript():
    import mimetypes

    assert mimetypes.guess_type("index.js")[0] == "text/javascript"


def test_cache_key_changes_when_source_changes(tmp_path, monkeypatch):
    from backend.app import runner

    problem = tmp_path / "cat" / "prob"
    problem.mkdir(parents=True)
    (tmp_path / "_common").mkdir()
    (problem / "meta.yaml").write_text("title: x")
    (problem / "main.py").write_text("a = 1")
    monkeypatch.setattr(runner, "PROBLEMS", tmp_path)
    before = runner._source_hash(problem)
    (problem / "main.py").write_text("a = 2")
    assert runner._source_hash(problem) != before


def test_verbose_run_traces_steps(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.app.runner.CACHE", tmp_path)
    monkeypatch.setattr(api, "CACHE", tmp_path)
    r = client.post("/api/run/mechanics/projectile", json={"params": {"n_angles": 2, "t_max": 1}, "verbose": True})
    assert r.status_code == 200, r.text
    console = r.json()["console"]
    assert "→ solve.solve(" in console and "Tensor(" in console
    assert "RK4 steps" in console and "\r" not in console  # tqdm bar collapsed to its final line


def test_run_stream_sends_lines_then_result(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr("backend.app.runner.CACHE", tmp_path)
    monkeypatch.setattr(api, "CACHE", tmp_path)
    body = {"params": {"n_angles": 2, "t_max": 1}, "verbose": True}
    with client.stream("POST", "/api/run-stream/mechanics/projectile", json=body) as r:
        assert r.status_code == 200
        events = [json.loads(line) for line in r.iter_lines() if line]
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "result" and kinds.count("line") > 3
    assert any(e.get("replace") for e in events if e["type"] == "line")  # tqdm redraws stream as replacements
    assert client.get(events[-1]["output"]).status_code == 200


def test_run_stream_reports_failures_as_events(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr("backend.app.runner.CACHE", tmp_path)
    monkeypatch.setattr(api, "ALLOW_CODE_EXEC", True)
    body = {"files": {"main.py": "print('before'); raise SystemExit(3)"}}
    with client.stream("POST", "/api/run-stream/mechanics/projectile", json=body) as r:
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert events[-1]["type"] == "error" and "before" in events[-1]["console"]


def test_public_site_may_call_a_local_backend():
    """Chrome's Private Network Access preflight from the github.io page to a backend on localhost."""
    res = client.options(
        "/api/health",
        headers={
            "Origin": "https://jackyko1991.github.io",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-private-network") == "true"
    assert res.headers.get("access-control-allow-origin") in {"*", "https://jackyko1991.github.io"}


def test_health_reports_the_queue():
    assert set(client.get("/api/health").json()["queue"]) == {"running", "waiting", "concurrent"}
