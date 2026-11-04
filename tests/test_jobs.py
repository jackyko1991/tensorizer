"""Detached jobs: runs that keep going after the page closes, replayable event logs, ownership, limits, retention."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from backend.app import limits
from backend.app import main as api
from tensorizer_physics.jobs import JobManager

ME = {"X-Visitor": "me-0123456789abcdef"}
OTHER = {"X-Visitor": "other-0123456789abcdef"}
QUICK = {"category": "mechanics", "slug": "projectile", "params": {"n_angles": 2, "t_max": 1}}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr("backend.app.runner.CACHE", tmp_path)
    monkeypatch.setattr(api, "CACHE", tmp_path)
    monkeypatch.setattr(api, "JOBS", JobManager(tmp_path / "_jobs"))
    with TestClient(api.app) as c:  # one event loop for the whole test, so background jobs keep running
        yield c


def _wait(client, job_id, headers=ME, timeout=120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}", headers=headers).json()
        if job["status"] in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def _events(client, job_id, start=0, headers=ME) -> list[dict]:
    with client.stream("GET", f"/api/jobs/{job_id}/events?from={start}", headers=headers) as r:
        assert r.status_code == 200
        return [json.loads(line) for line in r.iter_lines() if line]


def _forever(beat) -> dict:
    """Edited main.py that heartbeats into a file until it is killed."""
    code = "\n".join(
        [
            "import time",
            "print('start', flush=True)",
            "while True:",
            f"    open({str(beat)!r}, 'w').write(str(time.time()))",
            "    time.sleep(0.05)",
        ]
    )
    return {**QUICK, "files": {"main.py": code}}


def _until(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError("condition not reached")


def test_health_lists_the_jobs_feature(client):
    assert "jobs" in client.get("/api/health").json()["features"]


def test_job_keeps_running_after_the_client_disconnects(client):
    job_id = client.post("/api/jobs", json=QUICK, headers=ME).json()["job_id"]
    with client.stream("GET", f"/api/jobs/{job_id}/events", headers=ME) as r:
        first = json.loads(next(line for line in r.iter_lines() if line))  # one event, then the page closes
    assert first["seq"] == 0
    job = _wait(client, job_id)
    assert job["status"] == "done", job
    assert job["finished"] >= job["created"] and job["result"]["cached"] is False
    assert client.get(job["result"]["output"]).status_code == 200
    events = _events(client, job_id)
    assert events[-1]["type"] == "result" and events[-1]["output"] == job["result"]["output"]
    assert [e["seq"] for e in events] == list(range(len(events))) and job["events"] == len(events)

    # the same run again is served from the result cache: the job completes at once
    again = client.post("/api/jobs", json=QUICK, headers=ME).json()["job_id"]
    assert _wait(client, again)["result"]["cached"] is True


def test_events_replay_from_n(client):
    job_id = client.post("/api/jobs", json=QUICK, headers=ME).json()["job_id"]
    _wait(client, job_id)
    every = _events(client, job_id)
    assert len(every) > 4
    assert _events(client, job_id, start=3) == every[3:]
    assert _events(client, job_id, start=len(every)) == []


def test_visitors_see_only_their_own_jobs(client):
    job_id = client.post("/api/jobs", json=QUICK, headers=ME).json()["job_id"]
    for path in (f"/api/jobs/{job_id}", f"/api/jobs/{job_id}/events"):
        assert client.get(path, headers=OTHER).status_code == 404
    assert client.post(f"/api/jobs/{job_id}/cancel", headers=OTHER).status_code == 404
    assert client.get("/api/jobs", headers=OTHER).json()["jobs"] == []
    assert [j["id"] for j in client.get("/api/jobs", headers=ME).json()["jobs"]] == [job_id]
    assert client.get("/api/jobs").status_code == 400  # no visitor id
    assert client.post("/api/jobs", json=QUICK, headers={"X-Visitor": "short"}).status_code == 400
    _wait(client, job_id)
    stored = (api.JOBS.root / job_id / "job.json").read_text(encoding="utf-8")
    assert ME["X-Visitor"] not in stored  # only the hash is kept


def test_cancel_kills_the_process(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "ALLOW_CODE_EXEC", True)
    beat = tmp_path / "beat.txt"
    job_id = client.post("/api/jobs", json=_forever(beat), headers=ME).json()["job_id"]
    _until(beat.exists)
    assert client.get(f"/api/jobs/{job_id}", headers=ME).json()["status"] == "running"
    assert client.post(f"/api/jobs/{job_id}/cancel", headers=ME).json()["cancelled"] is True
    job = _wait(client, job_id, timeout=10)
    assert job["status"] == "cancelled"
    assert _events(client, job_id)[-1]["type"] == "error"
    time.sleep(0.5)
    last = beat.read_text()
    time.sleep(0.5)
    assert beat.read_text() == last  # no more heartbeats: the process was killed
    assert client.post(f"/api/jobs/{job_id}/cancel", headers=ME).json()["cancelled"] is False


def test_one_job_at_a_time_per_visitor(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "ALLOW_CODE_EXEC", True)
    monkeypatch.setattr(limits, "PER_CLIENT", 1)
    beat = tmp_path / "beat.txt"
    job_id = client.post("/api/jobs", json=_forever(beat), headers=ME).json()["job_id"]
    second = client.post("/api/jobs", json=QUICK, headers=ME)
    assert second.status_code == 429 and "already" in second.json()["detail"]
    client.post(f"/api/jobs/{job_id}/cancel", headers=ME)
    _wait(client, job_id, timeout=10)
    assert client.post("/api/jobs", json=QUICK, headers=ME).status_code == 200  # free again


def test_finished_jobs_expire_after_the_ttl(client, monkeypatch):
    monkeypatch.setattr(limits, "JOB_TTL_S", 60)
    job_id = client.post("/api/jobs", json=QUICK, headers=ME).json()["job_id"]
    _wait(client, job_id)
    run_id = api.JOBS.jobs[job_id].result["run_id"]
    assert run_id in api.JOBS.protected()  # its output is kept by the cache pruning while the job lives
    api.JOBS.jobs[job_id].finished -= 120  # ended two minutes ago
    assert client.get("/api/jobs", headers=ME).json()["jobs"] == []
    assert client.get(f"/api/jobs/{job_id}", headers=ME).status_code == 404
    assert not (api.JOBS.root / job_id).exists()


def test_cache_pruning_skips_the_job_store_and_job_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(limits, "CACHE_MAX_BYTES", 0)
    for name in ("_jobs", "kept", "old"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "out.gif").write_bytes(b"x" * 100)
    monkeypatch.setattr(limits, "PRUNE_PROTECT", [lambda: {"kept"}])
    assert limits.prune_cache(tmp_path) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["_jobs", "kept"]


def test_restart_marks_running_jobs_failed(tmp_path):
    root = tmp_path / "_jobs"
    for job_id, status in (("a1", "running"), ("b2", "queued"), ("c3", "done")):
        (root / job_id).mkdir(parents=True)
        record = {"id": job_id, "visitor": "v", "category": "mechanics", "slug": "projectile", "status": status}
        (root / job_id / "job.json").write_text(json.dumps(record), encoding="utf-8")
    (root / "a1" / "events.ndjson").write_text('{"type": "line", "text": "hi", "seq": 0}\n', encoding="utf-8")
    jobs = JobManager(root)
    assert {j.id: j.status for j in jobs.jobs.values()} == {"a1": "failed", "b2": "failed", "c3": "done"}
    assert jobs.jobs["a1"].error == "server restarted" and jobs.jobs["a1"].events == 2
    assert json.loads((root / "a1" / "job.json").read_text(encoding="utf-8"))["status"] == "failed"
    last = (root / "a1" / "events.ndjson").read_text(encoding="utf-8").splitlines()[-1]
    assert json.loads(last) == {"type": "error", "message": "server restarted", "console": "", "seq": 1}
    assert JobManager(root).jobs["a1"].status == "failed"  # and it stays so on the next start


def test_run_stream_still_works(client):
    with client.stream("POST", "/api/run-stream/mechanics/projectile", json={"params": QUICK["params"]}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert events[-1]["type"] == "result"
