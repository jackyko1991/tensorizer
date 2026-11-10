"""The Your agent / MCP page reports real contact and can revoke its own connection."""

import pytest
from fastapi.testclient import TestClient

from backend.app import agent, mcp_server
from backend.app.main import app
from tensorizer_physics.auth import needs_token


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "ENABLED", True)
    monkeypatch.setattr(agent, "WORK_ROOT", tmp_path)
    monkeypatch.setattr(mcp_server, "_connectors", {})
    return TestClient(app)


def test_connection_waits_for_real_mcp_contact_and_disconnect_revokes_access(client):
    token = client.post("/api/agents/connector", json={}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    status = client.get("/api/agents/connector", headers=headers)
    assert status.status_code == 200
    assert status.json() == {"connected": False, "last_seen": None}
    r = client.post(
        f"/mcp/{token}",
        headers={
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18",
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
    )
    assert r.status_code == 200
    status = client.get("/api/agents/connector", headers=headers).json()
    assert status["connected"] and isinstance(status["last_seen"], float)
    assert client.delete("/api/agents/connector", headers=headers).status_code == 204
    assert client.get("/api/agents/connector", headers=headers).status_code == 401
    assert client.get(f"/mcp/{token}").status_code == 401


def test_connection_management_requires_a_cloud_token_and_only_revokes_that_token(client):
    first = client.post("/api/agents/connector", json={}).json()["token"]
    second = client.post("/api/agents/connector", json={}).json()["token"]
    local = mcp_server.session_token("some-chat", "test")
    for token in (None, "invalid", local):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        assert client.get("/api/agents/connector", headers=headers).status_code == 401
        assert client.delete("/api/agents/connector", headers=headers).status_code == 401
    assert client.delete("/api/agents/connector", headers={"Authorization": f"Bearer {first}"}).status_code == 204
    assert mcp_server.lookup(first) is None
    assert mcp_server.lookup(second) is not None
    assert mcp_server.lookup(local) is not None


def test_connection_controls_use_the_connector_credential_but_creation_requires_pairing():
    assert not needs_token("GET", "/api/agents/connector")
    assert not needs_token("DELETE", "/api/agents/connector")
    assert needs_token("POST", "/api/agents/connector")


def test_paired_backend_accepts_connector_controls_without_server_token(client, monkeypatch):
    from tensorizer_physics import auth

    monkeypatch.setattr(auth, "AUTH", auth.Auth("test-server-pairing-secret"))
    assert client.post("/api/agents/connector", json={}).status_code == 401
    token = client.post(
        "/api/agents/connector",
        json={},
        headers={
            "Authorization": "Bearer test-server-pairing-secret",
        },
    ).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get("/api/agents/connector", headers=headers).status_code == 200
    assert client.delete("/api/agents/connector", headers=headers).status_code == 204
