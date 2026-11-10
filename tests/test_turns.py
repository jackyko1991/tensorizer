"""Detached agent turns: a chat turn outlives the page, its log rebuilds the chat, Stop ends it, History lists it."""

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from backend.app import agent, routes_agents, turns
from backend.app import main as api

ME = {"X-Visitor": "me-0123456789abcdef"}
OTHER = {"X-Visitor": "other-0123456789abcdef"}
CHAT = {"problem": "mechanics/projectile", "message": "explain the drag term", "agent": "claude-code"}


class FakeAgent:
    """agent.chat's shape: a session event, some text, then done; `hold` keeps it running until released."""

    def __init__(self):
        self.started = 0
        self.closed = 0
        self.release = None
        self.sessions = []

    def chat(self, session_id, problem, message, *args, **kwargs):
        fake = self

        async def stream():
            fake.started += 1
            fake.sessions.append(session_id)
            try:
                yield {"type": "session", "id": session_id or "s" * 32}
                for word in ("The ", "drag ", "term."):
                    yield {"type": "text", "delta": word}
                if message.startswith("hold"):
                    fake.loop, fake.release = asyncio.get_running_loop(), asyncio.Event()
                    await fake.release.wait()
                yield {"type": "tool", "id": "t1", "name": "run_problem", "input": {}}
                yield {"type": "done", "cost_usd": None}
            finally:
                fake.closed += 1  # the agent process would be killed here

        return stream()


@pytest.fixture
def fake(monkeypatch, tmp_path):
    f = FakeAgent()
    monkeypatch.setattr(agent, "ENABLED", True)
    monkeypatch.setattr(agent, "chat", f.chat)
    monkeypatch.setattr(routes_agents, "TURNS", turns.TurnManager(tmp_path / "_turns"))
    return f


@pytest.fixture
def client(fake):
    with TestClient(api.app) as c:  # one event loop for the whole test, so background turns keep running
        yield c


def _until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached")


def _events(client, turn_id, start=0, headers=ME) -> list[dict]:
    with client.stream("GET", f"/api/agents/turns/{turn_id}/events?from={start}", headers=headers) as r:
        assert r.status_code == 200
        return [json.loads(line) for line in r.iter_lines() if line]


def _start(client, body=CHAT, headers=ME) -> str:
    with client.stream("POST", "/api/agents/chat", json=body, headers=headers) as r:
        first = json.loads(next(line for line in r.iter_lines() if line))  # one event, then the page closes
    assert first["type"] == "turn" and first["message"] == body["message"]
    return first["id"]


def test_a_turn_keeps_running_after_the_page_closes(client, fake):
    turn_id = _start(client, {**CHAT, "message": "hold on"})
    _until(lambda: fake.release is not None)
    assert client.get(f"/api/agents/turns/{turn_id}", headers=ME).json()["status"] == "running"
    assert fake.closed == 0  # disconnecting did not stop the agent
    fake.loop.call_soon_threadsafe(fake.release.set)  # the agent finishes while nobody watches
    _until(lambda: client.get(f"/api/agents/turns/{turn_id}", headers=ME).json()["status"] == "done")
    events = _events(client, turn_id)
    assert [e["type"] for e in events] == ["turn", "session", "text", "tool", "done"]
    assert events[2]["delta"] == "The drag term."  # deltas merged in the log
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert _events(client, turn_id, start=3) == events[3:]


def test_stop_cancels_the_agent(client, fake):
    turn_id = _start(client, {**CHAT, "message": "hold on"})
    _until(lambda: fake.release is not None)
    assert client.post(f"/api/agents/turns/{turn_id}/cancel", headers=ME).json()["cancelled"] is True
    _until(lambda: client.get(f"/api/agents/turns/{turn_id}", headers=ME).json()["status"] == "cancelled")
    assert fake.closed == 1
    assert _events(client, turn_id)[-1]["type"] == "error"


def test_conversations_group_turns_by_session_and_stay_private(client, fake):
    first = _start(client)
    _until(lambda: client.get(f"/api/agents/turns/{first}", headers=ME).json()["status"] == "done")
    second = _start(client, {**CHAT, "message": "and lift?", "session": "s" * 32})
    _until(lambda: client.get(f"/api/agents/turns/{second}", headers=ME).json()["status"] == "done")
    other = _start(client, {**CHAT, "message": "new question", "session": "t" * 32})
    _until(lambda: client.get(f"/api/agents/turns/{other}", headers=ME).json()["status"] == "done")

    convs = client.get("/api/agents/conversations?problem=mechanics/projectile", headers=ME).json()["conversations"]
    assert [c["turns"] for c in convs] == [[other], [first, second]]
    assert convs[1]["title"] == "explain the drag term" and convs[1]["session"] == "s" * 32

    assert client.get("/api/agents/conversations", headers=OTHER).json()["conversations"] == []
    assert client.get(f"/api/agents/turns/{first}/events", headers=OTHER).status_code == 404
    assert client.post(f"/api/agents/turns/{first}/cancel", headers=OTHER).status_code == 404
    assert client.get("/api/agents/conversations").status_code == 400  # no visitor id

    assert client.delete(f"/api/agents/conversations/{convs[1]['id']}", headers=ME).json() == {"deleted": 2}
    left = client.get("/api/agents/conversations", headers=ME).json()["conversations"]
    assert [c["turns"] for c in left] == [[other]]
    stored = (routes_agents.TURNS.root / other / "turn.json").read_text(encoding="utf-8")
    assert ME["X-Visitor"] not in stored  # only the hash is kept


def test_a_restart_marks_running_turns_interrupted(tmp_path):
    root = tmp_path / "_turns"

    async def scenario():
        manager = turns.TurnManager(root)
        gate = asyncio.Event()

        async def stream():
            yield {"type": "session", "id": "a" * 32}
            await gate.wait()

        turn = manager.start("v", "mechanics/projectile", "codex", "hi", stream())
        await asyncio.sleep(0.05)
        manager._close_log(turn)  # the process dies here: nothing else is written
        return turn.id

    turn_id = asyncio.run(scenario())
    again = turns.TurnManager(root)
    turn = again.turns[turn_id]
    assert turn.status == "interrupted" and turn.session == "a" * 32

    async def read():
        return [e async for e in again.events(turn)]

    events = asyncio.run(read())
    assert events[-1] == {"type": "error", "message": turn.error, "interrupted": True, "seq": len(events) - 1}


def test_a_session_is_restored_from_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "WORK_ROOT", tmp_path)
    s = agent._session(None, "mechanics/projectile")
    s.claude_session = "claude-conversation"
    agent._persist(s)
    agent._sessions.clear()  # the backend restarted
    again = agent._session(s.id, "mechanics/projectile")
    assert again.id == s.id and again.workspace == s.workspace and again.claude_session == "claude-conversation"
    agent._sessions.clear()
    assert agent._session(s.id, "mechanics/simple_pendulum").id != s.id  # another problem: a new conversation
    assert agent._restore("../etc") is None


def test_continuing_a_chat_whose_session_is_gone_keeps_it_in_one_conversation(client, fake, monkeypatch):
    first = _start(client)
    _until(lambda: client.get(f"/api/agents/turns/{first}", headers=ME).json()["status"] == "done")
    conv = client.get("/api/agents/conversations", headers=ME).json()["conversations"][0]
    sent = []
    original = fake.chat

    def spy(session_id, problem, message, *args, **kwargs):
        sent.append(message)
        return original("u" * 32, problem, message, *args, **kwargs)  # the agent starts a new session

    monkeypatch.setattr(agent, "chat", spy)
    monkeypatch.setattr(agent, "known_session", lambda s: False)  # its folder was deleted
    body = {**CHAT, "message": "and the lift?", "session": conv["session"], "conversation": conv["id"]}
    second = _start(client, body)
    _until(lambda: client.get(f"/api/agents/turns/{second}", headers=ME).json()["status"] == "done")
    assert "explain the drag term" in sent[0] and "The drag term." in sent[0] and sent[0].endswith("and the lift?")
    convs = client.get("/api/agents/conversations", headers=ME).json()["conversations"]
    assert [c["turns"] for c in convs] == [[first, second]] and convs[0]["session"] == "u" * 32
    # someone else's conversation id is not joined
    stranger = _start(client, {**CHAT, "conversation": conv["id"]}, headers=OTHER)
    _until(lambda: client.get(f"/api/agents/turns/{stranger}", headers=OTHER).json()["status"] == "done")
    assert [c["turns"] for c in client.get("/api/agents/conversations", headers=ME).json()["conversations"]] == [
        [first, second]
    ]


def test_a_saved_conversation_folder_outlives_the_idle_session(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "WORK_ROOT", tmp_path)
    s = agent._session(None, "mechanics/projectile")
    s.last_used -= agent.IDLE_S + 1
    agent._cleanup()
    assert s.id not in agent._sessions and (tmp_path / s.id / agent.SESSION_FILE).is_file()
    assert agent.known_session(s.id) and not agent.known_session("../x")
    assert agent._session(s.id, "mechanics/projectile").workspace == s.workspace  # History continues it


def test_a_long_mcp_render_returns_a_render_id_and_is_collected_later(monkeypatch, tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError

    from backend.app import mcp_server

    async def scenario():
        gate = asyncio.Event()

        async def slow(s, params, client):
            await gate.wait()
            return "Run succeeded. Output: /api/outputs/x/result.gif", False

        monkeypatch.setattr(agent, "run_in_session", slow)
        c = mcp_server.Connector(id="c", client="local")
        s = agent.Session("a" * 32, "mechanics/projectile", tmp_path)
        task = asyncio.get_running_loop().create_task(mcp_server._render(c, s, {}, "call"))
        first = await mcp_server._collect(c, task, 0.05)
        assert "still running" in first and len(c.renders) == 1
        render_id = next(iter(c.renders))
        again = await mcp_server._collect(c, c.renders[render_id], 0.05, render_id)
        assert "still running" in again  # waiting timed out; the render goes on
        gate.set()
        done = await mcp_server._collect(c, c.renders[render_id], 1.0, render_id)
        assert done.startswith("Run succeeded") and c.renders == {}

        stopped = asyncio.get_running_loop().create_task(mcp_server._render(c, s, {}, "call2"))
        gate.clear()
        await mcp_server._collect(c, stopped, 0.01)
        stopped.cancel()
        with pytest.raises(ToolError):
            await mcp_server._collect(c, stopped, 0.5, next(iter(c.renders)))

    asyncio.run(scenario())


def test_ask_user_puts_a_question_in_the_chat_and_ends_the_wait(tmp_path):
    s = agent.Session("a" * 32, "mechanics/projectile", tmp_path)
    s.sink = asyncio.Queue()
    text, is_error = agent.ask_in_session(s, " Which drag model? ", ["linear", "quadratic", "quadratic", ""], False)
    assert not is_error and "End your turn" in text
    event = s.sink.get_nowait()
    assert event["type"] == "question" and event["question"] == "Which drag model?"
    assert event["options"] == ["linear", "quadratic"] and event["multi"] is False and len(event["id"]) == 8
    assert agent.ask_in_session(s, "Only one?", ["yes"], False)[1] is True  # needs 2+ options
    assert agent.ask_in_session(s, "", ["a", "b"], False)[1] is True
    many = agent.ask_in_session(s, "q" * 999, [f"o{i}" * 50 for i in range(9)], True)
    event = s.sink.get_nowait()
    assert not many[1] and len(event["question"]) == agent.MAX_QUESTION and len(event["options"]) == 6
    assert all(len(o) <= agent.MAX_OPTION for o in event["options"]) and event["multi"] is True
