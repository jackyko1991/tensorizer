"""Detached agent turns: a chat turn keeps going when the page closes or reloads, and can be watched again later.

turn = TURNS.start(visitor, problem, agent, message, agent.chat(...))   # → Turn, running in the background
async for event in TURNS.events(turn, start=0): ...                    # replay from event N, then follow live
TURNS.cancel(turn)                                                     # the Stop button (kills the agent process)

The same idea as tensorizer_physics.jobs, for the AI Assistant: the agent's events (text, tool, tool_result, run,
queued, files, notebook, problem, done | error) go to an append-only log on disk instead of to one HTTP response,
so closing the tab only stops *watching*. Each turn lives in <cache>/_turns/<id>/: turn.json (status, session,
problem, the user's message) and events.ndjson. The first event of every log is {"type": "turn", ...} with the
user's message, so a log alone rebuilds the chat.

Conversations: the turns that share an agent session id (the chat's conversation with the agent) form one
conversation; `conversations(visitor)` lists them, newest first, for the panel's History.

Ownership as for jobs: only the hash of the browser's X-Visitor id is stored, and a visitor sees only their own turns.
A server restart marks turns that were running as "interrupted" (the chat says so and the next message continues
the agent's conversation, when its session still exists).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import shutil
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from tensorizer_physics import limits

ACTIVE = ("running",)
FINISHED = ("done", "failed", "cancelled", "interrupted")
RECORD = ("id", "visitor", "session", "problem", "agent", "message", "status", "created", "finished", "error")
RECORD += ("events", "conversation")
RECAP_CHARS = 6000  # how much of an earlier chat is quoted when its agent session can't be resumed
MAX_MESSAGE = 2000  # characters of the user's message kept in turn.json (the log keeps it whole)
MAX_TEXT_EVENT = 4000  # text deltas are merged in the log, up to this many characters per event
TEXT_EVERY_S = 0.2  # ... and written at least this often, so a live answer still streams smoothly


@dataclass(eq=False)
class Turn:
    id: str
    visitor: str  # jobs.visitor_hash(X-Visitor); "" = an anonymous turn (not listed)
    problem: str
    agent: str
    message: str
    session: str | None = None  # the agent session (conversation) id, once the agent reported it
    status: str = "running"
    created: float = field(default_factory=time.time)
    finished: float | None = None
    error: str | None = None
    events: int = 0
    conversation: str | None = None  # the History entry this turn belongs to (stays when the agent session changes)
    # in memory only (live turns)
    task: asyncio.Task | None = None
    cancel_requested: bool = False
    _wake: asyncio.Event | None = None
    _log: object = None
    _text: str = ""  # text deltas waiting to be written as one event
    _text_since: float = 0.0

    def summary(self) -> dict:
        return {k: getattr(self, k) for k in RECORD if k != "visitor"}

    def record(self) -> dict:
        return {k: getattr(self, k) for k in RECORD}


class TurnManager:
    def __init__(self, root: Path):
        self.root = root
        self.turns: dict[str, Turn] = {}
        self._load()

    # --- storage ------------------------------------------------------------------------------------------------

    def _dir(self, turn: Turn) -> Path:
        return self.root / turn.id

    def _save(self, turn: Turn) -> None:
        d = self._dir(turn)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "turn.json.tmp"
        tmp.write_text(json.dumps(turn.record()), encoding="utf-8")
        tmp.replace(d / "turn.json")

    def _load(self) -> None:
        if not self.root.is_dir():
            return
        for d in self.root.iterdir():
            try:
                rec = json.loads((d / "turn.json").read_text(encoding="utf-8"))
                turn = Turn(**{k: rec[k] for k in RECORD if k in rec})
            except (OSError, ValueError, TypeError, KeyError):
                shutil.rmtree(d, ignore_errors=True)
                continue
            turn.events = _count_lines(d / "events.ndjson")
            if turn.status not in FINISHED:
                message = "the server restarted during this answer"
                turn.status, turn.error, turn.finished = "interrupted", message, time.time()
                self._write(turn, {"type": "error", "message": message, "interrupted": True})
                self._close_log(turn)
                self._save(turn)
            self.turns[turn.id] = turn

    def _write(self, turn: Turn, event: dict) -> None:
        if turn._log is None:
            self._dir(turn).mkdir(parents=True, exist_ok=True)
            turn._log = open(self._dir(turn) / "events.ndjson", "a", encoding="utf-8")  # noqa: SIM115
        turn._log.write(json.dumps({**event, "seq": turn.events}) + "\n")
        turn._log.flush()
        turn.events += 1
        if turn._wake is not None:
            old, turn._wake = turn._wake, asyncio.Event()
            old.set()

    def _flush_text(self, turn: Turn) -> None:
        if turn._text:
            text, turn._text = turn._text, ""
            self._write(turn, {"type": "text", "delta": text})

    def _append(self, turn: Turn, event: dict) -> None:
        """Log one agent event. Text deltas (often one word each) are merged for up to TEXT_EVERY_S or
        MAX_TEXT_EVENT characters (and before any other event), so a log is not thousands of lines."""
        if event.get("type") == "text":
            if not turn._text:
                turn._text_since = time.monotonic()
            turn._text += str(event.get("delta", ""))
            if len(turn._text) >= MAX_TEXT_EVENT or time.monotonic() - turn._text_since >= TEXT_EVERY_S:
                self._flush_text(turn)
            return
        self._flush_text(turn)
        self._write(turn, event)

    @staticmethod
    def _close_log(turn: Turn) -> None:
        if turn._log is not None:
            turn._log.close()
            turn._log = None

    # --- lifecycle ----------------------------------------------------------------------------------------------

    def start(
        self,
        visitor: str,
        problem: str,
        agent: str,
        message: str,
        stream: AsyncIterator[dict],
        conversation: str | None = None,
    ) -> Turn:
        """Run `stream` (agent.chat's events) in the background on the running event loop. `conversation`: the
        History entry it continues (by default the agent session it reports)."""
        self.prune()
        turn = Turn(secrets.token_hex(8), visitor, problem, agent, message[:MAX_MESSAGE])
        turn.conversation = conversation if conversation and self._owns(visitor, conversation) else None
        turn._wake = asyncio.Event()
        self.turns[turn.id] = turn
        self._write(turn, {"type": "turn", "id": turn.id, "message": message, "problem": problem, "agent": agent})
        self._save(turn)
        turn.task = asyncio.get_running_loop().create_task(self._run(turn, stream))
        return turn

    async def _run(self, turn: Turn, stream: AsyncIterator[dict]) -> None:
        status, error = "failed", None
        try:
            async for event in stream:
                kind = event.get("type")
                if kind == "session" and event.get("id") != turn.session:
                    turn.session = str(event["id"])
                    turn.conversation = turn.conversation or turn.session
                    self._save(turn)
                elif kind == "done":
                    status = "done"
                elif kind == "error":
                    error = str(event.get("message", ""))[:500]
                self._append(turn, event)
            if status != "done" and error is None:
                error = "the agent stopped without an answer"
                self._append(turn, {"type": "error", "message": error})
        except asyncio.CancelledError:  # the Stop button, or the server shutting down
            status = "cancelled" if turn.cancel_requested else "interrupted"
            error = "stopped" if turn.cancel_requested else "the server stopped during this answer"
            self._append(turn, {"type": "error", "message": error, "interrupted": status == "interrupted"})
        except Exception as e:  # noqa: BLE001 - a turn must always end with a status
            error = f"internal error: {e}"[:500]
            self._append(turn, {"type": "error", "message": error})
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()  # runs agent.chat's cleanup (kills the CLI, frees the session) at once
        self._flush_text(turn)
        turn.status, turn.error, turn.finished = status, error if status != "done" else None, time.time()
        self._close_log(turn)
        self._save(turn)
        if turn._wake is not None:  # followers see the end
            old, turn._wake = turn._wake, asyncio.Event()
            old.set()

    def cancel(self, turn: Turn) -> bool:
        if turn.status not in ACTIVE or turn.task is None or turn.task.done():
            return False
        turn.cancel_requested = True
        turn.task.cancel()
        return True

    # --- reading ------------------------------------------------------------------------------------------------

    def get(self, turn_id: str, visitor: str) -> Turn | None:
        turn = self.turns.get(turn_id)
        return turn if turn is not None and turn.visitor and turn.visitor == visitor else None

    def mine(self, visitor: str, session: str | None = None, problem: str | None = None) -> list[Turn]:
        self.prune()
        turns = [
            t
            for t in self.turns.values()
            if visitor
            and t.visitor == visitor
            and (session is None or t.session == session)
            and (problem is None or t.problem == problem)
        ]
        return sorted(turns, key=lambda t: t.created)

    @staticmethod
    def _key(t: Turn) -> str:
        return t.conversation or t.session or f"turn:{t.id}"

    def _owns(self, visitor: str, conversation: str) -> bool:
        return any(self._key(t) == conversation for t in self.mine(visitor))

    def conversations(self, visitor: str, problem: str | None = None) -> list[dict]:
        """The visitor's conversations (turns grouped by History entry), newest activity first."""
        groups: dict[str, list[Turn]] = {}
        for t in self.mine(visitor, problem=problem):
            groups.setdefault(self._key(t), []).append(t)
        out = []
        for key, turns in groups.items():
            first, last = turns[0], turns[-1]
            out.append(
                {
                    "id": key,
                    "session": last.session,
                    "problem": first.problem,
                    "agent": last.agent,
                    "title": first.message[:120],
                    "turns": [t.id for t in turns],
                    "created": first.created,
                    "updated": last.finished or last.created,
                    "status": last.status,
                }
            )
        return sorted(out, key=lambda c: -c["updated"])

    def recap(self, visitor: str, conversation: str) -> str:
        """The conversation so far as plain text (the user's messages and the agent's answers, newest kept), for an
        agent that starts afresh: its earlier session can't be resumed (expired, or another agent)."""
        parts: list[str] = []
        for t in self.mine(visitor):
            if self._key(t) != conversation or t.status == "running":
                continue
            answer = []
            for line in _read_lines(self._dir(t) / "events.ndjson"):
                if line.get("type") == "text":
                    answer.append(str(line.get("delta", "")))
            parts.append(f"User: {t.message}\nAssistant: {''.join(answer).strip() or '(no text answer)'}")
        text = "\n\n".join(parts)
        return text[-RECAP_CHARS:]

    async def events(self, turn: Turn, start: int = 0, poll_s: float = 5.0) -> AsyncIterator[dict]:
        """The turn's events from number `start`, then new ones as they come, until the turn has ended."""
        path = self._dir(turn) / "events.ndjson"
        index, offset, buffer = 0, 0, b""
        while True:
            wake = turn._wake
            ended = turn.status in FINISHED
            if path.is_file():
                with open(path, "rb") as f:
                    f.seek(offset)
                    chunk = f.read()
                offset += len(chunk)
                buffer += chunk
                *lines, buffer = buffer.split(b"\n")
                for line in lines:
                    if index >= start and line:
                        yield json.loads(line)
                    index += 1
            if ended or wake is None:
                return
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(wake.wait(), poll_s)

    def delete(self, turn: Turn) -> None:
        if turn.status in ACTIVE:
            self.cancel(turn)
        self.turns.pop(turn.id, None)
        self._close_log(turn)
        shutil.rmtree(self._dir(turn), ignore_errors=True)

    def prune(self, now: float | None = None) -> int:
        """Delete turns that ended more than JOB_TTL_S ago (the same retention as jobs; 0 = kept)."""
        ttl = limits.JOB_TTL_S
        if not ttl:
            return 0
        now = time.time() if now is None else now
        old = [t for t in self.turns.values() if t.status in FINISHED and t.finished and now - t.finished > ttl]
        for turn in old:
            self.delete(turn)
        return len(old)


def _read_lines(path: Path) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        return []


def _count_lines(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0
