"""OpenCode adapter (experimental: built from the docs and source, not tested against a live install here).

Command (https://opencode.ai/docs/cli/):

    opencode run --format json --dir <workspace> [-m provider/model] [-s <session id>]     (prompt on stdin)

`opencode run` reads stdin when it isn't a terminal (packages/cli/src/run/run.ts: mergeInput(..., readStdin())).
Permissions and the MCP server come from OPENCODE_CONFIG_CONTENT (inline JSON config, https://opencode.ai/docs/config/):
bash and webfetch denied, edits allowed (https://opencode.ai/docs/permissions/); without --auto, anything else that
asks for permission is rejected in a non-interactive run. Tensorizer's run_problem is a remote MCP server
(https://opencode.ai/docs/mcp-servers/).

`--format json` lines are {type, timestamp, sessionID, ...} (packages/cli/src/run/noninteractive.ts):
text {part: {text}}, reasoning, step_start, step_finish {part: {cost, tokens}},
tool_use {part: {id, tool, state: {status: "completed" | "error", input, output | error}}}, error {error: {message}}.
"""

from __future__ import annotations

import json

from .cli import tool_input, tool_name

LABEL = "OpenCode"
PERMISSION = {"edit": "allow", "bash": "deny", "webfetch": "deny"}


def command(exe: str, workspace: str, model: str | None, resume: str | None) -> list[str]:
    argv = [exe, "run", "--format", "json", "--dir", workspace]
    if model:
        argv += ["-m", model]
    if resume:
        argv += ["-s", resume]
    return argv


def env(mcp_url: str) -> dict[str, str]:
    config = {
        "$schema": "https://opencode.ai/config.json",
        "permission": PERMISSION,
        "mcp": {"tensorizer": {"type": "remote", "url": mcp_url, "enabled": True}},
    }
    return {"OPENCODE_CONFIG_CONTENT": json.dumps(config)}


class Mapper:
    def __init__(self) -> None:
        self.session: str | None = None
        self.cost = 0.0
        self._texts = 0

    def __call__(self, e: dict) -> list[dict]:
        self.session = e.get("sessionID") or self.session
        kind, part = e.get("type"), e.get("part") or {}
        if kind == "text":
            text = str(part.get("text", ""))
            if not text:
                return []
            self._texts += 1
            return [{"type": "text", "delta": ("\n\n" if self._texts > 1 else "") + text}]
        if kind == "tool_use":
            state = part.get("state") or {}
            tid = str(part.get("id") or part.get("partID") or "")
            ok = state.get("status") == "completed"
            text = state.get("output") if ok else state.get("error")
            return [
                {
                    "type": "tool",
                    "id": tid,
                    "name": tool_name(str(part.get("tool", ""))),
                    "input": tool_input(state.get("input")),
                },
                {"type": "tool_result", "id": tid, "ok": ok, "text": str(text or "")[:2000]},
            ]
        if kind == "step_finish":
            if isinstance(part.get("cost"), (int, float)):
                self.cost += part["cost"]
            return []
        if kind == "error":
            err = e.get("error") or {}
            msg = err.get("message") or (err.get("data") or {}).get("message") or "OpenCode failed"
            return [{"type": "error", "message": str(msg)}]
        return []
