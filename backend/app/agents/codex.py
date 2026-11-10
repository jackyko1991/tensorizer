"""OpenAI Codex CLI adapter, verified end to end with codex-cli 0.144.1.

Command (flags as documented at https://learn.chatgpt.com/docs/non-interactive-mode and
https://learn.chatgpt.com/docs/developer-commands?surface=cli, checked against codex-rs/exec/src/cli.rs):

    codex exec --json --cd <workspace> --sandbox workspace-write --skip-git-repo-check
               --ignore-user-config --ignore-rules
               -c features.shell_tool=false -c web_search="disabled"
               -c approval_policy="never" -c sandbox_workspace_write.network_access=false
               -c mcp_servers.tensorizer.url="<mcp url with token>"
               -c mcp_servers.tensorizer.required=true
               -c mcp_servers.tensorizer.default_tools_approval_mode="approve" [-m MODEL] [resume <thread id>] -

`-` reads the prompt from stdin. workspace-write lets Codex write only inside --cd (plus temp dirs) and keeps the
network off. Shell and web tools are disabled; scoped MCP tools handle reads, writes and renders. User config
and command rules are ignored, while the existing login is retained, like Claude's setting_sources=[].
The required MCP connection fails startup rather than silently losing render access. Only this server's tools
are pre-approved for a noninteractive turn. Events (`--json`, codex-rs/exec/src/exec_events.rs):
thread.started {thread_id}, turn.started, item.started / item.updated / item.completed {item: {id, type, ...}},
turn.completed {usage}, turn.failed {error: {message}}, error {message}. Item types used here: agent_message {text},
command_execution {command, aggregated_output, exit_code, status}, file_change {changes: [{path, kind}], status},
mcp_tool_call {server, tool, arguments, result, error, status}; reasoning and others are not shown.
"""

from __future__ import annotations

import json

from .cli import RUN_TOOL, tool_input, tool_name

LABEL = "Codex"


def command(exe: str, workspace: str, model: str | None, resume: str | None, mcp_url: str) -> list[str]:
    argv = [
        exe,
        "exec",
        "--json",
        "--cd",
        workspace,
        "--sandbox",
        "workspace-write",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "-c",
        "features.shell_tool=false",
        "-c",
        'web_search="disabled"',
        "-c",
        'approval_policy="never"',
        "-c",
        "sandbox_workspace_write.network_access=false",
        "-c",
        f"mcp_servers.tensorizer.url={json.dumps(mcp_url)}",
        "-c",
        "mcp_servers.tensorizer.required=true",
        "-c",
        'mcp_servers.tensorizer.default_tools_approval_mode="approve"',
    ]
    if model:
        argv += ["-m", model]
    if resume:
        argv += ["resume", resume]
    return [*argv, "-"]


class Mapper:
    """Codex JSONL events → panel events. `session` is the thread id, for `resume` on the next turn."""

    def __init__(self) -> None:
        self.session: str | None = None
        self._texts = 0
        self._shown: set[str] = set()

    def __call__(self, e: dict) -> list[dict]:
        kind = e.get("type")
        if kind == "thread.started":
            self.session = e.get("thread_id") or self.session
            return []
        if kind in ("turn.failed", "error"):
            err = e.get("error") if kind == "turn.failed" else e
            return [{"type": "error", "message": str((err or {}).get("message") or "Codex failed")}]
        if kind not in ("item.started", "item.completed"):
            return []
        item = e.get("item") or {}
        itype, iid = item.get("type"), str(item.get("id", ""))
        started = kind == "item.started"
        if itype == "agent_message" and not started:
            text = str(item.get("text", ""))
            self._texts += 1
            return [{"type": "text", "delta": ("\n\n" if self._texts > 1 else "") + text}]
        if itype == "command_execution":
            if started:
                return [
                    {"type": "tool", "id": iid, "name": "Bash", "input": tool_input({"command": item.get("command")})}
                ]
            code = item.get("exit_code")
            ok = code == 0 if code is not None else item.get("status") == "completed"
            return [{"type": "tool_result", "id": iid, "ok": ok, "text": str(item.get("aggregated_output", ""))[:2000]}]
        if itype == "file_change":
            paths = [c.get("path", "") for c in item.get("changes") or [] if isinstance(c, dict)]
            out = []
            if iid not in self._shown:  # a patch may only be reported once it is applied
                self._shown.add(iid)
                out.append({"type": "tool", "id": iid, "name": "Edit", "input": {"file_path": ", ".join(paths)}})
            if not started:
                out.append({"type": "tool_result", "id": iid, "ok": item.get("status") == "completed", "text": ""})
            return out
        if itype == "mcp_tool_call":
            name = RUN_TOOL if item.get("tool") == "run_problem" else f"mcp__{item.get('server')}__{item.get('tool')}"
            if item.get("server") == "tensorizer":
                name = tool_name(str(item.get("tool", "")))
            if started:
                return [{"type": "tool", "id": iid, "name": name, "input": tool_input(item.get("arguments"))}]
            err = item.get("error")
            result = item.get("result") or {}
            text = "\n".join(c.get("text", "") for c in result.get("content") or [] if isinstance(c, dict))
            return [
                {
                    "type": "tool_result",
                    "id": iid,
                    "ok": not err and not result.get("isError") and item.get("status") != "failed",
                    "text": (err or {}).get("message", "") if err else text[:2000],
                }
            ]
        return []
