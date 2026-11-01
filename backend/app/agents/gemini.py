"""Gemini CLI adapter (experimental: built from the docs, not tested against a live install here).

Command (https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/headless.md, docs/cli/cli-reference.md):

    gemini --output-format stream-json --approval-mode auto_edit [-m MODEL] [--resume latest]     (prompt on stdin)

Piped stdin makes it non-interactive. auto_edit approves file edits only; anything else that needs confirmation
(the shell) can't be approved in headless mode. On top of that the run gets its own settings file through
GEMINI_CLI_SYSTEM_SETTINGS_PATH (system settings override user/project ones, docs/reference/configuration.md):
tools.core limits the built-in tools to file reading/writing/searching, tools.exclude drops the shell and web tools,
and mcpServers.tensorizer.httpUrl (trust: true) adds run_problem. Each chat has its own workspace folder, and Gemini
keeps sessions per project folder, so `--resume latest` continues this chat.

stream-json events (packages/core/src/output/types.ts): init {session_id, model}, message {role, content, delta},
tool_use {tool_name, tool_id, parameters}, tool_result {tool_id, status, output, error}, error {severity, message},
result {status, error, stats}.
"""

from __future__ import annotations

import json
from pathlib import Path

from .cli import tool_input, tool_name

LABEL = "Gemini CLI"
CORE_TOOLS = ["read_file", "read_many_files", "write_file", "replace", "glob", "grep_search", "list_directory"]
EXCLUDED = ["run_shell_command", "web_fetch", "google_web_search"]


def command(exe: str, model: str | None, resume: bool) -> list[str]:
    argv = [exe, "--output-format", "stream-json", "--approval-mode", "auto_edit"]
    if model:
        argv += ["-m", model]
    if resume:
        argv += ["--resume", "latest"]
    return argv


def settings(mcp_url: str) -> dict:
    return {
        "tools": {"core": CORE_TOOLS, "exclude": EXCLUDED},
        "mcpServers": {"tensorizer": {"httpUrl": mcp_url, "trust": True}},
    }


def env(settings_file: Path, mcp_url: str) -> dict[str, str]:
    settings_file.write_text(json.dumps(settings(mcp_url), indent=2), encoding="utf-8")
    return {"GEMINI_CLI_SYSTEM_SETTINGS_PATH": str(settings_file)}


class Mapper:
    def __init__(self) -> None:
        self.session: str | None = None
        self._last = "start"  # what came before: text runs are joined, separated from text after a tool

    def __call__(self, e: dict) -> list[dict]:
        kind = e.get("type")
        if kind == "init":
            self.session = e.get("session_id") or self.session
            return []
        if kind == "message" and e.get("role") == "assistant":
            text = str(e.get("content", ""))
            if not text:
                return []
            sep = "\n\n" if self._last == "tool" else ""
            self._last = "text"
            return [{"type": "text", "delta": sep + text}]
        if kind == "tool_use":
            self._last = "tool"
            return [
                {
                    "type": "tool",
                    "id": str(e.get("tool_id", "")),
                    "name": tool_name(str(e.get("tool_name", ""))),
                    "input": tool_input(e.get("parameters")),
                }
            ]
        if kind == "tool_result":
            ok = e.get("status") == "success"
            text = e.get("output") if ok else (e.get("error") or {}).get("message")
            return [{"type": "tool_result", "id": str(e.get("tool_id", "")), "ok": ok, "text": str(text or "")[:2000]}]
        if kind == "error" and e.get("severity") != "warning":
            return [{"type": "error", "message": str(e.get("message") or "Gemini CLI failed")}]
        if kind == "result" and e.get("status") == "error":
            err = e.get("error") or {}
            return [{"type": "error", "message": str(err.get("message") if isinstance(err, dict) else err)}]
        return []
