"""Adapters that let the AI Assistant drive different coding agents on one problem's workspace.

Every adapter is an async generator `run(...)` yielding the panel's NDJSON events (text, tool, tool_result, run,
error, done). `agent.chat` adds session/files/problem around them, so all agents look the same to the page.

- claude_code: Claude Code through the Claude Agent SDK (this machine's login, or an API key / OpenRouter).
- codex, opencode, gemini: the OpenAI Codex CLI, OpenCode and the Gemini CLI, run as non-interactive subprocesses
  with JSON output (experimental: implemented from their docs). They reach `run_problem` through Tensorizer's own
  MCP server (mcp_server.py) with a token scoped to the chat's workspace.
- scan: which of these CLIs are installed on this machine (and, where cheap to tell, logged in).
"""
