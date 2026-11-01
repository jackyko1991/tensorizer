"""Claude Code adapter: the Claude Agent SDK driving the Claude Code CLI on one problem's workspace.

Read / Write / Edit / Glob / Grep confined to the workspace folder, plus `run_problem` as an in-process MCP tool.
Used for the LOCAL mode (this machine's own Claude Code login) and the API mode (the visitor's Anthropic key, or an
OpenRouter key through OpenRouter's Anthropic-compatible endpoint), which differ only in `env`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path


def _inside(path: str, root: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def short(data: dict) -> dict:
    """Tool inputs for display: long strings (whole files) trimmed."""
    return {k: (v[:300] + "…" if isinstance(v, str) and len(v) > 300 else v) for k, v in data.items()}


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return json.dumps(content)[:2000]


async def run(
    s,
    message: str,
    *,
    model: str,
    env: dict[str, str],
    system_prompt: str,
    max_turns: int,
    run_tool: Callable[[dict], Awaitable[tuple[str, bool]]],
    sink: asyncio.Queue,
    secrets: list[str],
    login_hint: str,
) -> AsyncIterator[dict]:
    """One turn. Yields text, tool, tool_result, run (via `sink`), done {cost_usd} or error."""
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ResultMessage,
        TextBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
        create_sdk_mcp_server,
        tool,
    )
    from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny, StreamEvent

    @tool(
        "run_problem",
        "Render the problem with the current files (and optional parameter overrides). "
        "Returns the terminal output and whether an animation/plot was produced.",
        {"type": "object", "properties": {"params": {"type": "object"}}},
    )
    async def run_problem(args: dict) -> dict:
        text, is_error = await run_tool(args.get("params") or {})
        return {"content": [{"type": "text", "text": text}], **({"is_error": True} if is_error else {})}

    async def can_use_tool(name: str, data: dict, _context) -> PermissionResultAllow | PermissionResultDeny:
        path = data.get("file_path") or data.get("path") or data.get("notebook_path")
        if path is not None and not _inside(path, s.workspace):
            return PermissionResultDeny(message="Only the files of this problem can be read or changed.")
        return PermissionResultAllow(updated_input=data)

    options = ClaudeAgentOptions(
        cwd=str(s.workspace),
        tools=["Read", "Write", "Edit", "Glob", "Grep"],
        disallowed_tools=["Bash", "WebFetch", "WebSearch", "Task"],
        allowed_tools=["mcp__tensorizer__run_problem"],
        mcp_servers={"tensorizer": create_sdk_mcp_server("tensorizer", tools=[run_problem])},
        can_use_tool=can_use_tool,
        permission_mode="default",
        system_prompt={"type": "preset", "preset": "claude_code", "append": system_prompt},
        model=model,
        max_turns=max_turns,
        include_partial_messages=True,
        setting_sources=[],
        resume=s.claude_session,
        env=env,
    )

    def scrub(text: str) -> str:
        for secret in secrets:
            if secret:
                text = text.replace(secret, "***")
        return text

    async def drive() -> None:
        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(message)
                async for m in client.receive_response():
                    await sink.put(("sdk", m))
        except Exception as e:  # noqa: BLE001 - shown to the user, never contains the key
            text = scrub(str(e))
            if "logged in" in text.lower() or "auth" in text.lower():
                text += login_hint
            await sink.put({"type": "error", "message": text[:500]})
        finally:
            await sink.put(None)

    task = asyncio.create_task(drive())
    try:
        while (item := await sink.get()) is not None:
            if isinstance(item, dict):
                yield item
                continue
            m = item[1]
            if isinstance(m, StreamEvent):
                ev = m.event
                if ev.get("type") == "content_block_delta" and ev.get("delta", {}).get("type") == "text_delta":
                    yield {"type": "text", "delta": ev["delta"]["text"]}
            elif isinstance(m, AssistantMessage):
                for b in m.content:
                    if isinstance(b, ToolUseBlock):
                        yield {"type": "tool", "id": b.id, "name": b.name, "input": short(b.input)}
                    elif isinstance(b, TextBlock) and not options.include_partial_messages:
                        yield {"type": "text", "delta": b.text}
            elif isinstance(m, UserMessage) and isinstance(m.content, list):
                for b in m.content:
                    if isinstance(b, ToolResultBlock):
                        yield {
                            "type": "tool_result",
                            "id": b.tool_use_id,
                            "ok": not b.is_error,
                            "text": text_of(b.content)[:2000],
                        }
            elif isinstance(m, ResultMessage):
                s.claude_session = m.session_id
                yield {"type": "done", "cost_usd": m.total_cost_usd}
    finally:
        task.cancel()  # the visitor pressed stop or closed the tab: end Claude Code too
