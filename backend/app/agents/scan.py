"""Find agent CLIs on the machine the backend runs on (only when the user asks: "Search this computer").

The result only says what is there: installed (bool), a version number, logged_in (True / False / None when it
can't be told cheaply, e.g. a login kept in the OS keychain) and whether Tensorizer's adapter is experimental.
Executable paths stay on the server (`executable()`), and no file contents or keys are ever read or returned:
credentials are detected by the file *existing* or an environment variable being *set*.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

VERSION_TIMEOUT_S = 5.0
_VERSION = re.compile(r"\d+\.\d+(?:\.\d+)?(?:[-.][0-9A-Za-z]+)?")


@dataclass(frozen=True)
class Cli:
    id: str
    label: str
    commands: tuple[str, ...]  # names looked up on PATH
    experimental: bool
    docs: str


CLIS = (
    Cli("claude-code", "Claude Code", ("claude",), False, "https://docs.claude.com/en/docs/claude-code"),
    Cli("codex", "OpenAI Codex CLI", ("codex",), True, "https://github.com/openai/codex"),
    Cli("opencode", "OpenCode", ("opencode",), True, "https://opencode.ai/docs/cli/"),
    Cli("gemini", "Gemini CLI", ("gemini",), True, "https://github.com/google-gemini/gemini-cli"),
)
BY_ID = {c.id: c for c in CLIS}

_found: dict[str, str] = {}  # id → executable path, from the last scan (server-side only)


def _bundled_claude() -> str | None:
    """The Claude Code CLI shipped inside claude-agent-sdk."""
    try:
        import claude_agent_sdk
    except ImportError:
        return None
    for p in sorted((Path(claude_agent_sdk.__file__).parent / "_bundled").glob("claude*")):
        if p.is_file():
            return str(p)
    return None


def _which(cli: Cli) -> str | None:
    for name in cli.commands:
        if path := shutil.which(name):
            return path
    return _bundled_claude() if cli.id == "claude-code" else None


def _version(path: str) -> str | None:
    """`<cli> --version`, reduced to the version number (so no paths or banners leak)."""
    try:
        out = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=VERSION_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    m = _VERSION.search(out.stdout or out.stderr or "")
    return m.group(0)[:32] if m else None


def _env(*names: str) -> bool:
    return any(os.environ.get(n) for n in names)


def _home(var: str, default: str) -> Path:
    return Path(os.environ[var]) if os.environ.get(var) else Path.home() / default


def logged_in(cli_id: str) -> bool | None:
    """True: a credentials file exists or an API-key variable is set. None: unknown (may be in a keychain)."""
    if cli_id == "claude-code":
        if (_home("CLAUDE_CONFIG_DIR", ".claude") / ".credentials.json").is_file() or _env("ANTHROPIC_API_KEY"):
            return True
        return None if sys.platform == "darwin" else False  # macOS keeps the login in the Keychain
    if cli_id == "codex":  # ~/.codex/auth.json (CODEX_HOME), unless the login is in the OS keyring
        return True if (_home("CODEX_HOME", ".codex") / "auth.json").is_file() or _env("CODEX_API_KEY") else None
    if cli_id == "opencode":  # written by `opencode auth login`
        data = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
        return True if (data / "opencode" / "auth.json").is_file() else None
    if cli_id == "gemini":
        home = Path(os.environ.get("GEMINI_CLI_HOME") or Path.home()) / ".gemini"  # GEMINI_CLI_HOME: the home root
        if (home / "oauth_creds.json").is_file() or _env("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            return True
        return None
    return None


def scan() -> list[dict]:
    """Every known CLI with its status (JSON-safe: booleans, a version number, labels)."""
    out = []
    for cli in CLIS:
        path = _which(cli)
        if path:
            _found[cli.id] = path
        else:
            _found.pop(cli.id, None)
        out.append(
            {
                "id": cli.id,
                "label": cli.label,
                "installed": path is not None,
                "version": _version(path) if path else None,
                "logged_in": logged_in(cli.id) if path else False,
                "experimental": cli.experimental,
                "docs": cli.docs,
            }
        )
    return out


def executable(cli_id: str) -> str | None:
    """Where the CLI is (from the last scan, else looked up now)."""
    if cli_id in _found and Path(_found[cli_id]).exists():
        return _found[cli_id]
    cli = BY_ID.get(cli_id)
    path = _which(cli) if cli else None
    if path:
        _found[cli_id] = path
    return path
