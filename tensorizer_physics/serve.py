"""Start the local runner for the website and say, once it is listening, which port is ready.

    python -m tensorizer_physics.serve [--port 8765] [--host 127.0.0.1]

`pixi run api` runs this on port 8000 (extra arguments pass through: `pixi run api --port 8765`). The website
connects from Backend > Local with just the port number, or, for a server elsewhere, from Backend > Remote with its
URL and pairing token.

Pairing token: set TENSORIZER_TOKEN (or TENSORIZER_TOKEN_SHA256) and every route except /api/health needs it
(tensorizer_physics.auth). Without a token, a server listening on a network address (--host 0.0.0.0) never runs
edited or user-written code (limits.code_exec_policy).
"""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn

SITE = "https://jackyko1991.github.io/tensorizer/"


def _devices() -> str:
    import torch

    found = ["cpu"]
    if torch.cuda.is_available():
        found.append(f"cuda ({torch.cuda.get_device_name(0)})")
    if torch.backends.mps.is_available():
        found.append("mps (Apple GPU)")
    return ", ".join(found)


def banner(port: int, devices: str, auth: str = "none", code: bool = False, host: str = "127.0.0.1") -> str:
    line = "─" * 64
    local = host in ("127.0.0.1", "localhost", "::1")
    where = f"Backend → Local → port {port}" if auth == "none" else "Backend → Remote → your servers"
    return "\n".join(
        [
            line,
            f"  Tensorizer runner is ready on port {port}",
            f"  address : http://localhost:{port}" + ("" if local else f" (listening on {host})"),
            f"  devices : {devices}",
            f"  token   : {'required (paste it in the website)' if auth == 'token' else 'none'}",
            f"  code    : {'edited and Workspace code may run' if code else 'library problems only'}",
            f"  website : {SITE}  →  {where}",
            "  stop    : Ctrl+C",
            line,
        ]
    )


class _Server(uvicorn.Server):
    """uvicorn.Server that prints the banner after the socket is listening (not before)."""

    def __init__(self, config: uvicorn.Config, port: int):
        super().__init__(config)
        self._port = port

    async def startup(self, sockets=None) -> None:
        await super().startup(sockets)
        if self.started:
            from backend.app import main as app_module

            auth = "token" if app_module.AUTH.enabled else "none"
            code = app_module.ALLOW_CODE_EXEC
            print(banner(self._port, _devices(), auth, code, self.config.host), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tensorizer local runner")
    parser.add_argument("--port", type=int, default=int(os.environ.get("TENSORIZER_PORT", "8765")))
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only (the default)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # the banner's arrows on Windows consoles
    # the code-execution policy needs to know whether other machines can reach this server (limits.BIND)
    os.environ["TENSORIZER_BIND"] = args.host
    has_token = os.environ.get("TENSORIZER_TOKEN") or os.environ.get("TENSORIZER_TOKEN_SHA256")
    if not has_token and args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"warning: listening on {args.host} without a pairing token: anyone who reaches this port can run library"
            " problems; edited and Workspace code stay off. Set TENSORIZER_TOKEN on your own server.",
            file=sys.stderr,
            flush=True,
        )
    config = uvicorn.Config("backend.app.main:app", host=args.host, port=args.port, log_level="info")
    _Server(config, args.port).run()


if __name__ == "__main__":
    main()
