"""Start the local runner for the website and say, once it is listening, which port is ready.

    python -m tensorizer_physics.serve [--port 8000] [--host 127.0.0.1]

`pixi run api` runs this (extra arguments pass through: `pixi run api --port 8765`). The website connects from
Backend > Local with just the port number.
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


def banner(port: int, devices: str) -> str:
    line = "─" * 64
    return "\n".join(
        [
            line,
            f"  Tensorizer runner is ready on port {port}",
            f"  address : http://localhost:{port}",
            f"  devices : {devices}",
            f"  website : {SITE}  →  Backend → Local → port {port}",
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
            print(banner(self._port, _devices()), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tensorizer local runner")
    parser.add_argument("--port", type=int, default=int(os.environ.get("TENSORIZER_PORT", "8000")))
    parser.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only (the default)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # the banner's arrows on Windows consoles
    config = uvicorn.Config("backend.app.main:app", host=args.host, port=args.port, log_level="info")
    _Server(config, args.port).run()


if __name__ == "__main__":
    main()
