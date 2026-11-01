"""Start the API and the Vite dev server together; Ctrl+C stops both.

pixi run dev        →  open http://localhost:5173

- The API restarts when backend/ or problems/_common/ change. This script watches those files itself
  instead of using uvicorn --reload, whose reloader can hang on Windows and keep serving old code.
- Vite updates the page live (React Fast Refresh, plus manifest updates when problems/ change).
- If either server crashes it is restarted (after printing its exit code).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NPM = "npm.cmd" if os.name == "nt" else "npm"
API_SOURCES = [ROOT / "backend", ROOT / "problems" / "_common"]

SERVERS = {
    "api": dict(
        args=[sys.executable, "-m", "uvicorn", "backend.app.main:app", "--port", "8000"],
        cwd=ROOT,
        env={
            **os.environ,
            "ALLOW_CODE_EXEC": "1",
            "AGENT_ENABLED": "1",
            "AGENT_AUTH": "local",
            "TENSORIZER_DEV_SITE": "http://localhost:5173",
        },
    ),
    "web": dict(args=[NPM, "run", "dev"], cwd=ROOT / "frontend", env=None),
}


def start(name: str) -> subprocess.Popen:
    spec = SERVERS[name]
    return subprocess.Popen(spec["args"], cwd=spec["cwd"], env=spec["env"])


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        if os.name == "nt":  # take the whole process tree down, including a render in progress
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        else:
            proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def api_snapshot() -> dict[Path, float]:
    return {f: f.stat().st_mtime for d in API_SOURCES for f in d.rglob("*.py")}


def main() -> None:
    # Make sure the code views are current before the first page load (rendering stays incremental).
    subprocess.run([sys.executable, "scripts/build_assets.py", "--manifest-only"], cwd=ROOT, check=True)

    procs = {name: start(name) for name in SERVERS}
    snapshot = api_snapshot()
    print("\n  Tensorizer dev: http://localhost:5173  (API on :8000)  — Ctrl+C to stop\n", flush=True)

    try:
        while True:
            time.sleep(1)
            current = api_snapshot()
            if current != snapshot:
                snapshot = current
                print("\n[dev] backend code changed — restarting the API\n", flush=True)
                stop(procs["api"])
                procs["api"] = start("api")
                continue
            for name, proc in procs.items():
                if proc.poll() is not None:
                    print(f"\n[dev] {name} server exited with code {proc.returncode} — restarting in 2 s\n", flush=True)
                    time.sleep(2)
                    procs[name] = start(name)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in procs.values():
            stop(proc)


if __name__ == "__main__":
    main()
