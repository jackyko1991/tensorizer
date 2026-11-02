"""The local runner as one zip, for people who don't use git: download, unzip, `pixi run api`.

    python scripts/runner_bundle.py [--out dist]

writes <out>/tensorizer-runner.zip with a tensorizer-runner/ folder: pixi.toml, pixi.lock, LICENSE, NOTICE, the
backend and the problems (code, tests, meta.yaml, pictures), plus a short README. The release workflow
(.github/workflows/runner-release.yml) publishes it at a fixed URL:

    https://github.com/jackyko1991/tensorizer/releases/download/runner-latest/tensorizer-runner.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = "tensorizer-runner"
FILES = ["pixi.toml", "pixi.lock", "LICENSE", "NOTICE", "pyproject.toml"]
DIRS = ["backend", "problems"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache"}

README = """# Tensorizer local runner

Runs Tensorizer problems on this computer for the website (https://jackyko1991.github.io/tensorizer/).

1. Install pixi once: https://pixi.sh
2. In this folder, start the runner:

   pixi run api            # CPU (and Apple GPU via MPS on Macs)
   pixi run -e cuda api    # NVIDIA GPU (Windows, Linux)

3. On the website open Backend > Local: it connects to http://localhost:8000 by itself.

The runner executes Python code on your computer, including code you or the AI edit. Only run code you trust.
"""


def members() -> list[Path]:
    paths = [ROOT / f for f in FILES if (ROOT / f).is_file()]
    for d in DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and not SKIP_DIRS & set(p.relative_to(ROOT).parts) and p.suffix != ".pyc":
                paths.append(p)
    return paths


def build(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    target = out / "tensorizer-runner.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{TOP}/README.md", README)
        for p in members():
            z.write(p, f"{TOP}/{p.relative_to(ROOT).as_posix()}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "dist"))
    path = build(Path(parser.parse_args().out))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
