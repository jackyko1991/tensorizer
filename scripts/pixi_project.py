"""Package one problem as a standalone pixi project (zip) that runs on any machine with pixi.

    python scripts/pixi_project.py mechanics/projectile out.zip

build_assets.py writes one per problem next to its GIF; the web app's "Download pixi project"
patches the user's edited files and parameters into that zip before saving it.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "problems"
REPO_URL = "https://github.com/jackyko1991/tensorizer"
FILES = ["meta.yaml", "main.py", "equation.py", "tensors.py", "solve.py", "plot.py", "diagram.svg"]
FIXED_TIME = (2026, 1, 1, 0, 0, 0)  # constant zip timestamps → identical bytes for identical content


def files(problem: Path) -> dict[str, str]:
    """Relative path → text for every file of the project."""
    cat, slug = problem.parent.name, problem.name
    meta = yaml.safe_load((problem / "meta.yaml").read_text(encoding="utf-8"))
    module = f"problems.{cat}.{slug}.main"
    defaults = {k: v["default"] for k, v in meta.get("params", {}).items()}

    out = {
        "pixi.toml": f"""[workspace]
name = "tensorizer-{slug.replace("_", "-")}"
description = "{meta["title"].replace('"', "'")} (from Tensorizer, {REPO_URL})"
channels = ["conda-forge"]
platforms = ["win-64", "osx-arm64", "linux-64"]

[dependencies]
python = "3.12.*"
pytorch = ">=2.4"
numpy = ">=1.26"
matplotlib = ">=3.8"
pillow = ">=10"
pyyaml = ">=6"
tqdm = ">=4.66"

[tasks]
# Render the result into out/ using the values in params.json
simulate = "python -m {module} --out out --params @params.json"
# Same, tracing every step with tensor shapes, timings and progress bars
verbose = "python -m {module} --out out --params @params.json --verbose"
""",
        "params.json": json.dumps(defaults, indent=2, ensure_ascii=False) + "\n",
        "README.md": f"""# {meta["title"]}

A standalone [Tensorizer]({REPO_URL}) problem ({cat}, {meta["level"]}).

{" ".join(str(meta.get("summary", "")).split())}

## Run

1. Install [pixi](https://pixi.sh) (one command, no admin rights needed).
2. In this folder:

   ```bash
   pixi run simulate   # renders out/{slug}.{"gif" if meta.get("animated", True) else "png"}
   pixi run verbose    # the same, printing every step with tensor shapes and timings
   ```

The first run downloads Python, PyTorch and matplotlib into `.pixi/` (a few minutes). Change values in
`params.json`, or edit the code in `problems/{cat}/{slug}/`, and run again. With an NVIDIA GPU and a CUDA build
of PyTorch the code uses the GPU automatically.

| File | Role |
|---|---|
| `main.py` | main loop: parameters → tensors → solve → plot |
| `equation.py` | physics → tensor translation |
| `tensors.py` | builds the PyTorch tensors |
| `solve.py` | the solver |
| `plot.py` | renders the GIF/PNG |
| `meta.yaml` | equations, parameters and their ranges |

Licensed under the Apache License 2.0 — see {REPO_URL}.
""",
        "problems/__init__.py": "",
        f"problems/{cat}/__init__.py": "",
    }
    for f in sorted((PROBLEMS / "_common").glob("*.py")):
        out[f"problems/_common/{f.name}"] = f.read_text(encoding="utf-8")
    out[f"problems/{cat}/{slug}/__init__.py"] = ""
    for name in FILES:
        if (problem / name).is_file():
            out[f"problems/{cat}/{slug}/{name}"] = (problem / name).read_text(encoding="utf-8")
    # Extra inputs a problem reads, e.g. a measurements CSV (translations and caches are not needed to run it).
    for f in sorted(problem.rglob("*")):
        rel = f.relative_to(problem).as_posix()
        if f.is_file() and f.name not in FILES and not rel.startswith(("i18n/", "__pycache__/")) and f.suffix != ".pyc":
            out[f"problems/{cat}/{slug}/{rel}"] = f.read_text(encoding="utf-8")
    return out


def zip_bytes(problem: Path) -> bytes:
    """The project as a zip under a top-level folder named after the problem (deterministic bytes)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for path, text in files(problem).items():
            info = zipfile.ZipInfo(f"{problem.name}/{path}", date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, text.replace("\r\n", "\n"))
    return buffer.getvalue()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python scripts/pixi_project.py <category>/<slug> <out.zip>")
    Path(sys.argv[2]).write_bytes(zip_bytes(PROBLEMS / sys.argv[1]))
    print(f"wrote {sys.argv[2]}")
