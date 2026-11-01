"""Build a Jupyter notebook for each problem (opens in Colab or any Jupyter).

    python scripts/notebooks.py            # write notebooks/<category>/<slug>.ipynb for every problem
    python scripts/notebooks.py --check    # exit 1 if any committed notebook is out of date (used by CI)

Each notebook is self-contained: it clones the repository for the shared helpers (problems/_common),
writes the problem's five files from editable %%writefile cells, then runs it and shows the animation.
Colab opens notebooks straight from GitHub, which is why they are committed under notebooks/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "problems"
OUT = ROOT / "notebooks"
REPO = "jackyko1991/tensorizer"
SITE = "https://jackyko1991.github.io/tensorizer/"
FILES = {
    "main.py": "main loop",
    "equation.py": "physics → tensor translation",
    "tensors.py": "PyTorch tensors",
    "solve.py": "solver",
    "plot.py": "2D/3D plot → GIF",
}


def _cell(kind: str, source: str, tag: str | None = None) -> dict:
    cell = {
        "cell_type": kind,
        "id": hashlib.sha1(f"{kind}:{tag}:{source}".encode()).hexdigest()[:8],
        "metadata": {"tensorizer": tag} if tag else {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def params_source(package: str, params: dict) -> str:
    """The run cell; `params` holds the values to run with (the problem defaults unless edited)."""
    lines = ",\n".join(f"    {name!r}: {value!r}" for name, value in params.items())
    return f"""import importlib
from pathlib import Path

from IPython.display import Image, display

from problems._common.runtime import load_meta, resolve_params

params = {{
{lines}
}}

# Re-import so edits in the %%writefile cells above take effect.
for name in [m for m in sys.modules if m.startswith("{package}")]:
    del sys.modules[name]
main = importlib.import_module("{package}.main")
problem_dir = Path("{package.replace(".", "/")}")
output = main.run(resolve_params(load_meta(problem_dir), params), Path("out"))
display(Image(filename=str(output)))
"""


def build(problem: Path) -> dict:
    cat, slug = problem.parent.name, problem.name
    meta = yaml.safe_load((problem / "meta.yaml").read_text(encoding="utf-8"))
    package = f"problems.{cat}.{slug}"

    rows = "\n".join(
        f"| ${r['symbol'].replace('|', chr(92) + 'vert ')}$ | `{r['code'].replace('|', chr(92) + '|')}` "
        f"| {r['shape']} | {r['meaning'].replace('|', '/')} |"
        for r in meta.get("translation", [])
    )
    intro = f"""# {meta["title"]}

*{cat} · {meta["level"]}* · [open in Tensorizer]({SITE}#/{cat}/{slug})

{" ".join(str(meta.get("summary", "")).split())}

**Vector form**

$${meta["vector_form"]}$$

**Index (tensor) form**

$${meta["index_form"]}$$

| Symbol | Code | Shape | Meaning |
|---|---|---|---|
{rows}

{" ".join(str(meta.get("notes", "")).split())}
"""
    setup = f"""import os
import subprocess
import sys

# Shared helpers (problems/_common) come from the repository; skip the clone if we are already inside it.
if not os.path.isdir("problems"):
    if not os.path.isdir("tensorizer"):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/{REPO}.git"], check=True)
    os.chdir("tensorizer")
sys.path.insert(0, os.getcwd())
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "tqdm"], check=True)

import torch

print("torch", torch.__version__, "· device:", "cuda" if torch.cuda.is_available() else "cpu")
"""
    cells = [
        _cell("markdown", intro),
        _cell(
            "markdown",
            "## Setup\n\nOn Colab, choose **Runtime → Change runtime type → GPU** to run on a GPU — "
            "the code uses it automatically.",
        ),
        _cell("code", setup, "setup"),
    ]
    for i, (name, role) in enumerate(FILES.items(), 1):
        code = (problem / name).read_text(encoding="utf-8")
        cells.append(_cell("markdown", f"## {i}. `{name}` — {role}"))
        cells.append(_cell("code", f"%%writefile {package.replace('.', '/')}/{name}\n{code}", f"file:{name}"))
    defaults = {k: v["default"] for k, v in meta.get("params", {}).items()}
    cells.append(_cell("markdown", "## Run\n\nChange the parameters (or any file above) and run again."))
    cells.append(_cell("code", params_source(package, defaults), "run"))

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "name": f"{slug}.ipynb"},
            "tensorizer": {"problem": f"{cat}/{slug}", "package": package},
        },
        "cells": cells,
    }


def dumps(notebook: dict) -> str:
    return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


def hidden(meta_file: Path) -> bool:
    """`hidden: true` in meta.yaml keeps a problem in the repository (code and tests) but off the site."""
    return bool(yaml.safe_load(meta_file.read_text(encoding="utf-8")).get("hidden"))


def problems() -> list[Path]:
    return sorted(
        p.parent
        for p in PROBLEMS.glob("*/*/meta.yaml")
        if all((p.parent / f).is_file() for f in FILES) and not p.parent.parent.name.startswith("_") and not hidden(p)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if notebooks/ is out of date")
    args = parser.parse_args()
    expected = {OUT / p.parent.name / f"{p.name}.ipynb": dumps(build(p)) for p in problems()}
    if args.check:
        stale = [
            str(f.relative_to(ROOT)) for f, text in expected.items() if not f.is_file() or f.read_text("utf-8") != text
        ]
        extra = [str(f.relative_to(ROOT)) for f in OUT.glob("*/*.ipynb") if f not in expected]
        if stale or extra:
            print("notebooks out of date — run `pixi run notebooks`:", *stale, *extra, sep="\n  ")
            sys.exit(1)
        print(f"{len(expected)} notebooks up to date")
        return
    for f in OUT.glob("*/*.ipynb"):
        if f not in expected:
            f.unlink()
    for f, text in expected.items():
        f.parent.mkdir(parents=True, exist_ok=True)
        if not f.is_file() or f.read_text("utf-8") != text:
            f.write_text(text, encoding="utf-8")
    print(f"wrote {len(expected)} notebooks to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
