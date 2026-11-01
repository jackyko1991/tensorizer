"""Shared plumbing every problem uses: device choice, parameters and the CLI entry point.

Each problem's ``main.py`` defines ``run(params, out_dir) -> Path`` and ends with::

    if __name__ == "__main__":
        cli(run, __file__)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import yaml


def device() -> torch.device:
    """Use the GPU when there is one; every problem runs fine on CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fast_mode() -> bool:
    """Tests set TENSORIZER_FAST=1 to render only a handful of frames."""
    return os.environ.get("TENSORIZER_FAST") == "1"


def load_meta(problem_dir: str | Path) -> dict[str, Any]:
    with open(Path(problem_dir) / "meta.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_params(meta: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Defaults from meta.yaml, overridden by the caller, cast and clamped to the declared range."""
    schema = meta.get("params", {})
    overrides = overrides or {}
    unknown = set(overrides) - set(schema)
    if unknown:
        raise ValueError(f"unknown parameter(s): {', '.join(sorted(unknown))}")

    params: dict[str, Any] = {}
    for name, spec in schema.items():
        value = overrides.get(name, spec["default"])
        kind = spec.get("type", "float")
        if kind == "choice":
            if value not in spec["options"]:
                raise ValueError(f"{name} must be one of {spec['options']}")
        else:
            value = int(value) if kind == "int" else float(value)
            if "min" in spec:
                value = max(spec["min"], value)
            if "max" in spec:
                value = min(spec["max"], value)
        params[name] = value
    return params


def cli(run: Callable[[dict[str, Any], Path], Path], main_file: str) -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252 and choke on ° or →
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="out", help="directory for the rendered GIF/PNG")
    parser.add_argument("--params", default="{}", help="JSON overrides, e.g. '{\"v0\": 40}'")
    parser.add_argument("--verbose", action="store_true", help="trace each step with tensor shapes and timings")
    args = parser.parse_args()

    meta = load_meta(Path(main_file).parent)
    params = resolve_params(meta, json.loads(args.params))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.verbose:
        from . import trace

        os.environ["TENSORIZER_VERBOSE"] = "1"  # turns on progress bars (see progress.py)
        print(f"[verbose] torch {torch.__version__} · device {device()} · params {params}", flush=True)
        trace.install(sys.modules[run.__module__])
        run = sys.modules[run.__module__].run
    output = run(params, out_dir)
    print(json.dumps({"output": str(output)}))
