"""Every problem package follows the convention and renders an output in fast mode."""

import importlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from problems._common.runtime import load_meta, resolve_params

ROOT = Path(__file__).resolve().parents[1] / "problems"
# CI splits the suite by topic: TENSORIZER_CATEGORIES=waves,fluids runs only those problems.
_ONLY = {c for c in os.environ.get("TENSORIZER_CATEGORIES", "").split(",") if c}
PROBLEMS = sorted(p.parent for p in ROOT.glob("*/*/meta.yaml") if not _ONLY or p.parent.parent.name in _ONLY)
LEVELS = ["beginner", "intermediate", "advanced"]
FILES = ["test.py", "main.py", "equation.py", "tensors.py", "solve.py", "plot.py"]
META_KEYS = {
    "title",
    "category",
    "level",
    "order",
    "summary",
    "animated",
    "vector_form",
    "index_form",
    "translation",
    "params",
}


def ids(p: Path) -> str:
    return f"{p.parent.name}/{p.name}"


@pytest.mark.parametrize("problem", PROBLEMS, ids=ids)
def test_convention(problem: Path):
    for name in FILES:
        assert (problem / name).is_file(), f"missing {name}"
    meta = load_meta(problem)
    assert META_KEYS <= set(meta), f"meta.yaml missing {META_KEYS - set(meta)}"
    assert meta["category"] == problem.parent.name
    assert meta["level"] in LEVELS
    resolve_params(meta)


@pytest.mark.parametrize("problem", PROBLEMS, ids=ids)
def test_has_setup_diagram(problem: Path):
    svg = problem / "diagram.svg"
    assert svg.is_file(), "missing diagram.svg (sketch of the physical setup)"
    root = ET.parse(svg).getroot()
    assert root.get("viewBox") == "0 0 240 160"
    assert "<script" not in svg.read_text(encoding="utf-8")


@pytest.mark.parametrize("problem", PROBLEMS, ids=ids)
def test_renders(problem: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TENSORIZER_FAST", "1")
    module = importlib.import_module(f"problems.{problem.parent.name}.{problem.name}.main")
    meta = load_meta(problem)
    output = module.run(resolve_params(meta), tmp_path)
    assert output.exists() and output.stat().st_size > 0
    assert output.suffix == (".gif" if meta["animated"] else ".png")
