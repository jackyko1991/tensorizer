"""The generated Colab/Jupyter notebooks contain the real problem files and their run cell works."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import notebooks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def cells(nb: dict, tag: str) -> list[str]:
    return ["".join(c["source"]) for c in nb["cells"] if c["metadata"].get("tensorizer") == tag]


def test_notebook_embeds_the_problem_files():
    problem = ROOT / "problems/mechanics/projectile"
    nb = notebooks.build(problem)
    for name in notebooks.FILES:
        (source,) = cells(nb, f"file:{name}")
        header, _, code = source.partition("\n")
        assert header == f"%%writefile problems/mechanics/projectile/{name}"
        assert code == (problem / name).read_text(encoding="utf-8")


def test_run_cell_executes(monkeypatch, tmp_path):
    shown = []
    monkeypatch.setenv("TENSORIZER_FAST", "1")
    monkeypatch.chdir(ROOT)

    nb = notebooks.build(ROOT / "problems/mechanics/projectile")
    (run,) = cells(nb, "run")
    # IPython isn't installed here: replace its display helpers with stand-ins.
    run = run.replace("from IPython.display import Image, display\n", "")
    run = run.replace('Path("out")', f"Path({str(tmp_path)!r})")
    exec(compile(run, "run-cell", "exec"), {"sys": sys, "Image": lambda filename: filename, "display": shown.append})
    assert len(shown) == 1 and Path(shown[0]).suffix == ".gif" and Path(shown[0]).is_file()


@pytest.mark.skipif(not (ROOT / "notebooks").is_dir(), reason="notebooks not generated")
def test_committed_notebooks_are_up_to_date():
    for problem in notebooks.problems():
        f = ROOT / "notebooks" / problem.parent.name / f"{problem.name}.ipynb"
        assert f.is_file(), f"missing {f.relative_to(ROOT)} — run `pixi run notebooks`"
        assert f.read_text(encoding="utf-8") == notebooks.dumps(notebooks.build(problem)), (
            f"{f.relative_to(ROOT)} is out of date — run `pixi run notebooks`"
        )
