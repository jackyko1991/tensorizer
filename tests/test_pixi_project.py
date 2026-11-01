"""The downloadable pixi project is complete and runs on its own."""

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pixi_project  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_zip_contents_and_determinism():
    problem = ROOT / "problems/mechanics/simple_pendulum"
    data = pixi_project.zip_bytes(problem)
    assert data == pixi_project.zip_bytes(problem)  # identical bytes → no needless rewrites
    names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
    for f in ["pixi.toml", "params.json", "README.md", "problems/_common/runtime.py", "problems/mechanics/__init__.py"]:
        assert f"simple_pendulum/{f}" in names
    for f in ["meta.yaml", "main.py", "equation.py", "tensors.py", "solve.py", "plot.py", "__init__.py"]:
        assert f"simple_pendulum/problems/mechanics/simple_pendulum/{f}" in names


def test_extracted_project_runs(tmp_path):
    zipfile.ZipFile(io.BytesIO(pixi_project.zip_bytes(ROOT / "problems/mechanics/projectile"))).extractall(tmp_path)
    project = tmp_path / "projectile"
    params = json.loads((project / "params.json").read_text())
    params["n_angles"] = 2
    (project / "params.json").write_text(json.dumps(params))
    # what `pixi run simulate` executes (pixi.toml), with the current interpreter instead of pixi's
    task = "python -m problems.mechanics.projectile.main --out out --params @params.json"
    assert f'simulate = "{task}"' in (project / "pixi.toml").read_text()
    env = {**os.environ, "TENSORIZER_FAST": "1", "PYTHONPATH": ""}
    proc = subprocess.run([sys.executable, *task.split()[1:]], cwd=project, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert (project / "out" / "projectile.gif").is_file()
