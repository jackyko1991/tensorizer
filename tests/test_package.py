"""tensorizer_physics: the old import paths are aliases, and copies of the shared code run on their own."""

import io
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import tensorizer_physics
from tensorizer_physics import runner

ROOT = Path(__file__).resolve().parents[1]


def test_old_paths_are_the_same_modules():
    import backend.app.limits
    import backend.app.run_queue
    import backend.app.runner
    import problems._common.plotting
    import problems._common.runtime
    from tensorizer_physics import limits, run_queue
    from tensorizer_physics.core import plotting, runtime

    assert problems._common.runtime is runtime and problems._common.plotting is plotting
    assert backend.app.runner is runner and backend.app.run_queue is run_queue and backend.app.limits is limits
    assert tensorizer_physics.FORMAT_VERSION == 1 and runner.PROBLEMS == ROOT / "problems"


def test_a_sandbox_with_copy_core_runs_a_problem(tmp_path):
    """Workspace and agent sandboxes run from their own folder: the copied shared code must be enough."""
    shutil.copytree(ROOT / "problems" / "mechanics" / "projectile", tmp_path / "problems" / "mechanics" / "projectile")
    for pkg in (tmp_path / "problems", tmp_path / "problems" / "mechanics"):
        (pkg / "__init__.py").touch()
    runner.copy_core(tmp_path)
    out = subprocess.run(
        [sys.executable, "-m", "problems.mechanics.projectile.main", "--out", "out", "--params", '{"t_max": 0.5}'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**runner._env(tmp_path), "TENSORIZER_FAST": "1", "PYTHONPATH": ""},
        timeout=300,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert (tmp_path / "out" / "projectile.gif").is_file()


def test_exported_project_ships_the_package():
    sys.path.insert(0, str(ROOT / "scripts"))
    import pixi_project

    data = pixi_project.zip_bytes(ROOT / "problems" / "mechanics" / "projectile")
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert any(n.endswith("tensorizer_physics/core/runtime.py") for n in names)
    assert any(n.endswith("problems/_common/runtime.py") for n in names)
