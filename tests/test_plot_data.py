"""Every saved plot also writes <stem>_plot_data.csv with the numbers it draws (Results page download)."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorizer_physics.core.plotting import export_plot_data, save_png  # noqa: E402


def _rows(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def test_png_writes_lines_scatter_and_3d(tmp_path):
    fig = plt.figure()
    ax = fig.add_subplot(121, title="energy")
    ax.plot([0, 1, 2], [3, 4, 5], label="E")
    ax.scatter([1], [2])
    ax.set_xlabel("t (s)")
    ax3 = fig.add_subplot(122, projection="3d")
    ax3.plot([0, 1], [0, 1], [0, 2])
    save_png(fig, tmp_path / "x.png")
    rows = _rows(tmp_path / "x_plot_data.csv")
    assert [r["series"] for r in rows] == ["E", "E", "E", "points 1", "line 1", "line 1"]
    assert rows[0]["panel"] == "energy"
    assert "# axes of 'energy': x = t (s)" in (tmp_path / "x_plot_data.csv").read_text(encoding="utf-8")
    assert rows[-1]["z"] == "2" and rows[0]["z"] == ""


def test_no_lines_no_csv(tmp_path):
    fig, ax = plt.subplots()
    ax.imshow([[0, 1], [1, 0]])
    assert export_plot_data(fig, tmp_path / "y.png") is None
    plt.close(fig)
    assert not (tmp_path / "y_plot_data.csv").exists()
