"""describe() of bundled .npz files: image datasets get kind "images" with shape and class counts;
others stay "array"."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_assets import describe  # noqa: E402


def test_images_and_labels(tmp_path):
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (12, 5, 4), dtype=np.uint8)
    labels = np.array([0, 1, 2] * 4, dtype=np.int64)
    labels[0] = 2
    f = tmp_path / "digits.npz"
    np.savez(f, images=images, labels=labels)
    entry = describe(f, "data/digits.npz", "ai/x/data/digits.npz")
    assert entry["kind"] == "images"
    assert entry["rows"] == 12
    (s,) = entry["images"]
    got = (s["name"], s["count"], s["height"], s["width"], s["channels"], s["dtype"])
    assert got == ("images", 12, 5, 4, 1, "uint8")
    assert s["labels"] == "labels"
    assert s["classes"] == {"0": 3, "1": 4, "2": 5}


def test_train_test_splits_compressed(tmp_path):
    f = tmp_path / "mnist_subset.npz"
    np.savez_compressed(
        f,
        x_train=np.zeros((6, 3, 3), np.uint8),
        y_train=np.arange(6, dtype=np.uint8) % 2,
        x_test=np.zeros((2, 3, 3), np.uint8),
        y_test=np.array([7, 7], np.uint8),
        idx_train=np.arange(6, dtype=np.int32),
    )
    entry = describe(f, "data/mnist_subset.npz", "u")
    assert entry["kind"] == "images"
    sets = {s["name"]: s for s in entry["images"]}
    assert set(sets) == {"x_train", "x_test"}
    assert sets["x_train"]["labels"] == "y_train" and sets["x_train"]["classes"] == {"0": 3, "1": 3}
    assert sets["x_test"]["classes"] == {"7": 2}


def test_images_without_labels(tmp_path):
    f = tmp_path / "frames.npz"
    np.savez(f, images=np.zeros((3, 2, 2), np.uint8), labels=np.zeros(5, np.int32))  # wrong length: not paired
    (s,) = describe(f, "data/frames.npz", "u")["images"]
    assert s["labels"] is None and s["classes"] == {}


def test_non_image_npz_falls_back_to_array(tmp_path):
    f = tmp_path / "field.npz"
    np.savez(f, u=np.zeros((4, 4), np.float64), t=np.linspace(0, 1, 5))
    entry = describe(f, "data/field.npz", "u")
    assert entry["kind"] == "array" and "images" not in entry
    bad = tmp_path / "broken.npz"
    bad.write_bytes(b"not a zip")
    assert describe(bad, "data/broken.npz", "u")["kind"] == "array"
