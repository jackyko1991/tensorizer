"""Build the concrete tensors: the bundled MNIST subset and the synthetic Swiss roll (a 2D sheet rolled up in 3D, so
its true flat coordinates are known), plus the bundled umap-learn reference embedding.

data/mnist_subset.npz holds `images` (2500, 28, 28) uint8 and `labels` (2500,) uint8: the first 250 images of every
digit in the MNIST training file, in file order. build_subset() made it once from the raw MNIST files, read with
gzip and numpy only (the raw folder is untrusted data, nothing in it is executed).
"""

import csv
import gzip
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).parent / "data"
SUBSET = DATA / "mnist_subset.npz"
REF_NPZ = DATA / "umap_learn_reference.npz"
REF_CSV = DATA / "umap_learn_reference.csv"
PER_CLASS, CLASSES = 250, 10
N_ROLL, ROLL_SEED, ROLL_NOISE = 1500, 0, 0.05
MD5 = {  # the published checksums of the raw files (verified before use)
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
}


def _idx(path: Path) -> np.ndarray:
    """Read one IDX file (big-endian header: magic, dims; then uint8 data) with gzip + numpy only."""
    raw = gzip.decompress(path.read_bytes())
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def build_subset(raw_dir: Path, out: Path = SUBSET) -> Path:
    """One-off: the first PER_CLASS training images of every digit, in file order."""
    import hashlib

    for name, md5 in MD5.items():
        assert hashlib.md5((raw_dir / name).read_bytes()).hexdigest() == md5, f"checksum mismatch: {name}"
    x, y = _idx(raw_dir / "train-images-idx3-ubyte.gz"), _idx(raw_dir / "train-labels-idx1-ubyte.gz")
    keep = np.sort(np.concatenate([np.flatnonzero(y == c)[:PER_CLASS] for c in range(CLASSES)]))
    np.savez_compressed(out, images=x[keep], labels=y[keep])
    return out


def mnist(device: torch.device, n: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """X (N, 784) pixels in [0, 1] (each image flattened to one row), labels (N,) int64. n keeps an evenly spread,
    still balanced subset (rows are in file order, digits interleaved)."""
    with np.load(SUBSET) as d:
        x = torch.from_numpy(d["images"].reshape(len(d["images"]), -1).astype(np.float32) / 255.0)
        y = torch.from_numpy(d["labels"].astype(np.int64))
    if n is not None and n < len(y):
        keep = torch.cat([torch.nonzero(y == c).flatten()[: n // CLASSES] for c in range(CLASSES)]).sort().values
        x, y = x[keep], y[keep]
    return x.to(device), y.to(device)


def swiss_roll_numpy(n: int = N_ROLL, seed: int = ROLL_SEED, noise: float = ROLL_NOISE) -> tuple:
    """The Swiss roll as in scikit-learn's make_swiss_roll, with numpy's PCG64 generator (stable across numpy
    versions, so the offline umap-learn reference sees exactly these points): roll angle t ∈ [1.5π, 4.5π],
    height h ∈ [0, 21]; x = t cos t, y = h, z = t sin t, plus Gaussian noise. Returns X (n, 3), t (n,), h (n,)."""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    h = 21.0 * rng.random(n)
    x = np.stack([t * np.cos(t), h, t * np.sin(t)], axis=1) + noise * rng.standard_normal((n, 3))
    return x.astype(np.float32), t.astype(np.float32), h.astype(np.float32)


def swiss_roll(device: torch.device, n: int = N_ROLL) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """X (n, 3) points in space; U (n, 2) their true flat coordinates (arc length along the spiral, height);
    t (n,) roll angle (for colouring). The arc length s(t) = ∫ √(1 + t²) dt makes U an isometric unrolling."""
    x, t, h = (torch.from_numpy(a) for a in swiss_roll_numpy(n))
    s = 0.5 * (t * torch.sqrt(1 + t**2) + torch.asinh(t))  # arc length of the spiral r = t
    return x.to(device), torch.stack([s, h], dim=1).to(device), t.to(device)


def reference() -> dict | None:
    """The offline umap-learn run (data/make_reference.py): embeddings (N, 2) and its metric rows, or None."""
    if not REF_NPZ.is_file():
        return None
    with np.load(REF_NPZ) as d:
        out = {k: d[k] for k in d.files}
    with open(REF_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    out["rows"] = rows
    return out
