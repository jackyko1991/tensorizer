"""Make data/mnist_subset.npz from the four original MNIST .gz files (run once, offline):

    python -I data/make_subset.py <folder with the MNIST .gz files> data/mnist_subset.npz

A balanced subset: 300 images per digit from the training set and 100 per digit from the test set, drawn without
replacement with numpy's default_rng(0). Only numpy and gzip are used; the IDX headers are checked.
"""

import gzip
import sys
from pathlib import Path

import numpy as np


def idx(path: Path, magic: int) -> np.ndarray:
    raw = gzip.decompress(path.read_bytes())
    head = np.frombuffer(raw[:4], ">u4")[0]
    assert head == magic, f"{path.name}: bad IDX magic {head}"
    ndim = magic & 0xFF
    shape = tuple(int(s) for s in np.frombuffer(raw[4 : 4 + 4 * ndim], ">u4"))
    data = np.frombuffer(raw[4 + 4 * ndim :], np.uint8)
    assert data.size == int(np.prod(shape)), f"{path.name}: size mismatch"
    return data.reshape(shape)


def balanced(labels: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    pick = [rng.choice(np.flatnonzero(labels == d), per_class, replace=False) for d in range(10)]
    return rng.permutation(np.concatenate(pick))


def main(src: str, out: str) -> None:
    src_dir, rng = Path(src), np.random.default_rng(0)
    xs, ys = idx(src_dir / "train-images-idx3-ubyte.gz", 2051), idx(src_dir / "train-labels-idx1-ubyte.gz", 2049)
    xt, yt = idx(src_dir / "t10k-images-idx3-ubyte.gz", 2051), idx(src_dir / "t10k-labels-idx1-ubyte.gz", 2049)
    assert xs.shape == (60000, 28, 28) and xt.shape == (10000, 28, 28)
    i, j = balanced(ys, 300, rng), balanced(yt, 100, rng)
    np.savez_compressed(out, images=np.concatenate([xs[i], xt[j]]), labels=np.concatenate([ys[i], yt[j]]),
                        split=np.repeat(np.array([0, 1], np.uint8), [len(i), len(j)]),  # 0 train, 1 test
                        source_index=np.concatenate([i, j]).astype(np.int32))  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
