"""Build the concrete tensors: the bundled MNIST subset, the balanced small training sets, the fixed filter bank and the
fresh (untrained) parameters of the supernet and of the CNN.

data/mnist_subset.npz holds `images` (3000, 28, 28) uint8 and `labels` (3000,) uint8: the first 100 images of every
digit in the MNIST training file (rows 0–999, a pool from which the small training sets are drawn) followed by the
first 200 of every digit in the MNIST test file (rows 1000–2999, the fixed test set). build_subset() made it once
from the raw MNIST files, read with gzip and numpy only (the raw folder is untrusted data, nothing in it is executed).
"""

import gzip
import math
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).parent / "data" / "mnist_subset.npz"
POOL_PER_CLASS, TEST_PER_CLASS, CLASSES = 100, 200, 10
N_POOL = POOL_PER_CLASS * CLASSES  # rows [0, N_POOL) are the training pool, the rest the test set
SIZES = (1, 5, 10, 50, 100)  # training images per class in the data study
LAMBDA = 0.25  # soft-threshold level S_λ (pixels are in [0, 1])
Q_PCA = 8  # principal components kept per channel
FFT_KEEP = 4  # |F(Z)|_{uv} kept for 0 ≤ u, v < 4 (the low frequencies)
MD5 = {  # the published checksums of the raw files (verified before use)
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _idx(path: Path) -> np.ndarray:
    """Read one IDX file (big-endian header: magic, dims; then uint8 data) with gzip + numpy only."""
    raw = gzip.decompress(path.read_bytes())
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def build_subset(raw_dir: Path, out: Path = DATA) -> Path:
    """One-off: the first POOL_PER_CLASS training and TEST_PER_CLASS test images of every digit, in file order."""
    import hashlib

    for name, md5 in MD5.items():
        assert hashlib.md5((raw_dir / name).read_bytes()).hexdigest() == md5, f"checksum mismatch: {name}"
    parts = []
    for prefix, per in (("train", POOL_PER_CLASS), ("t10k", TEST_PER_CLASS)):
        x, y = _idx(raw_dir / f"{prefix}-images-idx3-ubyte.gz"), _idx(raw_dir / f"{prefix}-labels-idx1-ubyte.gz")
        keep = np.sort(np.concatenate([np.flatnonzero(y == c)[:per] for c in range(CLASSES)]))
        parts.append((x[keep], y[keep]))
    np.savez_compressed(out, images=np.concatenate([p[0] for p in parts]), labels=np.concatenate([p[1] for p in parts]))
    return out


def load(device: torch.device) -> dict:
    """Pool and test set as float images in [0, 1] at half resolution, (N, 1, 14, 14) (each pixel the mean of a 2×2
    block: the whole 30-run study then fits in about a minute of one CPU core), and int64 labels (N,)."""
    with np.load(DATA) as d:
        x = torch.from_numpy(d["images"].astype(np.float32) / 255.0)[:, None]
        x = torch.nn.functional.avg_pool2d(x, 2).to(device)
        y = torch.from_numpy(d["labels"].astype(np.int64)).to(device)
    return {"x_pool": x[:N_POOL], "y_pool": y[:N_POOL], "x_test": x[N_POOL:], "y_test": y[N_POOL:]}


def small_set(y_pool: torch.Tensor, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """n images per class drawn (without replacement) from the pool, split per class into a train half (⌈n/2⌉) and
    a validation half (⌊n/2⌋). With n = 1 there is no second image: train and validation are the same 10 images."""
    gen = torch.Generator().manual_seed(1000 * seed + n)
    tr, va = [], []
    for c in range(CLASSES):
        idx = torch.nonzero(y_pool.cpu() == c).flatten()
        pick = idx[torch.randperm(len(idx), generator=gen)[:n]]
        tr.append(pick[: math.ceil(n / 2)])
        va.append(pick[math.ceil(n / 2) :] if n > 1 else pick)
    return torch.cat(tr).to(y_pool.device), torch.cat(va).to(y_pool.device)


def balanced_batch(y: torch.Tensor, size: int, gen: torch.Generator) -> torch.Tensor:
    """Indices of a class-balanced mini-batch of about `size` images (all of them when the set is smaller). y must be
    class-blocked with equal counts, as small_set returns it: class c occupies rows c m … c m + m − 1."""
    m = len(y) // CLASSES
    if len(y) <= size:
        return torch.arange(len(y), device=y.device)
    per = max(1, size // CLASSES)
    pick = torch.rand(CLASSES, m, generator=gen).argsort(1)[:, :per] + m * torch.arange(CLASSES)[:, None]
    return pick.flatten().to(y.device)


def fixed_kernels() -> torch.Tensor:
    """The fixed filter bank (4, 5, 5): Sobel-x, Sobel-y, Laplacian (3×3, zero-padded to 5×5) and a zero-mean Gabor
    (θ = 45°, wavelength 4 px, σ = 1.5 px), applied as cross-correlations with zero padding, as in CNNs."""
    sx = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    lap = torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]])
    u = torch.arange(-2.0, 3.0)
    U, V = torch.meshgrid(u, u, indexing="ij")
    g = torch.exp(-(U**2 + V**2) / (2 * 1.5**2)) * torch.cos(2 * math.pi * (U + V) / math.sqrt(2) / 4)
    g = g - g.mean()
    g = g / g.abs().sum() * 4  # same total weight as the Sobel kernel
    bank = torch.zeros(4, 5, 5)
    for k, m in enumerate((sx, sx.T.contiguous(), lap)):
        bank[k, 1:4, 1:4] = m
    bank[3] = g
    return bank


def moment_basis(h: int) -> torch.Tensor:
    """(6, h, h): 1, u, v, u², v², uv on the pixel grid with u, v ∈ [−1, 1] (u along rows, v along columns)."""
    u = torch.linspace(-1, 1, h)
    U, V = torch.meshgrid(u, u, indexing="ij")
    return torch.stack([torch.ones_like(U), U, V, U**2, V**2, U * V])


def init_supernet(n_edges: int, dims: dict, seed: int, device: torch.device) -> dict:
    """Trainable parameters: one 3×3 filter K_e per cell edge (6, 3, 3); per reduction r a linear read-out
    W_r (10, D_r), b_r (10,) and 10 prototypes P_r (10, D_r) with a log-temperature τ_r ()."""
    gen = torch.Generator().manual_seed(seed)

    def p(t):
        return t.to(device).requires_grad_(True)

    th = {"K": p(0.3 * torch.randn(n_edges, 3, 3, generator=gen))}
    for r, d in dims.items():
        th[f"W_{r}"] = p(0.01 * torch.randn(CLASSES, d, generator=gen))
        th[f"b_{r}"] = p(torch.zeros(CLASSES))
        th[f"P_{r}"] = p(0.5 * torch.randn(CLASSES, d, generator=gen))
        th[f"tau_{r}"] = p(torch.zeros(()))
    return th


def init_cnn(seed: int, device: torch.device, c1: int = 8, c2: int = 16) -> dict:
    """LeNet-style CNN on 14×14: conv 3×3 (1→c1), ReLU, max-pool 2; conv 3×3 (c1→c2), ReLU, max-pool 2 (7 → 3);
    linear (c2·3·3 → 10).
    He-uniform initialisation (as torch.nn's default bound 1/√fan_in for the biases)."""
    gen = torch.Generator().manual_seed(seed)

    def u(shape, fan_in, gain=6.0):
        b = math.sqrt(gain / fan_in)
        return ((torch.rand(shape, generator=gen) * 2 - 1) * b).to(device).requires_grad_(True)

    return {"W1": u((c1, 1, 3, 3), 9), "b1": u((c1,), 9, 1.0), "W2": u((c2, c1, 3, 3), 9 * c1),
            "b2": u((c2,), 9 * c1, 1), "W3": u((CLASSES, c2 * 9), c2 * 9), "b3": u((CLASSES,), c2 * 9, 1)}  # fmt: skip


def cnn_param_count(c1: int = 8, c2: int = 16) -> int:
    """9 c1 + c1 + 9 c1 c2 + c2 + 9 c2 · 10 + 10."""
    return 10 * c1 + 9 * c1 * c2 + c2 + 90 * c2 + 10
