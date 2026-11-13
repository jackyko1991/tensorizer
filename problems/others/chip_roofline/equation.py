"""The roofline model: how fast a kernel can go is the smaller of the chip's compute peak and what its memory can feed.

    P(I) = min(π, β · I)

with π the peak arithmetic rate (FLOP/s), β the memory bandwidth (byte/s) and I the kernel's arithmetic intensity
(FLOP per byte moved). A matrix product C_ij = A_ik B_kj of n×n matrices does 2n³ FLOP (n³ multiply-adds) and, at
best, moves the three matrices once: 3 n² b bytes for b-byte elements, so I = 2n / (3b).
"""


def matmul_flops(n: int) -> float:
    """FLOP of an n×n by n×n matrix product: n³ multiply-adds, 2 FLOP each."""
    return 2.0 * n**3


def matmul_bytes(n: int, elem_bytes: int) -> float:
    """Bytes of A, B and C, each read or written once (the best case a tiled kernel approaches)."""
    return 3.0 * n * n * elem_bytes


def intensity(n: int, elem_bytes: int) -> float:
    """Arithmetic intensity of the matrix product, FLOP per byte: 2n / (3b)."""
    return matmul_flops(n) / matmul_bytes(n, elem_bytes)


def copy_bytes(numel: int, elem_bytes: int) -> float:
    """A copy dst ← src reads every element once and writes it once."""
    return 2.0 * numel * elem_bytes


def roofline(peak_flops: float, bandwidth: float, intensity_: float) -> float:
    """Attainable FLOP/s at a given arithmetic intensity."""
    return min(peak_flops, bandwidth * intensity_)


def ridge(peak_flops: float, bandwidth: float) -> float:
    """The intensity (FLOP/byte) where the slope meets the flat ceiling: below it a kernel waits for memory."""
    return peak_flops / bandwidth
