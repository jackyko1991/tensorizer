r"""Mathematics → tensor translation.

1. Perron tree (Perron 1928, after Besicovitch 1919). Cut the triangle with apex (0, 1) and base [−h, h] on the x-axis
   into N = 2^k thin triangles T_i with bases [b_i, b_i + w₀], w₀ = 2h/N. Only horizontal shifts s_i are allowed, so
   every T_i keeps all its directions. At height y (0 ≤ y ≤ 1) triangle i covers the interval

        ℓ_i(y) = b_i (1 − y) + s_i,     r_i(y) = ℓ_i(y) + w₀ (1 − y),

   and the area of the union is ∫₀¹ |∪_i [ℓ_i(y), r_i(y)]| dy: a sort along i and a running maximum per row y.

   The shifts come from k rounds of Perron's step with a parameter α ∈ (½, 1): two neighbouring "hearts" (triangles
   with a common apex and adjacent bases [a, b], [b, c]) are made adjacent, then the right one is pushed left by
   d = (1 − α)(c − a). Their outer sides then cross at height α, so the new heart is the old pair scaled by α and the
   overlap above it is saved area. After k rounds the heart has height α^k and

        |∪ T_i| ≤ (α^{2k} + 2(1 − α)) |T|,

   which tends to 0 for α = 1 − ln k / k (so the area falls only like ln k / k = O(ln ln N / ln N)).

2. Tubes in ℝⁿ. A δ-tube is the δ-neighbourhood of a unit segment c + t u, |t| ≤ ½. Covering a union of tubes with
   cubes of side δ gives N(δ) cubes; its volume is ≈ N(δ) δⁿ and the Minkowski dimension is
   lim log N(δ) / log(1/δ). The Kakeya conjecture says that with one tube in every δ-separated direction this
   dimension is n: the union cannot be much smaller than a ball, |∪ tubes| ≳_ε δ^ε.
"""

import math

import torch


def perron_shifts(k: int, alpha: torch.Tensor, half_width: float) -> torch.Tensor:
    """Horizontal shifts s_i of the N = 2^k elementary triangles after k Perron rounds, for A values of α at once.

    alpha: (A,) → (A, N). Each round pairs the hearts (g, g+1): left heart base [L, L + w], right heart [R, R + w]."""
    n = 2**k
    a = alpha.double()[:, None]  # (A, 1)
    w = torch.full_like(a, 2 * half_width / n)  # heart base width (A, 1)
    left = -half_width + 2 * half_width * torch.arange(n, dtype=torch.float64, device=alpha.device) / n
    heart = left[None, :].repeat(len(alpha), 1)  # (A, G) base-left of every heart, G = N at the start
    shift = torch.zeros_like(heart)  # (A, N)
    for level in range(k):
        size = 2**level  # elementary triangles per heart before this round
        lhs, rhs = heart[:, 0::2], heart[:, 1::2]  # (A, G/2)
        delta = (lhs + w) - rhs - (1 - a) * 2 * w  # make adjacent, then push left by (1 − α)·(c − a)
        shift = shift.view(len(alpha), -1, 2, size)  # (A, G/2, 2, size): which half each triangle is in
        shift[:, :, 1, :] += delta[:, :, None]
        shift = shift.reshape(len(alpha), n)
        heart, w = lhs, 2 * a * w  # the new heart is the pair scaled by α
    return shift


def intervals(y: torch.Tensor, base_left: torch.Tensor, w0: float, shift: torch.Tensor) -> tuple:
    """ℓ_i(y) and r_i(y) for every row y and triangle i.   (Y,), (N,), (..., N) → (..., Y, N) twice"""
    lo = base_left[None, :] * (1 - y[:, None]) + shift[..., None, :]
    return lo, lo + w0 * (1 - y[:, None])


def union_length(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Length of ∪_i [lo_i, hi_i] along the last axis: sort by lo, then add what sticks out past the running max."""
    lo, order = torch.sort(lo, dim=-1)
    hi = torch.gather(hi, -1, order)
    reach = torch.cummax(hi, dim=-1).values  # furthest right end so far
    before = torch.cat([torch.full_like(reach[..., :1], -math.inf), reach[..., :-1]], dim=-1)
    return (hi - torch.maximum(lo, before)).clamp(min=0).sum(dim=-1)


def union_area(base_left: torch.Tensor, w0: float, shift: torch.Tensor, n_rows: int) -> torch.Tensor:
    """∫₀¹ |∪_i [ℓ_i(y), r_i(y)]| dy by the midpoint rule.   shift (..., N) → (...)"""
    y = (torch.arange(n_rows, dtype=torch.float64, device=shift.device) + 0.5) / n_rows
    lo, hi = intervals(y, base_left, w0, shift)
    return union_length(lo, hi).mean(dim=-1)


def inside_union(points: torch.Tensor, base_left: torch.Tensor, w0: float, shift: torch.Tensor) -> torch.Tensor:
    """Is each point (x, y), 0 ≤ y ≤ 1, in at least one shifted triangle?   (M, 2), (N,) → (M,) bool"""
    x, y = points[:, 0:1], points[:, 1:2]
    lo = base_left[None, :] * (1 - y) + shift[None, :]
    return ((x >= lo - 1e-12) & (x <= lo + w0 * (1 - y) + 1e-12)).any(dim=1)


def segment_cells(start: torch.Tensor, end: torch.Tensor, delta: float) -> torch.Tensor:
    """Indices of the δ-cubes met by every segment from start_j to end_j (sampled every δ/2 or finer).

    start, end: (J, n) → (J·S, n) int64 cube coordinates."""
    n_samples = int(math.ceil((end - start).norm(dim=1).max().item() / (delta / 2))) + 1
    t = torch.linspace(0, 1, n_samples, dtype=torch.float64, device=start.device)  # (S,)
    pts = start[:, None, :] + t[None, :, None] * (end - start)[:, None, :]  # (J, S, n)
    return torch.floor(pts / delta).long().reshape(-1, start.shape[1])


def count_cells(cells: torch.Tensor) -> int:
    """N(δ): number of distinct cubes (rows) in a (M, n) integer tensor, via one integer key per cube."""
    low = cells.min(dim=0).values
    span = int((cells.max(dim=0).values - low).max()) + 1
    radix = span ** torch.arange(cells.shape[1], device=cells.device)  # (n,) mixed-radix place values
    keys = torch.einsum("mi,i->m", cells - low, radix)  # repeated i ⇒ summed
    return len(torch.unique(keys))
