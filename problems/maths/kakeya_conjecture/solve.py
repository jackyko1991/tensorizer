"""Build Perron trees with the best α, assemble a Kakeya set from three of them, and count δ-cubes of tube unions."""

import math

import torch

from . import tensors
from .equation import count_cells, perron_shifts, segment_cells, union_area


def perron_tree(k: int, device: torch.device, n_alpha: int = 24, n_rows: int = 2000) -> dict:
    """Perron tree with 2^k triangles; α chosen by trying n_alpha values at once (coarse rows), then a fine area."""
    base, w0 = tensors.triangle_bases(k, device)
    alphas = torch.linspace(0.5, 0.99, n_alpha, dtype=torch.float64, device=device)
    coarse = union_area(base, w0, perron_shifts(k, alphas, tensors.HALF_WIDTH), 200)  # (A,)
    alpha = alphas[coarse.argmin()]
    shift = perron_shifts(k, alpha[None], tensors.HALF_WIDTH)[0]  # (N,)
    area = union_area(base, w0, shift, n_rows).item()
    return {"k": k, "alpha": alpha.item(), "base": base, "w0": w0, "shift": shift, "area": area}


def tree_polygons(tree: dict) -> torch.Tensor:
    """Vertices of every shifted triangle: apex (s_i, 1), base corners (b_i + s_i, 0) and (b_i + w₀ + s_i, 0).

    → (N, 3, 2)"""
    s, b, w0 = tree["shift"], tree["base"], tree["w0"]
    one, zero = torch.ones_like(s), torch.zeros_like(s)
    return torch.stack(
        [torch.stack([s, one], 1), torch.stack([b + s, zero], 1), torch.stack([b + w0 + s, zero], 1)], dim=1
    )


def tree_centre(tree: dict) -> torch.Tensor:
    """Middle of the tree's bounding box, the point the three copies are turned about.   → (2,)"""
    x = tree_polygons(tree)[..., 0]
    return torch.stack([(x.min() + x.max()) / 2, torch.tensor(0.5, dtype=x.dtype, device=x.device)])


def kakeya_polygons(tree: dict) -> torch.Tensor:
    """Three copies of the tree turned by 0°, 60° and 120° about its centre: every line direction is covered.

    → (3N, 3, 2)"""
    poly = tree_polygons(tree) - tree_centre(tree)
    rot = torch.stack([tensors.rotation(a * math.pi / 3) for a in range(3)]).to(poly.device)  # (3, 2, 2)
    turned = torch.einsum("cij,nvj->cnvi", rot, poly)  # repeated j ⇒ summed: rotate every vertex of every copy
    return turned.reshape(-1, 3, 2)


def tree_segments(tree: dict, delta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Segments from each apex to base points spaced ≤ δ/2: they sweep every triangle.   → (J, 2), (J, 2)"""
    m = max(2, int(math.ceil(tree["w0"] / (delta / 2))) + 1)
    frac = torch.linspace(0, 1, m, dtype=torch.float64, device=tree["shift"].device)
    s, b = tree["shift"][:, None], tree["base"][:, None]
    end_x = (b + tree["w0"] * frac[None, :] + s).reshape(-1)  # (N·m,)
    start = torch.stack([s.expand(-1, m).reshape(-1), torch.ones_like(end_x)], 1)
    return start, torch.stack([end_x, torch.zeros_like(end_x)], 1)


def kakeya_cells(tree: dict, delta: float) -> int:
    """N(δ) for the three-copy Kakeya set built from this tree."""
    start, end = tree_segments(tree, delta)
    shift = tree_centre(tree)
    rot = torch.stack([tensors.rotation(a * math.pi / 3) for a in range(3)]).to(start.device)
    s3 = torch.einsum("cij,mj->cmi", rot, start - shift).reshape(-1, 2)
    e3 = torch.einsum("cij,mj->cmi", rot, end - shift).reshape(-1, 2)
    return count_cells(segment_cells(s3, e3, delta))


def tube_counts(n: int, deltas: list[float], arrangement: str, device: torch.device, seed: int = 0) -> list[int]:
    """N(δ) for one unit tube in each δ-separated direction of ℝⁿ, centred by the given arrangement."""
    out = []
    for delta in deltas:
        u = tensors.directions(n, delta, device)  # (D, n)
        c = tensors.centres(arrangement, len(u), n, seed, device)  # (D, n)
        out.append(count_cells(segment_cells(c - u / 2, c + u / 2, delta)))
    return out


def dimension_slope(deltas: list[float], counts: list[int], last: int = 2) -> float:
    """Minkowski-dimension estimate: slope of log N(δ) against log(1/δ) over the last `last` + 1 scales."""
    x = torch.log(1 / torch.tensor(deltas[-last - 1 :], dtype=torch.float64))
    y = torch.log(torch.tensor(counts[-last - 1 :], dtype=torch.float64))
    x, y = x - x.mean(), y - y.mean()
    return (torch.einsum("i,i->", x, y) / torch.einsum("i,i->", x, x)).item()
