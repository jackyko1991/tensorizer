r"""Physics → tensor translation.

In the paraxial (small-angle) limit a ray at axial position z is a 2-vector y = (height h, angle θ),
and every optical element acts on it linearly: a 2×2 ray-transfer ("ABCD") matrix,

    free space of length d:   P(d) = [[1, d], [0, 1]]       (h grows by θ d, θ unchanged)
    thin lens of focal f:     L(f) = [[1, 0], [-1/f, 1]]    (h unchanged, θ kinked by -h/f)

Light meeting P(s), then L_1, then P(d), then L_2 sees the matrix product (rightmost acts first)

    M = L_2\, P(d)\, L_1\, P(s),    in index form  M_{im} = L^{(2)}_{ij} P^{(d)}_{jk} L^{(1)}_{kl} P^{(s)}_{lm}

and every ray r of a batch is mapped by the same matrix:

    y'_{ri} = M_{ij}\, y_{rj}        (repeated j ⇒ summed: einsum('ij,rj->ri'))

After the last lens the ray height is h' = (A + zC) h + (B + zD) θ. An image forms where h' no longer
depends on the starting angle θ, i.e. at z = -B/D. Since det P = det L = 1, also det M = 1.
"""

import torch


def free_space(d: torch.Tensor) -> torch.Tensor:
    """P(d) = [[1, d], [0, 1]].   d: any shape (...) → (..., 2, 2)"""
    d = torch.as_tensor(d, dtype=torch.float32)
    one, zero = torch.ones_like(d), torch.zeros_like(d)
    return torch.stack([torch.stack([one, d], -1), torch.stack([zero, one], -1)], -2)


def thin_lens(f: float) -> torch.Tensor:
    """L(f) = [[1, 0], [-1/f, 1]]: shape (2, 2)."""
    return torch.tensor([[1.0, 0.0], [-1.0 / f, 1.0]])


def chain(*elements: torch.Tensor) -> torch.Tensor:
    r"""Product of the elements in the order light meets them: M = E_n … E_2 E_1.

    Each step is the matrix product  M_{ik} ← E_{ij} M_{jk}  (repeated j summed).
    Same result as torch.linalg.multi_dot(list(reversed(elements))).
    """
    m = torch.eye(2)
    for e in elements:
        m = torch.einsum("ij,jk->ik", e, m)  # (2, 2)
    return m


def system_matrix(s: float, f1: float, d: float, f2: float) -> torch.Tensor:
    """Object → lens 1 → gap d → lens 2, as one (2, 2) matrix  M = L_2 P(d) L_1 P(s)."""
    return torch.linalg.multi_dot([thin_lens(f2), free_space(d), thin_lens(f1), free_space(s)])


def apply(m: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
    r"""y'_{ri} = M_{ij} y_{rj}.   m: (2, 2), rays: (R, 2) → (R, 2)"""
    return torch.einsum("ij,rj->ri", m, rays)


def image_distance(m: torch.Tensor) -> torch.Tensor:
    """Distance behind the last element where all rays from one object point meet: z = -B/D."""
    return -m[0, 1] / m[1, 1]
