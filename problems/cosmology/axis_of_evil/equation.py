r"""Physics → tensor translation: spherical harmonics, angular momentum and the preferred axis of a multipole.

The temperature of the microwave sky is expanded in spherical harmonics,

    T(\hat n) = \sum_{ℓm} a_{ℓm}\,Y_{ℓm}(\hat n),        a_{ℓ,−m} = (−1)^m a^*_{ℓm}  (T is real).

Each multipole ℓ is a vector a_m with 2ℓ+1 complex components, on which the angular-momentum matrices L_i of spin ℓ
act. De Oliveira-Costa et al. (2004) define the axis of a multipole as the direction \hat n around which the
power is most "planar" (concentrated in m = ±ℓ), i.e. the one that maximises the angular-momentum dispersion

    \sum_m m^2 |a_{ℓm}(\hat n)|^2 = \langle a | (\hat n\cdot L)^2 | a\rangle = \hat n_i\,M_{ij}\,\hat n_j,
    M_{ij} = \mathrm{Re}\; a^*_m\,(L_i L_j)_{mm'}\,a_{m'}       (m, m' summed).

So the axis is simply the top eigenvector of a 3 × 3 symmetric matrix, and a batch of skies is a batch of eigh.
The alignment statistic is |\hat n_2 \cdot \hat n_3| (axes have no head or tail).
"""

import math

import numpy as np
import torch


def gauss_legendre(n: int):
    """Nodes and weights of n-point Gauss–Legendre quadrature on [−1, 1] (for exact sphere integrals)."""
    x, w = np.polynomial.legendre.leggauss(n)
    return torch.tensor(x), torch.tensor(w)


def index(ell: int, m: int) -> int:
    """Position of (ℓ, m) in the flat coefficient vector: ℓ = 0 first, m = −ℓ … ℓ within each ℓ."""
    return ell * ell + ell + m


def spherical_harmonics(n: torch.Tensor, lmax: int) -> torch.Tensor:
    r"""Complex Y_{ℓm}(\hat n) with the Condon–Shortley phase.   n: (P, 3) unit vectors → (P, (lmax+1)²)

    The associated Legendre functions P_ℓ^m(cos θ) come from the standard three-term recurrence in ℓ; the loops
    run over (ℓ, m) only, every pixel is handled at once.
    """
    x = n[:, 2].clamp(-1, 1)  # cos θ
    s = torch.sqrt((1 - x * x).clamp_min(0))  # sin θ
    phi = torch.atan2(n[:, 1], n[:, 0])
    Y = torch.zeros(n.shape[0], (lmax + 1) ** 2, dtype=torch.complex128, device=n.device)
    for m in range(lmax + 1):
        p_mm = (-1) ** m * float(math.prod(range(1, 2 * m, 2))) * s**m  # P_m^m = (−1)^m (2m−1)!! sin^m θ
        p_prev, p = torch.zeros_like(x), p_mm
        for ell in range(m, lmax + 1):
            if ell > m:  # P_ℓ^m from P_{ℓ−1}^m and P_{ℓ−2}^m
                p_prev, p = p, ((2 * ell - 1) * x * p - (ell + m - 1) * p_prev) / (ell - m)
            norm = math.sqrt((2 * ell + 1) / (4 * math.pi) * math.factorial(ell - m) / math.factorial(ell + m))
            y = norm * p * torch.exp(1j * m * phi)
            Y[:, index(ell, m)] = y
            Y[:, index(ell, -m)] = (-1) ** m * y.conj()  # Y_{ℓ,−m} = (−1)^m Y*_{ℓm}
    return Y


def real_harmonics(n: torch.Tensor, lmax: int) -> torch.Tensor:
    r"""Real orthonormal harmonics: Y_{ℓ0}, and \sqrt2(−1)^m Re / Im Y_{ℓm} for m > 0 / m < 0.   → (P, (lmax+1)²)"""
    Y = spherical_harmonics(n, lmax)
    R = torch.empty(Y.shape, dtype=torch.float64, device=n.device)
    for ell in range(lmax + 1):
        R[:, index(ell, 0)] = Y[:, index(ell, 0)].real
        for m in range(1, ell + 1):
            R[:, index(ell, m)] = math.sqrt(2) * (-1) ** m * Y[:, index(ell, m)].real
            R[:, index(ell, -m)] = math.sqrt(2) * (-1) ** m * Y[:, index(ell, m)].imag
    return R


def real_to_complex(r: torch.Tensor, ell: int) -> torch.Tensor:
    r"""a_{ℓm} from real coefficients r_{ℓm} (…, 2ℓ+1): a_{ℓm} = (−1)^m (r_{ℓm} − i r_{ℓ,−m})/\sqrt2 for m > 0."""
    a = torch.zeros(r.shape, dtype=torch.complex128, device=r.device)
    a[..., ell] = r[..., ell]
    for m in range(1, ell + 1):
        a_m = (-1) ** m * (r[..., ell + m] - 1j * r[..., ell - m]) / math.sqrt(2)
        a[..., ell + m] = a_m
        a[..., ell - m] = (-1) ** m * a_m.conj()
    return a


def angular_momentum(ell: int) -> torch.Tensor:
    r"""Spin-ℓ matrices (L_x, L_y, L_z) in the basis m = −ℓ … ℓ: shape (3, 2ℓ+1, 2ℓ+1), complex.

    L_z = diag(m),  L_+|m⟩ = \sqrt{ℓ(ℓ+1) − m(m+1)}\,|m+1⟩,  L_x = (L_+ + L_−)/2,  L_y = (L_+ − L_−)/2i.
    """
    m = torch.arange(-ell, ell + 1, dtype=torch.float64)
    up = torch.sqrt(ell * (ell + 1) - m[:-1] * (m[:-1] + 1))  # ⟨m+1|L_+|m⟩
    Lp = torch.diag(up, -1).to(torch.complex128)  # row m+1, column m
    Lm = Lp.T.conj()
    return torch.stack([(Lp + Lm) / 2, (Lp - Lm) / 2j, torch.diag(m).to(torch.complex128)])


def dispersion_tensor(a: torch.Tensor, ell: int) -> torch.Tensor:
    r"""M_{ij} = Re a^*_m (L_i L_j)_{mm'} a_{m'} / Σ|a|², symmetrised.   a: (…, 2ℓ+1) → (…, 3, 3)"""
    L = angular_momentum(ell).to(a.device)
    M = torch.einsum("...m,imn,jnk,...k->...ij", a.conj(), L, L, a).real  # m, n, k summed
    M = (M + M.transpose(-1, -2)) / 2
    return M / (a.abs() ** 2).sum(-1)[..., None, None]


def principal_axis(M: torch.Tensor) -> torch.Tensor:
    """Unit eigenvector of the largest eigenvalue of each symmetric 3×3 matrix (batched eigh).   (…, 3, 3) → (…, 3)"""
    return torch.linalg.eigh(M).eigenvectors[..., -1]


def alignment(n2: torch.Tensor, n3: torch.Tensor) -> torch.Tensor:
    r"""|\hat n_2 \cdot \hat n_3| (headless axes): 1 = aligned, 0 = perpendicular.   (…, 3) → (…)"""
    return torch.einsum("...i,...i->...", n2, n3).abs()
