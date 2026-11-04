"""Build the concrete tensors: collocation points, the spectral bases that represent a profile, and the batch of
initial exponents λ that the search starts from.

Burgers. A profile is odd, U(0) = 0, U′(0) = −1, and grows like |y|^{λ/(1+λ)} far away. Both facts are built into the
representation U(y) = y P(r), with the compactified coordinate r = (1 + y²/L²)^{−g} ∈ (0, 1], g = 1/(2(1+λ)): r = 1 at
y = 0 and r → 0 as |y| → ∞. P is a Chebyshev series in x = 2r − 1 with P(1) = −1 (so U′(0) = −1) and P(0) = 0 (so
U ~ y^{λ/(1+λ)}). The 2025 paper calls this kind of choice the inductive bias of the network: coordinate transforms
and output shaping.

CCF. The vorticity profile Ω(y) is odd and decays like |y|^{−β}, β = 1/(1+λ). It is expanded in the boundary values of
(1 − iy)^{−s}, which are analytic in the upper half plane, so their Hilbert transforms are known exactly. The exponents
are s = jβ + m (j = 0 … J, m = 0 … M, s ≠ 0): the powers that appear in the far-field expansion.
"""

import math

import torch

L_BURGERS = 0.3  # length scale of the compactification; keeps the complex singularities of U far from r ∈ [0, 1]


def chebyshev_points(n: int, device: torch.device) -> torch.Tensor:
    """Chebyshev (first kind) points mapped to r ∈ (0, 1): (n,). Dense near r = 0 (far field) and r = 1 (y = 0)."""
    k = torch.arange(n, dtype=torch.float64, device=device)
    return 0.5 * (1 - torch.cos(math.pi * (k + 0.5) / n))


def chebyshev(x: torch.Tensor, n_modes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """T_k(x) and T_k′(x) for k = 1 … K by the three-term recurrence: (…, K) each."""
    t, dt = [torch.ones_like(x), x], [torch.zeros_like(x), torch.ones_like(x)]
    for _ in range(n_modes - 1):
        t.append(2 * x * t[-1] - t[-2])
        dt.append(2 * t[-2] + 2 * x * dt[-1] - dt[-2])
    return torch.stack(t[1 : n_modes + 1], -1), torch.stack(dt[1 : n_modes + 1], -1)


def burgers_basis(r: torch.Tensor, n_modes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """b_k(r) = T_k(x) − (1+x)/2 − (−1)^k (1−x)/2, x = 2r − 1, which vanish at r = 0 and r = 1, and db_k/dr.
    r (N,) → (N, K), (N, K)."""
    x = 2 * r - 1
    t, dt = chebyshev(x, n_modes)
    sign = (-1.0) ** torch.arange(1, n_modes + 1, dtype=r.dtype, device=r.device)  # T_k(−1) = (−1)^k
    b = t - (1 + x[..., None]) / 2 - sign * (1 - x[..., None]) / 2
    db = 2 * dt - 1 + sign
    return b, db


def basis_derivatives_at_one(n_modes: int, device: torch.device) -> torch.Tensor:
    """d^m b_k / dr^m at r = 1 (y = 0) for m = 1, 2, 3: (3, K). Uses T_k^{(m)}(1) = Π_{j<m} (k² − j²)/(2j + 1)."""
    k2 = torch.arange(1, n_modes + 1, dtype=torch.float64, device=device) ** 2
    sign = (-1.0) ** torch.arange(1, n_modes + 1, dtype=torch.float64, device=device)
    d1 = 2 * k2 - 1 + sign
    d2 = 4 * k2 * (k2 - 1) / 3
    d3 = 8 * k2 * (k2 - 1) * (k2 - 4) / 15
    return torch.stack([d1, d2, d3])


def circle_grid(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """φ_j = −π + 2π(j + ½)/n and y_j = tan(φ_j/2): the real line wrapped onto the circle (Möbius map), where the
    Hilbert transform of the line becomes the periodic Hilbert transform. (n,), (n,)"""
    phi = -math.pi + 2 * math.pi * (torch.arange(n, dtype=torch.float64, device=device) + 0.5) / n
    return phi, torch.tan(phi / 2)


def ccf_points(n: int, device: torch.device) -> torch.Tensor:
    """Collocation points y = tan(φ/2) for φ ∈ (0, 0.999π): y from ≈ 0 to ≈ 1300 (the profile is odd): (n,)."""
    phi = (torch.arange(n, dtype=torch.float64, device=device) + 0.5) / n * math.pi * 0.999
    return torch.tan(phi / 2)


def ccf_exponents(beta: torch.Tensor, n_j: int, n_m: int) -> torch.Tensor:
    """s = jβ + m for j = 0 … J, m = 0 … M without s = 0: (K,) with K = (J+1)(M+1) − 1."""
    j = torch.arange(n_j + 1, dtype=beta.dtype, device=beta.device)[:, None]
    m = torch.arange(n_m + 1, dtype=beta.dtype, device=beta.device)[None]
    return (j * beta + m).flatten()[1:]


def ccf_initial_coefficients(lam: torch.Tensor, n_j: int, n_m: int) -> torch.Tensor:
    """Start every member of the batch from Ω = −(1+λ) Im (1 − iy)^{−β} (so Ω′(0) = −1): (B, K)."""
    a = torch.zeros(lam.shape[0], (n_j + 1) * (n_m + 1) - 1, dtype=lam.dtype, device=lam.device)
    a[:, n_m] = -(1 + lam)  # the (j, m) = (1, 0) column, s = β
    return a


N_FAMILIES = 1  # powers w^j (j < N_FAMILIES) of w = 1/(1 + y²/L²) = r^{1/g}; extra families slowed convergence here


def burgers_features(r: torch.Tensor, g: torch.Tensor, n_modes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Features w^j b_k(r) (j = 0 … 2, k = 1 … K) and their r-derivatives. The far field of U contains powers of
    y^{−2g} ∝ r and of y^{−2} ∝ w = r^{1/g}; both are covered, and w = 1/(1 + y²/L²) is analytic at y = 0 too.
    r and g broadcast against each other (e.g. r (N,), g (B, 1) → (B, N)) → features (…, 3K), d/dr (…, 3K)."""
    g = torch.as_tensor(g, dtype=r.dtype, device=r.device)
    r, g = torch.broadcast_tensors(r, g)
    b, db = burgers_basis(r, n_modes)  # (…, K)
    j = torch.arange(N_FAMILIES, dtype=r.dtype, device=r.device)
    wj = (r ** (1 / g))[..., None, None] ** j  # (…, 1, 3): w^j
    phi = (wj * b[..., None]).flatten(-2)  # (…, K·3), index 3k + j
    dwj = j * wj / (g * r)[..., None, None]  # d w^j / dr = (j/g) w^j / r
    dphi = (dwj * b[..., None] + wj * db[..., None]).flatten(-2)
    return phi, dphi


def feature_derivatives_at_one(n_modes: int, g: torch.Tensor) -> torch.Tensor:
    """P′(1) and P″(1) contributions of every feature (b_k(1) = 0, w(1) = 1): (B, 2, 3K) for g (B,)."""
    d = basis_derivatives_at_one(n_modes, g.device)  # (3, K)
    j = torch.arange(N_FAMILIES, dtype=g.dtype, device=g.device)
    d1 = d[0][None, :, None].expand(len(g), -1, N_FAMILIES)  # (B, K, 3)
    d2 = d[1][None, :, None] + 2 * (j / g[:, None, None]) * d[0][None, :, None]
    return torch.stack([d1.flatten(1), d2.flatten(1)], 1)
