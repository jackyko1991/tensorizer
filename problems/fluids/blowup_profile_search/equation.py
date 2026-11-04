r"""Physics → tensor translation: self-similar blow-up profiles.

Ansatz u(x, t) = τ^λ U(y), τ = T − t, y = x / τ^{1+λ}. Then u_t = τ^{λ−1}[−λU + (1+λ) y U′] and u u_x = τ^{λ−1} U U′,
so inviscid Burgers u_t + u u_x = 0 becomes the profile equation

    −λ U + ((1 + λ) y + U) U′ = 0,

with λ unknown. Solving it as a linear ODE for y(U) gives y = −U − C U^{1 + 1/λ}: the profile is smooth (C^∞) only
when 1 + 1/λ is an odd integer 2i + 3, i.e. λ = 1/(2i + 2); with U′(0) = −1 and C = 1, y = −U − U^{2i+3}.

CCF, θ_t − (Hθ) θ_x = 0 (Hilbert transform H, H[cos] = sin). With ω = θ_x = Ω(y)/τ the velocity becomes
V(y) = ∫₀^y HΩ, and the vorticity profile obeys

    Ω + ((1 + λ) y − V) Ω′ − Ω HΩ = 0.

For F(y) = (1 − iy)^{−s} (analytic in the upper half plane): H[Re F] = Im F, so H[Im F] = −Re F.
"""

import torch

from . import tensors


# ---------------------------------------------------------------- Burgers
def burgers_residual(y: torch.Tensor, U: torch.Tensor, dU: torch.Tensor, lam) -> torch.Tensor:
    """R = −λU + ((1+λ)y + U) U′, elementwise (any shape)."""
    return -lam * U + ((1 + lam) * y + U) * dU


def burgers_exact(y: torch.Tensor, i: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The exact smooth profile of order i (λ = 1/(2i+2)): solve y = −U − U^n, n = 2i + 3, by Newton on every point
    at once. Returns U and U′ = −1/(1 + n U^{n−1})."""
    n = 2 * i + 3
    U = -y / (1 + y.abs()) ** (1 - 1 / n)  # right size near 0 and at infinity
    for _ in range(60):
        U = U - (U + U**n + y) / (1 + n * U ** (n - 1))
    return U, -1 / (1 + n * U ** (n - 1))


def compactify(y: torch.Tensor, lam: torch.Tensor, L: float = tensors.L_BURGERS) -> torch.Tensor:
    """r = (1 + y²/L²)^{−g}, g = 1/(2(1+λ))."""
    return (1 + (y / L) ** 2) ** (-1 / (2 * (1 + lam)))


def burgers_profile(
    c: torch.Tensor, lam: torch.Tensor, y: torch.Tensor, L: float = tensors.L_BURGERS
) -> tuple[torch.Tensor, torch.Tensor]:
    """U(y) = y P(r) with P(r) = −r + Σ c_k φ_k(r), and U′ = P + y P′(r) dr/dy.   c (3K,), y (N,) → (N,), (N,)"""
    g = 1 / (2 * (1 + lam))
    q = (y / L) ** 2
    r = (1 + q) ** (-g)
    phi, dphi = tensors.burgers_features(r, g, c.shape[-1] // tensors.N_FAMILIES)
    P = -r + phi @ c
    dP = -1 + dphi @ c
    drdy = -2 * g * y * r / (L**2 * (1 + q))
    return y * P, P + y * dP * drdy


def taylor_at_origin(c: torch.Tensor, lam: torch.Tensor, L: float = tensors.L_BURGERS) -> torch.Tensor:
    """Taylor coefficients (c₃, c₅) of U(y) = −y + c₃y³ + c₅y⁵ + …, from the series of y P(r(y)) at y = 0:
    c₃ = −g P′(1)/L², c₅ = g((g + 1)P′(1) + g P″(1))/(2L⁴). A profile with c₃ = 0 cannot sit at λ = 1/2 (the y³
    balance of the profile equation reads c₃(2λ − 1) = 0), one with c₃ = c₅ = 0 not at 1/4 either."""
    g = 1 / (2 * (1 + lam))
    d = tensors.feature_derivatives_at_one(c.shape[-1] // tensors.N_FAMILIES, g.reshape(1))[0]  # (2, 3K)
    p1, p2 = d[0] @ c - 1, d[1] @ c  # P = −r + …
    return torch.stack([-p1 * g / L**2, g * ((g + 1) * p1 + g * p2) / (2 * L**4)])


def max_slope(tau: torch.Tensor, lam: float) -> torch.Tensor:
    """max |u_x| = τ^λ |U′(0)| / τ^{1+λ} = 1/τ for the normalised profile U′(0) = −1: the same for every λ."""
    return tau**lam * 1.0 / tau ** (1 + lam)


# ---------------------------------------------------------------- Hilbert transform and CCF
def hilbert_periodic(f: torch.Tensor) -> torch.Tensor:
    """Periodic Hilbert transform along the last axis by FFT: Ĥf(k) = −i sgn(k) f̂(k), so H[cos] = sin."""
    n = f.shape[-1]
    k = torch.fft.fftfreq(n, 1 / n, dtype=f.dtype, device=f.device)
    return torch.fft.ifft(-1j * torch.sign(k) * torch.fft.fft(f)).real


def hilbert_line(f: torch.Tensor) -> torch.Tensor:
    """Hilbert transform on the real line for samples on the circle grid y = tan(φ/2) (tensors.circle_grid): the
    periodic transform in φ, minus its value at φ = π (y = ±∞), so that Hf → 0 at infinity."""
    n = f.shape[-1]
    k = torch.fft.fftfreq(n, 1 / n, dtype=f.dtype, device=f.device)
    f_hat = torch.fft.fft(f)
    shift = torch.exp(1j * torch.pi * k * (2 - 1 / n))  # e^{ik(π − φ₀)}, the grid starts at φ₀ = −π + π/n
    at_infinity = (-1j * torch.sign(k) * f_hat * shift).sum().real / n
    return hilbert_periodic(f) - at_infinity


def ccf_basis_pair(y: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Odd basis functions Im (1 − iy)^{−s} and their exact Hilbert transforms −Re (1 − iy)^{−s}: (N, K) each."""
    F = (1 - 1j * y)[:, None] ** (-s)
    return F.imag, -F.real


def ccf_fields(a: torch.Tensor, lam: torch.Tensor, y: torch.Tensor, n_j: int, n_m: int) -> dict:
    """Ω = Σ a_k Im F_k, HΩ, V = ∫₀^y HΩ and Ω′ for F_k = (1 − iy)^{−s_k}.   a (K,), y (N,) → dict of (N,)"""
    s = tensors.ccf_exponents(1 / (1 + lam), n_j, n_m)  # (K,)
    z = (1 - 1j * y)[:, None]  # (N, 1)
    F = z ** (-s)  # (N, K)
    one = (s - 1).abs() < 1e-12  # s = 1: ∫ 1/(1+y²) = arctan y
    V_basis = torch.where(one, -torch.atan(y)[:, None], (z ** (1 - s)).imag / torch.where(one, 1.0, 1 - s))
    dF = (1j * s * z ** (-s - 1)).imag  # d/dy Im (1 − iy)^{−s}
    return {"omega": F.imag @ a, "h_omega": -F.real @ a, "V": V_basis @ a, "d_omega": dF @ a, "d_omega0": s @ a}


def ccf_residual(a: torch.Tensor, lam: torch.Tensor, y: torch.Tensor, n_j: int, n_m: int) -> torch.Tensor:
    """Ω + ((1+λ)y − V)Ω′ − Ω HΩ at the collocation points, scaled by (1 + y²)^{β/2} so the far field counts as
    much as the core, plus the normalisation Ω′(0) = −1 (weight 10): (N + 1,)."""
    f = ccf_fields(a, lam, y, n_j, n_m)
    R = f["omega"] + ((1 + lam) * y - f["V"]) * f["d_omega"] - f["omega"] * f["h_omega"]
    R = R * (1 + y**2) ** (0.5 / (1 + lam))
    return torch.cat([R, 10 * (f["d_omega0"] + 1)[None]])


# ---------------------------------------------------------------- batched residuals with their Jacobians
def burgers_batch(C: torch.Tensor, lam: torch.Tensor, r: torch.Tensor, level: int, need_jac: bool = True) -> tuple:
    """Residual vectors R (B, M) of a batch of candidate profiles C (B, 3K) at exponents λ (B,), and their exact
    Jacobians J = ∂R/∂C (B, M, 3K). U = y(−r + φ·c) is linear in c, so R is quadratic and J has a closed form.
    Rows: the profile equation at the collocation points r (N,) (weighted by r ≈ 1/|U| far away); U(2) = −1, which
    fixes C = 1 in y = −U − C U^{2i+3}; and for level ≥ 1 the constraints c₃ = 0 (level 2: also c₅ = 0).
    With need_jac=False only R is computed (J is None)."""
    L = tensors.L_BURGERS
    n_modes = C.shape[-1] // tensors.N_FAMILIES
    g = (1 / (2 * (1 + lam)))[:, None]  # (B, 1)
    y = L * (r ** (-1 / g) - 1).clamp_min(0).sqrt()  # (B, N): the points follow λ
    phi, dphi = tensors.burgers_features(r, g, n_modes)  # (B, N, 3K)
    P = -r + torch.einsum("bnk,bk->bn", phi, C)
    dP = -1 + torch.einsum("bnk,bk->bn", dphi, C)
    drdy = -2 * g * y * r / (L**2 + y**2)
    U, dU = y * P, P + y * dP * drdy
    R = burgers_residual(y, U, dU, lam[:, None]) * r
    if need_jac:
        JU = y[..., None] * phi  # (B, N, 3K)
        JdU = phi + (y * drdy)[..., None] * dphi
        J = ((-lam[:, None] + dU)[..., None] * JU + ((1 + lam[:, None]) * y + U)[..., None] * JdU) * r[:, None]
    r2 = (1 + (2 / L) ** 2) ** (-g)  # (B, 1) the point y = 2
    phi2, _ = tensors.burgers_features(r2, g, n_modes)  # (B, 1, 3K)
    rows = [R, 20 * (-r2 + torch.einsum("bnk,bk->bn", phi2, C) + 0.5)]
    jac = [J, 20 * phi2] if need_jac else []
    if level >= 1:  # c₃L² = −g p₁ and c₅L⁴ = g((g+1)p₁ + g p₂)/2 with p_m = P^{(m)}(1)
        d = tensors.feature_derivatives_at_one(n_modes, g[:, 0])  # (B, 2, 3K)
        p1, p2 = (d[:, 0] * C).sum(-1) - 1, (d[:, 1] * C).sum(-1)
        rows.append(10 * (-g[:, 0] * p1)[:, None])
        jac.append(10 * (-g * d[:, 0])[:, None])
        if level >= 2:
            rows.append(10 * (g[:, 0] * ((g[:, 0] + 1) * p1 + g[:, 0] * p2) / 2)[:, None])
            jac.append(10 * (g * ((g + 1) * d[:, 0] + g * d[:, 1]) / 2)[:, None])
    return torch.cat(rows, 1), torch.cat(jac, 1) if need_jac else None


def ccf_batch(A: torch.Tensor, lam: torch.Tensor, y: torch.Tensor, n_j: int, n_m: int, need_jac: bool = True) -> tuple:
    """CCF residual vectors R (B, N + 1) and exact Jacobians ∂R/∂a (B, N + 1, K) for a batch of coefficient vectors
    A (B, K) at exponents λ (B,). Ω, HΩ, V and Ω′ are all linear in a, so R is quadratic in a."""
    s = torch.stack([tensors.ccf_exponents(1 / (1 + lv), n_j, n_m) for lv in lam])[:, None, :]  # (B, 1, K)
    z = (1 - 1j * y)[None, :, None]  # (1, N, 1)
    F = z ** (-s)  # (B, N, K)
    one = (s - 1).abs() < 1e-12  # s = 1: ∫ 1/(1+y²) dy = arctan y
    Om, Hm = F.imag, -F.real
    Vm = torch.where(one, -torch.atan(y)[None, :, None], (z ** (1 - s)).imag / torch.where(one, 1.0, 1 - s))
    Dm = (1j * s * z ** (-s - 1)).imag
    om, h, v, d = (torch.einsum("bnk,bk->bn", M, A) for M in (Om, Hm, Vm, Dm))
    c = (1 + lam[:, None]) * y - v  # characteristic speed (B, N)
    w = (1 + y**2) ** (0.5 / (1 + lam[:, None]))  # far-field weight (B, N)
    R = (om + c * d - om * h) * w
    s0 = s[:, 0, :]  # Ω′(0) = Σ s_k a_k
    R = torch.cat([R, 10 * ((s0 * A).sum(-1) + 1)[:, None]], 1)
    if not need_jac:
        return R, None
    J = (Om + c[..., None] * Dm - d[..., None] * Vm - h[..., None] * Om - om[..., None] * Hm) * w[..., None]
    return R, torch.cat([J, 10 * s0[:, None, :]], 1)
