r"""Physics → tensor translation.

The Kerr–Newman metric (Newman et al. 1965) is the unique stationary black hole of Einstein–Maxwell theory with
mass M, angular momentum J = aM and charge Q. In Boyer–Lindquist coordinates (t, r, θ, φ), units G = c = 1
(and Gaussian units for Q), with

    Σ = r^2 + a^2\cos^2θ,        Δ = r^2 - 2Mr + a^2 + Q^2,

the (t, φ) block of the metric is

    g_{tt} = -\left(1 - \frac{2Mr - Q^2}{Σ}\right),\quad g_{tφ} = -\frac{a\sin^2θ\,(2Mr - Q^2)}{Σ},\quad
    g_{φφ} = \frac{(r^2 + a^2)^2 - a^2Δ\sin^2θ}{Σ}\sin^2θ

(and g_{rr} = Σ/Δ, g_{θθ} = Σ). Everything here is a broadcast over a batch of (M, a, Q) at once:

    horizons       Δ = 0:        r_± = M ± \sqrt{M^2 - a^2 - Q^2}
    ergosurface    g_{tt} = 0:   r_E(θ) = M + \sqrt{M^2 - Q^2 - a^2\cos^2θ}
    frame dragging ω = -g_{tφ}/g_{φφ} = \frac{a(2Mr - Q^2)}{(r^2 + a^2)^2 - a^2Δ\sin^2θ}
    naked          a^2 + Q^2 > M^2: Δ has no real root, no horizon hides the ring singularity Σ = 0

Equatorial circular orbits (θ = π/2) have angular velocity Ω = dφ/dt solving the geodesic condition
∂_r g_{tt} + 2Ω ∂_r g_{tφ} + Ω^2 ∂_r g_{φφ} = 0, i.e.

    Ω_± = \frac{-∂_r g_{tφ} ± \sqrt{(∂_r g_{tφ})^2 - ∂_r g_{tt}\,∂_r g_{φφ}}}{∂_r g_{φφ}}

(+ prograde, − retrograde; the derivatives come from autograd). The energy per unit mass is
E = -(g_{tt} + g_{tφ}Ω)/\sqrt{N} with N = -(g_{tt} + 2g_{tφ}Ω + g_{φφ}Ω^2): the photon orbit is where N → 0
(the orbit needs the speed of light) and the ISCO is where E(r) is minimal (dE/dr = 0).
"""

import math

import torch

TOL = 1e-12  # a² + Q² within TOL·M² of M² counts as extremal, not naked


def _disc(M, a, Q):
    d = M**2 - a**2 - Q**2
    return torch.where(d.abs() < TOL * M**2, torch.zeros_like(d), d)


def is_naked(M: torch.Tensor, a: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """a² + Q² > M²: no horizon (cosmic censorship conjectures nature forbids this). Any shape → same shape."""
    return _disc(M, a, Q) < 0


def horizons(M, a, Q):
    """Outer and inner horizon r_± = M ± √(M² − a² − Q²); NaN when naked. Any (broadcast) shape."""
    root = torch.sqrt(_disc(M, a, Q))  # NaN for negative discriminant
    return M + root, M - root


def ergosphere(theta, M, a, Q, inner: bool = False):
    """Static limit g_tt = 0: r_E(θ) = M ± √(M² − Q² − a² cos²θ) (NaN where it does not exist)."""
    d = M**2 - Q**2 - (a * torch.cos(theta)) ** 2
    d = torch.where(d.abs() < TOL * M**2, torch.zeros_like(d), d)
    return M - torch.sqrt(d) if inner else M + torch.sqrt(d)


def metric_t_phi(r, theta, M, a, Q):
    """(g_tt, g_tφ, g_φφ) of Kerr–Newman in Boyer–Lindquist coordinates, broadcast over all inputs."""
    s2 = torch.sin(theta) ** 2
    sigma = r**2 + (a * torch.cos(theta)) ** 2
    delta = r**2 - 2 * M * r + a**2 + Q**2
    m = 2 * M * r - Q**2  # "mass function" 2Mr − Q²
    g_tt = -(1 - m / sigma)
    g_tp = -a * s2 * m / sigma
    g_pp = ((r**2 + a**2) ** 2 - a**2 * delta * s2) * s2 / sigma
    return g_tt, g_tp, g_pp


def frame_dragging(r, theta, M, a, Q):
    """ω = −g_tφ / g_φφ: the angular velocity dφ/dt of an observer with zero angular momentum (ZAMO)."""
    delta = r**2 - 2 * M * r + a**2 + Q**2
    return a * (2 * M * r - Q**2) / ((r**2 + a**2) ** 2 - a**2 * delta * torch.sin(theta) ** 2)


def surface_gravity(M, a, Q):
    """κ = (r₊ − r₋) / (2(r₊² + a²)): Hawking temperature T = κ/2π, zero for an extremal hole."""
    r_plus, r_minus = horizons(M, a, Q)
    return (r_plus - r_minus) / (2 * (r_plus**2 + a**2))


def kerr_isco(M, a, prograde: bool = True):
    """Bardeen–Press–Teukolsky (1972) closed form for Q = 0: r = M(3 + Z₂ ∓ √((3 − Z₁)(3 + Z₁ + 2Z₂)))."""
    x = a / M
    z1 = 1 + (1 - x**2) ** (1 / 3) * ((1 + x) ** (1 / 3) + (1 - x).clamp(min=0) ** (1 / 3))
    z2 = torch.sqrt(3 * x**2 + z1**2)
    root = torch.sqrt((3 - z1) * (3 + z1 + 2 * z2))
    return M * (3 + z2 - root) if prograde else M * (3 + z2 + root)


def circular_orbits(r, M, a, Q):
    """Equatorial circular geodesics on a radius grid. r: (B, R), M, a, Q: (B, 1).

    Returns dict of (B, R) tensors per branch ('pro', 'retro'): Omega = dφ/dt, N (> 0 ⇔ timelike), E.
    """
    r = r.detach().requires_grad_(True)
    half_pi = torch.full_like(r, math.pi / 2)
    g = metric_t_phi(r, half_pi, M, a, Q)
    # each g depends on its own r only, so the gradient of the sum is the elementwise derivative ∂_r g
    dg = [torch.autograd.grad(gi.sum(), r, retain_graph=True)[0] for gi in g]
    g_tt, g_tp, g_pp = (gi.detach() for gi in g)
    d_tt, d_tp, d_pp = dg
    root = torch.sqrt(d_tp**2 - d_tt * d_pp)
    out = {}
    for name, sign in (("pro", 1.0), ("retro", -1.0)):
        omega = (-d_tp + sign * root) / d_pp
        N = -(g_tt + 2 * g_tp * omega + g_pp * omega**2)
        E = -(g_tt + g_tp * omega) / torch.sqrt(N.clamp(min=1e-300))
        out[name] = {"Omega": omega, "N": N, "E": torch.where(N > 0, E, torch.full_like(E, math.inf))}
    return out


def equatorial_orbits(M, a, Q, n_r: int = 40001, r_max: float = 12.0):
    """Photon orbits and ISCOs (prograde and retrograde) for a batch of holes. M, a, Q: (B,) → dict of (B,).

    Grid: log-spaced r from just outside r₊ to r_max·M. NaN for naked singularities.
    """
    r_plus, _ = horizons(M, a, Q)
    r_in = torch.nan_to_num(r_plus, nan=1.0) * (1 + 1e-7)
    s = torch.linspace(0, 1, n_r, dtype=M.dtype, device=M.device)  # (R,)
    log_r = torch.log(r_in)[:, None] + s[None, :] * torch.log(r_max * M / r_in)[:, None]  # (B, R)
    r = torch.exp(log_r)
    orbits = circular_orbits(r, M[:, None], a[:, None], Q[:, None])
    naked = is_naked(M, a, Q)
    rows = torch.arange(len(M), device=M.device)
    result = {}
    for name, orb in orbits.items():
        # photon orbit: outermost grid point with N ≤ 0, then linear interpolation of N to zero
        inside = orb["N"] <= 0
        k = torch.where(inside.any(1), (inside * torch.arange(n_r, device=M.device)).amax(1), 0).clamp(max=n_r - 2)
        n0, n1 = orb["N"][rows, k], orb["N"][rows, k + 1]
        lr = log_r[rows, k] + (log_r[rows, k + 1] - log_r[rows, k]) * n0 / (n0 - n1)
        # no N ≤ 0 point: the photon orbit has merged with the horizon (extremal prograde orbits, at r = r₊)
        result[f"photon_{name}"] = torch.where(inside.any(1), torch.exp(lr), r_in)
        # ISCO: minimum of E(r), refined by a parabola through the three grid points around it
        E = orb["E"]
        j = E.argmin(1).clamp(1, n_r - 2)
        e_m, e_0, e_p = E[rows, j - 1], E[rows, j], E[rows, j + 1]
        shift = 0.5 * (e_m - e_p) / (e_m - 2 * e_0 + e_p)
        h = log_r[:, 1] - log_r[:, 0]
        result[f"isco_{name}"] = torch.exp(log_r[rows, j] + torch.nan_to_num(shift).clamp(-1, 1) * h)
    return {k: torch.where(naked, torch.full_like(v, math.nan), v) for k, v in result.items()}


def zero_angular_momentum_spiral(M, a, Q, r0: float = 10.0, n_r: int = 400):
    """A particle dropped from rest at infinity with L = 0 (E = 1) in the equatorial plane: φ(r) by quadrature.

    u^φ = −g_tφ/Δ and u^r = −√(g_φφ − Δ)/r, so dφ/dr = g_tφ r / (Δ √(g_φφ − Δ)): it is dragged around although
    it has no angular momentum (φ diverges logarithmically at r₊ in these coordinates). M, a, Q: (B,).
    Returns r (B, R) and φ (B, R), from r0 inward to 1.03 r₊.
    """
    r_plus, _ = horizons(M, a, Q)
    s = torch.linspace(0, 1, n_r, dtype=M.dtype, device=M.device)
    r = r0 * M[:, None] + (1.03 * r_plus - r0 * M)[:, None] * s[None, :]  # (B, R), decreasing
    _, g_tp, g_pp = metric_t_phi(r, torch.full_like(r, math.pi / 2), M[:, None], a[:, None], Q[:, None])
    delta = r**2 - 2 * M[:, None] * r + (a**2 + Q**2)[:, None]
    dphi_dr = g_tp * r / (delta * torch.sqrt(g_pp - delta))
    steps = 0.5 * (dphi_dr[:, 1:] + dphi_dr[:, :-1]) * r.diff(dim=1)  # trapezoid rule
    phi = torch.cat([torch.zeros_like(r[:, :1]), torch.cumsum(steps, 1)], 1)
    return r, phi
