r"""Physics -> tensor translation: three telescope tests of extra dimensions and of a fold in the projection.

(a) Gravitational waves. In D spacetime dimensions the wave's energy spreads over a (D-2)-sphere, so the strain
falls as h ~ d^{-(D-2)/2} instead of 1/d. A GW detector, assuming 1/d, then infers the distance

    d_GW = d_* (d_EM / d_*)^{(D-2)/2}                            (no screening, d_* = 1 Mpc, Pardo et al. 2018)
    d_GW = d_EM [1 + (d_EM / R_c)^{n (D-4)/2}]^{1/n}             (screened: leakage only beyond R_c, D > 4)

The GW likelihood of d_GW is a kernel density of the posterior samples with the d^2 volume prior divided out:

    L(x) = \sum_i w_i K_h(x - d_i) / \sum_i w_i,     w_i = d_i^{-2}       (i summed over samples)

and the posterior of D marginalises the electromagnetic distance:  p(D) \propto \int L(d_GW(d, D)) N(d; d_EM, σ) dd.

(b) Black-hole shadows. For ds^2 = -f dt^2 + dr^2/f + r^2 dΩ^2 a photon of impact parameter b obeys
(du/dφ)^2 = 1/b^2 - u^2 f(1/u), u = 1/r; with f = 1 - r_s/r, u'' = -u + (3/2) r_s u^2. The shadow edge is
b_c = min_r r/\sqrt{f(r)}: r_ph = 3GM/c^2, b_c = \sqrt{27} GM/c^2. The fold (P4, the author's modelling choice
here): the projected geometry is Schwarzschild down to a fold at r_f, and light that reaches the fold never returns.
Then b_edge = max(b_c, r_f/\sqrt{1 - r_s/r_f}) and δ = b_edge/b_c - 1. A Kerr hole's shadow (Bardeen 1973) is
slightly smaller; its area-equivalent radius gives the spin band.

(c) Supernovae. Dvali & Turner (2003): H^2 - H^α / r_c^{2-α} = 8πGρ/3, i.e. for a flat universe
E^2 = Ω_m (1+z)^3 + (1 - Ω_m) E^α. α = 0 is ΛCDM, α = 1 is the self-accelerating DGP brane world (gravity leaking
into a fifth dimension beyond r_c).

Distances in Mpc, radii in units of r_s (or GM/c^2 for Kerr, where stated).
"""

import math

import torch

C_KMS = 299792.458
B_SCHW = math.sqrt(27) / 2  # Schwarzschild shadow radius in units of r_s = 2GM/c^2


# ---------------------------------------------------------------- (a) gravitational-wave leakage
def gw_distance(d_em, D, R_c=None, n: float = 1.0, d_scale: float = 1.0):
    """d_GW the detector infers for a source at d_EM if gravity spreads into D dimensions.  broadcasts, Mpc."""
    if R_c is None:
        return d_scale * (d_em / d_scale) ** ((D - 2) / 2)
    return d_em * (1 + (d_em / R_c) ** (n * (D - 4) / 2)) ** (1 / n)


def strain_amplitude(d, D, R_c=None, n: float = 1.0):
    """h(d) in units of h at 1 Mpc for 4D gravity: the detector reads 1/d_GW."""
    return 1 / gw_distance(d, D, R_c, n)


def scott_bandwidth(d: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted Scott's rule h = σ_w n_eff^{-1/5}, n_eff = (Σw)^2 / Σw^2."""
    p = w / w.sum()
    mean = (p * d).sum()
    sd = ((p * (d - mean) ** 2).sum()).sqrt()
    n_eff = 1 / (p**2).sum()
    return sd * n_eff ** (-0.2)


def gw_likelihood(x: torch.Tensor, d: torch.Tensor, w: torch.Tensor, h, chunk: int = 2048) -> torch.Tensor:
    r"""L(x) = Σ_i w_i N(x; d_i, h) / Σ_i w_i: einsum over samples i.  x: (...) -> (...); integrates to 1 over x.

    Chunked over x only to bound memory (x points × samples)."""
    flat = x.reshape(-1)
    p = w / w.sum()
    out = []
    for s in range(0, flat.numel(), chunk):
        z = (flat[s : s + chunk, None] - d[None, :]) / h  # (X, S)
        out.append(torch.einsum("xs,s->x", torch.exp(-0.5 * z**2), p) / (h * math.sqrt(2 * math.pi)))
    return torch.cat(out).reshape(x.shape)


def gw_log_likelihood(x: torch.Tensor, d: torch.Tensor, w: torch.Tensor, h) -> torch.Tensor:
    """log L(x) by logsumexp over samples: finite even far out in the tails (for gradient-based fits).  (...)"""
    z = (x[..., None] - d) / h  # (..., S)
    logp = torch.log(w / w.sum())
    return torch.logsumexp(-0.5 * z**2 + logp, -1) - torch.log(h * math.sqrt(2 * math.pi))


def interp(x: torch.Tensor, grid: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Linear interpolation on a uniform grid, zero outside it.  x: (...) -> (...)"""
    step = grid[1] - grid[0]
    t = (x - grid[0]) / step
    k = t.floor().long().clamp(0, len(grid) - 2)
    s = t - k
    v = values[k] * (1 - s) + values[k + 1] * s
    return torch.where((t >= 0) & (t <= len(grid) - 1), v, torch.zeros_like(v))


def gaussian(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


# ---------------------------------------------------------------- (b) shadows and the fold
def schwarzschild_f(r: torch.Tensor, rs: torch.Tensor) -> torch.Tensor:
    return 1 - rs / r


def critical_impact_parameter(rs: torch.Tensor, f=schwarzschild_f, n: int = 20001):
    """b_c = min_r r / sqrt(f(r)) over r in (r_s, 4 r_s] and the photon-sphere radius (grid + parabola).  (B,)"""
    r = rs[:, None] * torch.linspace(1.001, 4.0, n, dtype=rs.dtype, device=rs.device)  # (B, n)
    b = r / f(r, rs[:, None]).sqrt()
    k = b.argmin(1).clamp(1, n - 2)
    idx = torch.arange(len(rs), device=rs.device)
    y0, y1, y2 = b[idx, k - 1], b[idx, k], b[idx, k + 1]
    step = r[:, 1] - r[:, 0]
    shift = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)  # vertex of the parabola through 3 points
    return y1 - 0.25 * (y0 - y2) * shift, r[idx, k] + shift * step


def photon_rhs(y: torch.Tensor) -> torch.Tensor:
    """d/dφ (u, u') = (u', -u + 3/2 u^2) for Schwarzschild with r_s = 1.  y: (..., 2)"""
    u, v = y[..., 0], y[..., 1]
    return torch.stack([v, -u + 1.5 * u**2], -1)


def fold_shadow_radius(r_f: torch.Tensor) -> torch.Tensor:
    """Shadow radius (units r_s) with a fold at r_f (units r_s; r_f <= 1 means no fold outside the horizon)."""
    rf = r_f.clamp_min(1.0 + 1e-12)
    graze = rf / (1 - 1 / rf).sqrt()  # impact parameter of the ray that just touches r_f
    return torch.where(rf > 1.5, graze, torch.full_like(rf, B_SCHW))


def fold_delta(r_f: torch.Tensor) -> torch.Tensor:
    """Fractional shadow change δ = b_edge / b_c - 1 caused by the fold."""
    return fold_shadow_radius(r_f) / B_SCHW - 1


def fold_bound(allowed: torch.Tensor) -> torch.Tensor:
    """Largest r_f/r_s with δ(r_f) <= allowed: the largest root of r^3 - B^2 r + B^2 = 0, B = b_c (1 + allowed).

    The cubic is r^2 / (1 - 1/r) = B^2 rearranged; its largest root by the trigonometric formula. allowed <= 0
    leaves only r_f <= 1.5 (inside the photon sphere, invisible to the shadow)."""
    B2 = (B_SCHW * (1 + allowed.clamp_min(0))) ** 2
    p, q = -B2, B2
    arg = (3 * q / (2 * p) * (-3 / p).sqrt()).clamp(-1, 1)
    return 2 * (-p / 3).sqrt() * torch.cos(torch.arccos(arg) / 3)


def kerr_delta(a: torch.Tensor, inc_deg: torch.Tensor, n: int = 4000) -> torch.Tensor:
    r"""Fractional change of the Kerr shadow's area-equivalent radius from \sqrt{27} M.  a: (A,), inc: (I,) -> (A, I)

    Bardeen's curve (M = 1): photon orbits r in [r_1, r_2] have ξ = -(r^3 - 3r^2 + a^2 r + a^2)/(a(r - 1)) and
    η = r^3 (4a^2 - r(r - 3)^2)/(a^2 (r - 1)^2); on the sky α = -ξ/\sin θ, β^2 = η + a^2\cos^2θ - ξ^2\cot^2θ.
    Area = 2 \int β dα (trapezoid along r, β = 0 where the orbit is not visible)."""
    a = a[:, None, None]
    th = torch.deg2rad(inc_deg).clamp(1e-3, math.pi / 2)[None, :, None]
    r1 = 2 * (1 + torch.cos(2 / 3 * torch.arccos(-a)))
    r2 = 2 * (1 + torch.cos(2 / 3 * torch.arccos(a)))
    s = 0.5 - 0.5 * torch.cos(torch.linspace(0, math.pi, n, dtype=a.dtype, device=a.device))  # dense at the ends
    r = r1 + (r2 - r1) * s  # (A, 1, n)
    xi = -(r**3 - 3 * r**2 + a**2 * r + a**2) / (a * (r - 1))
    eta = r**3 * (4 * a**2 - r * (r - 3) ** 2) / (a**2 * (r - 1) ** 2)
    alpha = -xi / torch.sin(th)
    beta = (eta + a**2 * torch.cos(th) ** 2 - xi**2 / torch.tan(th) ** 2).clamp_min(0).sqrt()
    area = 2 * torch.trapezoid(beta, alpha, dim=-1).abs()
    return (area / math.pi).sqrt() / math.sqrt(27) - 1


# ---------------------------------------------------------------- (c) supernovae and leakage
def leaky_expansion(z: torch.Tensor, Om, alpha, n_iter: int = 40) -> torch.Tensor:
    """E(z) solving E^2 = Ω_m (1+z)^3 + (1 - Ω_m) E^α by Newton's method.  z: (Z,), Om, alpha: (...) -> (..., Z)"""
    Om, alpha = torch.as_tensor(Om, dtype=z.dtype)[..., None], torch.as_tensor(alpha, dtype=z.dtype)[..., None]
    m = Om * (1 + z) ** 3
    E = (m + 1 - Om).sqrt()  # the ΛCDM value as the start
    for _ in range(n_iter):
        g = E**2 - m - (1 - Om) * E**alpha
        dg = 2 * E - alpha * (1 - Om) * E ** (alpha - 1)
        E = E - g / dg
    return E


def distance_modulus(z: torch.Tensor, nodes: torch.Tensor, Om, alpha) -> torch.Tensor:
    """μ(z) - 5 log10(c/H0 / Mpc) - 25: the shape of the Hubble diagram (H0 and M are one offset).  -> (..., N)"""
    f = 1 / leaky_expansion(nodes, Om, alpha)  # (..., Z) dχ/dz
    h = nodes[1] - nodes[0]
    chi = torch.cat([torch.zeros_like(f[..., :1]), (0.5 * h * (f[..., 1:] + f[..., :-1])).cumsum(-1)], -1)
    t = z / h
    k = t.floor().long().clamp(0, len(nodes) - 2)
    s = t - k
    chi_z = chi[..., k] * (1 - s) + chi[..., k + 1] * s
    return 5 * torch.log10((1 + z) * chi_z)


def chi2_offset_marginal(mu_shape: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    r"""min over a constant offset M of Σ_i ((μ_i - shape_i - M)/σ_i)^2 = Σ w r^2 - (Σ w r)^2 / Σ w (i summed)."""
    w = sigma**-2
    r = mu - mu_shape
    return torch.einsum("...i,i->...", r**2, w) - torch.einsum("...i,i->...", r, w) ** 2 / w.sum()
