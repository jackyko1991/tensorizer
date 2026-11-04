r"""Physics → tensor translation for a Keplerian orbit seen on the sky.

Units: time in years, lengths in AU, angles on the sky in milliarcseconds (mas), masses in M☉, distance R0 in kpc.
An angle of a mas at R0 kpc is a·R0 AU (1″ at 1 pc is 1 AU), so the semi-major axis is a_AU = a_mas · R0_kpc and
Kepler's third law reads P = 2π √(a³/GM), with GM ≈ 4π² AU³/yr² per solar mass.

Orbit in its own plane (eccentric anomaly E from Kepler's equation M = E − e sin E, mean anomaly M = 2π(t − t_p)/P):

    X = a (cos E − e),  Y = a √(1 − e²) sin E,   Ẋ = −a n sin E/(1 − e cos E),  Ẏ = a n √(1 − e²) cos E/(1 − e cos E)

Rotated onto the sky by R = R_z(Ω) R_x(i) R_z(ω):   r_{bti} = R_{bij} q_{btj}   (j summed), with i = (north = ΔDec,
east = ΔRA, line of sight away from us; the usual visual-binary convention). RV = v_z, positive = receding.
"""

import math

import torch

GM_SUN = 1.32712440018e20  # m³/s²
AU = 1.495978707e11  # m
YEAR = 365.25 * 86400.0  # Julian year, s
C = 299_792_458.0  # m/s
AU_PER_YR_KMS = AU / YEAR / 1000.0  # 1 AU/yr = 4.74 km/s
RV_SIGN = 1.0  # orbit-astronomy convention: the third axis points away from us, so RV = +v_z


def gm_au3_yr2(mass_msun: torch.Tensor) -> torch.Tensor:
    """GM in AU³/yr² for a mass in M☉ (≈ 4π² per M☉)."""
    return GM_SUN * mass_msun * YEAR**2 / AU**3


def semi_major_axis_au(p: torch.Tensor) -> torch.Tensor:
    """a_AU = a_mas · R0_kpc.   p (B, 8) = [a_mas, e, i°, Ω°, ω°, t_p, M/10⁶ M☉, R0_kpc] → (B,)"""
    return p[:, 0] * p[:, 7]


def period(p: torch.Tensor) -> torch.Tensor:
    """Kepler's third law P = 2π √(a³/GM) in years: (B,)."""
    return 2 * math.pi * torch.sqrt(semi_major_axis_au(p) ** 3 / gm_au3_yr2(p[:, 6] * 1e6))


def kepler_solve(mean: torch.Tensor, e: torch.Tensor, n_iter: int = 30) -> torch.Tensor:
    """E with E − e sin E = M, for any broadcastable shapes, by vectorised Newton iterations.

    The iterations run on detached tensors; one final Newton step on the live tensors then carries the exact
    implicit-function derivatives (dE/dM = 1/(1 − e cos E), dE/de = sin E/(1 − e cos E)) for autograd.
    """
    m0, e0 = mean.detach(), e.detach()
    big_e = m0 + 0.85 * e0 * torch.sign(torch.sin(m0))  # a start that converges for every e < 1
    for _ in range(n_iter):
        big_e = big_e - (big_e - e0 * torch.sin(big_e) - m0) / (1 - e0 * torch.cos(big_e))
    big_e = big_e.detach()
    return big_e - (big_e - e * torch.sin(big_e) - mean) / (1 - e * torch.cos(big_e))


def rotation(p: torch.Tensor) -> torch.Tensor:
    """R = R_z(Ω) R_x(i) R_z(ω) from the orbital plane to (north, east, away from us): (B, 3, 3)."""
    inc, big_om, om = (torch.deg2rad(p[:, k]) for k in (2, 3, 4))
    zero, one = torch.zeros_like(inc), torch.ones_like(inc)

    def rz(a):
        return torch.stack([torch.cos(a), -torch.sin(a), zero, torch.sin(a), torch.cos(a), zero, zero, zero, one], -1)

    rx = torch.stack([one, zero, zero, zero, torch.cos(inc), -torch.sin(inc), zero, torch.sin(inc), torch.cos(inc)], -1)
    shape = (-1, 3, 3)
    return torch.einsum("bij,bjk,bkl->bil", rz(big_om).view(shape), rx.view(shape), rz(om).view(shape))


def state_vectors(p: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Position (AU) and velocity (AU/yr) in (north, east, away from us).   p (B, 8), t (T,) → (B, T, 3) each"""
    a, e, tp = semi_major_axis_au(p)[:, None], p[:, 1:2], p[:, 5:6]
    n = 2 * math.pi / period(p)[:, None]  # mean motion, rad/yr
    big_e = kepler_solve(n * (t - tp), e)  # (B, T)
    cos_e, sin_e, root = torch.cos(big_e), torch.sin(big_e), torch.sqrt(1 - e**2)
    denom = 1 - e * cos_e
    zeros = torch.zeros_like(big_e)
    q = torch.stack([a * (cos_e - e), a * root * sin_e, zeros], -1)  # (B, T, 3) in the orbital plane
    dq = torch.stack([-a * n * sin_e / denom, a * n * root * cos_e / denom, zeros], -1)
    rot = rotation(p)
    return torch.einsum("bij,btj->bti", rot, q), torch.einsum("bij,btj->bti", rot, dq)  # j summed


def observables(p: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """What the telescopes measure: ΔRA (mas), ΔDec (mas), radial velocity (km/s), each (B, T)."""
    r, v = state_vectors(p, t)
    to_mas = 1.0 / p[:, 7:8]  # AU → mas at R0 kpc
    return r[..., 1] * to_mas, r[..., 0] * to_mas, RV_SIGN * v[..., 2] * AU_PER_YR_KMS


def precession_per_orbit(mass_msun: torch.Tensor, a_au: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    r"""Schwarzschild periapsis advance Δφ = 6πGM / (c² a (1 − e²)) in radians per orbit."""
    return 6 * math.pi * GM_SUN * mass_msun / (C**2 * a_au * AU * (1 - e**2))


def pericentre_redshift_kms(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Gravitational redshift GM/(c r_p) and transverse Doppler v_p²/(2c) at pericentre, as velocities (km/s)."""
    a = semi_major_axis_au(p) * AU
    e = p[:, 1]
    gm = GM_SUN * p[:, 6] * 1e6
    r_p = a * (1 - e)
    v_p2 = gm / a * (1 + e) / (1 - e)  # vis-viva at pericentre
    return gm / (r_p * C) / 1000, v_p2 / (2 * C) / 1000
