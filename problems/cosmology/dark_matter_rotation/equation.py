r"""Physics → tensor translation: circular speeds of baryons and halos, and MOND.

In a disk galaxy the circular speed at radius r balances gravity, V²/r = g(r). Independent mass components add
their accelerations, so their squared speeds add (in quadrature):

    V²_{gr} = V²_gas + Υ_d V²_disk + Υ_b V²_bulge + V²_halo           (g = galaxy, r = radius)

SPARC gives V_disk and V_bulge for a mass-to-light ratio Υ = 1, so Υ just rescales V². Halo models:

    NFW:      ρ = ρ_s / (x (1+x)²),        V² = V_s² [ln(1+x) − x/(1+x)] / x,             x = r/r_s, V_s² = 4πGρ_s r_s²
    Burkert:  ρ = ρ_0 / ((1+x)(1+x²)),     V² = V_0² [ln(1+x²) + 2 ln(1+x) − 2 arctan x] / x,  V_0² = πGρ_0 r_0²

MOND / radial acceleration relation (McGaugh, Lelli & Schombert 2016): no halo, but the observed acceleration is a
universal function of the Newtonian baryonic one,  g_obs = g_bar / (1 − e^{−√(g_bar/a₀)}),  so for g_bar ≪ a₀
g_obs → √(g_bar a₀) and V⁴ = G M a₀ (the baryonic Tully–Fisher relation).
"""

import torch

G_NEWTON = 4.30091e-6  # kpc (km/s)² / M_sun


def baryonic_v2(gas, disk, bul, ups_d, ups_b):
    """V²_bar = V_gas|V_gas| + Υ_d V_disk|V_disk| + Υ_b V_bul|V_bul|, (G, R). Υ: (G,). SPARC signs V < 0 where a
    component pulls outwards (a central gas hole), so V|V| keeps that sign."""
    return gas * gas.abs() + ups_d[:, None] * disk * disk.abs() + ups_b[:, None] * bul * bul.abs()


def nfw_velocity(r, V_s, r_s):
    """Circular speed of an NFW halo, V² = V_s² [ln(1+x) − x/(1+x)]/x with x = r/r_s."""
    x = r / r_s
    return V_s * torch.sqrt((torch.log1p(x) - x / (1 + x)) / x)


def burkert_velocity(r, V0, r0):
    """Circular speed of a Burkert (cored) halo, V² = V_0² [ln(1+x²) + 2 ln(1+x) − 2 arctan x]/x, x = r/r_0."""
    x = r / r0
    return V0 * torch.sqrt((torch.log1p(x**2) + 2 * torch.log1p(x) - 2 * torch.atan(x)) / x)


def point_mass_velocity(r, M):
    """Kepler: V = √(G M / r)."""
    return torch.sqrt(G_NEWTON * M / r)


def exponential_disk_velocity(R, mass, Rd):
    r"""Thin exponential disk Σ = Σ_0 e^{−R/R_d} (Freeman 1970): V² = 4πGΣ_0 R_d y² [I_0K_0 − I_1K_1](y), y = R/2R_d.

    Exponentially scaled Bessel functions (i0e · k0e = I_0 K_0) keep the products finite at large y.
    """
    sigma0 = mass / (2 * torch.pi * Rd**2)
    y = R / (2 * Rd)
    bessel = torch.special.i0e(y) * torch.special.scaled_modified_bessel_k0(y)
    bessel = bessel - torch.special.i1e(y) * torch.special.scaled_modified_bessel_k1(y)
    return torch.sqrt(4 * torch.pi * G_NEWTON * sigma0 * Rd * y**2 * bessel)


def total_velocity(V2_bar, V2_halo):
    """Squares add: V = √(V²_bar + V²_halo)."""
    return torch.sqrt((V2_bar + V2_halo).clamp(min=0))


def mond_acceleration(g_bar, a0):
    """Radial acceleration relation g_obs = g_bar / (1 − exp(−√(g_bar/a₀))) (McGaugh et al. 2016)."""
    g = g_bar.clamp(min=1e-6)
    return g / (-torch.expm1(-torch.sqrt(g / a0)))


def model_v2(model: str, p: dict, data: dict):
    """Model V² for every galaxy and radius, (G, R). p holds the positive parameters of that model."""
    R = data["R"]
    V2_bar = baryonic_v2(data["gas"], data["disk"], data["bul"], p["ups_d"], p["ups_b"])
    if model == "baryons":
        return V2_bar
    if model == "nfw":
        return V2_bar + nfw_velocity(R, p["V_s"][:, None], p["r_s"][:, None]) ** 2
    if model == "burkert":
        return V2_bar + burkert_velocity(R, p["V_s"][:, None], p["r_s"][:, None]) ** 2
    return mond_acceleration(V2_bar / R, p["a0"]) * R  # MOND: V² = g_obs r


def chi2(V2_model, data):
    """χ²_g = Σ_r ((V_obs − V_model)/σ)² over the real points of each galaxy, (G,)."""
    resid = (data["V"] - total_velocity(V2_model, torch.zeros_like(V2_model))) / data["err"]
    return (resid**2 * data["mask"]).sum(1)
