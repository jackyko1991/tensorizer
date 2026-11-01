"""Build the concrete tensors: energy grid E_e, basis attenuation μ_{em}, tube spectrum S_e, bin response R_{be},
the phantom's basis-material maps a_{myx} and the projection angles θ.

Mass attenuation coefficients (cm²/g) are rounded NIST XCOM values (the same sources as xray_attenuation),
interpolated log-log onto a 1 keV grid. Iodine has its K-edge at 33.17 keV, so its table lists that energy
twice: the value just below and just above the edge. Good enough to see the physics, not for clinical use.
"""

import math

import torch

MATERIALS = ["water", "bone", "iodine"]  # basis materials m = 0, 1, 2
K_EDGE = 33.169  # keV, iodine K-shell binding energy

# (energy keV, μ/ρ cm²/g) tables
WATER = ([20, 30, 40, 50, 60, 80, 100, 150], [0.8096, 0.3756, 0.2683, 0.2269, 0.2059, 0.1837, 0.1707, 0.1505])
BONE = ([20, 30, 40, 50, 60, 80, 100, 150], [4.001, 1.331, 0.6655, 0.4242, 0.3148, 0.2229, 0.1855, 0.1480])
IODINE = (
    [20, 30, K_EDGE, K_EDGE, 40, 50, 60, 80, 100, 120, 150],
    [25.4, 8.65, 6.55, 36.0, 22.1, 12.3, 7.6, 3.5, 1.94, 1.2, 0.69],
)
ALUMINIUM = ([20, 30, 40, 50, 60, 80, 100, 150], [3.441, 1.128, 0.5685, 0.3681, 0.2778, 0.2018, 0.1704, 0.1378])
RHO_WATER, RHO_BONE, RHO_AL = 1.0, 1.92, 2.699  # g/cm³ (ICRU cortical bone)

FOV_HALF_WIDTH = 12.0  # cm: the image grid [-1, 1]² covers a 24 cm field of view
IODINE_INSERTS = [2.5, 5.0, 10.0, 15.0]  # mg/mL iodine dissolved in water
MATCHED_IODINE = 10.0  # the bone insert is diluted until it looks like this iodine insert on conventional CT


def energies(kvp: float, device: torch.device, dtype=torch.float64) -> torch.Tensor:
    """Photon energies E_e = 20, 21, … kVp−1 keV: shape (E,)."""
    return torch.arange(20.0, float(kvp), 1.0, device=device, dtype=dtype)


def loglog(e: torch.Tensor, table: tuple[list, list]) -> torch.Tensor:
    """Interpolate μ/ρ linearly in (log E, log μ). A repeated table energy is an absorption edge:
    searchsorted(right=True) puts energies above it in the interval that starts with the post-edge value."""
    e_tab = torch.tensor(table[0], dtype=e.dtype, device=e.device)
    mu_tab = torch.tensor(table[1], dtype=e.dtype, device=e.device)
    i = torch.searchsorted(e_tab, e, right=True).clamp(1, len(e_tab) - 1)  # table interval [i-1, i]
    lo, hi = torch.log(e_tab[i - 1]), torch.log(e_tab[i])
    w = (torch.log(e) - lo) / (hi - lo)
    return torch.exp((1 - w) * torch.log(mu_tab[i - 1]) + w * torch.log(mu_tab[i]))


def basis_attenuation(e: torch.Tensor) -> torch.Tensor:
    """μ_{em} of shape (E, 3): water and bone in 1/cm per (cm of material), iodine in 1/cm per (mg/cm³).

    A pixel with water fraction a_w, bone fraction a_b and c mg/mL iodine has μ_e = μ_{em} a_m.
    """
    water = loglog(e, WATER) * RHO_WATER
    bone = loglog(e, BONE) * RHO_BONE
    iodine = loglog(e, IODINE) / 1000.0  # (cm²/g)·(mg/cm³)/(1000 mg/g)
    return torch.stack([water, bone, iodine], dim=1)


def spectrum(kvp: float, e: torch.Tensor, filter_mm_al: float = 3.0) -> torch.Tensor:
    """S_e: Kramers' photon-number spectrum (kVp − E)/E behind an aluminium filter; sums to 1. (E,)"""
    kramers = (kvp - e).clamp_min(0.0) / e
    filtered = kramers * torch.exp(-loglog(e, ALUMINIUM) * RHO_AL * filter_mm_al / 10.0)
    return filtered / filtered.sum()


def bin_response(e: torch.Tensor, thresholds: list[float], sigma_kev: float) -> torch.Tensor:
    """R_{be}: probability that a photon of energy E_e is counted in bin b (thresholds T_b ≤ E' < T_{b+1}).

    The detector measures E' = E + Gaussian noise of width σ, so R_{be} = Φ((T_{b+1} − E)/σ) − Φ((T_b − E)/σ),
    with T_{B} = ∞. Photons below the lowest threshold are not counted (that threshold rejects electronic noise).
    """
    edges = torch.tensor([*thresholds, math.inf], dtype=e.dtype, device=e.device)  # (B+1,)
    cdf = 0.5 * (1 + torch.erf((edges[:, None] - e[None, :]) / (math.sqrt(2) * max(sigma_kev, 1e-3))))
    return cdf[1:] - cdf[:-1]  # (B, E)


def matched_bone_fraction(mu: torch.Tensor, s: torch.Tensor, e: torch.Tensor, conc: float) -> float:
    """Bone volume fraction w that an energy-integrating scanner cannot tell from `conc` mg/mL iodine.

    Use the spectrum hardened by 10 cm of water, energy-weighted as the detector weights it, and solve
    w μ̄_bone + (1 − w) μ̄_water = μ̄_water + c μ̄_iodine for w.
    """
    weight = s * e * torch.exp(-mu[:, 0] * 10.0)
    weight = weight / weight.sum()
    mw, mb, mi = (weight @ mu).tolist()
    return conc * mi / (mb - mw)


def grid(n: int, device: torch.device, dtype=torch.float64) -> torch.Tensor:
    """Pixel centres in [-1, 1] (units of the FOV half width): (N,). The detector bins s use the same grid."""
    return (torch.arange(n, device=device, dtype=dtype) + 0.5) * (2.0 / n) - 1.0


def phantom(x: torch.Tensor, bone_fraction: float):
    """A 20 cm water cylinder with five 3.2 cm inserts on a ring: four iodine concentrations and one diluted bone.

    Returns the basis maps a_{myx} (3, N, N): water fraction, bone fraction, iodine mg/mL;
    the insert masks (5, N, N) bool (slightly shrunk, for ROI statistics) and their labels.
    """
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    r = FOV_HALF_WIDTH
    maps = torch.zeros(3, len(x), len(x), dtype=x.dtype, device=x.device)
    body = xx**2 + yy**2 <= (10.0 / r) ** 2
    maps[0][body] = 1.0
    labels = [f"I {c:g}" for c in IODINE_INSERTS] + ["bone"]
    rois = []
    for k in range(5):
        phi = math.pi / 2 + 2 * math.pi * k / 5
        cx, cy = 5.5 / r * math.cos(phi), 5.5 / r * math.sin(phi)
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        inside = d2 <= (1.6 / r) ** 2
        if k < 4:
            maps[2][inside] = IODINE_INSERTS[k]  # iodine dissolved in water: water fraction stays 1
        else:
            maps[0][inside] = 1.0 - bone_fraction
            maps[1][inside] = bone_fraction
        rois.append(d2 <= (1.1 / r) ** 2)
    return maps, torch.stack(rois), labels


def angles(n: int, device: torch.device, dtype=torch.float64) -> torch.Tensor:
    """θ_a = π a / A: a half turn of parallel-beam projections. (A,)"""
    return torch.arange(n, device=device, dtype=dtype) * (math.pi / n)
