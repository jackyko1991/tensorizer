"""Build the concrete tensors: the batch of potentials, their starting points and the observational bounds.

Units: reduced Planck mass M_pl = 1. Starting points come from the textbook slow-roll e-fold counts (Liddle & Lyth;
Martin, Ringeval & Vennin, "Encyclopaedia Inflationaris", 2014); the exact integration then measures the true N.
"""

import math

import torch

KINDS = ["m2phi2", "phi4", "starobinsky", "natural", "hilltop"]
LABELS = {
    "m2phi2": "m²φ²",
    "phi4": "λφ⁴",
    "starobinsky": "Starobinsky R²",
    "natural": "natural",
    "hilltop": "quartic hilltop",
}

# Observations (pivot k = 0.05 Mpc⁻¹):
NS_MEAN, NS_SIGMA = 0.9649, 0.0042  # Planck 2018 X (TT,TE,EE+lowE+lensing), 68 % (arXiv:1807.06211)
R_MAX = 0.036  # BICEP/Keck 2021 (BK18 + Planck + BAO), 95 % upper limit (arXiv:2110.00483)
A_S = 2.1e-9  # Planck 2018 amplitude of the scalar power spectrum, ln(10¹⁰ A_s) = 3.044


def model(kinds: list[str], f: float, mu: float, device: torch.device) -> dict:
    """One-hot rows w_{bp} (B, 5) picking each field's potential, and the constants f (natural) and μ (hilltop)."""
    index = torch.tensor([KINDS.index(k) for k in kinds], device=device)
    w = torch.nn.functional.one_hot(index, len(KINDS)).to(torch.float64)
    return {"kinds": list(kinds), "w": w, "f": f, "mu": mu}


def start_field(m: dict, N_start: float) -> torch.Tensor:
    """φ_i (B,) roughly N_start e-folds before the end of inflation, from the slow-roll count N(φ) of each potential."""
    f, mu, N = m["f"], m["mu"], N_start
    s = math.sqrt(2 / 3)
    # natural: ε_V = tan²(φ/2f)/(2f²) = 1 at the end, N = 2f² ln[sin(φ_e/2f) / sin(φ/2f)]
    phi_end_nat = 2 * f * math.atan(math.sqrt(2) * f)
    # Starobinsky: N = (3/4)[e^{sφ} − e^{sφ_e} − s(φ − φ_e)], ends at φ_e = √(3/2) ln(1 + 2/√3)
    phi_end_star = math.log(1 + 2 / math.sqrt(3)) / s
    phi_star = phi_end_star
    for _ in range(30):  # fixed-point iteration e^{sφ} = 4N/3 + e^{sφ_e} + s(φ − φ_e)
        phi_star = math.log(4 * N / 3 + math.exp(s * phi_end_star) + s * (phi_star - phi_end_star)) / s
    # quartic hilltop, x = φ/μ: N = (μ²/8)(1/x² + x² − 1/x_e² − x_e²), with ε_V = 1 at √8 x_e³/μ = 1 − x_e⁴
    x_end = 1.0
    for _ in range(60):
        x_end = max(1 - math.sqrt(8) * x_end**3 / mu, 1e-6) ** 0.25
    c = 8 * N / mu**2 + 1 / x_end**2 + x_end**2  # 1/u + u = c with u = x²
    x_hill = math.sqrt((c - math.sqrt(c * c - 4)) / 2)
    table = {
        "m2phi2": math.sqrt(4 * N + 2),  # N = (φ² − φ_e²)/4, φ_e = √2
        "phi4": math.sqrt(8 * N + 8),  # N = (φ² − φ_e²)/8, φ_e = √8
        "starobinsky": phi_star,
        "natural": 2 * f * math.asin(math.sin(phi_end_nat / (2 * f)) * math.exp(-N / (2 * f**2))),
        "hilltop": mu * x_hill,
    }
    return torch.tensor([table[k] for k in m["kinds"]], dtype=torch.float64, device=m["w"].device)


def allowed(ns: torch.Tensor, r: torch.Tensor, sigmas: float = 2) -> torch.Tensor:
    """Inside the Planck n_s band (± sigmas σ) and below the BICEP/Keck r limit (a box, not the published contour)."""
    return ((ns - NS_MEAN).abs() < sigmas * NS_SIGMA) & (r < R_MAX)
