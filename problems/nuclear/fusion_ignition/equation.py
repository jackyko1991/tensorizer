r"""Physics → tensor translation for a 0D inertial-fusion hot spot.

A hot spot of DT at temperature T, fixed density ρ and radius R (areal density ρR) with burn-up fraction φ.
Power densities (W/m³), with n = ρ/m̄ ions per m³ (half D, half T, m̄ = 2.5 u):

    P_α    = ¼ n² (1 − φ)² ⟨σv⟩(T) E_α,         E_α = 3.5 MeV          alpha heating (a fraction f_α is absorbed)
    P_br   = C_B n² √T,                        C_B = 5.34e-37 W m³ keV^-1/2   bremsstrahlung (Z = 1)
    P_pdV  = 3 p c_s / R,  p = 2 n T,  c_s = √(2T/m̄)                 expansion (PdV) work
    P_cond = 3 κ₀ T^{7/2} / R²                                        electron conduction into the cold shell

Energy balance of the hot spot, u = 3 n T (ions + electrons):

    u dT/dt / T = f_α P_α − L (P_br + P_pdV)                         L = loss multiplier
    (1/M) dM/dt = (P_cond + (1 − f_α) P_α) / u                        burn propagation: heat leaving the
                                                                      hot spot ablates cold fuel into it
    dφ/dt = ½ n (1 − φ)² ⟨σv⟩ − φ (1/M) dM/dt                        burn-up, diluted by fresh fuel

Conduction does not appear in the temperature equation: the heat it carries out is recycled as new hot
mass (the ablative hot-spot picture), so it grows ρR at fixed T instead of cooling the hot spot.
With fixed ρ, R ∝ M^{1/3}, so d ln(ρR)/dt = (1/3) d ln M/dt.

⟨σv⟩ is the Bosch–Hale (Nucl. Fusion 32, 611, 1992) parameterisation for D + T → α + n, valid 0.2–100 keV.
"""

import torch

KEV = 1.602176634e-16  # J
AMU = 1.66053907e-27  # kg
M_BAR = 2.5 * AMU  # mean DT ion mass
E_ALPHA = 3.5e3 * KEV  # J
E_FUSION = 17.6e3 * KEV  # J per reaction (alpha + neutron)
C_BREMS = 5.34e-37  # W m³ keV^-1/2
KAPPA0 = 1e14  # W m⁻¹ keV^-7/2: Spitzer electron conductivity with ln Λ ≈ 10 (approximate)
RHO_L_ALPHA = 0.3  # g/cm²: alpha range at 10 keV (illustrative)

# Bosch–Hale coefficients for T(d,n)α, ⟨σv⟩ in cm³/s with T in keV
BH = dict(
    BG=34.3827,
    mrc2=1124656.0,
    C1=1.17302e-9,
    C2=1.51361e-2,
    C3=7.51886e-2,
    C4=4.60643e-3,
    C5=1.35000e-2,
    C6=-1.06750e-4,
    C7=1.36600e-5,
)


def sigma_v(T: torch.Tensor) -> torch.Tensor:
    r"""⟨σv⟩(T) in m³/s (Bosch–Hale):  θ = T / [1 − T(C2 + T(C4 + T C6)) / (1 + T(C3 + T(C5 + T C7)))],
    ξ = (B_G²/4θ)^{1/3},  ⟨σv⟩ = C1 θ √(ξ / (m_r c² T³)) e^{−3ξ}."""
    c = BH
    T = T.clamp(0.2, 100.0)
    theta = T / (1 - T * (c["C2"] + T * (c["C4"] + T * c["C6"])) / (1 + T * (c["C3"] + T * (c["C5"] + T * c["C7"]))))
    xi = (c["BG"] ** 2 / (4 * theta)) ** (1 / 3)
    sv_cm3 = c["C1"] * theta * torch.sqrt(xi / (c["mrc2"] * T**3)) * torch.exp(-3 * xi)
    return sv_cm3 * 1e-6


def number_density(rho_gcc: float) -> float:
    """Ions per m³ for DT at density ρ in g/cm³."""
    return rho_gcc * 1e3 / M_BAR


def alpha_fraction(rho_r: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    r"""Fraction of alpha energy absorbed in the hot spot, f_α = 1 − exp(−ρR / ρℓ_α), ρℓ_α ∝ T^{5/4} (illustrative)."""
    range_alpha = RHO_L_ALPHA * (T / 10.0) ** 1.25
    return 1 - torch.exp(-rho_r / range_alpha)


def powers(T: torch.Tensor, rho_r: torch.Tensor, phi: torch.Tensor, rho_gcc: float) -> dict:
    """All power densities (W/m³) for a batch of hot spots; T in keV, ρR in g/cm², shapes (B,)."""
    n = number_density(rho_gcc)
    radius = rho_r * 10.0 / (rho_gcc * 1e3)  # ρR [g/cm²] → kg/m² (×10), divided by ρ [kg/m³] → m
    p_alpha = 0.25 * n**2 * (1 - phi) ** 2 * sigma_v(T) * E_ALPHA
    p_brems = C_BREMS * n**2 * torch.sqrt(T)
    sound = torch.sqrt(2 * T * KEV / M_BAR)
    p_pdv = 3 * (2 * n * T * KEV) * sound / radius
    p_cond = 3 * KAPPA0 * T**3.5 / radius**2
    return {"alpha": p_alpha, "brems": p_brems, "pdv": p_pdv, "cond": p_cond, "u": 3 * n * T * KEV}


def rhs(y: torch.Tensor, rho_gcc: float, loss: float, rho_r_max: float) -> torch.Tensor:
    r"""d/dt of the state y = (ln T, ln ρR, φ), shape (B, 3), in 1/s.

    ln T:   (f_α P_α − L(P_br + P_pdV)) / u
    ln ρR:  ⅓ (P_cond + (1 − f_α)P_α) / u × (1 − ρR/ρR_max)   (growth stops when the cold fuel is used up)
    φ:      ½ n (1 − φ)² ⟨σv⟩ − φ d ln M/dt
    """
    T, rho_r, phi = torch.exp(y[:, 0]), torch.exp(y[:, 1]), y[:, 2]
    p = powers(T, rho_r, phi, rho_gcc)
    f_a = alpha_fraction(rho_r, T)
    d_ln_t = (f_a * p["alpha"] - loss * (p["brems"] + p["pdv"])) / p["u"]
    d_ln_m = (p["cond"] + (1 - f_a) * p["alpha"]) / p["u"] * (1 - rho_r / rho_r_max).clamp(min=0.0)
    d_phi = 0.5 * number_density(rho_gcc) * (1 - phi) ** 2 * sigma_v(T) - phi * d_ln_m
    return torch.stack([d_ln_t, d_ln_m / 3, d_phi], dim=1)


def heating_balance(T: torch.Tensor, rho_r: torch.Tensor, rho_gcc: float, loss: float) -> torch.Tensor:
    r"""Initial self-heating margin f_α P_α − L(P_br + P_pdV) in W/m³ (> 0 ⇒ the hot spot heats up)."""
    p = powers(T, rho_r, torch.zeros_like(T), rho_gcc)
    return alpha_fraction(rho_r, T) * p["alpha"] - loss * (p["brems"] + p["pdv"])


def ideal_ignition_temperature() -> float:
    r"""T where alpha heating equals bremsstrahlung (f_α = 1, no other losses): ¼⟨σv⟩E_α = C_B √T."""
    T = torch.linspace(1.0, 20.0, 19001, dtype=torch.float64)
    margin = 0.25 * sigma_v(T) * E_ALPHA - C_BREMS * torch.sqrt(T)
    return T[int(torch.nonzero(margin > 0)[0])].item()


def confinement_time(T: torch.Tensor, rho_r: torch.Tensor, rho_gcc: float) -> torch.Tensor:
    """Expansion time R / (3 c_s) implied by the PdV term, in seconds."""
    radius = rho_r * 10.0 / (rho_gcc * 1e3)
    return radius / (3 * torch.sqrt(2 * T * KEV / M_BAR))
