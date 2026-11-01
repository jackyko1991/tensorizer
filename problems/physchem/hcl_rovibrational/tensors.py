"""Build the tensors: spectroscopic constants of each isotopologue, the J grid, the wavenumber grid, temperatures.

Units: wavenumbers in cm⁻¹, temperatures in K.
"""

import torch

# Approximate spectroscopic constants of H³⁵Cl (ground electronic state), rounded literature values.
HCL35 = {
    "we": 2990.9,  # ω_e, harmonic wavenumber
    "wexe": 52.8,  # ω_e x_e, anharmonicity
    "Be": 10.593,  # B_e, equilibrium rotational constant
    "ae": 0.307,  # α_e, vibration–rotation coupling
    "De": 5.3e-4,  # D_e, centrifugal distortion
}
# atomic masses (u) and natural abundances
M_H = 1.007825
ISOTOPES = [("H³⁵Cl", 34.968853, 0.758), ("H³⁷Cl", 36.965903, 0.242)]


def isotope_names() -> list[str]:
    return [name for name, _, _ in ISOTOPES]


def reduced_masses(device: torch.device) -> torch.Tensor:
    """μ_i = m_H m_Cl / (m_H + m_Cl)   (I,) in u."""
    m_cl = torch.tensor([m for _, m, _ in ISOTOPES], device=device)
    return M_H * m_cl / (M_H + m_cl)


def abundances(device: torch.device) -> torch.Tensor:
    """Natural abundance of each isotopologue   (I,)."""
    return torch.tensor([a for _, _, a in ISOTOPES], device=device)


def constants(mu: torch.Tensor) -> dict[str, torch.Tensor]:
    """Constants of every isotopologue (each (I,)), scaled from H³⁵Cl with ρ = √(μ₃₅/μ_i).

    Same potential curve, different mass: ω_e ∝ ρ, ω_e x_e ∝ ρ², B_e ∝ ρ², α_e ∝ ρ³, D_e ∝ ρ⁴.
    """
    rho = torch.sqrt(mu[0] / mu)
    powers = {"we": 1, "wexe": 2, "Be": 2, "ae": 3, "De": 4}
    return {key: HCL35[key] * rho ** powers[key] for key in powers}


def rotational_levels(J_max: int, device: torch.device) -> torch.Tensor:
    """J = 0, 1, …, J_max   (J,): the rotational quantum number of the lower (v = 0) level."""
    return torch.arange(J_max + 1, device=device, dtype=torch.get_default_dtype())


def wavenumber_grid(lo: float, hi: float, step: float, device: torch.device) -> torch.Tensor:
    """ν̃ grid of the simulated spectrum   (G,) cm⁻¹."""
    return torch.arange(lo, hi, step, device=device)


def temperatures(T_end: float, n_frames: int, device: torch.device) -> torch.Tensor:
    """A temperature ramp from 40 K up to T_end   (F,): one spectrum per animation frame."""
    return torch.linspace(40.0, max(T_end, 41.0), n_frames, device=device)
